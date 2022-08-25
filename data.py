import numpy as np
import tensorflow as tf
from ddsp.training.data import DataProvider
from ddsp.spectral_ops import compute_loudness
import customPath
import os
import csv
from scipy.io.wavfile import read
import pandas as pd
from tqdm import tqdm
import utils
import librosa as lr
from math import ceil
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

_max_dataset_length = 100000

### Generic dataset ###
class GenericDataset(DataProvider):
    def __init__(self, audio_length=4, sample_rate=16000, frame_rate=250):
        super().__init__(sample_rate, frame_rate)
        self.frame_length = int(self.sample_rate/self.frame_rate)
        self.audio_length = audio_length
        self.sample_rate_LB = sample_rate//2
    
    def generate_tfrecord(self, path, split, estimate_f0=False):
            file_list = [f for f in os.listdir(os.path.join(path, split)) if f.endswith('.wav')]
            with tf.io.TFRecordWriter(os.path.join(path, f'{split}.tfrecord')) as file_writer:
                for file_name in tqdm(file_list):
                    _, x = read(os.path.join(path, split, file_name))
                    x = x.astype('float64')
                    x = x/np.max(np.abs(x))

                    n_chunks = x.size//(self.sample_rate*self.audio_length)
                    n_sample_chunk = self.sample_rate*self.audio_length
                    for i_chunk in range(n_chunks+1):
                        x_chunk = x[i_chunk*n_sample_chunk:(i_chunk+1)*n_sample_chunk]
                        if x_chunk.size < n_sample_chunk:
                            x_chunk = np.concatenate((x_chunk, np.zeros((n_sample_chunk-x_chunk.size))))
                        x_chunk_LB_downsampled = lr.resample(x_chunk, orig_sr=self.sample_rate, target_sr=self.sample_rate_LB)
                        x_chunk_LB_resampled = lr.resample(x_chunk_LB_downsampled, orig_sr=self.sample_rate_LB, target_sr=self.sample_rate)
                        x_chunk_stft = lr.stft(x_chunk, n_fft=1024)
                        x_chunk_stft[:ceil(x_chunk_stft.shape[0]/2)] = np.zeros((ceil(x_chunk_stft.shape[0]/2), x_chunk_stft.shape[1]))
                        x_chunk_HB = lr.istft(x_chunk_stft, n_fft=1024)
                        note, velocity = self.get_note_velocity(file_name)                       
                        if estimate_f0:
                            f0_hz = 0
                        else:
                            f0_hz = utils.midi_to_hz(utils._PITCHES_MIDI_NUMBER[utils._PITCHES.index(note)])*np.ones((self.frame_rate*self.audio_length))
                        loudness_db = compute_loudness(x_chunk, sample_rate=self.sample_rate, frame_rate=self.frame_rate, use_tf=False)
                        loudness_db = loudness_db[:self.frame_rate*self.audio_length]
                        # loudness_db = -10*np.ones((self.frame_rate*self.audio_length))

                        record_bytes = tf.train.Example(features=tf.train.Features(feature={
                            "audio": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x_chunk_LB_resampled).numpy()])),
                            "audio_HB": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x_chunk_HB).numpy()])),
                            "audio_WB": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x_chunk).numpy()])),
                            "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_name.encode()])),
                            "chunk": tf.train.Feature(int64_list=tf.train.Int64List(value=[i_chunk])),
                            "f0_hz": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(f0_hz).numpy()])),
                            "loudness_db": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(loudness_db).numpy()])),
                        })).SerializeToString()
                        file_writer.write(record_bytes)

    def decode_tfrecord(self, record_bytes):
        batch = tf.io.parse_single_example(
            record_bytes,
            {'audio': tf.io.FixedLenFeature([], dtype=tf.string),
            'audio_WB': tf.io.FixedLenFeature([], dtype=tf.string),
            'audio_HB': tf.io.FixedLenFeature([], dtype=tf.string),
            'filename': tf.io.FixedLenFeature([], dtype=tf.string),
            'chunk': tf.io.FixedLenFeature([], dtype=tf.int64),
            'f0_hz': tf.io.FixedLenFeature([], dtype=tf.string),
            'loudness_db': tf.io.FixedLenFeature([], dtype=tf.string)}
        )

        audio = tf.ensure_shape(tf.io.parse_tensor(batch['audio'], out_type=tf.float64), shape=(int(self.audio_length*self.sample_rate)))
        audio_WB = tf.ensure_shape(tf.io.parse_tensor(batch['audio_WB'], out_type=tf.float64), shape=(int(self.audio_length*self.sample_rate)))
        audio_HB = tf.ensure_shape(tf.io.parse_tensor(batch['audio_HB'], out_type=tf.float64), shape=(int(self.audio_length*self.sample_rate)))
        filename = batch['filename']
        chunk = batch['chunk']
        f0_hz = tf.ensure_shape(tf.io.parse_tensor(batch['f0_hz'], out_type=tf.float64), shape=(int(self.audio_length*self.frame_rate)))
        loudness_db = tf.ensure_shape(tf.io.parse_tensor(batch['loudness_db'], out_type=tf.float64), shape=(int(self.audio_length*self.frame_rate)))

        return {
            'audio': audio,
            'audio_WB': audio_WB,
            'audio_HB': audio_HB,
            'f0_hz': f0_hz,
            'loudness_db': loudness_db,
            'filename': filename
        }

    def get_note_velocity(self, filename):
        raise NotImplementedError


### OrchideaSol dataset ###
class OrchideaSol(GenericDataset):
    def __init__(self, split, audio_length=4, sample_rate=16000, frame_rate=250):
        super().__init__(audio_length, sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.orchideaSOL(), f'{split}.tfrecord')

    def get_dataset(self, shuffle = False):
        if not os.path.isfile(self.path):
            print(f'\n Creating TFrecord file for split {self.split} ...')
            self.generate_tfrecord(os.path.dirname(self.path), self.split)
            print(f'{self.path} created.\n')

        dataset = tf.data.TFRecordDataset(self.path).map(self.decode_tfrecord)
        if shuffle:
            return dataset.shuffle(_max_dataset_length)
        else:
            return dataset
    
    def get_note_velocity(self, filename):
        note = filename.split('-')[2]
        velocity = filename.split('-')[3]
        return note, velocity

### OrchideaSol_tiny dataset ###
class OrchideaSolTiny(GenericDataset):
    def __init__(self, split, audio_length=4, sample_rate=16000, frame_rate=250):
        super().__init__(audio_length, sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.orchideaSOL_tiny(), f'{split}.tfrecord')

    def get_dataset(self, shuffle = False):
        if not os.path.isfile(self.path):
            print(f'\n Creating TFrecord file for split {self.split} ...')
            self.generate_tfrecord(os.path.dirname(self.path), self.split)
            print(f'{self.path} created.\n')

        dataset = tf.data.TFRecordDataset(self.path).map(self.decode_tfrecord)
        if shuffle:
            return dataset.shuffle(_max_dataset_length)
        else:
            return dataset
    
    def get_note_velocity(self, filename):
        note = filename.split('-')[2]
        velocity = filename.split('-')[3]
        return note, velocity

### MedleySolosDB Data Generator ###
class MedleySolosDB(GenericDataset):
    def __init__(self, split, audio_length=4, sample_rate=16000, frame_rate=250):
        super().__init__(audio_length, sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.medleySolosDB(), f'{split}.tfrecord')

    def get_dataset(self, shuffle = False):
        if not os.path.isfile(self.path):
            print(f'\n Creating TFrecord file for split {self.split} ...')
            self.generate_tfrecord(os.path.dirname(self.path), self.split)
            print(f'{self.path} created.\n')

        dataset = tf.data.TFRecordDataset(self.path).map(self.decode_tfrecord)
        if shuffle:
            return dataset.shuffle(_max_dataset_length)
        else:
            return dataset
    
    def get_note_velocity(self, filename):
        all_metadata = pd.read_csv(os.path.join(os.path.dirname(self.path), 'medley-solos-DB_metadata.csv'))
        uuid4 = filename[:-4].split('_')[2]
        metadata = all_metadata[all_metadata['uuid4'].str.contains(uuid4)]
        note = 'A0'
        velocity = None
        return note, velocity

### Gtzan dataset ###
class Gtzan(GenericDataset):
    def __init__(self, split, audio_length=4, sample_rate=16000, frame_rate=250):
        super().__init__(audio_length, sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.gtzan(), f'{split}.tfrecord')

    def get_dataset(self, shuffle = False):
        if not os.path.isfile(self.path):
            print(f'\n Creating TFrecord file for split {self.split} ...')
            self.generate_tfrecord(os.path.dirname(self.path), self.split)
            print(f'{self.path} created.\n')

        dataset = tf.data.TFRecordDataset(self.path).map(self.decode_tfrecord)
        if shuffle:
            return dataset.shuffle(_max_dataset_length)
        else:
            return dataset
    
    def get_note_velocity(self, filename):
        note = 'A0'
        velocity = 'mf'
        return note, velocity
import re
import numpy as np
import tensorflow as tf
from ddsp.training.data import DataProvider
import customPath
import os
import csv
from scipy.io.wavfile import read
import pandas as pd

class GenericDataset(DataProvider):
    def __init__(self, audio_length=4, sample_rate=16000, frame_rate=250, batch_size=1):
        super().__init__(sample_rate, frame_rate)
        self.frame_length = int(self.sample_rate/self.frame_rate)
        self.audio_length = audio_length
        self.batch_size = batch_size
    
    def decode_tfrecord(self, record_bytes):
        batch = tf.io.parse_single_example(
            record_bytes,
            {'audio': tf.io.FixedLenFeature([], dtype=tf.string),
            'filename': tf.io.FixedLenFeature([], dtype=tf.string),
            'chunk': tf.io.FixedLenFeature([], dtype=tf.int64),
            'instrument': tf.io.FixedLenFeature([], dtype=tf.string),
            'f0_hz': tf.io.FixedLenFeature([], dtype=tf.string),
            'loudness_db': tf.io.FixedLenFeature([], dtype=tf.string)}
        )

        audio = tf.ensure_shape(tf.io.parse_tensor(batch['audio'], out_type=tf.float64), shape=(int(self.audio_length*self.sample_rate)))
        filename = batch['filename']
        chunk = batch['chunk']
        instrument = batch['instrument']
        f0_hz = tf.ensure_shape(tf.io.parse_tensor(batch['f0_hz'], out_type=tf.float64), shape=(int(self.audio_length*self.frame_rate)))
        loudness_db = tf.ensure_shape(tf.io.parse_tensor(batch['loudness_db'], out_type=tf.float64), shape=(int(self.audio_length*self.frame_rate)))

        return {
            'audio': audio,
            'f0_hz': f0_hz,
            'loudness_db': loudness_db
        }

### OrchideaSol Data Generator ###
class OrchideaSol(GenericDataset):
    def __init__(self, split, audio_length=4, sample_rate=16000, frame_rate=250, batch_size=1):
        super().__init__(audio_length, sample_rate, frame_rate, batch_size)
        self.split = split
        self.path = os.path.join(customPath.orchideaSOL(), f'{split}.tfrecord')

    def get_dataset(self, shuffle = False):
        if not os.path.isfile(self.path):
            generate_tfrecord(self.path)
            
        dataset = tf.data.TFRecordDataset(self.path).map(self.decode_tfrecord)
        if shuffle:
            return dataset.shuffle(self.batch_size)
        else:
            return dataset

### OrchideaSol_tiny Data Generator ###
class OrchideaSolTiny(GenericDataset):
    def __init__(self, split, audio_length=4, sample_rate=16000, frame_rate=250, batch_size=1):
        super().__init__(audio_length, sample_rate, frame_rate, batch_size)
        self.split = split
        self.path = os.path.join(customPath.orchideaSOL_tiny(), f'{split}.tfrecord')

    def get_dataset(self, shuffle = False):
        dataset = tf.data.TFRecordDataset(self.path).map(self.decode_tfrecord)
        if shuffle:
            return dataset.shuffle(self.batch_size)
        else:
            return dataset

### MedleySolosDB Data Generator ###
class MedleySolosDB(DataProvider):
    def __init__(self, split, sample_rate, frame_rate, batch_size):
        super().__init__(sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.medleySolosDB(), split)
        self.frame_length = int(self.sample_rate/self.frame_rate)

        # scan split folder to create ids
        # self.list_ids = []
        # for f in os.listdir(self.path):
        #     if f.endswith('.wav'):
        #         _, x = read(os.path.join(self.path, f))
        #         n_frames = int(x.size/(self.sample_rate/self.frame_rate))
        #         for frame in range(n_frames):
        #             self.list_ids.append((f, frame))

        self.list_ids = []
        for f in os.listdir(self.path):
            if f.endswith('.wav'):
                self.list_ids.append(f)

        self.batch_size = batch_size

        super().__init__(sample_rate, frame_rate)

    def get_length(self):
        return len(self.list_ids)

    def get_dataset(self, shuffle):
        return tf.data.Dataset.from_generator(self.get_features,
            output_signature = (tf.TensorSpec(shape=(None,), dtype=tf.dtypes.float32))
            )

    def get_features(self):
        all_metadata = pd.read_csv(os.path.join(os.path.dirname(self.path), 'medley-solos-DB_metadata.csv'))
        for audio_id in self.list_ids:
            # retrieve the instrument from .csv file
            uuid4 = audio_id[:-4].split('_')[2]
            metadata = all_metadata[all_metadata['uuid4'].str.contains(uuid4)]
            instrument = metadata['instrument'].values[0]

            # load frame into wavfile
            _, x = read(os.path.join(self.path, audio_id))
            # frame = x[audio_id[1]*self.frame_length: (audio_id[1]+1)*self.frame_length]

            # # padding if the frame at the end of the wavfile is too short
            # if frame.size < self.frame_length:
            #     frame = np.concatenate((frame, np.zeros((self.frame_length-frame.size))))

            # yield
            yield (x)

### Gtzan Data Generator ###
class Gtzan(DataProvider):
    def __init__(self, split, sample_rate, frame_rate, batch_size):
        super().__init__(sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.gtzan(), split)
        self.frame_length = int(self.sample_rate/self.frame_rate)

        # scan split folder to create ids
        # self.list_ids = []
        # for f in os.listdir(self.path):
        #     if f.endswith('.wav'):
        #         _, x = read(os.path.join(self.path, f))
        #         n_frames = int(x.size/(self.sample_rate/self.frame_rate))
        #         for frame in range(n_frames):
        #             self.list_ids.append((f, frame))

        self.list_ids = []
        for f in os.listdir(self.path):
            if f.endswith('.wav'):
                self.list_ids.append(f)

        self.batch_size = batch_size

        super().__init__(sample_rate, frame_rate)

    def get_length(self):
        return len(self.list_ids)

    def get_dataset(self, shuffle):
        return tf.data.Dataset.from_generator(self.get_features,
            output_signature = (tf.TensorSpec(shape=(None,), dtype=tf.dtypes.float32))
            )

    def get_features(self):
        for audio_id in self.list_ids:
            # load wavfile
            _, x = read(os.path.join(self.path, audio_id))

            # yield
            yield (x)
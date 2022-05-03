import numpy as np
import tensorflow as tf
from ddsp.training.data import DataProvider
import customPath
import os
import csv
from scipy.io.wavfile import read
import pandas as pd

_PITCHES = ['A0', 'A#0', 'B0', 'C0', 'C#0', 'D0', 'D#0', 'E0', 'F0', 'F#0', 'G0', 'G#0',
            'A1', 'A#1', 'B1', 'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1',
            'A2', 'A#2', 'B2', 'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2',
            'A3', 'A#3', 'B3', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3',
            'A4', 'A#4', 'B4', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4',
            'A5', 'A#5', 'B5', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5',
            'A6', 'A#6', 'B6', 'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6',
            'A7', 'A#7', 'B7', 'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7',
            'A8', 'A#8', 'B8', 'C8', 'C#8', 'D8', 'D#8', 'E8', 'F8', 'F#8', 'G8', 'G#8',]

_VELOCITIES = ['pp', 'p', 'mf', 'f', 'ff']

### OrchideaSol Data Generator ###
class OrchideaSol(DataProvider):
    def __init__(self, split, sample_rate, frame_rate, batch_size):
        super().__init__(sample_rate, frame_rate)
        self.split = split
        self.path = os.path.join(customPath.orchideaSOL(), split)
        self.frame_length = int(self.sample_rate/self.frame_rate)

        # scan split folder to create ids
        self.list_ids = []
        for f in os.listdir(self.path):
            if f.endswith('.wav'):
                # _, x = read(os.path.join(self.path, f))
                # n_frames = int(x.size/(self.sample_rate/self.frame_rate))
                # for frame in range(n_frames):
                self.list_ids.append(f)

        self.batch_size = batch_size

        super().__init__(sample_rate, frame_rate)

    def get_length(self):
        return len(self.list_ids)

    def get_dataset(self, shuffle):
        return tf.data.Dataset.from_generator(self.get_features,
            output_signature = (tf.TensorSpec(shape=(None,), dtype=tf.dtypes.float32),
                                tf.TensorSpec(shape=(), dtype=tf.int32),
                                tf.TensorSpec(shape=(), dtype=tf.int32))
            )

    def get_features(self):
        for audio_id in self.list_ids:
            _, x = read(os.path.join(self.path, audio_id))
            # frame = x[audio_id[1]*self.frame_length: (audio_id[1]+1)*self.frame_length]
            instrument = audio_id.split('-')[0]
            pitch = _PITCHES.index(audio_id.split('-')[2])
            velocity = _VELOCITIES.index(audio_id.split('-')[3])
            # if frame.size < self.frame_length:
            #     frame = np.concatenate((frame, np.zeros((self.frame_length-frame.size))))
            yield (x, pitch, velocity)

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
            output_signature = (tf.TensorSpec(shape=(), dtype=tf.dtypes.float32))
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
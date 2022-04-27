import numpy as np
import tensorflow as tf
from ddsp.training.data import DataProvider
import customPath
import os
from scipy.io.wavfile import read

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
                _, x = read(os.path.join(self.path, f))
                n_frames = int(x.size/(self.sample_rate/self.frame_rate))
                for frame in range(n_frames):
                    self.list_ids.append((f, frame))

        self.batch_size = batch_size

        super().__init__(sample_rate, frame_rate)

    def get_dataset(self, shuffle):
        return tf.data.Dataset.from_generator(self.get_features,
            output_signature = tf.TensorSpec(shape=(self.frame_length,), dtype=tf.dtypes.float32))

    def get_features(self):
        for audio_id in self.list_ids:
            _, x = read(os.path.join(self.path, audio_id[0]))
            frame = x[audio_id[1]*self.frame_length: (audio_id[1]+1)*self.frame_length]
            if frame.size < self.frame_length:
                frame = np.concatenate((frame, np.zeros((self.frame_length-frame.size))))
            yield frame

import numpy as np
import librosa as lr
import os
from scipy.io.wavfile import read, write

class GtzanDataset:
    """ Class to load Gtzan dataset
    """
    def __init__(self,
                 filter_genre = None,
                 shuffle = False):
        self.path = '/home/user/Documents/Datasets/gtzan/'
        self.track_list = []
        self.genre_list = []
        self.shuffle = shuffle

        # load data in track_list
        for dir_name in sorted([e for e in os.listdir(self.path) if os.path.isdir(self.path + e)]):
            if filter_genre is None or dir_name in filter_genre:
                path_genre = self.path + dir_name
                self.genre_list.append(dir_name)
                for track_name in sorted(os.listdir(path_genre)):
                    track_dict = {'name': track_name, 'genre': dir_name}
                    self.track_list.append(track_dict)
        
        
        # compute some (hopefully) useful attributes
        self.n_tracks = len(self.track_list)
        self.n_genre  = len(self.genre_list)
        
        # id list to be randomized if self.shuffle = True
        self.id_list = np.arange(len(self.track_list))
        if self.shuffle:
            print("shuffle")
            np.random.shuffle(self.id_list)

    def load_wav(self, track_id):
        ''' extract the waveform from the track corresponding to track_id in self.track_list
        '''
        path_track = self.path + self.track_list[track_id]['genre'] + '/' + self.track_list[track_id]['name']
        return read(path_track)


class GoodSoundsDataset:
    """ Class to load good-sounds dataset
    """
    def __init__(self):
        self.path = "/home/user/Documents/Datasets/good-sounds/"
        self.instrument_list = ['flute', 'cello', 'clarinet', 'trumpet', 'violin',
                                'sax_alto', 'sax_tenor', 'sax_baritone', 'sax_soprano',
                                'oboe', 'piccolo', 'bass']
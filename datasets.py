import numpy as np
import librosa as lr
import os

class GtzanDataset:
    """ Class to load Gtzan dataset
    """
    def __init__(self,
                 filter_genre = None):
        self.path = '/home/user/Documents/Datasets/gtzan/'
        self.track_list = []
        self.genre_list = []

        # load data in track_list
        for dir_name in [e for e in os.listdir(self.path) if os.path.isdir(self.path + e)]:
            if filter_genre is None or dir_name in filter_genre:
                path_genre = self.path + dir_name
                self.genre_list.append(dir_name)
                for track_name in os.listdir(path_genre):
                    track_dict = {'name': track_name, 'genre': dir_name}
                    self.track_list.append(track_dict)
        
        # compute some (hopefully) useful attributes
        self.n_tracks = len(self.track_list)
        self.n_genre  = len(self.genre_list)


class GoodSoundsDataset:
    """ Class to load good-sounds dataset
    """
    def __init__(self):
        self.path = "/home/user/Documents/Datasets/good-sounds/"
        self.instrument_list = ['flute', 'cello', 'clarinet', 'trumpet', 'violin',
                                'sax_alto', 'sax_tenor', 'sax_baritone', 'sax_soprano',
                                'oboe', 'piccolo', 'bass']
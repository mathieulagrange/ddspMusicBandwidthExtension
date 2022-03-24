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


class OrchideaSOLDataset:
    """ Class to load orchideaSOL dataset
    """
    def __init__(self,
                 shuffle = False):
        self.path = "/home/user/Documents/Datasets/orchideaSOL/OrchideaSOL2020/"
        self.instrument_families = ['Brass', 'Keyboards', 'PluckedStrings', 'Strings', 'Winds']
        # self.instruments_dict = {'Brass': ['Bass_Tuba', 'Horn', 'Trombone', 'Trumpet']}
        self.all_playing_styles = ['ordinario']
        self.track_list = []
        self.id_list = []
        self.shuffle = shuffle

        # load data in track_list
        for instrument_family in self.instrument_families:
            cur_instrument_list = sorted([e for e in os.listdir(os.path.join(self.path,instrument_family))
                                                if os.path.isdir(os.path.join(self.path,instrument_family,e))])
            for instrument in cur_instrument_list:
                cur_instru_playing_styles = sorted([e for e in os.listdir(os.path.join(self.path,instrument_family,instrument))
                                                        if os.path.isdir(os.path.join(self.path,instrument_family,instrument,e))])
                for playing_style in self.all_playing_styles:
                    if playing_style in cur_instru_playing_styles:
                        for wav_file in sorted([e for e in os.listdir(os.path.join(self.path,instrument_family,instrument,playing_style))
                                                    if e.endswith('.wav')]):
                            wav_file_dict = {'name': wav_file,
                                             'playing_style': playing_style,
                                             'instrument': instrument,
                                             'instrument_family': instrument_family}
                            self.track_list.append(wav_file_dict)
                    else:
                        print(f'Warning: Playing style {playing_style} does not exist for instrument {instrument} from family {instrument_family}')


        self.id_list = np.arange(len(self.track_list))
        if self.shuffle:
            np.random.shuffle(self.id_list)

    def load_wav(self, track_id):
        ''' extract the waveform from the track corresponding to track_id in self.track_list
        '''
        path_track = os.path.join(self.path,
                             self.track_list[track_id]['instrument_family'],
                             self.track_list[track_id]['instrument'],
                             self.track_list[track_id]['playing_style'],
                             self.track_list[track_id]['name'])
        fs, x = read(path_track)
        return fs, np.float32(x)
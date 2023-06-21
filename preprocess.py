import yaml
import pathlib
import librosa as li
from core import extract_loudness, extract_pitch, extract_pitch_from_filename, samples_to_frames, extract_pitches_and_loudnesses_from_filename
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile
import customPath
import os
import pickle
from sys import getsizeof

def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))

def preprocess(f, model, sampling_rate, block_size, signal_length, oneshot, downsampling_factor=2, **kwargs):
    x, sr = li.load(f, sr=sampling_rate)
    x = x/np.max(np.abs(x))
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))
    x_downsampled = li.resample(x, orig_sr=sampling_rate, target_sr=sampling_rate//downsampling_factor)
    x_LB = li.resample(x_downsampled, orig_sr=sampling_rate//downsampling_factor, target_sr=sampling_rate)

    if oneshot:
        x = x[..., :signal_length]

    if model == 'ddsp':
        pitch = extract_pitch(x_LB, alg='crepe', sampling_rate=sampling_rate, block_size=block_size)
        # indiv_pitches = extract_pitch(x_LB, alg='bittner', sampling_rate=sampling_rate, block_size=block_size)
        loudness = extract_loudness(x_LB, sampling_rate, block_size)
        indiv_pitches = None
        # if len(str(f).split('/')[-1][:-4].split('_')) == 8:
        #     pitch = extract_pitch_from_filename(f, 'synthetic', sampling_rate, frame_size=block_size)
        #     pitch_list = [pitch]
        # else:
        #     pitch_list, _ = extract_pitches_and_loudnesses_from_filename(str(f).split('/')[-1][:-4], sampling_rate, signal_length)
        #     pitch_list, _ = samples_to_frames(pitch_list, pitch_list, sampling_rate, block_size)

        x = x.reshape(-1, signal_length)
        x_LB = x_LB.reshape(-1, signal_length)
        pitch = pitch.reshape(x.shape[0], -1)
        indiv_pitches = indiv_pitches.reshape(x.shape[0], indiv_pitches.shape[0], -1)
        loudness = loudness.reshape(x.shape[0], -1)

    elif model == 'resnet':
        x = x.reshape(-1, signal_length)
        x_LB = x_LB.reshape(-1, signal_length)

    if model == 'ddsp':
        if indiv_pitches is not None:
            return x, x_LB, pitch, loudness, indiv_pitches
        else:
            return x, x_LB, pitch, loudness
    elif model == 'resnet':
        return x, x_LB

class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir, model, max_sources=5):
        super().__init__()
        self.signals_WB = np.load(path.join(out_dir, "signals_WB.npy"))
        self.signals_LB = np.load(path.join(out_dir, "signals_LB.npy"))
        self.model = model
        self.max_sources = max_sources

        if 'ddsp' in model:
            self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
            self.loudness = np.load(path.join(out_dir, "loudness.npy"))
            if os.path.isfile(path.join(out_dir, "mono_loudnesses.npy")):
                self.mono_loudnesses = np.load(path.join(out_dir, "mono_loudnesses.npy"))
            else:
                self.mono_loudnesses = None
            
            if model == 'ddsp_decoder_multi' and os.path.isfile(path.join(out_dir, "all_pitches.pkl")):
                with open(path.join(out_dir, "all_pitches.pkl"), 'rb') as f:
                    self.all_pitches = pickle.load(f)
            else:
                self.all_pitches = None

        with open(path.join(out_dir, "filenames.pkl"), 'rb') as f:
            self.filenames = pickle.load(f)
        self.model = model

    def __len__(self):
        return self.signals_WB.shape[0]

    def __getitem__(self, idx):
        s_WB = torch.from_numpy(self.signals_WB[idx])
        s_LB = torch.from_numpy(self.signals_LB[idx])
        if 'ddsp' in self.model:
            if self.all_pitches is not None:
                all_p = self.all_pitches[idx]
                p = np.zeros((self.max_sources, all_p.shape[1]))
                for i_frame in range(all_p.shape[1]):
                    cur_pitches = all_p[:, i_frame]
                    smallest_pitches = sorted(cur_pitches[cur_pitches>0])[:self.max_sources]
                    if len(smallest_pitches) > 0:
                        p[:len(smallest_pitches), i_frame] = smallest_pitches
                if p.shape[0] < self.max_sources:
                    n_sources = p.shape[0]
                    p_void = np.zeros((self.max_sources-n_sources,p.shape[1]), dtype=np.float32)
                    p = np.concatenate((p, p_void), axis=0)
                p = torch.from_numpy(p)
                p = p.float()
            else:
                p = torch.from_numpy(self.pitchs[idx])
                if self.model == 'ddsp_decoder_multi':
                    p = p.unsqueeze(0)
            l = torch.from_numpy(self.loudness[idx])
            return s_WB, s_LB, p, l
        elif self.model == 'resnet':
            return s_WB, s_LB

def main():
    class args(Config):
        NAME = "config"
        SPLIT = 'train'

    args.parse_args()
    with open(os.path.join(customPath.config(), f'{args.NAME}.yaml'), "r") as config:
        config = yaml.safe_load(config)

    model = config['train']['model']
    
    if config['preprocess']['downsampling_factor'] == 2:
        if config['preprocess']['sampling_rate'] == 16000:
            if config['data']['dataset'] == 'synthetic':
                data_location = os.path.join(customPath.synthetic(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_crepe':
                data_location = os.path.join(customPath.synthetic_crepe(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_crepe(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly':
                data_location = os.path.join(customPath.synthetic_poly(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_2':
                data_location = os.path.join(customPath.synthetic_poly_2(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_2(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_3':
                data_location = os.path.join(customPath.synthetic_poly_3(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_3(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_mono':
                data_location = os.path.join(customPath.synthetic_poly_mono(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_mono(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_mono_2':
                data_location = os.path.join(customPath.synthetic_poly_mono_2(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_mono_2(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'sol':
                data_location = os.path.join(customPath.orchideaSOL(),args.SPLIT)
                out_dir = os.path.join(customPath.orchideaSOL(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'tiny':
                data_location = os.path.join(customPath.orchideaSOL_tiny(),args.SPLIT)
                out_dir = os.path.join(customPath.orchideaSOL_tiny(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'medley':
                data_location = os.path.join(customPath.medleySolosDB(),args.SPLIT)
                out_dir = os.path.join(customPath.medleySolosDB(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'dsd_sources':
                data_location = os.path.join(customPath.dsd_sources(),args.SPLIT)
                out_dir = os.path.join(customPath.dsd_sources(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'dsd_mixtures':
                data_location = os.path.join(customPath.dsd_mixtures(),args.SPLIT)
                out_dir = os.path.join(customPath.dsd_mixtures(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'gtzan':
                data_location = os.path.join(customPath.gtzan(),args.SPLIT)
                out_dir = os.path.join(customPath.gtzan(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'medleyDB_stems':
                data_location = os.path.join(customPath.medleyDB_stems(),args.SPLIT)
                out_dir = os.path.join(customPath.medleyDB_stems(), f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'medleyDB_mixtures':
                data_location = os.path.join(customPath.medleyDB_mixtures(),args.SPLIT)
                out_dir = os.path.join(customPath.medleyDB_mixtures(), f'preprocessed_{model}/{args.SPLIT}')

        elif config['preprocess']['sampling_rate'] == 8000:
            if config['data']['dataset'] == 'synthetic':
                data_location = os.path.join(customPath.synthetic(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.synthetic(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_crepe':
                data_location = os.path.join(customPath.synthetic_crepe(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_crepe(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly':
                data_location = os.path.join(customPath.synthetic_poly(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'sol':
                data_location = os.path.join(customPath.orchideaSOL(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.orchideaSOL(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'tiny':
                data_location = os.path.join(customPath.orchideaSOL_tiny(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.orchideaSOL_tiny(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'medley':
                data_location = os.path.join(customPath.medleySolosDB(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.medleySolosDB(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'dsd_sources':
                data_location = os.path.join(customPath.dsd_sources(), '8000', args.SPLIT)
                out_dir = os.path.join(customPath.dsd_sources(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'dsd_mixtures':
                data_location = os.path.join(customPath.dsd_mixtures(),'8000', args.SPLIT)
                out_dir = os.path.join(customPath.dsd_mixtures(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'gtzan':
                data_location = os.path.join(customPath.gtzan(),'8000', args.SPLIT)
                out_dir = os.path.join(customPath.gtzan(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'medleyDB_stems':
                data_location = os.path.join(customPath.medleyDB_stems(),'8000', args.SPLIT)
                out_dir = os.path.join(customPath.medleyDB_stems(), '8000', f'preprocessed_{model}/{args.SPLIT}')
            elif config['data']['dataset'] == 'medleyDB_mixtures':
                data_location = os.path.join(customPath.medleyDB_mixtures(),'8000', args.SPLIT)
                out_dir = os.path.join(customPath.medleyDB_mixtures(), '8000', f'preprocessed_{model}/{args.SPLIT}')

    elif config['preprocess']['downsampling_factor'] == 4:
        if config['preprocess']['sampling_rate'] == 16000:
            if config['data']['dataset'] == 'synthetic':
                data_location = os.path.join(customPath.synthetic(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_crepe':
                data_location = os.path.join(customPath.synthetic_crepe(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_crepe(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly':
                data_location = os.path.join(customPath.synthetic_poly(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_2':
                data_location = os.path.join(customPath.synthetic_poly_2(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_2(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_3':
                data_location = os.path.join(customPath.synthetic_poly_3(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_3(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_mono':
                data_location = os.path.join(customPath.synthetic_poly_mono(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_mono(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'synthetic_poly_mono_2':
                data_location = os.path.join(customPath.synthetic_poly_mono_2(),args.SPLIT)
                out_dir = os.path.join(customPath.synthetic_poly_mono_2(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'sol':
                data_location = os.path.join(customPath.orchideaSOL(),args.SPLIT)
                out_dir = os.path.join(customPath.orchideaSOL(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'tiny':
                data_location = os.path.join(customPath.orchideaSOL_tiny(),args.SPLIT)
                out_dir = os.path.join(customPath.orchideaSOL_tiny(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'medley':
                data_location = os.path.join(customPath.medleySolosDB(),args.SPLIT)
                out_dir = os.path.join(customPath.medleySolosDB(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'dsd_sources':
                data_location = os.path.join(customPath.dsd_sources(),args.SPLIT)
                out_dir = os.path.join(customPath.dsd_sources(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'dsd_mixtures':
                data_location = os.path.join(customPath.dsd_mixtures(),args.SPLIT)
                out_dir = os.path.join(customPath.dsd_mixtures(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'gtzan':
                data_location = os.path.join(customPath.gtzan(),args.SPLIT)
                out_dir = os.path.join(customPath.gtzan(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'medleyDB_stems':
                data_location = os.path.join(customPath.medleyDB_stems(),args.SPLIT)
                out_dir = os.path.join(customPath.medleyDB_stems(), f'preprocessed_{model}_4/{args.SPLIT}')
            elif config['data']['dataset'] == 'medleyDB_mixtures':
                data_location = os.path.join(customPath.medleyDB_mixtures(),args.SPLIT)
                out_dir = os.path.join(customPath.medleyDB_mixtures(), f'preprocessed_{model}_4/{args.SPLIT}')

    makedirs(out_dir, exist_ok=True)

    files = get_files(data_location, config['data']['extension'])
    pb = tqdm(files)

    signals_WB = []
    signals_LB = []
    pitchs = []
    loudness = []
    filenames = []
    mono_loudnesses = []
    all_pitches = []

    for f in pb:
        pb.set_description(str(f))
        if model == 'ddsp':
            features = preprocess(f, model, **config["preprocess"])
            if len(features) == 5:
                x_WB, x_LB, p, l, indiv_pitches = features
                for i_pitch in range(indiv_pitches.shape[0]):
                    all_pitches.append(indiv_pitches[i_pitch])
            else:
                x_WB, x_LB, p, l = features
            pitchs.append(p)
            loudness.append(l)
        elif model == 'resnet':
            x_WB, x_LB = preprocess(f, model, **config["preprocess"])
        signals_LB.append(x_LB)
        signals_WB.append(x_WB)
        for i_filename in range(x_LB.shape[0]):
            filenames.append((str(f), i_filename))
        if os.path.isfile(path.join(data_location, str(f)[:-4]+"_loudness.npy")):
            mono_loudness = np.load(os.path.join(path.join(data_location, str(f)[:-4]+"_loudness.npy")))
            mono_loudnesses.append(mono_loudness)

    signals_WB = np.concatenate(signals_WB, 0).astype(np.float32)
    signals_LB = np.concatenate(signals_LB, 0).astype(np.float32)
    if model == 'ddsp':
        pitchs = np.concatenate(pitchs, 0).astype(np.float32)
        loudness = np.concatenate(loudness, 0).astype(np.float32)
        if len(mono_loudnesses) > 0:
            mono_loudnesses = np.stack(mono_loudnesses, 0).astype(np.float32)
        # if len(all_pitches) > 0:
        #     max_n_sources = max([p.shape[0] for p in all_pitches])
        #     for i_pitch in range(len(all_pitches)):
        #         p = all_pitches[i_pitch]
        #         if p.shape[0] < max_n_sources:
        #             p_zero_pad = np.zeros((max_n_sources-p.shape[0], p.shape[1]))
        #             new_p = np.concatenate((p, p_zero_pad), axis=0)
        #             all_pitches[i_pitch] = new_p
        #     all_pitches = np.stack(all_pitches, 0).astype(np.float32)

    np.save(path.join(out_dir, "signals_WB.npy"), signals_WB)
    np.save(path.join(out_dir, "signals_LB.npy"), signals_LB)
    if 'ddsp' in model:
        np.save(path.join(out_dir, "pitchs.npy"), pitchs)
        np.save(path.join(out_dir, "loudness.npy"), loudness)
        if 'synthetic_poly' in config['data']['dataset']:
            np.save(path.join(out_dir, "mono_loudnesses.npy"), mono_loudnesses)
        if len(all_pitches) > 0:
            with open(path.join(out_dir, 'all_pitches.pkl'), 'wb') as f:
                pickle.dump(all_pitches, f)

    with open(path.join(out_dir, "filenames.pkl"), 'wb') as f:
        pickle.dump(filenames, f)

if __name__ == "__main__":
    main()
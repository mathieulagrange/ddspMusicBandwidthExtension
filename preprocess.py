import yaml
import pathlib
import librosa as li
from core import extract_loudness, extract_pitch, extract_pitch_from_filename
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


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sampling_rate)
    x = x/np.max(np.abs(x))
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))
    x_downsampled = li.resample(x, orig_sr=sampling_rate, target_sr=sampling_rate//2)
    x_LB = li.resample(x_downsampled, orig_sr=sampling_rate//2, target_sr=sampling_rate)

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    # pitch = extract_pitch_from_filename(f, 'synthetic', sampling_rate, frame_size=block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    x_LB = x_LB.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, x_LB, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals_WB = np.load(path.join(out_dir, "signals_WB.npy"))
        self.signals_LB = np.load(path.join(out_dir, "signals_LB.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))
        with open(path.join(out_dir, "filenames.pkl"), 'rb') as f:
            self.filenames = pickle.load(f)

    def __len__(self):
        return self.signals_WB.shape[0]

    def __getitem__(self, idx):
        s_WB = torch.from_numpy(self.signals_WB[idx])
        s_LB = torch.from_numpy(self.signals_LB[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s_WB, s_LB, p, l


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    if config['data']['dataset'] == 'synthetic':
        data_location = os.path.join(customPath.synthetic(),'valid')
        out_dir = os.path.join(customPath.synthetic(), 'preprocessed/valid')
    elif config['data']['dataset'] == 'sol':
        data_location = os.path.join(customPath.orchideaSOL(),'valid')
        out_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed/valid')

    makedirs(out_dir, exist_ok=True)

    files = get_files(data_location, config['data']['extension'])
    pb = tqdm(files)

    signals_WB = []
    signals_LB = []
    pitchs = []
    loudness = []
    filenames = []

    for f in pb:
        pb.set_description(str(f))
        x_WB, x_LB, p, l = preprocess(f, **config["preprocess"])
        signals_LB.append(x_LB)
        signals_WB.append(x_WB)
        pitchs.append(p)
        loudness.append(l)
        filenames.append(str(f))

    signals_WB = np.concatenate(signals_WB, 0).astype(np.float32)
    signals_LB = np.concatenate(signals_LB, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    np.save(path.join(out_dir, "signals_WB.npy"), signals_WB)
    np.save(path.join(out_dir, "signals_LB.npy"), signals_LB)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)

    with open(path.join(out_dir, "filenames.pkl"), 'wb') as f:
        pickle.dump(filenames, f)

if __name__ == "__main__":
    main()
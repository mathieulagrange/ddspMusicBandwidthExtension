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


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sampling_rate)
    x = x/np.max(np.abs(x))
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    # pitch = extract_pitch_from_filename(f, 'synthetic', sampling_rate, frame_size=block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    if config['data']['dataset'] == 'synthetic':
        data_location = os.path.join(customPath.synthetic(),'train')
        out_dir = os.path.join(customPath.synthetic(), 'preprocessed/acc')
    elif config['data']['dataset'] == 'acc':
        data_location = os.path.join(customPath.orchideaSOL(),'acc')
        out_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed/acc')
    elif config['data']['dataset'] == 'sol':
        data_location = os.path.join(customPath.orchideaSOL(),'train')
        out_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed/sol')

    makedirs(out_dir, exist_ok=True)

    files = get_files(data_location, config['data']['extension'])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()
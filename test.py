import torch
from model import DDSP
import yaml
from effortless_config import Config
import os
from preprocess import Dataset
import matplotlib.pyplot as plt
import librosa as lr
from math import ceil
from metrics import lsd, sdr
import numpy as np
from tqdm import tqdm

torch.set_grad_enabled(False)

class args(Config):
    RUN = './runs/first'
    DATA = False
    OUT_DIR = "export"
    BATCH = 1

params = {'model': 'first',
          'batch': 1,
          'nfft': 1024,
         }

args.parse_args()
os.makedirs(args.OUT_DIR, exist_ok=True)

with open(os.path.join(args.RUN, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

model = DDSP(**config["model"], device='cpu')
model.load_state_dict(torch.load(os.path.join(args.RUN, "state.pth"), map_location=torch.device('cpu')))
model.eval()

dataset = Dataset(config["preprocess"]["out_dir"])

dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    True,
    drop_last=True,
)

device = torch.device("cpu")

all_sdr = np.empty((len(dataloader)))
all_lsd = np.empty((len(dataloader)))

for i_batch, (s, p, l) in tqdm(enumerate(dataloader)):
    s = s.to(device)
    p = p.unsqueeze(-1).to(device)
    l = l.unsqueeze(-1).to(device)
    print(s.shape, p.shape, l.shape)
    raise ValueError("ok")
    y = model(s, p, l)
    
    # metrics computation
    orig_signal = s.numpy()[0]
    orig_stft = lr.stft(orig_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)

    rec_signal = y.numpy()[0, :, 0]
    rec_stft = lr.stft(rec_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)
    rec_stft[:ceil(params['nfft']//4), :] = orig_stft[:ceil(params['nfft']//4), :]

    cur_sdr = sdr(orig_signal, rec_signal)
    cur_lsd = lsd(orig_stft[ceil(params['nfft']//4):], rec_stft[ceil(params['nfft']//4):])
    all_sdr[i_batch] = cur_sdr
    all_lsd[i_batch] = cur_lsd


print(all_sdr, all_lsd)
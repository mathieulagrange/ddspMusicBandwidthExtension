#%%
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
from torchinfo import summary
import customPath
from IPython.display import Audio, display
from scipy.io.wavfile import read

torch.set_grad_enabled(False)
device = torch.device("cpu")
seed = 4
torch.manual_seed(seed)

#%% parameters and config
#########################
params = {'batch': 1,
          'nfft': 1024,
          'run': './models/sol_100harmo_25000steps_batch32_newData',
        #   'run': './models/bwe_sol_100harmo_25000steps_batch32_newData',
        #   'run': './models/synth_100harmo_5000steps_batch32_newData',
        #   'run': './models/bwe_synth_100harmo_5000steps_batch32_newData',
         }

with open(os.path.join(params['run'], "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

# prepare model and dataset
model = DDSP(**config["model"])
model.load_state_dict(torch.load(os.path.join(params['run'], "state.pth"), map_location=torch.device('cpu')))
model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        f0 = output['f0'].detach()
        amp = output['amp'].detach()
        harmo_amps = output['harmo_amps'].detach()
        noise_filter = output['noise_filter'].detach()
        activation[name] = dict(f0 = f0, amp = amp, harmo_amps = harmo_amps, noise_filter = noise_filter)
    return hook

model.decoder.register_forward_hook(get_activation('decoder'))

if config['data']['dataset'] == 'synthetic':
    out_dir = os.path.join(customPath.synthetic(), 'preprocessed/test')
elif config['data']['dataset'] == "sol":
    out_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed/test')

dataset = Dataset(out_dir)

dataloader = torch.utils.data.DataLoader(
    dataset,
    1,
    False,
    drop_last=True
)

mean_loudness = config["data"]["mean_loudness"]
std_loudness = config["data"]["std_loudness"]


# print(model.decoder.mlp_f0)

#%% visualize outputs
#####################
### ORCHIDEASOL ###
# filename, frame = 'Acc-ord-C6-mf-alt2-N', 20   # accordion
# filename, frame = 'Vc-ord-G#4-ff-2c-N', 20      # violoncello
# filename, frame = 'Hp-ord-A#4-ff-N-N', 2      # harp
filename, frame = 'Cb-ord-D4-mf-1c-T13d', 20     # contrabass
# filename, frame = 'Gtr-ord-A#4-mf-3c-T21d', 2 # guitar
# filename, frame = 'ASax-ord-B3-ff-N-N', 10     # saxophone
# filename, frame = 'Vn-ord-F#5-ff-3c-N', 10     # violin
# filename, frame = 'TpC-ord-F#5-ff-N-N', 10     # trumpet
# filename, frame = 'Tbn-ord-F#4-pp-N-T10u', 10  # trombone
# filename, frame = 'Fl-ord-F4-mf-N-N', 10       # flute
# filename, frame = 'Ob-ord-C#6-mf-N-N', 10      # oboe
# filename, frame = 'Hn-ord-D5-mf-N-N', 10       # horn

plt.rcParams['figure.figsize'] = [14, 10]
for data_idx, (s_WB, s_LB, p, l) in enumerate(dataloader):
    filename_idx = dataset.filenames[data_idx][0].split('/')[-1][:-4]
    if filename_idx == filename:
        s_WB, s_LB, p, l = s_WB, s_LB, p, l
        break

s_WB = s_WB.to(device)
s_LB = s_LB.to(device)
p = p.unsqueeze(-1).to(device)
l = l.unsqueeze(-1).to(device)
l = (l - mean_loudness) / std_loudness

if config['data']['input'] == 'LB':
    y = model(s_LB, p, l).squeeze(-1)
elif config['data']['input'] == 'WB':
    y = model(s_WB, p, l).squeeze(-1)

print(f'F0: {p.numpy()[0,0,0]}')

# plot intermediate outputs
f0 = activation['decoder']['f0'].numpy()[0, :, 0]
amp = activation['decoder']['amp'].numpy()[0]
harmo_amps = activation['decoder']['harmo_amps'].numpy()[0]
noise_filter = activation['decoder']['noise_filter'].numpy()[0]

# waveform and STFT
orig_signal = s_WB[0].detach().cpu().numpy()
orig_signal = orig_signal/np.max(np.abs(orig_signal))
orig_stft = lr.stft(orig_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)

input_signal = s_LB[0].detach().cpu().numpy()
# orig_signal = orig_signal/np.max(np.abs(orig_signal))
input_stft = lr.stft(input_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)

rec_signal = y[0].detach().cpu().numpy()
# rec_signal = rec_signal/np.max(np.abs(rec_signal))
rec_stft = lr.stft(rec_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)
rec_stft[:params['nfft']//4, :] = orig_stft[:params['nfft']//4, :]
rec_signal = lr.istft(rec_stft, n_fft = params['nfft'], hop_length = params['nfft']//2)

# listen
print(f'Original signal')
display(Audio(data = orig_signal, rate=config['preprocess']['sampling_rate']))
print(f'Input signal')
display(Audio(data = input_signal, rate=config['preprocess']['sampling_rate']))
print(f'Regenerated signal')
display(Audio(data = rec_signal, rate=config['preprocess']['sampling_rate']))

# plot stfts
fix, ax = plt.subplots(4)
ax[0].imshow(np.log(np.abs(orig_stft)+1), aspect='auto')
ax[1].imshow(np.log(np.abs(rec_stft)+1), aspect='auto')
ax[2].plot(p.numpy()[0, :, 0])
ax[3].plot(l.numpy()[0, :, 0])

ax[0].invert_yaxis()
ax[1].invert_yaxis()

ax[0].set_title('Original STFT', fontsize=8)
ax[1].set_title('Reconstructed STFT', fontsize=8)
ax[2].set_title('F0', fontsize=8)
ax[3].set_title('Estimated loudness', fontsize=8)

fig, ax = plt.subplots(5)
ax[0].plot(f0)
ax[1].plot(amp)
ax[2].imshow(harmo_amps, aspect='auto')
ax[3].plot(harmo_amps[:, 10])
ax[4].imshow(noise_filter, aspect='auto')

ax[2].invert_yaxis()
ax[4].invert_yaxis()

ax[0].set_title('Decoder output: F0', fontsize=8)
ax[1].set_title('Decoder output: Gain', fontsize=8)
ax[2].set_title('Decoder output: Harmonic amplitudes', fontsize=8)
ax[3].set_title('Decoder output: Harmonic amplitudes - 1 frame', fontsize=8)
ax[4].set_title('Decoder output: Noise filter', fontsize=8)

plt.figure()
orig_spectrum_frame = np.log(np.abs(orig_stft[:, frame])+1)
rec_spectrum_frame = np.log(np.abs(rec_stft[:, frame])+1)
rec_spectrum_frame[:params['nfft']//4] = np.zeros((params['nfft']//4))
plt.plot(orig_spectrum_frame, color='black', linestyle='dashed', alpha=1.)
plt.plot(rec_spectrum_frame, color='red', alpha=0.7)
plt.axvline(x=params['nfft']//4, color='grey', linewidth=1.)

plt.show()

#%% listen and display specific signals
#######################################
split = 'train'
signal_ids = sorted(list(set([int(f[:-4].split('_')[2]) for f in os.listdir(params['run']) if f.endswith(".wav") and f.startswith(split)])))

for signal_id in signal_ids[:3]:
    orig_name = f'{split}_orig_{signal_id}.wav'
    fs, x_orig = read(os.path.join(params['run'], orig_name))
    regen_name = f'{split}_regen_{signal_id}.wav'
    fs, x_regen = read(os.path.join(params['run'], regen_name))

    x_orig = x_orig/np.max(np.abs(x_orig))
    x_regen = x_regen/np.max(np.abs(x_regen))

    print(f'Original signal n°{signal_id}')
    display(Audio(data = x_orig, rate=fs))
    print(f'Regenerated signal n°{signal_id}')
    display(Audio(data = x_regen, rate=fs))

    orig_stft = lr.stft(x_orig, n_fft=1024)
    regen_stft = lr.stft(x_regen, n_fft=1024)

    fig, ax = plt.subplots(2)
    ax[0].imshow(np.log(np.abs(orig_stft)+1), aspect='auto')
    ax[1].imshow(np.log(np.abs(regen_stft)+1), aspect='auto')
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    plt.show()


# %%

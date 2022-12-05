#%%
import torch
from model_ddsp import DDSP
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
        #   'run': './models/bwe_synth_100harmo_5000steps_batch32_lossWB',
        #   'run': './models/bwe_synth_100harmo_5000steps_batch32_lossHB',
        #   'run': './models/bwe_sol_100harmo_25000steps_batch32_lossWB',
        #   'run': './models/bwe_sol_100harmo_25000steps_batch32_lossHB',
        #   'run': './models/bwe_medley_100harmo_25000steps_batch32_lossWB',
        #   'run': './models/bwe_medley_100harmo_25000steps_batch32_lossHB',
          'run': './models/bwe_dsd_sources_100harmo_25000steps_batch32_lossWB',
        #   'run': './models/bwe_dsd_sources_100harmo_25000steps_batch32_lossHB_3',
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
    out_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/test')
elif config['data']['dataset'] == "sol":
    out_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test')
elif config['data']['dataset'] == "medley":
    out_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/test')
elif config['data']['dataset'] == "dsd_sources":
    out_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/test')

dataset = Dataset(out_dir, 'resnet')

dataloader = torch.utils.data.DataLoader(
    dataset,
    1,
    False,
    drop_last=True
)

mean_loudness = config["data"]["mean_loudness"]
std_loudness = config["data"]["std_loudness"]

#%% visualize outputs
#####################
### ORCHIDEASOL ###
# filename, frame = 'Acc-ord-C6-mf-alt2-N', 20   # accordion
# filename, frame = 'Vc-ord-G#4-ff-2c-N', 20      # violoncello
# filename, frame = 'Hp-ord-A#4-ff-N-N', 2      # harp
# filename, frame = 'Cb-ord-D4-mf-1c-T13d', 20     # contrabass
# filename, frame = 'Gtr-ord-A#4-mf-3c-T21d', 2 # guitar
# filename, frame = 'ASax-ord-B3-ff-N-N', 10     # saxophone
# filename, frame = 'Vn-ord-F#5-ff-3c-N', 10     # violin
# filename, frame = 'TpC-ord-F#5-ff-N-N', 10     # trumpet
# filename, frame = 'Tbn-ord-F#4-pp-N-T10u', 10  # trombone
# filename, frame = 'Fl-ord-F4-mf-N-N', 10       # flute
# filename, frame = 'Ob-ord-C#6-mf-N-N', 10      # oboe
# filename, frame = 'Hn-ord-D5-mf-N-N', 10       # horn

### SYNTHETIC ###
# filename, frame = 'A4_f440.0_gain0.9_15harmo_att0.18_sus0.5_dec0.67_4sec', 20
# filename, frame = 'A#4_f466.1637615180899_gain0.75_10harmo_att0.12_sus0.53_dec1.17_4sec', 20
# filename, frame = 'B4_f493.8833012561241_gain0.89_20harmo_att0.09_sus0.66_dec0.38_4sec', 20
# filename, frame = 'C5_f523.2511306011972_gain0.76_15harmo_att0.02_sus0.51_dec0.42_4sec', 20
# filename, frame = 'C#4_f277.1826309768721_gain0.95_20harmo_att0.0_sus0.99_dec0.31_4sec', 20
# filename, frame = 'D4_f293.6647679174076_gain0.9_15harmo_att0.25_sus0.78_dec0.49_4sec', 20
# filename, frame = 'D#5_f622.2539674441618_gain0.77_10harmo_att0.18_sus0.59_dec1.45_4sec', 20
# filename, frame = 'E6_f1318.5102276514797_gain0.82_10harmo_att0.19_sus0.78_dec0.99_4sec', 20
# filename, frame = 'F6_f1396.9129257320155_gain0.91_10harmo_att0.12_sus0.57_dec1.43_4sec', 20
# filename, frame = 'F#4_f369.9944227116344_gain0.76_10harmo_att0.17_sus0.89_dec1.84_4sec', 20
# filename, frame = 'G6_f1567.981743926997_gain0.92_10harmo_att0.28_sus0.5_dec0.57_4sec', 20
# filename, frame = 'G#4_f415.3046975799451_gain1.0_20harmo_att0.25_sus0.71_dec0.57_4sec', 20

### MEDLEY ###
# filename, frame = 'Medley-solos-DB_test-0_0e6a2886-8c00-5e39-f1eb-53144f0cd7c7', 20
# filename, frame = 'Medley-solos-DB_test-1_4c793bdb-087f-5584-f918-4222077a867b', 20
# filename, frame = 'Medley-solos-DB_test-2_5a2474d8-f2ef-56c3-f67c-b403a12cbf1b', 65
# filename, frame = 'Medley-solos-DB_test-3_1e12111e-b736-517f-f0e6-8ec6d1e6b272', 20
# filename, frame = 'Medley-solos-DB_test-4_2bf51e03-f29f-5705-f5cd-de074b802127', 8
# filename, frame = 'Medley-solos-DB_test-5_bba1a98b-e7cc-5dfe-f47c-2f1c132b641e', 20
# filename, frame = 'Medley-solos-DB_test-6_657752be-26ac-5213-fca7-77c0ae2d097c', 15
# filename, frame = 'Medley-solos-DB_test-7_4c39a27e-344e-5089-f039-1cb0092221bc', 20

### DSD sources ###
# filename, frame = 'actions_one-minute-smile_bass', 20
# filename, frame = 'actions_one-minute-smile_drums', 1000
# filename, frame = 'actions_one-minute-smile_other', 200
# filename, frame = 'actions_one-minute-smile_vocals', 100
# filename, frame = 'al-james_schoolboy-facination_bass', 100
# filename, frame = 'al-james_schoolboy-facination_drums', 100
# filename, frame = 'al-james_schoolboy-facination_other', 100
# filename, frame = 'al-james_schoolboy-facination_vocals', 100
# filename, frame = 'animal_clinic-a_bass', 100
# filename, frame = 'animal_clinic-a_drums', 100
# filename, frame = 'animal_clinic-a_other', 100
# filename, frame = 'animal_clinic-a_vocals', 100
# filename, frame = 'drumtracks_ghost-bitch_bass', 100
# filename, frame = 'drumtracks_ghost-bitch_drums', 100
# filename, frame = 'drumtracks_ghost-bitch_other', 100
# filename, frame = 'drumtracks_ghost-bitch_vocals', 100
# filename, frame = 'girls-under-glass_we-feel-alright_bass', 100
# filename, frame = 'girls-under-glass_we-feel-alright_drums', 100
# filename, frame = 'girls-under-glass_we-feel-alright_other', 100
# filename, frame = 'girls-under-glass_we-feel-alright_vocals', 100
# filename, frame = 'motor-tapes_shore_bass', 100
# filename, frame = 'motor-tapes_shore_drums', 100
# filename, frame = 'motor-tapes_shore_other', 100
# filename, frame = 'motor-tapes_shore_vocals', 100

s_WB_list = []
s_LB_list = []
p_list = []
l_list = []
y_list = []
f0_list = []
amp_list = []
harmo_amps_list = []
noise_filter_list = []

plt.rcParams['figure.figsize'] = [14, 10]
for data_idx, (s_WB, s_LB, p, l) in enumerate(dataloader):
    if config['data']['dataset'] == 'synthetic':
        filename_idx = dataset.filenames[data_idx].split('/')[-1][:-4]
    elif config['data']['dataset'] == 'sol':
        filename_idx = dataset.filenames[data_idx][0].split('/')[-1][:-4]
    elif config['data']['dataset'] == 'medley':
        filename_idx = dataset.filenames[data_idx][0].split('/')[-1][:-4]
    elif config['data']['dataset'] == 'dsd_sources':
        filename_idx = dataset.filenames[data_idx][0].split('/')[-1][:-4]

    if filename_idx == filename:
        s_WB_list.append(s_WB)
        s_LB_list.append(s_LB)
        p_list.append(p)
        l_list.append(l)
        if config['data']['dataset'] == 'dsd_sources' and len(s_WB_list) >= 10:
            break

for i in range(len(s_WB_list)):
    s_WB = s_WB_list[i].to(device)
    s_LB = s_LB_list[i].to(device)
    p = p_list[i].unsqueeze(-1).to(device)
    l = l_list[i].unsqueeze(-1).to(device)
    l = (l - mean_loudness) / std_loudness

    if config['data']['input'] == 'LB':
        y = model(s_LB, p, l).squeeze(-1)
    elif config['data']['input'] == 'WB':
        y = model(s_WB, p, l).squeeze(-1)

    y_list.append(y)

    # intermediate outputs
    f0_list.append(activation['decoder']['f0'].numpy()[0, :, 0])
    amp_list.append(activation['decoder']['amp'].numpy()[0])
    harmo_amps_list.append(activation['decoder']['harmo_amps'].numpy()[0])
    noise_filter_list.append(activation['decoder']['noise_filter'].numpy()[0])

# numpy transformation
for i in range(len(s_WB_list)):
    s_WB_temp = s_WB_list[i]
    s_WB_list[i] = s_WB_temp[0].detach().cpu().numpy()
    s_LB_temp = s_LB_list[i]
    s_LB_list[i] = s_LB_temp[0].detach().cpu().numpy()
    y_temp = y_list[i]
    y_list[i] = y_temp[0].detach().cpu().numpy()
    p_temp = p_list[i]
    p_list[i] = p_temp.numpy()[0, :]
    l_temp = l_list[i]
    l_list[i] = l_temp.numpy()[0, :]

# concatenate all signals
s_WB_concat = np.concatenate(s_WB_list)
s_LB_concat = np.concatenate(s_LB_list)
y_concat = np.concatenate(y_list)
p_concat = np.concatenate(p_list)
l_concat = np.concatenate(l_list)

f0_concat = np.concatenate(f0_list)
amp_concat = np.concatenate(amp_list)
harmo_amps_concat = np.concatenate(harmo_amps_list, axis=1)
noise_filter_concat = np.concatenate(noise_filter_list)

# waveform and STFT
orig_signal = s_WB_concat
orig_signal = orig_signal/np.max(np.abs(orig_signal))
orig_stft = lr.stft(orig_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)
# 
input_signal = s_LB_concat
input_stft = lr.stft(input_signal, n_fft = params['nfft'], hop_length = params['nfft']//2)

rec_signal = y_concat
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
ax[2].plot(p_concat)
ax[3].plot(l_concat)

ax[0].invert_yaxis()
ax[1].invert_yaxis()

ax[0].set_title('Original STFT', fontsize=8)
ax[1].set_title('Reconstructed STFT', fontsize=8)
ax[2].set_title('F0', fontsize=8)
ax[3].set_title('Estimated loudness', fontsize=8)

fig, ax = plt.subplots(5)
ax[0].plot(f0_concat)
ax[1].plot(amp_concat)
ax[2].imshow(harmo_amps_concat, aspect='auto')
ax[3].plot(harmo_amps_concat[:, 10])
ax[4].imshow(noise_filter_concat.T, aspect='auto')

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

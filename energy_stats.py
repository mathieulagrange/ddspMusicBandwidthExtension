#%%
import os
import customPath
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
from preprocess import Dataset
import torch
from math import ceil
from tqdm import tqdm
#%%
datasets = ['synthetic', 'sol', 'medley', 'dsd_sources', 'dsd_mixtures']

cutoff_list = [0, 2000, 4000]#, 8000, 11025]
all_HB_energy = [None for i in range(len(datasets))]

nfft = 1024
fs = 16000
i_dataset = 0

for dataset in datasets:
    if dataset == 'synthetic':
        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/test')
    elif dataset == 'sol':
        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test')
    elif dataset == 'medley':
        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/test')
    elif dataset == 'dsd_sources':
        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp/test')
    elif dataset == 'dsd_mixtures':
        data_dir = os.path.join(customPath.dsd_mixtures(), 'preprocessed_ddsp/test')

    dataset_obj = Dataset(data_dir, 'ddsp')

    dataloader = torch.utils.data.DataLoader(
        dataset_obj,
        1,
        False,
        drop_last=True
    )

    for i, (s_WB, s_LB, p, l) in tqdm(enumerate(dataloader)):
        audio = s_WB[0].detach().cpu().numpy()

        stft = lr.stft(audio, n_fft = nfft, hop_length = nfft//2)
        WB_energy = np.mean(np.sum(np.square(np.abs(stft)), axis=0))
        if WB_energy > 0.:
            cur_HB_energies = np.zeros((len(cutoff_list)))
            for i_cutoff, cutoff in enumerate(cutoff_list):
                HB_stft = stft[ceil(nfft*cutoff//fs):]
                HB_energy = np.mean(np.sum(np.square(np.abs(HB_stft)), axis=0))
                cur_HB_energies[i_cutoff] = HB_energy/WB_energy
            if all_HB_energy[i_dataset] is None:
                all_HB_energy[i_dataset] = cur_HB_energies
            elif len(all_HB_energy[i_dataset].shape) == 1:
                all_HB_energy[i_dataset] = np.stack((all_HB_energy[i_dataset], cur_HB_energies))
            else:
                cur_HB_energies_exp = np.expand_dims(cur_HB_energies, axis=0)
                all_HB_energy[i_dataset] = np.concatenate((all_HB_energy[i_dataset], cur_HB_energies_exp))

    i_dataset += 1
#%%
mean_per_dataset_per_cutoff = []
print(all_HB_energy)
for i_cutoff in range(len(cutoff_list)):
    mean_per_dataset_per_cutoff.append([np.mean(all_HB_energy[i][:, i_cutoff]) for i in range(len(all_HB_energy))])

#%%
fig, ax = plt.subplots()
dsd_labels = ['Synthetic', 'OrchideaSOL', 'MedleySolosDB', 'DSD sources', 'DSD mixtures']

# ylines = [, 5, 7.5, 10, 12.5, 15, 17.5]
# for y in ylines:
#     plt.axhline(y=y, linewidth=1, color='black', alpha=0.5)

x = np.arange(len(mean_per_dataset_per_cutoff[0]))
for i_cutoff in range(len(cutoff_list)):
    if i_cutoff == 0:
        ax.bar(x, mean_per_dataset_per_cutoff[i_cutoff], label=f'> {cutoff_list[i_cutoff]}', color='lightblue')
    else:
        ax.bar(x, mean_per_dataset_per_cutoff[i_cutoff], label=f'> {cutoff_list[i_cutoff]}')
ax.set_xticks(x, dsd_labels)
ax.set_title('Mean energy ratio HB/WB per instrument type, Nyquist freq = 8000 Hz')
ax.invert_yaxis()
plt.legend()
plt.show()

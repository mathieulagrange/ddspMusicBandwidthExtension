# from datasets import OrchideaSOLDataset, GtzanDataset#, NsynthDataset, MedleyDBSoloDataset
from datasets_generators import OrchideaSol, MedleySolosDB
from metrics import sdr, lsd
from sbr import sbr
import numpy as np
import os
import customPath
import warnings
import librosa as lr

def evaluate(setting, experiment):
    # if os.path.isdir(os.path.join(customPath.results(), setting.id())):
    #     print(f'The experiment with setting {setting.id()} has already been run and the metrics has been saved in folder {os.path.join(customPath.results(), setting.id())}')
    # else:
    #     os.mkdir(os.path.join(customPath.results(), experiment.name, setting.id()))
    # dataset instantiation
    if setting.data == 'orchideaSol':
        dataset = OrchideaSol('test', 16000, 100, 1)
    elif setting.data == 'medleySolosDB':
        dataset = MedleySolosDB('test', 16000, 100, 1)
    
    if setting.method == 'replication':
        harmonic_duplication = False

    all_sdr = np.empty((dataset.get_length()))
    all_lsd = np.empty((dataset.get_length()))

    ds = dataset.get_dataset(shuffle=False)

    for i, test_data in enumerate(ds):
        audio = test_data[0].numpy()

        # stft transformation
        stft = lr.stft(audio, n_fft = setting.nfft, hop_length = dataset.frame_length)
        reconstructed_stft = np.empty((stft.shape[0]-1, stft.shape[1]), np.complex64)

        for i_frame in range(stft.shape[1]):
            # SBR algorithm
            WB_spectrum, reconstructed_spectrum = sbr(stft[:, i_frame], 
                                        phase_reconstruction = setting.phase,
                                        energy_matching_size = setting.matchingEnergy,
                                        harmonic_duplication = harmonic_duplication)
            reconstructed_stft[:, i_frame] = reconstructed_spectrum

        # istft transformation
        reconstructed_audio = lr.istft(reconstructed_stft, n_fft = setting.nfft, hop_length = dataset.frame_length)

        # adjust audio lengths
        if audio.size > reconstructed_audio.size:
            audio = audio[:reconstructed_audio.size]
        elif reconstructed_audio.size > audio.size:
            reconstructed_audio.size = reconstructed_audio[:audio.size]

        # we compute the metrics
        cur_sdr = sdr(audio, reconstructed_audio)
        cur_lsd = lsd(audio, reconstructed_audio, n_fft = setting.nfft, hop_length = dataset.frame_length)

        print(cur_sdr, cur_lsd)

        # we save the metrics for this test data
        all_sdr[i] = cur_sdr
        all_lsd[i] = cur_lsd

    return all_sdr, all_lsd    
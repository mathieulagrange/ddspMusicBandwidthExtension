# from datasets import OrchideaSOLDataset, GtzanDataset#, NsynthDataset, MedleyDBSoloDataset
from data import OrchideaSol, OrchideaSolTiny, MedleySolosDB, Gtzan
from metrics import sdr, lsd
from sbr import sbr
import numpy as np
import time
import librosa as lr
from tqdm import tqdm
from scipy.io.wavfile import write

def evaluate(setting, experiment):
    tic = time.time()
    # if os.path.isdir(os.path.join(customPath.results(), setting.id())):
    #     print(f'The experiment with setting {setting.id()} has already been run and the metrics has been saved in folder {os.path.join(customPath.results(), setting.id())}')
    # else:
    #     os.mkdir(os.path.join(customPath.results(), experiment.name, setting.id()))
    # dataset instantiation
    if setting.data == 'sol':
        dataset = OrchideaSol('test', 16000, 250, 1)
    elif setting.data == 'tiny':
        dataset = OrchideaSolTiny('test', 16000, 250, 1)
    elif setting.data == 'medley':
        dataset = MedleySolosDB('test', 16000, 250, 1)
    elif setting.data == 'gtzan':
        dataset = Gtzan('test', 16000, 250, 1)
    
    ds = dataset.get_dataset(shuffle=False)

    ds_length = 0
    for ex in ds:
        ds_length += 1

    if 'replication' in setting.method:
        replication = True
    else:
        replication = False

    if 'harmonic' in setting.method:
        harmonic_duplication = True
    else:
        harmonic_duplication = False
    
    all_sdr = np.empty((ds_length))
    all_lsd = np.empty((ds_length))

    for i, test_data in tqdm(enumerate(ds)):
        audio = test_data['audio'].numpy()

        ### ORACLE ALGO ###
        if setting.alg == 'oracle':
            reconstructed_audio = audio
            cur_sdr = sdr(audio, reconstructed_audio)
            cur_lsd = lsd(audio, reconstructed_audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

        ### SBR ALGO ###
        elif setting.alg == 'sbr':
            # stft transformation
            stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//2)
            reconstructed_stft = np.empty((stft.shape), np.complex64)

            for i_frame in range(stft.shape[1]):
                # SBR algorithm
                WB_spectrum, reconstructed_spectrum = sbr(stft[:, i_frame], 
                                            replication = replication,
                                            phase_reconstruction = setting.phase,
                                            energy_matching_size = setting.matchingEnergy,
                                            harmonic_duplication = harmonic_duplication)
                reconstructed_stft[:, i_frame] = reconstructed_spectrum

            # istft transformation
            reconstructed_audio = lr.istft(reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

            # adjust audio lengths
            if audio.size > reconstructed_audio.size:
                audio = audio[:reconstructed_audio.size]
            elif reconstructed_audio.size > audio.size:
                reconstructed_audio.size = reconstructed_audio[:audio.size]

            # we compute the metrics
            cur_sdr = sdr(audio, reconstructed_audio)
            cur_lsd = lsd(stft, reconstructed_stft)

        # we save the metrics for this test data
        all_sdr[i] = cur_sdr
        all_lsd[i] = cur_lsd

        toc = time.time()
        elapsed_time = int(toc-tic)

    return all_sdr, all_lsd, elapsed_time
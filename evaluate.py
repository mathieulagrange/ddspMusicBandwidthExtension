# from datasets import OrchideaSOLDataset, GtzanDataset#, NsynthDataset, MedleyDBSoloDataset
from data import OrchideaSol, OrchideaSolTiny, MedleySolosDB, Gtzan
from metrics import sdr, lsd
from sbr import sbr
import numpy as np
import time
import librosa as lr
from tqdm import tqdm
from scipy.io.wavfile import write
import os
import customPath
import tensorflow as tf
import training

def evaluate(setting, experiment):
    tic = time.time()

    # dataset instantiation
    if setting.data == 'sol':
        dataset = OrchideaSol('test', 16000, 250)
    elif setting.data == 'tiny':
        dataset = OrchideaSolTiny('test', 16000, 250)
    elif setting.data == 'medley':
        dataset = MedleySolosDB('test', 16000, 250)
    elif setting.data == 'gtzan':
        dataset = Gtzan('test', 16000, 250)
    
    ds = dataset.get_dataset()
    
    # how may examples in dataset
    ds_length = 0
    for ex in ds:
        ds_length += 1

    # prepare metrics for each example
    all_sdr = np.empty((ds_length))
    all_lsd = np.empty((ds_length))
    
    if setting.alg != 'ddsp':
        if 'replication' in setting.method:
            replication = True
        else:
            replication = False

        if 'harmonic' in setting.method:
            harmonic_duplication = True
        else:
            harmonic_duplication = False

        for i, test_data in tqdm(enumerate(ds)):
            audio = test_data['audio'].numpy()

            ### ORACLE ALGO ###
            if setting.alg == 'oracle':
                reconstructed_audio = audio
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//2)
                reconstructed_stft = lr.stft(reconstructed_audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

                cur_sdr = sdr(audio, reconstructed_audio)
                cur_lsd = lsd(stft, reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

            ### DUMB ALGO ###
            elif setting.alg == 'dumb':
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

                if stft.shape[0] % 2 == 1:
                    nBands_LB = int(np.ceil(stft.shape[0]/2))
                    nBands_UB = int(stft.shape[0] - nBands_LB)

                reconstructed_stft = np.zeros((stft.shape))
                reconstructed_stft[:nBands_LB, :] = stft[:nBands_LB, :]
                reconstructed_audio = lr.istft(reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

                # adjust audio lengths
                if audio.size > reconstructed_audio.size:
                    audio = audio[:reconstructed_audio.size]
                elif reconstructed_audio.size > audio.size:
                    reconstructed_audio.size = reconstructed_audio[:audio.size]

                cur_sdr = sdr(audio, reconstructed_audio)
                cur_lsd = lsd(stft, reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

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

    else:
    ### DDSP ALGO ###
        model_name = 'ddsp'
        model_dir = os.path.join(customPath.models(), model_name)

        # if a training has already been started for this model
        if os.path.isdir(os.path.join(customPath.models(), model_name)):
            n_steps_total = setting.n_steps_total
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            if latest_checkpoint is not None:
                latest_checkpoint_n_steps = os.path.basename(latest_checkpoint).split('-')[0]
            else:
                latest_checkpoint_n_steps = 0
            # check if the training is completely done
            if latest_checkpoint_n_steps < setting.n_steps_total:
            # we train the model again for setting.n_steps_per_training steps
                training.train(model_name, model_dir, setting)
                # generateSomeOutputs()
            else:
                # generateSomeOutputs()
                # computeMetrics()
                raise NotImplementedError
        # if no training has ever been done for this model
        else:
            os.mkdir(model_dir)
            training.train(model_name, model_dir, setting)

    toc = time.time()
    elapsed_time = int(toc-tic)

    return all_sdr, all_lsd, elapsed_time
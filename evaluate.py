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
import logging
from math import ceil
from models import OriginalAutoencoder, SulunResNet
from generate import checkpoint_test_generation, checkpoint_train_generation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def evaluate(setting, experiment):
    tic = time.time()

    # test dataset instantiation
    if setting.data == 'sol':
        dataset_test = OrchideaSol('test', 4, 16000, 250)
    elif setting.data == 'tiny':
        dataset_test = OrchideaSolTiny('test', 4, 16000, 250)
    elif setting.data == 'medley':
        dataset_test = MedleySolosDB('test', 4, 16000, 250)
    elif setting.data == 'gtzan':
        dataset_test = Gtzan('test', 4, 16000, 250)
    
    ds_test = dataset_test.get_dataset()
    
    # how may examples in dataset
    ds_length = 0
    for ex in ds_test.batch(batch_size=1):
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

        for i, test_data in tqdm(enumerate(ds_test)):
            audio = test_data['audio_WB'].numpy()

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
                cur_lsd = lsd(stft[ceil(setting.nfft//4):], reconstructed_stft[ceil(setting.nfft//4):])
            
            # we save the metrics for this test data
            all_sdr[i] = cur_sdr
            all_lsd[i] = cur_lsd

    else:
        ### DDSP ALGO ###               
        ds_test = ds_test.batch(batch_size=1)

        # load model
        step = None
        if setting.data == 'sol':
            if setting.output == 'WB':
                if setting.longTraining:
                    model_name = 'ddsp_estimatedLoudness_outputWB_longTraining'
                else:
                    model_name = 'ddsp_estimatedLoudness_outputWB_sol_monitoringMetrics'
                    step = 60000
            else:
                if setting.longTraining:
                    model_name = 'ddsp_estimatedLoudness_outputHB_longTraining'
                else:
                    model_name = 'ddsp_estimatedLoudness_outputHB_sol_monitoringMetrics'
                    step = 50000
        elif setting.data == 'medley':
            if setting.output == 'WB':
                if setting.longTraining:
                    model_name = 'ddsp_estimatedLoudness_outputWB_medley_longTraining'
                else:
                    model_name = 'ddsp_estimatedLoudness_outputWB_medley_monitoringMetrics'
                    step = 50000
            else:
                if setting.longTraining:
                    model_name = 'ddsp_estimatedLoudness_outputHB_medley_longTraining'
                else:
                    model_name = 'ddsp_estimatedLoudness_outputHB_medley'

        if setting.model == 'original_autoencoder':
            model = OriginalAutoencoder()
        elif setting.model == 'resnet':
            model = SulunResNet()
            print('ResNet loaded')

        model_dir = os.path.join(customPath.models(), model_name)
        model.restore(os.path.join(model_dir, 'train_files'), step=step)
        print('Trained model loaded.')

        print('Evaluation on the whole test set ...')
        for i_batch, batch in tqdm(enumerate(ds_test)):
            # output generation from the ddsp model
            outputs = model(batch, training=False)

            # reconstructed signal + recontructed stft
            reconstructed_audio = model.get_audio_from_outputs(outputs).numpy()[0]
            reconstructed_stft = lr.stft(reconstructed_audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

            # original WB signal + stft
            audio = batch['audio_WB'].numpy()[0]
            stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

            # we replace the LB with the ground-truth before computing metrics
            reconstructed_stft[:ceil(setting.nfft//4), :] = stft[:ceil(setting.nfft//4), :]

            # we compute metrics and store them
            cur_sdr = sdr(audio, reconstructed_audio)
            cur_lsd = lsd(stft[ceil(setting.nfft//4):], reconstructed_stft[ceil(setting.nfft//4):])

            all_sdr[i_batch] = cur_sdr
            all_lsd[i_batch] = cur_lsd

        print('Evaluation done.')
            
    toc = time.time()
    elapsed_time = int(toc-tic)

    return all_sdr, all_lsd, elapsed_time
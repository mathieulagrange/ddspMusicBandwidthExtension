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
from models import OriginalAutoencoder
from generate import generate_audio, checkpoint_test_generation

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
        model_name = 'ddsp_longTraining'
        model_dir = os.path.join(customPath.models(), model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model = OriginalAutoencoder()
        logging.basicConfig(filename=os.path.join(model_dir, 'training.log'), level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s')

        n_steps_total = int(setting.n_steps_total)
        if os.path.isdir(os.path.join(model_dir, 'train_files')):
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(model_dir, 'train_files'))
            if latest_checkpoint is not None:
                latest_checkpoint_n_steps = int(os.path.basename(latest_checkpoint).split('-')[-1])
            else:
                latest_checkpoint_n_steps = 0

            # check if the training is completely done
            if latest_checkpoint_n_steps < setting.n_steps_total:
            # we train the model again for setting.n_steps_per_training steps
                logging.info(f'Training is restarted from step {latest_checkpoint_n_steps} out of n_steps_total.')
                
                training.train(model_name, os.path.join(model_dir, 'train_files'), setting)
                logging.info('Generating reconstructed audio from some test data ...')
                checkpoint_test_generation(model_dir, model, 'sol', latest_checkpoint_n_steps+setting.n_steps_per_training)
                logging.info('Reconstruction done.')
            
            else:
                # all training steps have been completed
                logging.info('The training has reached the maximum number of steps.')
                logging.info('Generating reconstructed audio from some test data ...')
                checkpoint_test_generation(model_dir, model, 'sol', latest_checkpoint_n_steps+setting.n_steps_per_training)
                logging.info('Reconstruction done.')
                
                # compute metrics for the whole test dataset
                tic = time.time()
                ds_test = ds_test.batch(batch_size=1)
                model.restore(os.path.join(model_dir, 'train_files'))
                for i_batch, batch in enumerate(ds_test):
                    outputs = model(batch, training=False)

                    reconstructed_audio = model.get_audio_from_outputs(outputs).numpy()[0]
                    print(reconstructed_audio)
                    reconstructed_stft = lr.stft(reconstructed_audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

                    audio = batch['audio_WB'].numpy()
                    stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

                    cur_sdr = sdr(audio, reconstructed_audio)
                    cur_lsd = lsd(stft, reconstructed_stft)

                    all_sdr[i_batch] = cur_sdr
                    all_lsd[i_batch] = cur_lsd
                
                toc = time.time()
                elapsed_time = int(toc-tic)
                logging.info('All metrics have been computed for the whole test dataset.')


        # if no training has ever been done for this model
        else:
            latest_checkpoint_n_steps = 0
            logging.info('No on-going trainings have been found.')
            os.mkdir(os.path.join(model_dir, 'train_files'))
            logging.info("Model dir have been created, with subdir 'train_files'.")
            training.train(model_name, os.path.join(model_dir, 'train_files'), setting)

            logging.info('Generating reconstructed audio from some test data ...')
            checkpoint_test_generation(model_dir, model, 'sol', latest_checkpoint_n_steps+setting.n_steps_per_training)
            logging.info('Reconstruction done.')

    toc = time.time()
    elapsed_time = int(toc-tic)

    return all_sdr, all_lsd, elapsed_time
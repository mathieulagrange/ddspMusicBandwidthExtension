from data import OrchideaSol, OrchideaSolTiny, MedleySolosDB, Gtzan
import numpy as np
import librosa as lr
from tqdm import tqdm
from scipy.io.wavfile import write
import os
import customPath
import tensorflow as tf
import training
import logging
from math import ceil
from models import OriginalAutoencoder
from generate import checkpoint_test_generation, checkpoint_train_generation
from metrics import compute_lsd, compute_mss

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

params = {
    'model_name': 'ddsp_estimatedLoudness_outputWB_sol_monitoringMetrics',
    'data': 'sol',
    'batch_size': 2,
    'model': 'original_autoencoder',
    'n_steps_total': 100000,
    'n_steps_per_training': 10000,
    'nfft': 1024,
    'early_stop_loss_value': None,
    'output': 'WB',
    'longTraining': False
}

model_dir = os.path.join(customPath.models(), params['model_name'])
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model = OriginalAutoencoder()
logging.basicConfig(filename=os.path.join(model_dir, 'training.log'), level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s')

n_steps_total = params['n_steps_total']
if os.path.isdir(os.path.join(model_dir, 'train_files')):
    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(model_dir, 'train_files'))
    if latest_checkpoint is not None:
        latest_checkpoint_n_steps = int(os.path.basename(latest_checkpoint).split('-')[-1])
    else:
        latest_checkpoint_n_steps = 0

    # check if the training is completely done
    if latest_checkpoint_n_steps < params['n_steps_total']:
        # we train the model again for setting.n_steps_per_training steps
        logging.info(f'Training is restarted from step {latest_checkpoint_n_steps} out of {n_steps_total}.')
        
        training.train(os.path.join(model_dir, 'train_files'), params)
        logging.info('Generating reconstructed audio from some test data ...')
        checkpoint_test_generation(model_dir, model, params['data'], latest_checkpoint_n_steps+params['n_steps_per_training'])
        logging.info('Generating reconstructed audio from some train data ...')
        checkpoint_train_generation(model_dir, model, params['data'], latest_checkpoint_n_steps+params['n_steps_per_training'])
        logging.info('Reconstruction done.')
        compute_lsd(model_dir, model, params['data'], 'train', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_lsd(model_dir, model, params['data'], 'valid', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_lsd(model_dir, model, params['data'], 'test', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_mss(model_dir, model, params['data'], 'train', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_mss(model_dir, model, params['data'], 'valid', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_mss(model_dir, model, params['data'], 'test', latest_checkpoint_n_steps+params['n_steps_per_training'])

    else:
        # all training steps have been completed
        logging.info('The training has reached the maximum number of steps.')
        logging.info('Generating reconstructed audio from some test data ...')
        checkpoint_test_generation(model_dir, model, 'sol', latest_checkpoint_n_steps+params['n_steps_per_training'])
        logging.info('Generating reconstructed audio from some train data ...')
        checkpoint_train_generation(model_dir, model, 'sol', latest_checkpoint_n_steps+params['n_steps_per_training'])
        logging.info('Reconstruction done.')

        compute_lsd(model_dir, model, params['data'], 'train', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_lsd(model_dir, model, params['data'], 'valid', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_lsd(model_dir, model, params['data'], 'test', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_mss(model_dir, model, params['data'], 'train', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_mss(model_dir, model, params['data'], 'valid', latest_checkpoint_n_steps+params['n_steps_per_training'])
        compute_mss(model_dir, model, params['data'], 'test', latest_checkpoint_n_steps+params['n_steps_per_training'])

# if no training has ever been done for this model
else:
    latest_checkpoint_n_steps = 0
    logging.info('No on-going trainings have been found.')
    os.mkdir(os.path.join(model_dir, 'train_files'))
    logging.info("Model dir have been created, with subdir 'train_files'.")
    training.train(os.path.join(model_dir, 'train_files'), params)

    logging.info('Generating reconstructed audio from some test data ...')
    checkpoint_test_generation(model_dir, model, 'sol', latest_checkpoint_n_steps+params['n_steps_per_training'])
    logging.info('Reconstruction done.')

    compute_lsd(model_dir, model, params['data'], 'train', latest_checkpoint_n_steps+params['n_steps_per_training'])
    compute_lsd(model_dir, model, params['data'], 'valid', latest_checkpoint_n_steps+params['n_steps_per_training'])
    compute_lsd(model_dir, model, params['data'], 'test', latest_checkpoint_n_steps+params['n_steps_per_training'])
    compute_mss(model_dir, model, params['data'], 'train', latest_checkpoint_n_steps+params['n_steps_per_training'])
    compute_mss(model_dir, model, params['data'], 'valid', latest_checkpoint_n_steps+params['n_steps_per_training'])
    compute_mss(model_dir, model, params['data'], 'test', latest_checkpoint_n_steps+params['n_steps_per_training'])

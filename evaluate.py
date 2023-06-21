from metrics import sdr, lsd
import numpy as np
import time
import librosa as lr
from tqdm import tqdm
from scipy.io.wavfile import write
import os
import customPath
from math import ceil
from model_ddsp import DDSP, DDSPNonHarmonic, DDSPMulti, DDSPNoise
from model_resnet import Resnet
import torch
from preprocess import Dataset
import yaml
from effortless_config import Config
from core import extract_pitch, extract_loudness, extract_pitch_from_filename, extract_pitches_and_loudnesses_from_filename, samples_to_frames, count_n_signals, gaussian_comb_filters
import logging
import matplotlib.pyplot as plt
from torchinfo import summary

logging.basicConfig(filename='evaluate.log', level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s')

np.set_printoptions(precision=10)
torch.set_printoptions(precision=10)
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 4
torch.manual_seed(4)
n_decimals_rounding = 5

def evaluate(setting, experiment):
    tic = time.time()

    # test dataset instantiation
    if setting.downsampling_factor == 2:
        if setting.sampling_rate == 16000:
            if not hasattr(setting, 'model') or 'ddsp' in setting.model:
                if setting.data == 'synthetic':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_crepe':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_crepe(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_crepe(), 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_poly':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_poly_2':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly_2(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly_2(), 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_poly_3':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly_3(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly_3(), 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_poly_mono':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly_mono(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly_mono(), 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_poly_mono_2':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly_mono_2(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly_mono_2(), 'preprocessed_ddsp/test')
                elif setting.data == 'sol':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test')
                elif setting.data == 'medley':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/train')
                    if setting.split == 'test':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/test')
                elif setting.data == 'dsd_sources':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp/test')
                elif setting.data == 'dsd_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_mixtures(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_mixtures(), 'preprocessed_ddsp/test')
                elif setting.data == 'gtzan':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp/test')
                elif setting.data == 'medleyDB_stems':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_stems(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_stems(), 'preprocessed_ddsp/test')
                elif setting.data == 'medleyDB_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_ddsp/test')

            elif setting.model == 'resnet':
                if setting.data == 'synthetic':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet/test')
                if setting.data == 'synthetic_poly':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_resnet/test')
                elif setting.data == 'sol':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_resnet/test')
                elif setting.data == 'medley':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/test')
                elif setting.data == 'dsd_sources':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/test')
                elif setting.data == 'dsd_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_mixtures(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_mixtures(), 'preprocessed_resnet/test')
                elif setting.data == 'gtzan':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_resnet/test')
                elif setting.data == 'medleyDB_stems':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_stems(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_stems(), 'preprocessed_resnet/test')
                elif setting.data == 'medleyDB_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_resnet/test')

        elif setting.sampling_rate == 8000:
            if not hasattr(setting, 'model') or 'ddsp' in setting.model:
                if setting.data == 'synthetic':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_crepe':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_crepe(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_crepe(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'synthetic_poly':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'sol':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.orchideaSOL(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.orchideaSOL(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'medley':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleySolosDB(), '8000', 'preprocessed_ddsp/train')
                    if setting.split == 'test':
                        data_dir = os.path.join(customPath.medleySolosDB(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'dsd_sources':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_sources(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_sources(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'dsd_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_mixtures(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_mixtures(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'gtzan':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.gtzan(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.gtzan(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'medleyDB_stems':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_stems(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_stems(), '8000', 'preprocessed_ddsp/test')
                elif setting.data == 'medleyDB_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), '8000', 'preprocessed_ddsp/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), '8000', 'preprocessed_ddsp/test')

            elif setting.model == 'resnet':
                if setting.data == 'synthetic':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic(), '8000', 'preprocessed_resnet/test')
                if setting.data == 'synthetic_poly':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'sol':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.orchideaSOL(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.orchideaSOL(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'medley':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleySolosDB(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleySolosDB(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'dsd_sources':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_sources(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_sources(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'dsd_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_mixtures(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_mixtures(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'gtzan':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.gtzan(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.gtzan(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'medleyDB_stems':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_stems(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_stems(), '8000', 'preprocessed_resnet/test')
                elif setting.data == 'medleyDB_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), '8000', 'preprocessed_resnet/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), '8000', 'preprocessed_resnet/test')
    
    elif setting.downsampling_factor == 4:
        if setting.sampling_rate == 16000:
            if not hasattr(setting, 'model') or 'ddsp' in setting.model:
                if setting.data == 'synthetic':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp_4/test')
                elif setting.data == 'synthetic_poly':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_ddsp_4/test')
                elif setting.data == 'sol':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp_4/test')
                elif setting.data == 'medley':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp_4/test')
                elif setting.data == 'dsd_sources':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp_4/test')
                elif setting.data == 'gtzan':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp_4/test')
                elif setting.data == 'medleyDB_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_ddsp_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_ddsp_4/test')

            elif setting.model == 'resnet':
                if setting.data == 'synthetic':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet_4/test')
                elif setting.data == 'sol':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_resnet_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_resnet_4/test')
                elif setting.data == 'medley':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet_4/test')
                elif setting.data == 'synthetic_poly':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_resnet_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_resnet_4/test')
                elif setting.data == 'gtzan':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_resnet_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_resnet_4/test')
                elif setting.data == 'medleyDB_mixtures':
                    if setting.split == 'train':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_resnet_4/train')
                    elif setting.split == 'test':
                        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_resnet_4/test')


    if setting.alg == 'ddsp':
        if 'ddsp' in setting.model:
            dataset = Dataset(data_dir, model='ddsp')
        elif setting.model == 'resnet':
            dataset = Dataset(data_dir, model='resnet')
    elif setting.alg == 'ddsp_poly_decoder':
        dataset = Dataset(data_dir, model='ddsp_decoder_multi')
    else:
        dataset = Dataset(data_dir, model='ddsp')
    dataloader = torch.utils.data.DataLoader(dataset, 1, False)
    
    # prepare metrics for each example
    all_sdr = []
    all_lsd = []
    
    if 'ddsp' not in setting.alg:
        if 'replication' in setting.method:
            replication = True
        else:
            replication = False

        if 'harmonic' in setting.method:
            harmonic_duplication = True
        else:
            harmonic_duplication = False

        first_example = 1
        orig_audio_list = []
        rec_audio_list = []

        for i, test_data in tqdm(enumerate(dataloader)):
            if setting.data == 'synthetic':
                if setting.sampling_rate == 16000:
                    if not hasattr(setting, 'model') or setting.model == 'ddsp_original_autoencoder':
                        filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                        chunk_idx = 0
                    elif setting.model == 'resnet':
                        filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                        chunk_idx = dataset.filenames[i][1]
                elif setting.sampling_rate == 8000:
                    filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                    chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'synthetic_crepe':
                if setting.sampling_rate == 16000:
                    if not hasattr(setting, 'model') or setting.model == 'ddsp_original_autoencoder':
                        filename_idx = dataset.filenames[i].split('/')[-1][:-4]
                        chunk_idx = 0
                    elif setting.model == 'resnet':
                        filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                        chunk_idx = dataset.filenames[i][1]
                elif setting.sampling_rate == 8000:
                    filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                    chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'synthetic_poly':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'synthetic_poly_2':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'sol':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'medley':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'dsd_sources':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'dsd_mixtures':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'gtzan':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'medleyDB_stems':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]
            elif setting.data == 'medleyDB_mixtures':
                filename_idx = dataset.filenames[i][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i][1]

            if chunk_idx == 0:
                if not first_example:
                    orig_audio_concat = np.concatenate(orig_audio_list)
                    rec_audio_concat = np.concatenate(rec_audio_list)

                    # original WB signal + stft
                    orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)

                    # recontructed stft
                    rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)

                    # we replace the LB with the ground-truth before computing metrics
                    rec_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :] = orig_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :]

                    # we compute metrics and store them
                    cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
                    cur_lsd = lsd(orig_stft[ceil(setting.nfft//(setting.downsampling_factor*2)):], rec_stft[ceil(setting.nfft//(setting.downsampling_factor*2)):])
                    all_sdr.append(cur_sdr)
                    all_lsd.append(cur_lsd)

                orig_audio_list = []
                rec_audio_list = []
                first_example = 0

            s_WB, s_LB, p, l = test_data
            audio = np.round(s_WB[0].detach().cpu().numpy(), decimals=n_decimals_rounding)

            ### ORACLE ALGO ###
            if setting.alg == 'oracle':
                reconstructed_audio = audio
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//4)
                reconstructed_stft = lr.stft(reconstructed_audio, n_fft = setting.nfft, hop_length = setting.nfft//4)

                orig_audio_list.append(audio)
                rec_audio_list.append(reconstructed_audio)
                # cur_sdr = sdr(audio, reconstructed_audio)
                # cur_lsd = lsd(stft, reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

            ### DUMB ALGO ###
            elif setting.alg == 'dumb':
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//4)

                if stft.shape[0] % 2 == 1:
                    nBands_LB = int(np.ceil(stft.shape[0]/setting.downsampling_factor))
                    nBands_UB = int(stft.shape[0] - nBands_LB)

                reconstructed_stft = np.zeros((stft.shape), dtype=np.complex)
                reconstructed_stft[:nBands_LB, :] = stft[:nBands_LB, :]
                reconstructed_audio = lr.istft(reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//4)

                # adjust audio lengths
                if audio.size > reconstructed_audio.size:
                    audio = audio[:reconstructed_audio.size]
                elif reconstructed_audio.size > audio.size:
                    reconstructed_audio = reconstructed_audio[:audio.size]

                orig_audio_list.append(audio)
                rec_audio_list.append(reconstructed_audio)

                # cur_sdr = sdr(audio, reconstructed_audio)
                # cur_lsd = lsd(stft[ceil(setting.nfft//4):], reconstructed_stft[ceil(setting.nfft//4):], n_fft = setting.nfft, hop_length = setting.nfft//2)

            ### NOISE ALGO ###      
            elif setting.alg == 'noise':
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//4)
                noise = np.random.rand(audio.size)*2-1
                noise_stft = lr.stft(noise, n_fft = setting.nfft, hop_length = setting.nfft//4)

                if stft.shape[0] % 2 == 1:
                    nBands_LB = int(np.ceil(stft.shape[0]/2))
                    nBands_UB = int(stft.shape[0] - nBands_LB)

                reconstructed_stft = stft
                reconstructed_stft[nBands_LB:, :] = noise_stft[nBands_LB:, :]
                reconstructed_audio = lr.istft(reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

                # adjust audio lengths
                if audio.size > reconstructed_audio.size:
                    audio = audio[:reconstructed_audio.size]
                elif reconstructed_audio.size > audio.size:
                    reconstructed_audio = reconstructed_audio[:audio.size]

                orig_audio_list.append(audio)
                rec_audio_list.append(reconstructed_audio)


            ### SBR ALGO ###
            elif setting.alg == 'sbr':
                from sbr import sbr
                # stft transformation
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//4)
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
                reconstructed_audio = lr.istft(reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//4)

                # adjust audio lengths
                if audio.size > reconstructed_audio.size:
                    audio = audio[:reconstructed_audio.size]
                elif reconstructed_audio.size > audio.size:
                    reconstructed_audio = reconstructed_audio[:audio.size]

                orig_audio_list.append(audio)
                rec_audio_list.append(reconstructed_audio)

                # we compute the metrics
                # cur_sdr = sdr(audio, reconstructed_audio)
                # cur_lsd = lsd(stft[ceil(setting.nfft//4):], reconstructed_stft[ceil(setting.nfft//4):])
            
            # we save the metrics for this test data
            # all_sdr[i] = cur_sdr
            # all_lsd[i] = cur_lsd

        # last batch    
        orig_audio_concat = np.concatenate(orig_audio_list)
        rec_audio_concat = np.concatenate(rec_audio_list)

        orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)
        rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)
        rec_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :] = orig_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :]

        cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
        cur_lsd = lsd(orig_stft[ceil(setting.nfft//(setting.downsampling_factor*2)):], rec_stft[ceil(setting.nfft//(setting.downsampling_factor*2)):])
        all_sdr.append(cur_sdr)
        all_lsd.append(cur_lsd)

    else:
        ### DDSP ALGO ###               
        # model name
        if setting.alg == 'ddsp':
            if setting.downsampling_factor == 2:
                if setting.sampling_rate == 16000:
                    # DDSP vanilla
                    if setting.model == 'ddsp_original_autoencoder':
                        if setting.data == 'synthetic':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_100harmos_5000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_100harmos_5000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_crepe':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_crepe_100harmos_5000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_crepe_100harmos_5000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_poly':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_poly_2':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_poly_3':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'sol':
                            if setting.loss == 'WB':
                                model_name = 'bwe_sol_100harmos_25000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_sol_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medley':
                            if setting.loss == 'WB':
                                model_name = 'bwe_medley_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_medley_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'dsd_sources':
                            if setting.loss == 'WB':
                                model_name = 'bwe_dsd_sources_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_dsd_sources_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'dsd_mixtures':
                            if setting.loss == 'WB':
                                model_name = 'bwe_dsd_mixtures_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_dsd_mixtures_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'gtzan':
                            if setting.loss == 'WB':
                                model_name = 'bwe_gtzan_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                if setting.noiseTraining:
                                    model_name = 'bwe_gtzan_100harmos_25000steps_batch32_lossHB'
                                else:
                                    model_name = 'bwe_gtzan_100harmos_25000steps_batch32_lossHB_noNoise'
                        elif setting.data == 'medleyDB_stems':
                            if setting.loss == 'WB':
                                model_name = 'bwe_medleyDB_stems_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_medleyDB_stems_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medleyDB_mixtures':
                            if setting.loss == 'WB':
                                model_name = 'bwe_medleyDB_mixtures_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                if setting.noiseTraining:
                                    model_name = 'bwe_medleyDB_mix_100harmos_25000steps_batch32_lossHB'
                                else:
                                    model_name = 'bwe_medleyDB_mix_100harmos_25000steps_batch32_lossHB_noNoise'
                    # DDSP non harmo
                    if setting.model == 'ddsp_non_harmo':
                        if setting.data == 'synthetic':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synth_ddspNonHarmo_100harmos_5000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synth_ddspNonHarmo_100harmos_5000steps_batch32_lossHB'
                        if setting.data == 'synthetic_crepe':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synth_crepe_ddspNonHarmo_100harmos_5000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synth_crepe_ddspNonHarmo_100harmos_5000steps_batch32_lossHB'

                    # Resnet
                    elif setting.model == 'resnet':
                        if setting.data == 'synthetic':
                            model_name = 'bwe_synth_resnet_64000steps_batch16'
                        elif setting.data == 'synthetic_poly':
                            model_name = 'bwe_spoly_resnet_125000steps_batch2'
                        elif setting.data == 'sol':
                            model_name = 'bwe_sol_resnet_125000steps_batch1'
                        elif setting.data == 'medley':
                            model_name = 'bwe_medley_resnet_250000steps_batch16'
                        elif setting.data == 'dsd_sources':
                            model_name = 'bwe_dsd_sources_resnet_250000steps_batch8'
                        elif setting.data == 'dsd_mixtures':
                            model_name = 'bwe_dsd_mixtures_resnet_250000steps_batch8'
                        elif setting.data == 'gtzan':
                            model_name = 'bwe_gtzan_resnet_125000steps_batch2'
                        elif setting.data == 'medleyDB_stems':
                            model_name = 'bwe_medleyDB_stems_resnet_250000steps_batch16'
                        elif setting.data == 'medleyDB_mixtures':
                            model_name = 'bwe_medleyDB_mix_resnet_125000steps_batch2'
                elif setting.sampling_rate == 8000:
                    # DDSP vanilla
                    if setting.model == 'ddsp_original_autoencoder':
                        if setting.data == 'synthetic':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_8000_100harmos_5000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_8000_100harmos_5000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_crepe':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_crepe_8000_100harmos_5000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_crepe_8000_100harmos_5000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_poly':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_8000_100harmos_25000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_synthetic_poly_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'sol':
                            if setting.loss == 'WB':
                                model_name = 'bwe_sol_8000_100harmos_25000steps_batch32_lossWB'
                            else:
                                model_name = 'bwe_sol_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medley':
                            if setting.loss == 'WB':
                                model_name = 'bwe_medley_8000_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_medley_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'dsd_sources':
                            if setting.loss == 'WB':
                                model_name = 'bwe_dsd_sources_8000_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_dsd_sources_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'dsd_mixtures':
                            if setting.loss == 'WB':
                                model_name = 'bwe_dsd_mixtures_8000_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_dsd_mixtures_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'gtzan':
                            if setting.loss == 'WB':
                                model_name = 'bwe_gtzan_8000_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_gtzan_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medleyDB_stems':
                            if setting.loss == 'WB':
                                model_name = 'bwe_medleyDB_stems_8000_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_medleyDB_stems_8000_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medleyDB_mixtures':
                            if setting.loss == 'WB':
                                model_name = 'bwe_medleyDB_mixtures_8000_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_medleyDB_mixtures_8000_100harmos_25000steps_batch32_lossHB'

                    # DDSP non harmo
                    if setting.model == 'ddsp_non_harmo':
                        if setting.data == 'synthetic':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synth_8000_ddspNonHarmo_100harmos_5000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synth_8000_ddspNonHarmo_100harmos_5000steps_batch32_lossHB'
                        if setting.data == 'synthetic_crepe':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synth_crepe_8000_ddspNonHarmo_100harmos_5000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synth_crepe_8000_ddspNonHarmo_100harmos_5000steps_batch32_lossHB'

                    # Resnet
                    elif setting.model == 'resnet':
                        if setting.data == 'synthetic':
                            model_name = 'bwe_synthetic_8000_resnet_64000steps_batch16'
                        elif setting.data == 'synthetic_poly':
                            model_name = 'bwe_synthetic_poly_8000_resnet_250000steps_batch16'
                        elif setting.data == 'sol':
                            model_name = 'bwe_sol_8000_resnet_64000steps_batch8'
                        elif setting.data == 'medley':
                            model_name = 'bwe_medley_8000_resnet_250000steps_batch16'
                        elif setting.data == 'dsd_sources':
                            model_name = 'bwe_dsd_sources_8000_resnet_250000steps_batch8'
                        elif setting.data == 'dsd_mixtures':
                            model_name = 'bwe_dsd_mixtures_8000_resnet_250000steps_batch8'
                        elif setting.data == 'gtzan':
                            model_name = 'bwe_gtzan_8000_resnet_250000steps_batch16'
                        elif setting.data == 'medleyDB_stems':
                            model_name = 'bwe_medleyDB_stems_8000_resnet_250000steps_batch16'
                        elif setting.data == 'medleyDB_mixtures':
                            model_name = 'bwe_medleyDB_mixtures_8000_resnet_250000steps_batch16'

            elif setting.downsampling_factor == 4:
                if setting.sampling_rate == 16000:
                    # DDSP vanilla
                    if setting.model == 'ddsp_original_autoencoder':
                        if setting.data == 'synthetic':
                            if setting.loss == 'HB':
                                model_name = 'bwe_synthetic_factor4_100harmos_5000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_poly':
                            if setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_factor4_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'sol':
                            if setting.loss == 'HB':
                                model_name = 'bwe_sol_factor4_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medley':
                            if setting.loss == 'HB':
                                model_name = 'bwe_medley_factor4_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'gtzan':
                            if setting.loss == 'HB':
                                model_name = 'bwe_gtzan_factor4_100harmos_25000steps_batch32_lossHB'
                        elif setting.data == 'medleyDB_mixtures':
                            if setting.loss == 'HB':
                                model_name = 'bwe_medleyDB_mix_factor4_100harmos_25000steps_batch32_lossHB'
                    # DDSP noise
                    elif setting.model == 'ddsp_noise':
                        if setting.data == 'synthetic':
                            if setting.loss == 'HB':
                                model_name = 'bwe_synthetic_factor4_ddspNoise_5000steps_batch32_lossHB'
                        elif setting.data == 'synthetic_poly':
                            if setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_factor4_ddspNoise_25000steps_batch32_lossHB'
                        elif setting.data == 'sol':
                            if setting.loss == 'HB':
                                model_name = 'bwe_sol_factor4_ddspNoise_25000steps_batch32_lossHB'
                        elif setting.data == 'medley':
                            if setting.loss == 'HB':
                                model_name = 'bwe_medley_factor4_ddspNoise_25000steps_batch32_lossHB'
                        elif setting.data == 'gtzan':
                            if setting.loss == 'HB':
                                model_name = 'bwe_gtzan_factor4_ddspNoise_25000steps_batch32_lossHB'
                        elif setting.data == 'medleyDB_mixtures':
                            if setting.loss == 'HB':
                                model_name = 'bwe_medleyDB_mixtures_factor4_ddspNoise_25000steps_batch32_lossHB'

                    # Resnet
                    elif setting.model == 'resnet':
                        if setting.data == 'synthetic':
                            model_name = 'bwe_synthetic_factor4_resnet_250000steps_batch8'
                        elif setting.data == 'synthetic_poly':
                            model_name = 'bwe_synthetic_poly_factor4_resnet_125000steps_batch2'
                        elif setting.data == 'sol':
                            model_name = 'bwe_sol_factor4_resnet_250000steps_batch8'
                        elif setting.data == 'medley':
                            model_name = 'bwe_medley_factor4_resnet_250000steps_batch8'
                        elif setting.data == 'dsd_sources':
                            model_name = 'bwe_dsd_sources_factor4_resnet_250000steps_batch8'
                        elif setting.data == 'medleyDB_stems':
                            model_name = 'bwe_medleyDB_stems_factor4_resnet_250000steps_batch8'
                        elif setting.data == 'gtzan':
                            model_name = 'bwe_gtzan_factor4_resnet_125000steps_batch2'
                        elif setting.data == 'gtzan':
                            model_name = 'bwe_dsd_mixtures_factor4_resnet_250000steps_batch8'
                        elif setting.data == 'medleyDB_mixtures':
                            model_name = 'bwe_medleyDB_mix_factor4_resnet_125000steps_batch2'

        # DDSP cyclic
        elif setting.alg == 'ddsp_multi':
            if setting.downsampling_factor == 2:
                if setting.sampling_rate == 16000:
                    if setting.model == 'ddsp_original_autoencoder':
                        if setting.train_data == 'synthetic':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_100harmos_5000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synthetic_100harmos_5000steps_batch32_lossHB'
                        if setting.train_data == 'synthetic_poly':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_100harmos_25000steps_batch32_lossHB'
                        elif setting.train_data == 'gtzan':
                            if setting.loss == 'WB':
                                model_name = 'bwe_gtzan_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                if setting.noiseTraining:
                                    model_name = 'bwe_gtzan_100harmos_25000steps_batch32_lossHB'
                                else:
                                    model_name = 'bwe_gtzan_100harmos_25000steps_batch32_lossHB_noNoise'
                        elif setting.train_data == 'dsd_sources':
                            model_name = 'bwe_dsd_sources_100harmo_25000steps_batch32_lossHB'
                        elif setting.train_data == 'dsd_mixtures':
                            model_name = 'bwe_dsd_mixtures_100harmo_25000steps_batch32_lossHB'
                        elif setting.train_data == 'medleyDB_mixtures':
                            if setting.noiseTraining:
                                model_name = 'bwe_medleyDB_mix_100harmos_25000steps_batch32_lossHB'
                            else:
                                model_name = 'bwe_medleyDB_mix_100harmos_25000steps_batch32_lossHB_noNoise'

                elif setting.sampling_rate == 8000:
                    if setting.model == 'ddsp_original_autoencoder':
                        if setting.train_data == 'synthetic':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_8000_100harmos_5000steps_batch32_lossWB_block128'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synth_8000_100harmo_5000steps_batch32_lossHB'
                        elif setting.train_data == 'dsd_sources':
                            model_name = 'bwe_dsd_sources_8000_100harmo_25000steps_batch32_lossHB'
                        elif setting.train_data == 'dsd_mixtures':
                            model_name = 'bwe_dsd_mixtures_8000_100harmo_25000steps_batch32_lossHB'
            elif setting.downsampling_factor == 4:
                if setting.sampling_rate == 16000:
                    if setting.model == 'ddsp_original_autoencoder':
                        if setting.train_data == 'synthetic':
                            if setting.loss == 'HB':
                                model_name = 'bwe_synthetic_factor4_5000steps_batch32_lossHB'
                        elif setting.train_data == 'gtzan':
                            if setting.loss == 'HB':
                                model_name = 'bwe_gtzan_factor4_100harmos_25000steps_batch32_lossHB'
                        elif setting.train_data == 'medleyDB_mixtures':
                            model_name = 'bwe_medleyDB_mix_factor4_100harmos_25000steps_batch32_lossHB'

        # DDSP poly decoder
        elif setting.alg == 'ddsp_poly_decoder':
            if setting.downsampling_factor == 2:
                if setting.sampling_rate == 16000:
                    if setting.model == 'ddsp_decoder_multi':
                        if setting.train_data == 'synthetic_poly':
                            if setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_decMulti_25000steps_batch32_lossHB'
                        if setting.train_data == 'synthetic_poly_2':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_2_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_2_100harmos_25000steps_batch32_lossHB'
                        elif setting.train_data == 'synthetic_poly_3':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_3_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_3_100harmos_25000steps_batch32_lossHB'
                        elif setting.train_data == 'synthetic_poly_mono':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_mono_decMulti_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_mono_decMulti_100harmos_25000steps_batch32_lossHB'
                        elif setting.train_data == 'synthetic_poly_mono_2':
                            if setting.loss == 'WB':
                                model_name = 'bwe_synthetic_poly_mono_2_decMulti_100harmos_25000steps_batch32_lossWB'
                            elif setting.loss == 'HB':
                                model_name = 'bwe_synthetic_poly_mono_2_decMulti_100harmos_25000steps_batch32_lossHB'
                        elif setting.train_data == 'gtzan':
                            if setting.loss == 'HB':
                                if setting.max_n_sources == 2:
                                    model_name = 'bwe_gtzan_decMulti_2sources_100harmos_25000steps_batch32_lossHB'
                                elif setting.max_n_sources == 3:
                                    model_name = 'bwe_gtzan_decMulti_3sources_100harmos_25000steps_batch32_lossHB'
                                elif setting.max_n_sources == 5:
                                    if setting.noiseTraining:
                                        model_name = 'bwe_gtzan_decMulti5_25000steps_batch32_lossHB_noBugMultiPitch'
                                    else:
                                        model_name = 'bwe_gtzan_100harmos_decMulti5_25000steps_batch32_lossHB_noNoise'
                            elif setting.loss == 'WB':
                                if setting.noiseTraining:
                                    model_name = 'bwe_gtzan_decMulti5_25000steps_batch32_lossWB'
                        elif setting.train_data == 'medleyDB_mixtures':
                            if setting.loss == 'HB':
                                if setting.noiseTraining:
                                    model_name = 'bwe_medleyDB_mix_decMulti_25000steps_batch32_lossHB'
                                else:
                                    model_name = 'bwe_medleyDB_mix_100harmos_decMulti5_25000steps_batch32_lossHB_noNoise'

            elif setting.downsampling_factor == 4:
                if setting.sampling_rate == 16000:
                    if setting.model == 'ddsp_decoder_multi':
                        if setting.train_data == 'synthetic_poly':
                            if setting.loss == 'HB':
                                if setting.max_n_sources == 5:
                                    model_name = 'bwe_synthetic_poly_factor4_100harmos_decMulti5_25000steps_batch32_lossHB'
                        if setting.train_data == 'gtzan':
                            if setting.loss == 'HB':
                                if setting.max_n_sources == 5:
                                    model_name = 'bwe_gtzan_factor4_100harmos_decMulti5_25000steps_batch32_lossHB'
                        elif setting.train_data == 'medleyDB_mixtures':
                            if setting.loss == 'HB':
                                model_name = 'bwe_medleyDB_mix_factor4_100harmos_decMulti5_25000steps_batch32_lossHB'

        print(f'Loading model: {model_name} ...')
        # config file loading
        with open(os.path.join(customPath.models(), model_name, "config.yaml"), "r") as config:
            config = yaml.safe_load(config)
        print('Model loaded.')
        
        # load model
        if 'ddsp' in setting.model:
            if setting.model == 'ddsp_original_autoencoder':
                model = DDSP(**config["model"])
            elif setting.model == 'ddsp_decoder_multi':
                model = DDSPMulti(**config['model'])
            elif setting.model == 'ddsp_non_harmo':
                model = DDSPNonHarmonic(**config['model'])
            elif setting.model == 'ddsp_noise':
                model = DDSPNoise(**config['model'])
            checkpoint = torch.load(os.path.join(customPath.models(), model_name, "state.pth"), map_location=device)
            if 'model_state_dict' in checkpoint.keys(): 
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            mean_loudness = config["data"]["mean_loudness"]
            std_loudness = config["data"]["std_loudness"]
            summary(model)
        elif setting.model == 'resnet':
            model = Resnet()
            checkpoint = torch.load(os.path.join(customPath.models(), model_name, "state.pth"), map_location=device)
            if 'model_state_dict' in checkpoint.keys(): 
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            summary(model)

        print('Trained model loaded.')

        print('Evaluation on the whole test set ...')
        first_example = 1
        orig_audio_list = []
        rec_audio_list = []
        for i_batch, batch in tqdm(enumerate(dataloader)):
            filename_idx = dataset.filenames[i_batch][0].split('/')[-1][:-4]
            chunk_idx = dataset.filenames[i_batch][1]
            if chunk_idx == 0:
                if not first_example:
                    orig_audio_concat = np.concatenate(orig_audio_list)
                    rec_audio_concat = np.concatenate(rec_audio_list)

                    # original WB signal + stft
                    orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)

                    # recontructed stft
                    rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)

                    # we replace the LB with the ground-truth before computing metrics
                    rec_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :] = orig_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :]
                    rec_signal_temp = lr.istft(rec_stft, n_fft = setting.nfft, hop_length = setting.nfft//4)

                    # we compute metrics and store them
                    cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
                    cur_lsd = lsd(orig_stft, rec_stft)
                    all_sdr.append(cur_sdr)
                    all_lsd.append(cur_lsd)

                orig_audio_list = []
                rec_audio_list = []
                first_example = 0

            # output generation from the ddsp model
            if 'ddsp' in setting.model:
                if setting.noiseTraining:
                    add_noise = True
                else:
                    add_noise = False
                if setting.alg in ['ddsp', 'ddsp_poly_decoder']:
                    s_WB, s_LB, p, l = batch
                    s_WB = s_WB.to(device)
                    s_LB = s_LB.to(device)
                    p = p.unsqueeze(-1).to(device)
                    l = l.unsqueeze(-1).to(device)
                    l = (l - mean_loudness) / std_loudness

                    if setting.alg == 'ddsp_poly_decoder':
                        n_sources = p.shape[1]
                        if n_sources < setting.max_n_sources:
                            n_missing_sources = setting.max_n_sources-n_sources
                            p_void = torch.Tensor(np.zeros((1, n_missing_sources, p.shape[2], 1)))                            
                            p = torch.cat((p, p_void), dim=1)
                    
                    if setting.alg == 'ddsp':
                        if setting.model == 'ddsp_noise':
                            y = model(s_LB, l, ).squeeze(-1)
                        else:
                            y = model(s_LB, p, l, add_noise=add_noise, reverb=False).squeeze(-1)
                    elif setting.alg == 'ddsp_poly_decoder':
                        y = model(s_LB, p, l, add_noise=add_noise, reverb=False, n_sources=setting.max_n_sources).squeeze(-1)


                if setting.alg == 'ddsp_multi':
                    s_WB, s_LB, p, l = batch
                    s_WB = s_WB.to(device)
                    s_LB = s_LB.to(device)

                    if s_LB.shape[0] > 1: # we didn't take into account batch_size > 1 here
                        raise ValueError("Not implemented if batch_size > 1")

                    # extract pitch and loudness for all signals per frames
                    if setting.data == 'synthetic' and setting.pitch == 'gt':
                        pitches = [p[0]]
                        loudnesses = [l[0]]
                    elif 'synthetic_poly' in setting.data and (setting.pitch == 'gt' or setting.loudness_gt):
                        pitches, loudnesses = extract_pitches_and_loudnesses_from_filename(filename_idx, fs=config["preprocess"]['sampling_rate'], signal_length=s_LB.shape[1])
                        pitches, loudnesses = samples_to_frames(pitches, loudnesses, config["preprocess"]['sampling_rate'], config["preprocess"]['block_size'])

                    if setting.pitch == 'bittner':
                        pitches = extract_pitch(s_LB.cpu().numpy()[0], alg='bittner', sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])

                    if setting.iteration == 0:
                        if setting.data == 'synthetic':
                            n_iteration = 1
                        elif 'synthetic_poly' in setting.data:
                            n_iteration = count_n_signals(dataset.filenames[i_batch][0].split('/')[-1][:-4])

                        if setting.pitch == 'bittner':
                            n_iteration = pitches.shape[0]
                    else:
                        n_iteration = setting.iteration
                    
                    # inference loop
                    for i_source in range(n_iteration):
                        # remove previously inferred signals
                        if i_source == 0:
                            s_LB_residual = s_LB
                        else:
                            y_mono_stft_mag = np.abs(lr.stft(y_mono.numpy()[0], n_fft=setting.nfft, hop_length=setting.nfft//4))
                            s_LB_residual_stft = lr.stft(s_LB_residual.numpy()[0], n_fft=setting.nfft, hop_length=setting.nfft//4)
                            s_LB_residual_stft_mag = np.abs(s_LB_residual_stft)
                            s_LB_residual_phase = np.angle(s_LB_residual_stft)

                            s_LB_residual_stft_mag[:setting.nfft//(setting.downsampling_factor*2), :] = s_LB_residual_stft_mag[:setting.nfft//(setting.downsampling_factor*2), :] - y_mono_stft_mag[:setting.nfft//(setting.downsampling_factor*2), :]
                            s_LB_residual_stft_mag = s_LB_residual_stft_mag.clip(min=0)

                            s_LB_residual_stft_new = s_LB_residual_stft_mag*np.exp(1j*s_LB_residual_phase)
                            s_LB_residual = np.real(lr.istft(s_LB_residual_stft_new, n_fft=setting.nfft, hop_length=setting.nfft//4, length=y_mono.numpy()[0].size))
                            s_LB_residual = torch.Tensor(s_LB_residual).unsqueeze(0)
                        
                        # pitch
                        if setting.pitch == 'gt':
                            pitch = pitches[i_source]
                        elif setting.pitch == 'crepe':
                            pitch = extract_pitch(s_LB_residual.cpu().numpy()[0], alg='crepe', sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])
                        elif setting.pitch == 'yin':
                            pitch = extract_pitch(s_LB_residual.cpu().numpy()[0], alg='yin', sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])
                        elif setting.pitch == 'bittner':
                            pitch = pitches[i_source]

                        pitch_numpy = np.copy(pitch)
                        pitch = torch.Tensor(pitch)
                        p = pitch.unsqueeze(0).unsqueeze(-1)
                        
                        # gaussian comb filter
                        # if setting.comb_filter:
                        #     comb_filters = gaussian_comb_filters(pitch_numpy, setting.sampling_rate, nfft=setting.nfft)
                        #     s_stft = lr.stft(s_LB_residual.numpy()[0], n_fft=setting.nfft)[:, :-1]
                        #     s_stft_mag = np.abs(s_stft)
                        #     s_stft_pha = np.angle(s_stft)
                        #     s_stft_filtered_mag = s_stft_mag*comb_filters # keeping out only peaks at harmonics
                        #     s_stft_filtered = s_stft_filtered_mag*np.exp(1j*s_stft_pha)
                        #     s_LB_filtered = np.real(lr.istft(s_stft_filtered, n_fft=setting.nfft, hop_length=setting.nfft//2, length=s_LB_residual.numpy()[0].size))

                        # loudness
                        if bool(setting.loudness_gt):
                            if 'synthetic_poly' in setting.data:
                                l = loudnesses[i_source]
                            else:
                                raise ValueError("Ground-truth loudness doesn't exist for other datasets than synthetic_poly")
                        else:
                            # if setting.comb_filter:
                            #     l = extract_loudness(s_LB_filtered, sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])
                            # else:
                            l = extract_loudness(s_LB_residual.cpu().numpy()[0], sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])
                            # if setting.pitch == 'bittner': # we use the activation vector of the pitch estimation alg to adjust the loudness vector
                            #     activation = activations[i_source]
                            #     l = np.log(np.exp(l)*activation+1e-7)
                                
                        l = torch.Tensor(l)
                        l = l.unsqueeze(0).unsqueeze(-1)
                        l = (l - mean_loudness) / std_loudness

                        # inference
                        if setting.noise == 'all' or i_source == (n_iteration-1):
                            # if setting.comb_filter:
                            #     s_LB_filtered = torch.Tensor(s_LB_filtered).unsqueeze(0)
                            #     y_mono = model(s_LB_filtered, p, l, add_noise=add_noise).squeeze(-1)
                            # else:
                            y_mono = model(s_LB_residual, p, l, add_noise=add_noise).squeeze(-1)
                        else:
                            # if setting.comb_filter:
                            #     s_LB_filtered = torch.Tensor(s_LB_filtered).unsqueeze(0)
                            #     y_mono = model(s_LB_filtered, p, l, add_noise=add_noise).squeeze(-1)
                            # else:
                            y_mono = model(s_LB_residual, p, l, add_noise=add_noise).squeeze(-1)

                        if i_source == 0:
                            y = y_mono
                        else:
                            y = y + y_mono
                                    
            elif setting.model == 'resnet':
                s_WB, s_LB = batch
                s_WB = s_WB.unsqueeze(1).to(device)
                s_LB = s_LB.unsqueeze(1).to(device)
                y = model(s_LB)[0]
                s_WB = s_WB[0]

            rec_audio = np.round(y[0].detach().cpu().numpy(), decimals=n_decimals_rounding)
            orig_audio = np.round(s_WB[0].detach().cpu().numpy(), decimals=n_decimals_rounding)
            
            orig_audio_list.append(orig_audio)
            rec_audio_list.append(rec_audio)

        # last batch    
        orig_audio_concat = np.concatenate(orig_audio_list)
        rec_audio_concat = np.concatenate(rec_audio_list)

        orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)
        rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//4)
        rec_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :] = orig_stft[:ceil(setting.nfft//(setting.downsampling_factor*2)), :]

        cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
        cur_lsd = lsd(orig_stft[ceil(setting.nfft//(setting.downsampling_factor*2)):], rec_stft[ceil(setting.nfft//(setting.downsampling_factor*2)):])
        all_sdr.append(cur_sdr)
        all_lsd.append(cur_lsd)
        logging.info(filename_idx+' '+str(cur_lsd))
        print('Evaluation done.')
            
    toc = time.time()
    elapsed_time = int(toc-tic)

    all_sdr = np.array(all_sdr)
    all_lsd = np.array(all_lsd)

    return all_sdr, all_lsd, elapsed_time
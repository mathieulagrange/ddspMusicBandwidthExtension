from metrics import sdr, lsd
from sbr import sbr
import numpy as np
import time
import librosa as lr
from tqdm import tqdm
from scipy.io.wavfile import write
import os
import customPath
from math import ceil
from model_ddsp import DDSP
from model_resnet import Resnet
import torch
from preprocess import Dataset
import yaml
from effortless_config import Config

torch.set_grad_enabled(False)
device = torch.device("cpu")
seed = 4
torch.manual_seed(4)

def evaluate(setting, experiment):
    tic = time.time()

    # test dataset instantiation
    if not hasattr(setting, 'model') or 'ddsp' in setting.model:
        if setting.data == 'sol':
            if setting.split == 'train':
                data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/train')
            elif setting.split == 'test':
                data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test')
        elif setting.data == 'medley':
            if setting.split == 'train':
                data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/train')
            if setting.split == 'test':
                data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/test')
        elif setting.data == 'gtzan':
            if setting.split == 'train':
                data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp/train')
            elif setting.split == 'test':
                data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp/test')
        elif setting.data == 'synthetic':
            if setting.split == 'train':
                data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/train')
            elif setting.split == 'test':
                data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/test')
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

    elif setting.model == 'resnet':
        if setting.data == 'synthetic':
            if setting.split == 'train':
                data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet/train')
            elif setting.split == 'test':
                data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet/test')
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


    if setting.alg == 'ddsp':
        if setting.model == 'ddsp_original_autoencoder':
            dataset = Dataset(data_dir, model='ddsp')
        elif setting.model == 'resnet':
            dataset = Dataset(data_dir, model='resnet')
    else:
        dataset = Dataset(data_dir, model='ddsp')
    dataloader = torch.utils.data.DataLoader(dataset, 1, False)
    
    # prepare metrics for each example
    all_sdr = []
    all_lsd = []
    
    if setting.alg != 'ddsp':
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
                if not hasattr(setting, 'model') or setting.model == 'ddsp_original_autoencoder':
                    filename_idx = dataset.filenames[i].split('/')[-1][:-4]
                    chunk_idx = 0
                elif setting.model == 'resnet':
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

            if chunk_idx == 0:
                if not first_example:
                    orig_audio_concat = np.concatenate(orig_audio_list)
                    rec_audio_concat = np.concatenate(rec_audio_list)

                    # original WB signal + stft
                    orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)

                    # recontructed stft
                    rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)

                    # we replace the LB with the ground-truth before computing metrics
                    rec_stft[:ceil(setting.nfft//4), :] = orig_stft[:ceil(setting.nfft//4), :]

                    # we compute metrics and store them
                    cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
                    cur_lsd = lsd(orig_stft[ceil(setting.nfft//4):], rec_stft[ceil(setting.nfft//4):])
                    all_sdr.append(cur_sdr)
                    all_lsd.append(cur_lsd)

                orig_audio_list = []
                rec_audio_list = []
                first_example = 0


            s_WB, s_LB, p, l = test_data
            audio = s_WB[0].detach().cpu().numpy()

            ### ORACLE ALGO ###
            if setting.alg == 'oracle':
                reconstructed_audio = audio
                stft = lr.stft(audio, n_fft = setting.nfft, hop_length = setting.nfft//2)
                reconstructed_stft = lr.stft(reconstructed_audio, n_fft = setting.nfft, hop_length = setting.nfft//2)

                orig_audio_list.append(audio)
                rec_audio_list.append(reconstructed_audio)
                # cur_sdr = sdr(audio, reconstructed_audio)
                # cur_lsd = lsd(stft, reconstructed_stft, n_fft = setting.nfft, hop_length = setting.nfft//2)

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

                orig_audio_list.append(audio)
                rec_audio_list.append(reconstructed_audio)

                # cur_sdr = sdr(audio, reconstructed_audio)
                # cur_lsd = lsd(stft[ceil(setting.nfft//4):], reconstructed_stft[ceil(setting.nfft//4):], n_fft = setting.nfft, hop_length = setting.nfft//2)

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

        orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)
        rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)
        rec_stft[:ceil(setting.nfft//4), :] = orig_stft[:ceil(setting.nfft//4), :]

        cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
        cur_lsd = lsd(orig_stft[ceil(setting.nfft//4):], rec_stft[ceil(setting.nfft//4):])
        all_sdr.append(cur_sdr)
        all_lsd.append(cur_lsd)

    else:
        ### DDSP ALGO ###               
        # model name
        if setting.sampling_rate == 16000:
            if setting.model == 'ddsp_original_autoencoder':
                if setting.data == 'synthetic':
                    if setting.loss == 'WB':
                        model_name = 'bwe_synth_8000_100harmo_5000steps_batch32_lossWB'
                    else:
                        model_name = 'bwe_synth_8000_100harmo_5000steps_batch32_lossHB'
                elif setting.data == 'sol':
                    if setting.loss == 'WB':
                        model_name = 'bwe_sol_8000_100harmo_25000steps_batch32_lossWB'
                    else:
                        model_name = 'bwe_sol_8000_100harmo_25000steps_batch32_lossHB'
                elif setting.data == 'medley':
                    if setting.loss == 'WB':
                        model_name = 'bwe_medley_8000_100harmo_25000steps_batch32_lossWB'
                    elif setting.loss == 'HB':
                        model_name = 'bwe_medley_8000_100harmo_25000steps_batch32_lossHB'
                elif setting.data == 'dsd_sources':
                    if setting.loss == 'WB':
                        model_name = 'bwe_dsd_sources_100harmo_25000steps_batch32_lossWB'
                    elif setting.loss == 'HB':
                        model_name = 'bwe_dsd_sources_100harmo_25000steps_batch32_lossHB'
                elif setting.data == 'dsd_mixtures':
                    if setting.loss == 'WB':
                        model_name = 'bwe_dsd_mixtures_100harmo_25000steps_batch32_lossWB'
                    elif setting.loss == 'HB':
                        model_name = 'bwe_dsd_mixtures_100harmo_25000steps_batch32_lossHB'

            elif setting.model == 'resnet':
                if setting.data == 'synthetic':
                    model_name = 'bwe_synth_resnet_64000steps_batch16'
                elif setting.data == 'sol':
                    model_name = 'bwe_sol_resnet_64000steps_batch8'
                elif setting.data == 'medley':
                    model_name = 'bwe_medley_resnet_250000steps_batch16'
                elif setting.data == 'dsd_sources':
                    model_name = 'bwe_dsd_sources_resnet_250000steps_batch8'
                elif setting.data == 'dsd_mixtures':
                    model_name = 'bwe_dsd_mixtures_resnet_250000steps_batch8'
 
        elif setting.sampling_rate == 8000:
            if setting.model == 'ddsp_original_autoencoder':
                if setting.data == 'sol':
                    if setting.loss == 'WB':
                        model_name = 'bwe_sol_100harmo_25000steps_batch32_lossWB'
                    else:
                        model_name = 'bwe_sol_100harmo_25000steps_batch32_lossHB'
                elif setting.data == 'medley':
                    if setting.loss == 'WB':
                        model_name = 'bwe_medley_100harmo_25000steps_batch32_lossWB'
                    elif setting.loss == 'HB':
                        model_name = 'bwe_medley_100harmo_25000steps_batch32_lossHB'
                elif setting.data == 'synthetic':
                    if setting.loss == 'WB':
                        model_name = 'bwe_synth_100harmo_5000steps_batch32_lossWB'
                    else:
                        model_name = 'bwe_synth_100harmo_5000steps_batch32_lossHB'
                elif setting.data == 'dsd_sources':
                    if setting.loss == 'WB':
                        model_name = 'bwe_dsd_sources_100harmo_25000steps_batch32_lossWB'
                    elif setting.loss == 'HB':
                        model_name = 'bwe_dsd_sources_100harmo_25000steps_batch32_lossHB'
                elif setting.data == 'dsd_mixtures':
                    if setting.loss == 'WB':
                        model_name = 'bwe_dsd_mixtures_100harmo_25000steps_batch32_lossWB'
                    elif setting.loss == 'HB':
                        model_name = 'bwe_dsd_mixtures_100harmo_25000steps_batch32_lossHB'


        # config file loading
        with open(os.path.join(customPath.models(), model_name, "config.yaml"), "r") as config:
            config = yaml.safe_load(config)
        
        # load model
        if setting.model == 'ddsp_original_autoencoder':
            model = DDSP(**config["model"])
            model.load_state_dict(torch.load(os.path.join(customPath.models(), model_name, "state.pth"), map_location=torch.device('cpu')))
            model.eval()
            mean_loudness = config["data"]["mean_loudness"]
            std_loudness = config["data"]["std_loudness"]
        elif setting.model == 'resnet':
            model = model = Resnet()
            model.load_state_dict(torch.load(os.path.join(customPath.models(), model_name, "state.pth"), map_location=torch.device('cpu')))
            model.eval()

        print('Trained model loaded.')

        print('Evaluation on the whole test set ...')
        first_example = 1
        orig_audio_list = []
        rec_audio_list = []
        for i_batch, batch in tqdm(enumerate(dataloader)):
            if config['data']['dataset'] == 'synthetic':
                if setting.model == 'ddsp_original_autoencoder':
                    filename_idx = dataset.filenames[i_batch].split('/')[-1][:-4]
                    chunk_idx = 0
                elif setting.model == 'resnet':
                    filename_idx = dataset.filenames[i_batch][0].split('/')[-1][:-4]
                    chunk_idx = dataset.filenames[i_batch][1]
            elif config['data']['dataset'] == 'sol':
                filename_idx = dataset.filenames[i_batch][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i_batch][1]
            elif config['data']['dataset'] == 'medley':
                filename_idx = dataset.filenames[i_batch][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i_batch][1]
            elif config['data']['dataset'] == 'dsd_sources':
                filename_idx = dataset.filenames[i_batch][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i_batch][1]
            elif config['data']['dataset'] == 'dsd_mixtures':
                filename_idx = dataset.filenames[i_batch][0].split('/')[-1][:-4]
                chunk_idx = dataset.filenames[i_batch][1]

            if chunk_idx == 0:
                if not first_example:
                    orig_audio_concat = np.concatenate(orig_audio_list)
                    rec_audio_concat = np.concatenate(rec_audio_list)

                    # original WB signal + stft
                    orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)

                    # recontructed stft
                    rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)

                    # we replace the LB with the ground-truth before computing metrics
                    rec_stft[:ceil(setting.nfft//4), :] = orig_stft[:ceil(setting.nfft//4), :]

                    # we compute metrics and store them
                    cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
                    cur_lsd = lsd(orig_stft[ceil(setting.nfft//4):], rec_stft[ceil(setting.nfft//4):])
                    all_sdr.append(cur_sdr)
                    all_lsd.append(cur_lsd)

                orig_audio_list = []
                rec_audio_list = []
                first_example = 0

            # output generation from the ddsp model
            if 'ddsp' in setting.model:
                s_WB, s_LB, p, l = batch
                s_WB = s_WB.to(device)
                s_LB = s_LB.to(device)
                p = p.unsqueeze(-1).to(device)
                l = l.unsqueeze(-1).to(device)
                l = (l - mean_loudness) / std_loudness
                y = model(s_LB, p, l).squeeze(-1)
            
            elif setting.model == 'resnet':
                s_WB, s_LB = batch
                s_WB = s_WB.unsqueeze(1).to(device)
                s_LB = s_LB.unsqueeze(1).to(device)
                y = model(s_LB)[0]
                s_WB = s_WB[0]

            rec_audio = y[0].detach().cpu().numpy()
            orig_audio = s_WB[0].detach().cpu().numpy()
            
            orig_audio_list.append(orig_audio)
            rec_audio_list.append(rec_audio)

        # last batch    
        orig_audio_concat = np.concatenate(orig_audio_list)
        rec_audio_concat = np.concatenate(rec_audio_list)

        orig_stft = lr.stft(orig_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)
        rec_stft = lr.stft(rec_audio_concat, n_fft = setting.nfft, hop_length = setting.nfft//2)
        rec_stft[:ceil(setting.nfft//4), :] = orig_stft[:ceil(setting.nfft//4), :]

        cur_sdr = sdr(orig_audio_concat, rec_audio_concat)
        cur_lsd = lsd(orig_stft[ceil(setting.nfft//4):], rec_stft[ceil(setting.nfft//4):])
        all_sdr.append(cur_sdr)
        all_lsd.append(cur_lsd)

        print('Evaluation done.')
            
    toc = time.time()
    elapsed_time = int(toc-tic)

    all_sdr = np.array(all_sdr)
    all_lsd = np.array(all_lsd)

    return all_sdr, all_lsd, elapsed_time
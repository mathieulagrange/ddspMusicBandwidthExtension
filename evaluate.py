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

class args(Config):
    NAME = "debug"
    DATASET = "synthetic"
    MAX_N_SOURCES = 5
    CYCLIC = False

args.parse_args()

with open(os.path.join(customPath.models(), args.NAME, f'{args.NAME}.yaml'), "r") as config:
    config = yaml.safe_load(config)

np.set_printoptions(precision=10)
torch.set_printoptions(precision=10)
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 4
torch.manual_seed(4)
n_decimals_rounding = 5

tic = time.time()

# load dataset
if 'ddsp' in config['train']['model']:
    if args.DATASET == 'synthetic':
        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_ddsp/test')
    elif args.DATASET == 'synthetic_poly':
        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_ddsp/test')
    elif args.DATASET == 'sol':
        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test')
    elif args.DATASET == 'medley':
        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/test')
    elif args.DATASET == 'gtzan':
        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_ddsp/test')
    elif args.DATASET == 'medleyDB_mixtures':
        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_ddsp/test')

elif config['train']['model'] == 'resnet':
    if args.DATASET == 'synthetic':
        data_dir = os.path.join(customPath.synthetic(), 'preprocessed_resnet/test')
    elif args.DATASET == 'synthetic_poly':
        data_dir = os.path.join(customPath.synthetic_poly(), 'preprocessed_resnet/test')
    elif args.DATASET == 'sol':
        data_dir = os.path.join(customPath.orchideaSOL(), 'preprocessed_resnet/test')
    elif args.DATASET == 'medley':
        data_dir = os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/test')
    elif args.DATASET == 'gtzan':
        data_dir = os.path.join(customPath.gtzan(), 'preprocessed_resnet/test')
    elif args.DATASET == 'medleyDB_mixtures':
        data_dir = os.path.join(customPath.medleyDB_mixtures(), 'preprocessed_resnet/test')


if config['train']['model'] == 'ddsp_poly_decoder':
    dataset = Dataset(data_dir, model='ddsp_poly_decoder')
else:
    if 'ddsp' in config['train']['model']:
        dataset = Dataset(data_dir, model='ddsp')
    elif config['train']['model'] == 'resnet':
        dataset = Dataset(data_dir, model='resnet')
dataloader = torch.utils.data.DataLoader(dataset, 1, False)

# prepare metric for each example
all_lsd = []

# loading trained model
print(f'Loading model: {args.NAME} ...')

# load model
print(config['train']['model'])
if 'ddsp' in config['train']['model']:
    if config['train']['model'] == 'ddsp':
        model = DDSP(**config["model"])
    elif config['train']['model'] == 'ddsp_poly_decoder':
        model = DDSPMulti(**config['model'])
    elif config['train']['model'] == 'ddsp_non_harmo':
        model = DDSPNonHarmonic(**config['model'])
    elif config['train']['model'] == 'ddsp_noise':
        model = DDSPNoise(**config['model'])

    checkpoint = torch.load(os.path.join(customPath.models(), args.NAME, "state.pth"), map_location=device)
    if 'model_state_dict' in checkpoint.keys(): 
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    mean_loudness = config["data"]["mean_loudness"]
    std_loudness = config["data"]["std_loudness"]

elif config['train']['model'] == 'resnet':
    model = Resnet()
    checkpoint = torch.load(os.path.join(customPath.models(), args.NAME, "state.pth"), map_location=device)
    if 'model_state_dict' in checkpoint.keys(): 
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

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
            orig_stft = lr.stft(orig_audio_concat, n_fft = config['model']['block_size']*4, hop_length = config['model']['block_size'])

            # recontructed stft
            rec_stft = lr.stft(rec_audio_concat, n_fft = config['model']['block_size']*4, hop_length = config['model']['block_size'])

            # we replace the LB with the ground-truth before computing metrics
            rec_stft[:ceil(config['model']['block_size']*4//(config['preprocess']['downsampling_factor']*2)), :] = orig_stft[:ceil(config['model']['block_size']*4//(config['preprocess']['downsampling_factor']*2)), :]
            rec_signal_temp = lr.istft(rec_stft, n_fft = config['model']['block_size']*4, hop_length = config['model']['block_size'])

            # we compute metrics and store them
            cur_lsd = lsd(orig_stft, rec_stft)
            all_lsd.append(cur_lsd)

        orig_audio_list = []
        rec_audio_list = []
        first_example = 0

    # output generation from the ddsp model
    if 'ddsp' in config['train']['model']:
        if not args.CYCLIC and config['train']['model'] in ['ddsp', 'ddsp_poly_decoder']:
            s_WB, s_LB, p, l = batch
            s_WB = s_WB.to(device)
            s_LB = s_LB.to(device)
            p = p.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)
            l = (l - mean_loudness) / std_loudness

            if config['train']['model'] == 'ddsp_poly_decoder':
                n_sources = p.shape[1]
                if n_sources < args.MAX_N_SOURCES:
                    n_missing_sources = args.MAX_N_SOURCES-n_sources
                    p_void = torch.Tensor(np.zeros((1, n_missing_sources, p.shape[2], 1)))                            
                    p = torch.cat((p, p_void), dim=1)
            
            if config['train']['model'] == 'ddsp':
                if config['train']['model'] == 'ddsp_noise':
                    y = model(s_LB, l, ).squeeze(-1)
                else:
                    y = model(s_LB, p, l, add_noise=True, reverb=False).squeeze(-1)
            elif config['train']['model'] == 'ddsp_poly_decoder':
                y = model(s_LB, p, l, add_noise=True, reverb=False, n_sources=args.MAX_N_SOURCES).squeeze(-1)


        if args.CYCLIC:
            s_WB, s_LB, p, l = batch
            s_WB = s_WB.to(device)
            s_LB = s_LB.to(device)

            if s_LB.shape[0] > 1: # we didn't take into account batch_size > 1 here
                raise ValueError("Not implemented if batch_size > 1")

            # extract pitch and loudness for all signals per frames
            pitches = extract_pitch(s_LB.cpu().numpy()[0], alg='bittner', sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])
            n_iteration = pitches.shape[0]

            # if args.DATASET == 'synthetic':
            #     n_iteration = 1
            # elif 'synthetic_poly' in args.DATASET:
            #     n_iteration = count_n_signals(dataset.filenames[i_batch][0].split('/')[-1][:-4])

            # inference loop
            for i_source in range(n_iteration):
                # remove previously inferred signals
                if i_source == 0:
                    s_LB_residual = s_LB
                else:
                    y_mono_stft_mag = np.abs(lr.stft(y_mono.numpy()[0], n_fft=config['model']['block_size']*4, hop_length=config['model']['block_size']))
                    s_LB_residual_stft = lr.stft(s_LB_residual.numpy()[0], n_fft=config['model']['block_size']*4, hop_length=config['model']['block_size'])
                    s_LB_residual_stft_mag = np.abs(s_LB_residual_stft)
                    s_LB_residual_phase = np.angle(s_LB_residual_stft)

                    s_LB_residual_stft_mag[:config['model']['block_size']//2, :] = s_LB_residual_stft_mag[:config['model']['block_size']//2, :] - y_mono_stft_mag[:config['model']['block_size']//2, :]
                    s_LB_residual_stft_mag = s_LB_residual_stft_mag.clip(min=0)

                    s_LB_residual_stft_new = s_LB_residual_stft_mag*np.exp(1j*s_LB_residual_phase)
                    s_LB_residual = np.real(lr.istft(s_LB_residual_stft_new, n_fft=config['model']['block_size']*4, hop_length=config['model']['block_size'], length=y_mono.numpy()[0].size))
                    s_LB_residual = torch.Tensor(s_LB_residual).unsqueeze(0)
                
                # pitch
                pitch = pitches[i_source]
                pitch_numpy = np.copy(pitch)
                pitch = torch.Tensor(pitch)
                p = pitch.unsqueeze(0).unsqueeze(-1)
                
                # loudness
                l = extract_loudness(s_LB_residual.cpu().numpy()[0], sampling_rate=config["preprocess"]['sampling_rate'], block_size=config["preprocess"]['block_size'])
                l = torch.Tensor(l)
                l = l.unsqueeze(0).unsqueeze(-1)
                l = (l - mean_loudness) / std_loudness

                # inference
                if i_source == (n_iteration-1):
                    y_mono = model(s_LB_residual, p, l, add_noise=True).squeeze(-1)
                else:
                    y_mono = model(s_LB_residual, p, l, add_noise=False).squeeze(-1)

                if i_source == 0:
                    y = y_mono
                else:
                    y = y + y_mono
                            
    elif config['train']['model'] == 'resnet':
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

orig_stft = lr.stft(orig_audio_concat, n_fft = config['model']['block_size']*4, hop_length = config['model']['block_size'])
rec_stft = lr.stft(rec_audio_concat, n_fft = config['model']['block_size']*4, hop_length = config['model']['block_size'])
rec_stft[::ceil(config['model']['block_size']*4//(config['preprocess']['downsampling_factor']*2)), :] = orig_stft[::ceil(config['model']['block_size']*4//(config['preprocess']['downsampling_factor']*2)), :]

cur_lsd = lsd(orig_stft[:ceil(config['model']['block_size']*4//(config['preprocess']['downsampling_factor']*2)):], rec_stft[:ceil(config['model']['block_size']*4//(config['preprocess']['downsampling_factor']*2)):])
all_lsd.append(cur_lsd)
print('Evaluation done.')
        
toc = time.time()
elapsed_time = int(toc-tic)

all_lsd = np.array(all_lsd)

print(f'Evaluation of model {args.NAME} on dataset {args.DATASET} done in {elapsed_time} seconds')
print(f'Average LSD: {np.mean(all_lsd)}')
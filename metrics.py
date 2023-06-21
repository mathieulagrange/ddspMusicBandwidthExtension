import numpy as np
import librosa as lr
import librosa as lr
from tqdm import tqdm
import numpy as np
import os
from math import ceil

epsilon = 10e-6

def sdr(ref, est):
    '''Calculate the signal-to-distortion ratio between a reference signal and its estimation
    '''
    ref_energy = np.sum(np.square(ref))
    dist_energy = np.sum(np.square(ref-est))

    if dist_energy != 0:
        return 10*np.log10(ref_energy/dist_energy)
    else:
        return np.inf

def lsd(ref, est, n_fft=1024, hop_length=512):
    '''Calculate the log-spectral distance between a reference signal and its estimation
    '''
    spec_ref = np.round(np.abs(ref), decimals=3)
    spec_est = np.round(np.abs(est), decimals=3)

    log_diff = 10*(np.log10(np.square(spec_est)+epsilon)-np.log10((np.square(spec_ref)+epsilon)))
    freq_mean = np.mean(np.square(log_diff[:, :-1]), axis=0)
    lsd = np.mean(np.sqrt(freq_mean))
    lsd = np.round(lsd, decimals=2)
    return lsd

# def compute_lsd(model_dir, model_type, data, split, step, nfft=1024, hop_length=512, sampling_rate=16000):
#     # dataset
#     if data == 'sol':
#         # dataset = OrchideaSol(split, 4, sampling_rate, 250)
#         tfrecord_pattern = os.path.join(customPath.orchideaSOL(), 'test.tfrecord-*')
#         dataset = ddsp.training.data.TFRecordProvider(tfrecord_pattern)
#     elif data == 'tiny':
#         dataset = OrchideaSolTiny(split, 4, sampling_rate, 250)
#     elif data == 'medley':
#         dataset = MedleySolosDB(split, 4, sampling_rate, 250)
#     elif data == 'gtzan':
#         dataset = Gtzan(split, 4, sampling_rate, 250)
#     ds = dataset.get_dataset()
#     ds = ds.batch(batch_size=1)

#     ds_length = 0
#     for ex in ds.batch(batch_size=1):
#         ds_length += 1

#     # initialize lsd array
#     all_lsd = np.empty((ds_length))

#     # load model
#     if model_type == 'original_autoencoder':
#         model = OriginalAutoencoder(sampling_rate = sampling_rate)
#     elif model_type == 'resnet':
#         model = SulunResNet()

#     model.restore(os.path.join(model_dir, 'train_files'))
#     print('Trained model loaded.')

#     print(f'Evaluation on the whole {split} set ...')
#     for i_batch, batch in tqdm(enumerate(ds)):
#         # output generation from the ddsp model
#         outputs = model(batch, training=False)

#         # reconstructed signal + recontructed stft
#         if model_type == 'original_autoencoder':
#             reconstructed_audio = model.get_audio_from_outputs(outputs).numpy()[0]
#         elif model_type == 'resnet':
#             reconstructed_audio = outputs['audio_synth'].numpy()[0, :, 0]
#         reconstructed_stft = lr.stft(reconstructed_audio, n_fft = nfft, hop_length = nfft//2)

#         # original WB signal + stft
#         audio = batch['audio'].numpy()[0]
#         stft = lr.stft(audio, n_fft = nfft, hop_length = nfft//2)

#         # we replace the LB with the ground-truth before computing metrics
#         reconstructed_stft[:ceil(nfft//4), :] = stft[:ceil(nfft//4), :]

#         # we compute metrics and store them
#         cur_lsd = lsd(stft[ceil(nfft//4):], reconstructed_stft[ceil(nfft//4):])

#         all_lsd[i_batch] = cur_lsd

#     np.save(os.path.join(model_dir, 'train_files', f'lsd_{split}_step{step}'), all_lsd)

# def compute_mss(model_dir, model_type, data, split, step, fft_sizes=[64, 128, 256, 512, 1024, 2048], sampling_rate=16000, loss='WB'):
#     # dataset
#     if data == 'sol':
#         # dataset = OrchideaSol(split, 4, sampling_rate, 250)
#         tfrecord_pattern = os.path.join(customPath.orchideaSOL(), 'test.tfrecord-*')
#         dataset = ddsp.training.data.TFRecordProvider(tfrecord_pattern)
#     elif data == 'tiny':
#         dataset = OrchideaSolTiny(split, 4, sampling_rate, 250)
#     elif data == 'medley':
#         dataset = MedleySolosDB(split, 4, sampling_rate, 250)
#     elif data == 'gtzan':
#         dataset = Gtzan('test', 4, sampling_rate, 250)
#     ds = dataset.get_dataset()
#     ds = ds.batch(batch_size=1)

#     ds_length = 0
#     for ex in ds.batch(batch_size=1):
#         ds_length += 1

#     # initialize lsd array
#     all_mss = np.empty((ds_length))

#     # initialize MSS loss
#     if loss == 'WB':
#         mss = losses.SpectralLoss(loss_type = 'L1', mag_weight = 1.0, logmag_weight = 1.0)
#     elif loss == 'HB':
#         mss = custom_losses.HF_SpectralLoss(
#                 loss_type = 'L1',
#                 mag_weight = 1.0,
#                 logmag_weight = 1.0
#             )

#     # load model
#     if model_type == 'original_autoencoder':
#         model = OriginalAutoencoder(sampling_rate = sampling_rate, loss=loss)
#     elif model_type == 'resnet':
#         model = SulunResNet()

#     model.restore(os.path.join(model_dir, 'train_files'))
#     print('Trained model loaded.')

#     print(f'Evaluation on the whole {split} set ...')
#     for i_batch, batch in tqdm(enumerate(ds)):
#         # output generation from the ddsp model
#         outputs = model(batch, training=False)

#         # reconstructed signal
#         reconstructed_audio = model.get_audio_from_outputs(outputs).numpy()[0]

#         # original WB signal
#         audio = batch['audio'].numpy()[0]

#         # we compute metrics and store them
#         cur_mss = mss(audio, reconstructed_audio)

#         all_mss[i_batch] = cur_mss

#     np.save(os.path.join(model_dir, 'train_files', f'mss_{split}_step{step}'), all_mss)
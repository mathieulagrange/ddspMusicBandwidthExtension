import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
import math
import matplotlib.pyplot as plt
from utils import _PITCHES, _PITCHES_MIDI_NUMBER
import librosa as lr
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict, unwrap_output
from basic_pitch.note_creation import model_output_to_notes


MIN_LOUDNESS = -16.71

def safe_log(x):
    return torch.log(x + 1e-7)


@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap, HF=False, downsampling_factor=2):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        if HF:
            S = S[:, s//(downsampling_factor*2):, :]
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate/2).float() + 1e-4
    aa = aa.permute(0, 2, 1)
    return amplitudes * aa

def remove_outside_HB(amplitudes, freqs, sampling_rate):
    n_harm = amplitudes.shape[1]
    up_out = (freqs < sampling_rate/2).float() + 1e-4
    down_out = (freqs > sampling_rate//2).float() + 1e-4
    # up_out = up_out.permute(0, 2, 1)
    # down_out = down_out.permute(0, 2, 1)
    return amplitudes * up_out * down_out

def pitch_estimation_bittner(signal, sampling_rate, hop_length=256, window_length=2, block_size=256):
    # resample signal
    new_fs = 22050
    signal_resampled = lr.resample(signal, orig_sr=sampling_rate, target_sr=new_fs)

    # constants
    audio_n_samples = new_fs*window_length-hop_length
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames*hop_length
    hop_size = audio_n_samples-overlap_len
    signal_length = signal_resampled.shape[-1]
    # padding input signal
    signal_pad = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), signal_resampled])

    # import here to save import loading time
    from tensorflow import expand_dims, saved_model
    from tensorflow import signal as signal_tf
    
    # framing signal
    audio_windowed = expand_dims(
        signal_tf.frame(signal_pad, audio_n_samples, hop_size, pad_end=True, pad_value=0),
        axis=-1,
    )
    window_times = [
        {
            "start": t_start,
            "end": t_start + (audio_n_samples / new_fs),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / new_fs
    ]

    # model inference
    model = saved_model.load(str(ICASSP_2022_MODEL_PATH))
    output = model(audio_windowed)
    unwrapped_output = {k: unwrap_output(output[k], signal_length, n_overlapping_frames) for k in output}

    # output management
    onset_threshold = 0.5
    frame_threshold = 0.3
    minimum_note_length = 127.70
    min_note_len = int(np.round(minimum_note_length / 1000 * (new_fs / 256)))

    minimum_frequency = 1
    maximum_frequency = 8000
    multiple_pitch_bends = False
    melodia_trick = True

    midi_data, note_events = model_output_to_notes(
        unwrapped_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,  # convert to frames
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
    )

    # transform midi_data to list of framewise pitches
    all_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_notes.append(note)

    pitches = []
    activations = []
    for note in all_notes:
        start = np.round(note.start, decimals=5)
        end = np.round(note.end, decimals=5)
        pitch = note.pitch

        sample_start = round(start*sampling_rate)
        sample_end = round(end*sampling_rate)
        f0 = lr.midi_to_hz(pitch)
        pitch_array = f0*np.ones(signal.shape[-1])
        activation_array = np.zeros((signal.shape[-1]))
        if sample_end >= signal.shape[-1]:
            activation_array[sample_start:] = np.ones((signal.shape[-1]-sample_start))
        else:
            activation_array[sample_start:sample_end] = np.ones((sample_end-sample_start))
    
        pitches.append(pitch_array)
        activations.append(activation_array)

    # if no notes found by bittner alg
    if pitches == []:
        pitches = [np.zeros((signal.shape[-1]))]
    if activations == []:
        activations = [np.zeros((signal.shape[-1]))]

    pitches = samples_to_frames_generic_signal(pitches, fs=sampling_rate, block_size=block_size, method='mean')
    activations = samples_to_frames_generic_signal(activations, fs=sampling_rate, block_size=block_size, method='max')

    different_pitches = sorted(list(set([e[0] for e in pitches])))
    n_diff_pitches = len(different_pitches)
    indiv_pitches = np.zeros((n_diff_pitches, pitches[0].size), dtype=np.float32)
    for i_note in range(len(pitches)):
        cur_pitch = pitches[i_note][0]
        idx_pitch = different_pitches.index(cur_pitch)
        indiv_pitches[idx_pitch] += pitches[i_note]*activations[i_note]

    for i_pitch in range(n_diff_pitches):
        indiv_pitches[i_pitch] = np.clip(indiv_pitches[i_pitch], a_min=0, a_max=different_pitches[i_pitch])

    return indiv_pitches, activations

def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sr = sampling_rate, n_fft = n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S

def extract_pitch(signal, alg, sampling_rate, block_size):
    length = signal.shape[-1] // block_size
    
    if alg == 'crepe':
        f0 = crepe.predict(
            signal,
            sampling_rate,
            step_size=int(1000 * block_size / sampling_rate),
            verbose=0,
            center=True,
            viterbi=True,
        )
        f0 = f0[1].reshape(-1)[:-1]
    
    elif alg == 'yin':
        f0 = lr.yin(
            signal,
            fmin = 1,
            fmax = sampling_rate//2,
            frame_length = block_size,
            hop_length = block_size,
        )
        f0 = f0[:-1]

    elif alg == 'bittner':
        f0, activations = pitch_estimation_bittner(signal, sampling_rate=sampling_rate)
    
    if alg != 'bittner' and f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    if alg == 'bittner':
        return f0#, activations
    else:
        return f0

def extract_pitch_from_filename(f, dataset, fs, frame_size):
    f = str(f)
    x, sr = li.load(f, fs)
    duration = x.size//fs

    if dataset == "synthetic":
        attributes = f.split('/')[-1].split('_')
        print(attributes)
        att = float(attributes[4][3:])
        sus = float(attributes[5][3:])
        dec = float(attributes[6][3:])
        pitch_duration = int((att+sus+dec)*fs//frame_size)
        f0 = float(attributes[1][1:])
    
    f0_per_sample = np.zeros((duration*fs//frame_size))
    f0_per_sample[:pitch_duration] = f0

    return f0_per_sample

def count_n_signals(f):
    attributes = f.split('_')
    n_signals = len(attributes)//6
    return n_signals

def extract_pitches_and_loudnesses_from_filename(f, fs, signal_length):
    attributes = f.split('_')
    n_signals = len(attributes)//6
    
    pitches = []
    loudnesses = []
    for i in range(n_signals):
        f0 = round(float(lr.midi_to_hz(_PITCHES_MIDI_NUMBER[_PITCHES.index(attributes[i*6+0])])), 2)
        gain = float(attributes[i*6+1][4:])
        max_loudness = float(attributes[i*6+2][2:])
        att = float(attributes[i*6+3][1:])
        sus = float(attributes[i*6+4][1:])
        dec = float(attributes[i*6+5][1:])
        pitch_duration = int((att+sus+dec)*fs)
        
        pitch = np.zeros((signal_length))
        pitch[:pitch_duration] = f0
        loudness_db = compute_loudness_from_asd(signal_length, fs, max_loudness, gain, att, sus, dec)

        pitches.append(pitch)
        loudnesses.append(loudness_db)

    return pitches, loudnesses

def compute_loudness_from_asd(signal_length, fs, max_loudness, gain, att, sus, dec):
    max_loudness_lin = pow(10, max_loudness)
    min_loudness_lin = pow(10, MIN_LOUDNESS)

    loudness = np.ones((signal_length))*min_loudness_lin

    loudness[:round(att*fs)] = np.linspace(min_loudness_lin, max_loudness_lin*gain, round(fs*att))
    loudness[round(att*fs):round((att+sus)*fs)] = np.ones(round(sus*fs))*max_loudness_lin*gain
    loudness[round((att+sus)*fs):round((att+sus+dec)*fs)] = np.linspace(max_loudness_lin*gain, min_loudness_lin, round(fs*dec))
    
    loudness_db = np.log10(loudness)

    return loudness_db

def samples_to_frames(pitches, loudnesses, fs, block_size):
    n_signals = len(pitches)
    n_frames = pitches[0].size//block_size
    pitches_frames = []
    loudnesses_frames = []
    for i in range(n_signals):
        cur_pitch = pitches[i]
        cur_loudness = loudnesses[i]
        new_pitch = np.zeros((n_frames))
        new_loudness = np.zeros((n_frames))

        for i_frame in range(n_frames):
            new_pitch[i_frame] = np.mean(cur_pitch[i_frame*block_size:(i_frame+1)*block_size])
            new_loudness[i_frame] = np.mean(cur_loudness[i_frame*block_size:(i_frame+1)*block_size])

        pitches_frames.append(new_pitch)
        loudnesses_frames.append(new_loudness)

    return pitches_frames, loudnesses_frames

def samples_to_frames_generic_signal(signals, fs, block_size, method='mean'):
    n_signals = len(signals)
    n_frames = signals[0].size//block_size
    signals_frames = []
    for i in range(n_signals):
        cur_signal = signals[i]
        new_signal = np.zeros((n_frames))

        for i_frame in range(n_frames):
            if method == 'mean':
                new_signal[i_frame] = np.mean(cur_signal[i_frame*block_size:(i_frame+1)*block_size])
            elif method == 'max':
                new_signal[i_frame] = np.max(cur_signal[i_frame*block_size:(i_frame+1)*block_size])
        
        signals_frames.append(new_signal)

    return signals_frames


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal

def additive_synth(frequencies, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omegas = torch.cumsum(2 * math.pi * frequencies / sampling_rate, 1)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal

def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output

def gaussian_comb_filters(f0_array, fs, nfft=1024, peak_width_semitones=1):
    comb_filter = np.zeros((nfft//2+1, f0_array.size))
    for i_f0, f0 in enumerate(f0_array):
        n_harmo_max = int((fs//2)//f0)
        for n_harmo in range(n_harmo_max):
            f_harmo = (n_harmo+1)*f0
            f_min = round(f_harmo/pow(2,(peak_width_semitones/2)/12), ndigits=2)
            f_max = round(f_harmo*pow(2,(peak_width_semitones/2)/12), ndigits=2)

            f_harmo_nfft = round(f_harmo*(nfft//2)/(fs//2))
            f_min_nfft = round(f_min*(nfft//2)/(fs//2))
            f_max_nfft = round(f_max*(nfft//2)/(fs//2))
            width_nfft = f_max_nfft-f_min_nfft
            width_nfft = width_nfft/4.29193 # to have a variance so that the gaussian width for 90% of the samples is exactly peak_width_semitones

            x = np.linspace(start=0, stop=nfft//2+1, num=nfft//2+1, dtype=int)
            gaussian = np.exp(-(np.square(x-f_harmo_nfft)/(2*width_nfft**2)))
            comb_filter[:, i_f0] = comb_filter[:, i_f0]+gaussian
            comb_filter[:, i_f0] = comb_filter[:, i_f0]/np.max(comb_filter[:, i_f0])

    return comb_filter
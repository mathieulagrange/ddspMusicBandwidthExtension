import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import os
import customPath
from utils import _PITCHES, _PITCHES_MIDI_NUMBER
from random import random
from tqdm import tqdm
from scipy.io.wavfile import write

vowel_formants = {'i': [250, 2250, 3000],
                  'y': [250, 1750, 2150],
                  'et': [400, 2050, 2650],
                  'u': [300, 750, 2300],
                  'o': [350, 750, 2550],
                  'ai': [600, 1750, 2600],
                  'oe': [500, 1350, 2350],
                  'eu': [350, 1350, 2250],
                  'a': [750, 1450, 2600],
                  'e': [550, 1550, 2550],
                  'c_inversed': [500, 1050, 2550],
                 }

def additive_synth(fs, f0, n_harmo, duration, gains_harmo, phase=None):
    # check if gains_harmo contain n_harmo gains
    if gains_harmo.size != n_harmo:
        raise ValueError("gains_harmo must contain exactly n_harmo gains.") 

    x = np.zeros((n_harmo, fs*duration))

    if phase is not None:
        phase = random()*2*np.pi-np.pi
    for i_harmo in range(n_harmo):
        if i_harmo*f0 <= fs//2:
            x_harmo = lr.tone(frequency=((i_harmo+1)*f0), sr=fs, duration=duration, phi=phase)
        else:
            x_harmo = np.zeros((fs*duration))
        x[i_harmo, :] = (1/gains_harmo[i_harmo])*x_harmo
    x = np.sum(x, axis=0)

    return x

def pink_noise(fs, duration, n_fft = 1024):
    n = np.random.rand(fs*duration)*2-1
    n_stft = lr.stft(n, n_fft=n_fft)
    for i_frame in range(n_stft.shape[1]):
        n_stft[:, i_frame] = n_stft[:, i_frame]/np.arange(1, n_fft//2+2)
    n = lr.istft(n_stft, n_fft=n_fft)
    return n

def asd_envelop(fs, attack, sustain, decay):
    env_attack = np.linspace(0, 1, int(fs*attack))
    env_sustain = np.ones((int(sustain*fs)))
    env_decay = np.linspace(1, 0, int(fs*decay))
    envelop = np.concatenate((env_attack, env_sustain, env_decay))
    return envelop

def add_vowel_formant(x, vowel, fs=16000, n_fft=1024, filter_width=100, formant_db = 12):
    x_stft = lr.stft(x, n_fft=n_fft)
    power = pow(10, formant_db/10)
    formant_freqs = vowel_formants[vowel]
    filter = np.ones(n_fft//2+1)
    for freq in formant_freqs:
        band_min = round((freq-filter_width//2)*(n_fft//2+1)/(fs//2))
        band_max = round((freq+filter_width//2)*(n_fft//2+1)/(fs//2))
        if band_max > fs//2:
            raise ValueError("Upper frequency of the filter is greater than Nyquist frequency")
        
        bandwidth = band_max-band_min+1
        if bandwidth % 2:
            bandwidth = bandwidth-1

        middle_point = bandwidth//2+1
        filter[band_min:band_min+middle_point] = power*np.arange(middle_point)/(middle_point-1)+1
        filter[band_max-middle_point:band_max] = power*np.flip(np.arange(middle_point)/(middle_point-1)+1)

    filter = np.expand_dims(filter, -1)
    x_stft_filtered = np.multiply(x_stft, filter)
    x_filtered = lr.istft(x_stft_filtered, n_fft=n_fft)

    return x_filtered, filter
        

def generate_signal(fs, f0, n_harmo, duration, attack, sustain, decay, phase=None):
    gains = np.arange(1 , n_harmo+1)**2
    dry_signal = additive_synth(fs, f0, n_harmo, duration, gains, phase=phase)
    envelop = asd_envelop(fs, attack, sustain, decay)

    if envelop.size < fs*duration:
        envelop = np.concatenate((envelop, np.zeros((fs*duration-envelop.size))))
    elif envelop.size > fs*duration:
        envelop = envelop[:fs*duration]

    n = pink_noise(fs, duration, n_fft=1024)
    gain_n = random()/10

    s = envelop*(dry_signal + gain_n*n)

    return s

if __name__ == "__main__":
    path = os.path.join(customPath.dataset(), 'synthetic/test/')
    fs = 16000
    duration = 4

    # changing params
    n_harmo_list = [10, 15, 20]
    note_list = [_PITCHES[i_note] for i_note in range(36, 81)]

    for n_harmo in n_harmo_list:
        print(f'Generating signals for {n_harmo} harmonics ...')
        for note in tqdm(note_list):
            f0 = lr.midi_to_hz(_PITCHES_MIDI_NUMBER[_PITCHES.index(note)])
            attack = round(random()*0.3, 2) # between 0 and 0.3
            sustain = round(random()/2+0.5, 2) # between 0.5 and 1
            decay = round(random()*2, 2) # between 0 and 2

            gain = round(random()/4+0.75, 2) #between 0.75 and 1

            s = generate_signal(fs, f0, n_harmo, duration, attack, sustain, decay, phase=True)

            x = gain*s

            name = f'{note}_f{f0}_gain{gain}_{n_harmo}harmo_att{attack}_sus{sustain}_dec{decay}_{duration}sec.wav'
            write(os.path.join(path, name), fs, x)
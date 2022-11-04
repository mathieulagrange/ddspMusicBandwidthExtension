import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import os
import customPath
from utils import _PITCHES, _PITCHES_MIDI_NUMBER
from random import random
from tqdm import tqdm
from scipy.io.wavfile import write


def additive_synth(fs, f0, n_harmo, duration, gains_harmo):
    # check if gains_harmo contain n_harmo gains
    if gains_harmo.size != n_harmo:
        raise ValueError("gains_harmo must contain exactly n_harmo gains.") 

    x = np.zeros((n_harmo, fs*duration))

    for i_harmo in range(n_harmo):
        if i_harmo*f0 <= fs//2:
            x_harmo = lr.tone(frequency=(i_harmo*f0), sr=fs, duration=duration)
        else:
            x_harmo = np.zeros((fs*duration))
        x[i_harmo, :] = (1/gains_harmo[i_harmo])*x_harmo
    x = np.sum(x, axis=0)

    return x

def asd_envelop(fs, attack, sustain, decay):
    env_attack = np.linspace(0, 1, int(fs*attack))
    env_sustain = np.ones((int(sustain*fs)))
    env_decay = np.linspace(1, 0, int(fs*decay))
    envelop = np.concatenate((env_attack, env_sustain, env_decay))
    return envelop

def generate_signal(fs, f0, n_harmo, duration, attack, sustain, decay):
    gains = np.arange(1 , n_harmo+1)**2
    dry_signal = additive_synth(fs, f0, n_harmo, duration, gains)
    envelop = asd_envelop(fs, attack, sustain, decay)

    if envelop.size < fs*duration:
        envelop = np.concatenate((envelop, np.zeros((fs*duration-envelop.size))))
    elif envelop.size > fs*duration:
        envelop = envelop[:fs*duration]

    n = np.random.rand(fs*duration)
    gain_n = random()/10

    s = envelop*(dry_signal + gain_n*n)

    return s

if __name__ == "__main__":
    path = os.path.join(customPath.dataset(), 'synthetic/train/')
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
            gain_n = round(random()/8, 2) #between 0.75 and 1

            s = generate_signal(fs, f0, n_harmo, duration, attack, sustain, decay)

            x = gain*s

            name = f'{note}_f{f0}_gain{gain}_{n_harmo}harmo_att{attack}_sus{sustain}_dec{decay}_{duration}sec.wav'
            write(os.path.join(path, name), fs, x)
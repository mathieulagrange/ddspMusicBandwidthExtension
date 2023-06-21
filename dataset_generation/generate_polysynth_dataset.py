import os
import numpy as np
from scipy.io.wavfile import read, write
import random
from tqdm import tqdm
from core import extract_loudness

def filter_out_notes_without_HF(signal_list, fc=8000):
    filtered_signal_list = []
    for filename in signal_list:
        f0 = extract_f0(filename)
        n_harmo = extract_n_harmos(filename)
        highest_harmo_freq = (n_harmo-1)*f0
        if highest_harmo_freq >= fc:
            filtered_signal_list.append(filename)
    return filtered_signal_list

def extract_note(filename):
    note = filename.split('_')[0]
    return note

def extract_f0(filename):
    f0 = round(float(filename.split('_')[1][1:]),2)
    return f0    

def extract_n_harmos(filename):
    n_harmos = int(filename.split('_')[3][:2])
    return n_harmos

def extract_asd(filename):
    att = float(filename.split('_')[4][3:])
    sus = float(filename.split('_')[5][3:])
    dec = float(filename.split('_')[6][3:])
    return att, sus, dec

def filter_out_same_note(mono_signals, note):
    mono_signals_filtered = [f for f in mono_signals if extract_note(f)[:-1] != note[:-1]]
    return mono_signals_filtered

def generate_poly_signal(signal_list, signal_gains, mono_dataset_path, poly_dataset_path, max_no_of_notes=5):
    if len(signal_list) != len(signal_gains):
        raise ValueError("The number of signals in signal_list is not the same as the number of gains in signal_gains")
    
    fs, x_mix = read(os.path.join(mono_dataset_path, signal_list[0]))

    # first signal
    note = extract_note(signal_list[0])
    gain = signal_gains[0]
    att, sus, dec = extract_asd(signal_list[0])
    first_loudness = extract_loudness(x_mix, fs, block_size=256)
    max_loudness = round(np.max(extract_loudness(x_mix, fs, block_size=256)), 2)
    poly_filename = note+'_gain'+str(gain)+'_ln'+str(max_loudness)+'_a'+str(att)+'_s'+str(sus)+'_d'+str(dec)
    all_loudnesses = np.zeros((max_no_of_notes, first_loudness.size))
    all_loudnesses[0] = first_loudness
    # other signals
    for i_signal, signal in enumerate(signal_list[1:]):
        _, x = read(os.path.join(mono_dataset_path, signal))
        loudness = extract_loudness(signal_gains[i_signal+1]*x, fs, block_size=256)
        all_loudnesses[i_signal+1] = loudness
        max_loudness = round(np.max(extract_loudness(x, fs, block_size=256)), 2)
        x_mix += signal_gains[i_signal+1]*x
        poly_filename += '_'
        poly_filename += extract_note(signal)
        poly_filename += '_gain'
        poly_filename += str(signal_gains[i_signal+1])
        poly_filename += '_ln'
        poly_filename += str(max_loudness)

        att, sus, dec = extract_asd(signal) 
        poly_filename += '_a'
        poly_filename += str(att)
        poly_filename += '_s'
        poly_filename += str(sus)
        poly_filename += '_d'
        poly_filename += str(dec)

    np.save(os.path.join(poly_dataset_path, poly_filename+'_loudness.npy'), all_loudnesses)
    write(os.path.join(poly_dataset_path, poly_filename+'.wav'), fs, x_mix)

if __name__ == "__main__":
    random.seed(4)

    fs = 16000
    split = 'train'
    
    mono_synth_path = '/home/user/Documents/Datasets/synthetic/'+split+'/'
    poly_synth_path = '/home/user/Documents/Datasets/synthetic_poly_3/'+split+'/'

    os.makedirs(poly_synth_path, exist_ok=True)
    os.makedirs(os.path.join(poly_synth_path, split), exist_ok=True)

    chord_lengths = [3, 3, 3, 3, 3]

    mono_signals = [f for f in os.listdir(mono_synth_path) if f.endswith('.wav')]
    mono_signals = sorted(filter_out_notes_without_HF(mono_signals, fc=8000))

    for signal in tqdm(mono_signals):
        first_signal_note = extract_note(signal)
        for chord_length in chord_lengths:
            available_signals = filter_out_same_note(mono_signals, first_signal_note)
            random.shuffle(available_signals)

            final_list_of_signals = [signal]
            signal_gains = [1]

            # select new signals up to the chord_length
            for i_signal in range(chord_length-1):
                final_list_of_signals.append(available_signals[0]) # select first random signal
                new_signal_note = extract_note(available_signals[0]) # extract note of the new signal
                available_signals = filter_out_same_note(available_signals, new_signal_note) # filter out signals with the same note
                random.shuffle(available_signals) # shuffle again for next loop stage
                gain = round(random.random()/2 + 0.5, 2)
                signal_gains.append(gain)

            generate_poly_signal(final_list_of_signals, signal_gains, mono_synth_path, poly_synth_path, max_no_of_notes = max(chord_lengths))
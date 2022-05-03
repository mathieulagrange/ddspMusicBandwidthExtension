import numpy as np
import librosa as lr

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
    spec_ref = np.abs(lr.stft(ref, n_fft=n_fft, hop_length=hop_length).T)
    spec_est = np.abs(lr.stft(est, n_fft=n_fft, hop_length=hop_length).T)
    log_diff = 10*(np.log10(np.square(spec_est)+epsilon)-np.log10((np.square(spec_ref)+epsilon)))
    freq_mean = np.mean(np.square(log_diff), axis=1)
    lsd = np.mean(np.sqrt(freq_mean))

    return lsd
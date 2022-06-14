import numpy as np
import librosa as lr
import peakutils

def sbr(spectrum_WB, phase_reconstruction = "oracle", replication = True, energy_matching_size = 0.25, harmonic_duplication = True, n_peaks = 10):
    '''Spectral Band Replication algorithm
    '''
    # if we have an odd number of frequency bands, we discard the last one to make it even
    if spectrum_WB.size % 2 == 1:
        nBands_LB = int(np.ceil(spectrum_WB.size/2))
        nBands_UB = int(spectrum_WB.size - nBands_LB)

    # compute the magnitude and phase components
    mag_spectrum_LB = np.abs(spectrum_WB[:nBands_LB])
    phase_spectrum_LB = np.angle(spectrum_WB[:nBands_LB])

    # replicate the magnitude spectrum from the lower band to the upper band, with energy continuity
    if replication:
        matching_energy_bandwidth = int(mag_spectrum_LB.size * energy_matching_size)
        energy_LB = np.sum(np.square(mag_spectrum_LB[-matching_energy_bandwidth:]))
        energy_UB = np.sum(np.square(mag_spectrum_LB[:matching_energy_bandwidth]))
        if energy_LB == 0.: # avoiding dividing by 0 if the signal has already faded out
            mag_spectrum_UB = np.zeros((nBands_UB))
        else:
            mag_spectrum_UB = mag_spectrum_LB[:nBands_UB] * energy_LB / energy_UB
            mag_spectrum_UB = mag_spectrum_LB[:nBands_UB]
    else:
        mag_spectrum_UB = np.zeros((nBands_UB))

    # get phase spectrum depending on the method given by phase parameter
    if phase_reconstruction == 'oracle':
        phase_spectrum_UB = np.angle(spectrum_WB[nBands_LB:])
    elif phase_reconstruction == 'flipped':
        phase_spectrum_UB = phase_spectrum_LB[:-1][::-1]
    elif phase_reconstruction == 'noise':
        phase_spectrum_UB = 2*np.pi*np.random.rand(nBands_UB)

    # harmonic peaks extraction from ground-truth and replication
    if harmonic_duplication:
        harmonics_UB = np.zeros((mag_spectrum_UB.size))
        indexes = peakutils.indexes(np.abs(spectrum_WB[nBands_LB:]))
        if indexes != []:
            highest_peak_indexes = indexes[np.argsort(np.abs(spectrum_WB[nBands_LB:])[indexes][::-1])[:n_peaks]]
            for peak in highest_peak_indexes:
                harmonics_UB[peak] = np.abs(spectrum_WB[nBands_LB:])[peak]

        # we replace the harmonics in the the replicated spectrum
        for freq in range(len(mag_spectrum_UB)):
            if harmonics_UB[freq] != 0:
                mag_spectrum_UB[freq] = harmonics_UB[freq]

    # we recontruct the complex spectrum
    reconstructed_mag_spectrum_WB = np.concatenate((mag_spectrum_LB, mag_spectrum_UB))
    reconstructed_phase_spectrum_WB = np.concatenate((phase_spectrum_LB, phase_spectrum_UB))
    reconstructed_spectrum_WB = reconstructed_mag_spectrum_WB*np.exp(1j*reconstructed_phase_spectrum_WB)

    return spectrum_WB, reconstructed_spectrum_WB
import torch
import torch.nn as nn
from core import mlp, gru, remove_above_nyquist, upsample, remove_outside_HB
from core import harmonic_synth, amp_to_impulse_response, fft_convolve, additive_synth
from core import resample
from modules import Z_Encoder, Decoder, DecoderNonHarmonic, DecoderMulti, DecoderNoise
from utils import scale_function
import matplotlib.pyplot as plt
import numpy as np

class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x

class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonics, n_bands, sampling_rate, block_size, nfft=1024, device="cuda:0", max_sources=None):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = Z_Encoder(n_fft=nfft, hop_length=block_size, sample_rate=sampling_rate)
        self.decoder = Decoder(n_harmonics = n_harmonics, n_bands = n_bands, device=device)

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.device = device

    def forward(self, signal, pitch, loudness, add_noise=True, reverb=False):
        latent = self.encoder(signal)
        decoder_outputs = self.decoder(latent, pitch, loudness)

        # harmonic part
        total_amp = decoder_outputs['amp']
        amplitudes = decoder_outputs['harmo_amps']

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes = amplitudes.permute(0, 2, 1)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        total_amp = total_amp.unsqueeze(-1)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        # amplitudes = upsample(amplitudes, 128)
        # pitch = upsample(pitch, 128)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = decoder_outputs['noise_filter']

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)
        if add_noise:
            signal = harmonic + noise
        else:
            signal = harmonic
            
        # reverb
        if reverb:
            signal = self.reverb(signal)

        return signal

class DDSPMulti(nn.Module):
    def __init__(self, hidden_size, n_harmonics, n_bands, max_sources, sampling_rate, block_size, nfft=1024, device="cuda:0"):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = Z_Encoder(n_fft=nfft, hop_length=block_size, sample_rate=sampling_rate)
        self.decoder = DecoderMulti(n_harmonics = n_harmonics, n_bands = n_bands, max_sources = max_sources, device=device)

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.device = device

    def forward(self, signal, pitch_list, loudness, add_noise=True, reverb=False, n_sources=0):
        latent = self.encoder(signal)
        decoder_outputs = self.decoder(latent, pitch_list, loudness)

        # harmonic part
        total_amp_list = decoder_outputs['amp_list']
        amplitudes_list = decoder_outputs['harmo_amps_list']

        for i_amp, amplitudes in enumerate(amplitudes_list):
            amplitudes = remove_above_nyquist(
                amplitudes,
                pitch_list[:, i_amp, :, :],
                self.sampling_rate,
            )
            amplitudes = amplitudes.permute(0, 2, 1)
            amplitudes /= amplitudes.sum(-1, keepdim=True)
            total_amp = total_amp_list[i_amp].unsqueeze(-1)
            amplitudes *= total_amp
            amplitudes_list[i_amp] = amplitudes

        upsample_pitch_list = []
        for i_source in range(len(amplitudes_list)):
            amplitudes = upsample(amplitudes_list[i_source], self.block_size)
            pitch = upsample(pitch_list[:, i_source, :, :], self.block_size)
            upsample_pitch_list.append(pitch)
            amplitudes_list[i_source] = amplitudes
        upsample_pitch_list = torch.stack(upsample_pitch_list)

        # for i_source in range(upsample_pitch_list.shape[0]):
        for i_source in range(n_sources):
            if i_source == 0:
                harmonic = harmonic_synth(upsample_pitch_list[i_source], amplitudes_list[i_source], self.sampling_rate)
            else:
                harmonic = harmonic + harmonic_synth(upsample_pitch_list[i_source], amplitudes_list[i_source], self.sampling_rate)
        
        # noise part
        param = decoder_outputs['noise_filter']

        impulse = amp_to_impulse_response(param, self.block_size)
        # torch.manual_seed(4)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)
        if add_noise:
            signal = harmonic + noise
        else:
            signal = harmonic

        # # reverb
        # if reverb:
        #     signal = self.reverb(signal)
        return signal


class DDSPNonHarmonic(nn.Module):
    def __init__(self, hidden_size, n_harmonics, n_bands, sampling_rate, block_size, device="cuda:0"):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = Z_Encoder()
        self.decoder = DecoderNonHarmonic(n_harmonics = n_harmonics, n_bands = n_bands, device=device, sampling_rate=sampling_rate)

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.device = device

    def forward(self, signal, pitch, loudness, reverb=False):
        latent = self.encoder(signal)
        decoder_outputs = self.decoder(latent, pitch, loudness)

        # harmonic part
        total_amp = decoder_outputs['amp']
        amplitudes = decoder_outputs['harmo_amps']
        freqs = decoder_outputs['harmo_freqs']

        amplitudes = remove_outside_HB(
            amplitudes,
            freqs,
            self.sampling_rate,
        )

        freqs = freqs.permute(0, 2, 1)
        amplitudes = amplitudes.permute(0, 2, 1)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        total_amp = total_amp.unsqueeze(-1)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        freqs = upsample(freqs, self.block_size)
        
        harmonic = additive_synth(freqs, amplitudes, self.sampling_rate)

        # noise part
        param = decoder_outputs['noise_filter']

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        # # reverb
        # if reverb:
        #     signal = self.reverb(signal)

        return signal
    
class DDSPNoise(nn.Module):
    def __init__(self, hidden_size, n_bands, n_harmonics, max_sources, sampling_rate, block_size, nfft=1024, device="cuda:0"):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = Z_Encoder(n_fft=nfft, hop_length=block_size, sample_rate=sampling_rate)
        self.decoder = DecoderNoise(n_bands = n_bands, device=device)

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.device = device

    def forward(self, signal, loudness):
        # print('input signal + loudness')
        # print(signal[:10], loudness[:10])
        # print('------------')
        latent = self.encoder(signal)
        decoder_outputs = self.decoder(latent, loudness)

        # noise part
        param = decoder_outputs['noise_filter']
        # print('decoder output noise filter')
        # print(param[:10])
        # print( '--------------')
        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)
        signal = noise
            
        return signal

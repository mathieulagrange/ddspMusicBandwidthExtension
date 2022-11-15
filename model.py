import torch
import torch.nn as nn
from core import mlp, gru, remove_above_nyquist, upsample
from core import harmonic_synth, amp_to_impulse_response, fft_convolve
from core import resample
from modules import Z_Encoder, Decoder, HarmonicOscillator, FilteredNoise, TrainableFIRReverb
from utils import scale_function
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self, config):
        """
        encoder_config
                use_z=False, 
                sample_rate=16000,
                z_units=16,
                n_fft=2048,
                hop_length=64,
                n_mels=128,
                n_mfcc=30,
                gru_units=512
        
        decoder_config
                mlp_units=512,
                mlp_layers=3,
                use_z=False,
                z_units=16,
                n_harmonics=101,
                n_freq=65,
                gru_units=512,
        components_config
                sample_rate
                hop_length
        """
        super().__init__()

        self.decoder = Decoder(config)
        self.encoder = Z_Encoder(config)

        hop_length = frame_length = int(config.sample_rate * config.frame_resolution)

        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate, frame_length=hop_length
        )

        self.filtered_noise = FilteredNoise(frame_length=hop_length)

        self.reverb = TrainableFIRReverb(reverb_length=config.sample_rate * 3)

        self.crepe = None
        self.config = config

    def forward(self, batch, add_reverb=True):
        """
        z
        input(dict(f0, z(optional), l)) : a dict object which contains key-values below
                f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
                z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
                loudness : torch.tensor w/ shape(B, time)
        """
        latent = self.encoder(batch)
        decoder_outputs = self.decoder(latent)

        harmonic = self.harmonic_oscillator(decoder_outputs)
        noise = self.filtered_noise(decoder_outputs)

        audio = dict(
            harmonic=harmonic, noise=noise, audio_synth=harmonic + noise[:, : harmonic.shape[-1]]
        )

        if self.config.use_reverb and add_reverb:
            audio["audio_reverb"] = self.reverb(audio)

        audio["amp"] = latent["amp"]
        audio["harmo_amps"] = latent["harmo_amps"]

        return audio

    def get_f0(self, x, sample_rate=16000, f0_threshold=0.5):
        """
        input:
            x = torch.tensor((1), wave sample)
        
        output:
            f0 : (n_frames, ). fundamental frequencies
        """
        if self.crepe is None:
            from components.ptcrepe.ptcrepe.crepe import CREPE

            self.crepe = CREPE(self.config.crepe)
            for param in self.parameters():
                self.device = param.device
                break
            self.crepe = self.crepe.to(self.device)
        self.eval()

        with torch.no_grad():
            time, f0, confidence, activation = self.crepe.predict(
                x,
                sr=sample_rate,
                viterbi=True,
                step_size=int(self.config.frame_resolution * 1000),
                batch_size=32,
            )

            f0 = f0.float().to(self.device)
            f0[confidence < f0_threshold] = 0.0
            f0 = f0[:-1]

        return f0

    def reconstruction(self, x, sample_rate=16000, add_reverb=True, f0_threshold=0.5, f0=None):
        """
        input:
            x = torch.tensor((1), wave sample)
            f0 (if exists) = (num_frames, )
        output(dict):
            f0 : (n_frames, ). fundamental frequencies
            a : (n_frames, ). amplitudes
            c : (n_harmonics, n_frames). harmonic constants
            sig : (n_samples)
            audio_reverb : (n_samples + reverb, ). reconstructed signal
        """
        self.eval()

        with torch.no_grad():
            if f0 is None:
                f0 = self.get_f0(x, sample_rate=sample_rate, f0_threshold=f0_threshold)

            batch = dict(f0=f0.unsqueeze(0), audio=x.to(self.device),)

            recon = self.forward(batch, add_reverb=add_reverb)

            # make shape consistent(removing batch dim)
            for k, v in recon.items():
                recon[k] = v[0]

            recon["f0"] = f0

            return 

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
    def __init__(self, hidden_size, n_harmonics, n_bands, sampling_rate, block_size, device="cuda:0"):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = Z_Encoder()
        self.decoder = Decoder(n_harmonics = n_harmonics, n_bands = n_bands, device=device)

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

        signal = harmonic + noise

        # # reverb
        # if reverb:
        #     signal = self.reverb(signal)

        return signal
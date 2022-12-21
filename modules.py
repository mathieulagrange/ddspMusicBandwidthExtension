import torch
import torch.nn as nn
import math
import torchaudio
from utils import scale_function
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True, device="cuda;0"):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units
        self.inplace = inplace
        self.device = device

        self.mlp_layer1 = nn.Sequential(
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(inplace=self.inplace),
            )
        self.mlp_layer_list = []
        for i in range(2, n_layer + 1):
            # mlp_name = f'mlp_layer{i}'
            self.mlp_layer_list.append(
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(inplace=self.inplace),
                ))

    def forward(self, x):
        x = self.mlp_layer1(x)
        for layer in self.mlp_layer_list:
            layer.to(self.device)
            x = layer(x)
        return x


class Z_Encoder(nn.Module):
    def __init__(
        self,
        n_fft = 1024,
        hop_length = 256,
        sample_rate=16000,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        z_units=16,
        bidirectional=False,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=20.0, f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(gru_units * 2 if bidirectional else gru_units, z_units)

    def forward(self, signal):
        x = signal
        x = self.mfcc(x)
        x = x[:, :, :-1]
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x

class Decoder(nn.Module):
    def __init__(self, use_z = True,
                 mlp_units = 512,
                 mlp_layers = 3, 
                 z_units = 16, 
                 gru_units = 512, 
                 bidirectional = False, 
                 n_harmonics = 101, 
                 n_bands = 65,
                 device = "cuda:0"):
        super().__init__()
        self.mlp_f0 = MLP(n_input=1, n_units=mlp_units, n_layer=mlp_layers, device=device)
        self.mlp_loudness = MLP(n_input=1, n_units=mlp_units, n_layer=mlp_layers, device=device)
        self.use_z = use_z
        if use_z:
            self.mlp_z = MLP(
                n_input=z_units, n_units=mlp_units, n_layer=mlp_layers, device=device
            )
            self.num_mlp = 3
        else:
            self.num_mlp = 2

        self.gru = nn.GRU(
            input_size=self.num_mlp * mlp_units,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.mlp_gru = MLP(
            n_input=gru_units * 2 if bidirectional else gru_units,
            n_units=mlp_units,
            n_layer=mlp_layers,
            inplace=True,
            device=device
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(mlp_units, n_harmonics + 1)
        self.dense_filter = nn.Linear(mlp_units, n_bands)

        self.device = device

    def forward(self, latent, f0, loudness):
        if self.use_z:
            z = latent
            self.mlp_z.to(self.device)
            latent_z = self.mlp_z(z)

        latent_f0 = self.mlp_f0(f0)

        latent_loudness = self.mlp_loudness(loudness)

        if self.use_z:
            latent = torch.cat((latent_f0, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        latent, (h) = self.gru(latent)
        latent = self.mlp_gru(latent)

        amplitude = self.dense_harmonic(latent)

        amp = amplitude[..., 0]
        amp = scale_function(amp)

        # a = torch.sigmoid(amplitude[..., 0])
        harmo_amps = torch.nn.functional.softmax(amplitude[..., 1:], dim=-1)

        noise_filter = self.dense_filter(latent)
        noise_filter = scale_function(noise_filter)

        harmo_amps = harmo_amps.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input.

        return dict(f0 = f0, amp = amp, harmo_amps = harmo_amps, noise_filter = noise_filter)

class DecoderNonHarmonic(nn.Module):
    def __init__(self, use_z = True,
                 mlp_units = 512,
                 mlp_layers = 3, 
                 z_units = 16, 
                 gru_units = 512, 
                 bidirectional = False, 
                 n_harmonics = 101, 
                 n_bands = 65,
                 device = "cuda:0"):
        super().__init__()
        self.mlp_f0 = MLP(n_input=1, n_units=mlp_units, n_layer=mlp_layers, device=device)
        self.mlp_loudness = MLP(n_input=1, n_units=mlp_units, n_layer=mlp_layers, device=device)
        self.use_z = use_z
        if use_z:
            self.mlp_z = MLP(
                n_input=z_units, n_units=mlp_units, n_layer=mlp_layers, device=device
            )
            self.num_mlp = 3
        else:
            self.num_mlp = 2

        self.gru = nn.GRU(
            input_size=self.num_mlp * mlp_units,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.mlp_gru = MLP(
            n_input=gru_units * 2 if bidirectional else gru_units,
            n_units=mlp_units,
            n_layer=mlp_layers,
            inplace=True,
            device=device
        )

        # one element for overall loudness
        self.dense_harmonic_freqs = nn.Linear(mlp_units, n_harmonics + 1)
        self.dense_harmonic_amps = nn.Linear(mlp_units, n_harmonics + 1)
        self.dense_filter = nn.Linear(mlp_units, n_bands)

        self.device = device

    def forward(self, latent, f0, loudness):
        if self.use_z:
            z = latent
            self.mlp_z.to(self.device)
            latent_z = self.mlp_z(z)

        latent_f0 = self.mlp_f0(f0)

        latent_loudness = self.mlp_loudness(loudness)

        if self.use_z:
            latent = torch.cat((latent_f0, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        latent, (h) = self.gru(latent)
        latent = self.mlp_gru(latent)

        amplitude = self.dense_harmonic_amps(latent)
        amp = amplitude[..., 0]
        amp = scale_function(amp)

        # a = torch.sigmoid(amplitude[..., 0])
        harmo_amps = torch.nn.functional.softmax(amplitude[..., 1:], dim=-1)
        harmo_freqs = self.dense_harmonic_freqs(latent)

        noise_filter = self.dense_filter(latent)
        noise_filter = scale_function(noise_filter)

        harmo_amps = harmo_amps.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input.
        harmo_freqs = harmo_amps.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input.
        
        return dict(f0 = f0, amp = amp, harmo_freqs = harmo_freqs, harmo_amps = harmo_amps, noise_filter = noise_filter)

class HarmonicOscillator(nn.Module):
    def __init__(self, sr=16000, frame_length=64, attenuate_gain=0.02, device="cuda"):
        super(HarmonicOscillator, self).__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain

        self.device = device

        self.framerate_to_audiorate = nn.Upsample(
            scale_factor=self.frame_length, mode="linear", align_corners=False
        )

    def forward(self, z):

        """
        Compute Addictive Synthesis
        Argument: 
            z['f0'] : fundamental frequency envelope for each sample
                - dimension (batch_num, frame_rate_time_samples)
            z['c'] : harmonic distribution of partials for each sample 
                - dimension (batch_num, partial_num, frame_rate_time_samples)
            z['a'] : loudness of entire sound for each sample
                - dimension (batch_num, frame_rate_time_samples)
        Returns:
            addictive_output : synthesized sinusoids for each sample 
                - dimension (batch_num, audio_rate_time_samples)
        """

        fundamentals = z["f0"]
        framerate_harmo_amps_bank = z["harmo_amps"]

        num_osc = framerate_harmo_amps_bank.shape[1]

        # Build a frequency envelopes of each partials from z['f0'] data
        partial_mult = (
            torch.linspace(1, num_osc, num_osc, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        framerate_f0_bank = (
            fundamentals.unsqueeze(-1).expand(-1, -1, num_osc).transpose(1, 2) * partial_mult
        )

        # Antialias z['c']
        mask_filter = (framerate_f0_bank < self.sr / 2).float()
        antialiased_framerate_c_bank = framerate_harmo_amps_bank * mask_filter

        # Upsample frequency envelopes and build phase bank
        audiorate_f0_bank = self.framerate_to_audiorate(framerate_f0_bank)
        audiorate_phase_bank = torch.cumsum(audiorate_f0_bank / self.sr, 2)

        # Upsample amplitude envelopes
        audiorate_a_bank = self.framerate_to_audiorate(antialiased_framerate_c_bank)

        # Build harmonic sinusoid bank and sum to build harmonic sound
        sinusoid_bank = (
            torch.sin(2 * np.pi * audiorate_phase_bank) * audiorate_a_bank * self.attenuate_gain
        )

        framerate_loudness = z["amp"]
        audiorate_loudness = self.framerate_to_audiorate(framerate_loudness.unsqueeze(0)).squeeze(0)

        additive_output = torch.sum(sinusoid_bank, 1) * audiorate_loudness

        return additive_output
    
class FilteredNoise(nn.Module):
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2, device = 'cuda'):
        super(FilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain
        
    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-varient in terms of frame by frame) filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Argument:
            z['H'] : filter coefficient bank for each batch, which will be used for constructing linear-phase filter.
                - dimension : (batch_num, frame_num, filter_coeff_length)
        
        """
        
        batch_num, frame_num, filter_coeff_length = z['noise_filter'].shape
        self.filter_window = nn.Parameter(torch.hann_window(filter_coeff_length * 2 - 1, dtype = torch.float32), requires_grad = False).to(self.device)
        
        INPUT_FILTER_COEFFICIENT = z['noise_filter']
        
        # Desired linear-phase filter can be obtained by time-shifting a zero-phase form (especially to a causal form to be real-time),
        # which has zero imaginery part in the frequency response. 
        # Therefore, first we create a zero-phase filter in frequency domain.
        # Then, IDFT & make it causal form. length IDFT-ed signal size can be both even or odd, 
        # but we choose odd number such that a single sample can represent the center of impulse response.
        ZERO_PHASE_FR_BANK = INPUT_FILTER_COEFFICIENT.unsqueeze(-1).expand(batch_num, frame_num, filter_coeff_length, 2).contiguous()
        ZERO_PHASE_FR_BANK[..., 1] = 0
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length, 2)
        zero_phase_ir_bank = torch.irfft(ZERO_PHASE_FR_BANK, 1, signal_sizes = (filter_coeff_length * 2 - 1,))
           
        # Make linear phase causal impulse response & Hann-window it.
        # Then zero pad + DFT for linear convolution.
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * self.filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.frame_length - 1))
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.rfft(zero_paded_windowed_linear_phase_ir_bank, 1)
        
        # Generate white noise & zero pad & DFT for linear convolution.
        noise = torch.rand(batch_num, frame_num, self.frame_length, dtype = torch.float32).view(-1, self.frame_length).to(self.device) * 2 - 1
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = torch.rfft(zero_paded_noise, 1)

        # Convolve & IDFT to make filtered noise frame, for each frame, noise band, and batch.
        FILTERED_NOISE = torch.zeros_like(ZERO_PADED_NOISE).to(self.device)
        FILTERED_NOISE[:, :, 0] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0] \
            - ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1]
        FILTERED_NOISE[:, :, 1] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1] \
            + ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0]
        filtered_noise = torch.irfft(FILTERED_NOISE, 1).view(batch_num, frame_num, -1) * self.attenuate_gain         
                
        # Overlap-add to build time-varying filtered noise.
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad = False).unsqueeze(1).to(self.device)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), 
                                                       overlap_add_filter, 
                                                       stride = self.frame_length, 
                                                       padding = 0).squeeze(1)
        
        return output_signal

class TrainableFIRReverb(nn.Module):
    def __init__(self, reverb_length=48000, device="cuda"):

        super(TrainableFIRReverb, self).__init__()

        # default reverb length is set to 3sec.
        # thus this model can max out t60 to 3sec, which corresponds to rich chamber characters.
        self.reverb_length = reverb_length
        self.device = device

        # impulse response of reverb.
        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32).to(self.device) * 2 - 1,
            requires_grad=True,
        )

        # Initialized drywet to around 26%.
        # but equal-loudness crossfade between identity impulse and fir reverb impulse is not implemented yet.
        self.drywet = nn.Parameter(
            torch.tensor([-1.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

        # Initialized decay to 5, to make t60 = 1sec.
        self.decay = nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

    def forward(self, z):
        """
        Compute FIR Reverb
        Input:
            z['audio_synth'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """

        # Send batch of input signals in time domain to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        input_signal = z["audio_synth"]
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))
        INPUT_SIGNAL = torch.rfft(zero_pad_input_signal, 1)

        # Build decaying impulse response and send it to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        # Dry-wet mixing is done by mixing impulse response, rather than mixing at the final stage.

        """ TODO 
        Not numerically stable decay method?
        """
        decay_envelope = torch.exp(
            -(torch.exp(self.decay) + 2)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device)
        )
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(self.device)
        ir_identity[:, 0] = 1

        """ TODO
        Equal-loudness(intensity) crossfade between to ir.
        """
        final_fir = (
            torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        )
        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))

        FIR = torch.rfft(zero_pad_final_fir, 1)

        # Convolve and inverse FFT to get original signal.
        OUTPUT_SIGNAL = torch.zeros_like(INPUT_SIGNAL).to(self.device)
        OUTPUT_SIGNAL[:, :, 0] = (
            INPUT_SIGNAL[:, :, 0] * FIR[:, :, 0] - INPUT_SIGNAL[:, :, 1] * FIR[:, :, 1]
        )
        OUTPUT_SIGNAL[:, :, 1] = (
            INPUT_SIGNAL[:, :, 0] * FIR[:, :, 1] + INPUT_SIGNAL[:, :, 1] * FIR[:, :, 0]
        )

        output_signal = torch.irfft(OUTPUT_SIGNAL, 1)

        return output_signal
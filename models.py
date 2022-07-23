from ddsp.training import train_util, models, trainers, encoders, decoders, preprocessing
from ddsp.training.models import Autoencoder
from ddsp import processors
from ddsp import synths
from ddsp import core
from ddsp import effects
from ddsp import losses

class OriginalAutoencoder(Autoencoder):
    def __init__(self):

        # preprocessor
        preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=1000)

        # encoder
        encoder = encoders.MfccTimeDistributedRnnEncoder(
            rnn_channels = 512,
            rnn_type = 'gru',
            z_dims = 16,
            z_time_steps = 250
        )

        # decoder
        decoder = decoders.RnnFcDecoder(
            rnn_channels = 512,
            rnn_type = 'gru',
            ch = 512,
            layers_per_stack = 3,
            input_keys = ('ld_scaled', 'f0_scaled', 'z'),
            output_splits = (('amps', 1), ('harmonic_distribution', 100), ('noise_magnitudes', 65))
        )

        # synths
        harmonic = synths.Harmonic(
            n_samples = 64000,
            sample_rate = 16000,
            scale_fn = core.exp_sigmoid,
            normalize_below_nyquist = True,
            name = 'harmonic'
        )
        filtered_noise = synths.FilteredNoise(
            n_samples = 64000,
            window_size = 257,
            scale_fn = core.exp_sigmoid,
            name = 'filtered_noise'
        )
        harmonic_plus_fn= processors.Add(name='add')

        # processor group
        dag = [
            (harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
            (filtered_noise, ['noise_magnitudes']),
            (harmonic_plus_fn, ['filtered_noise/signal', 'harmonic/signal']),
        ]
        processor_group = processors.ProcessorGroup(dag = dag, name='processor_group')

        # losses
        fft_sizes = [64, 128, 256, 512, 1024, 2048]
        spectral_loss = losses.SpectralLoss(
            loss_type = 'L1',
            mag_weight = 1.0,
            logmag_weight = 1.0
        )

        super().__init__(preprocessor = preprocessor,
                         encoder = encoder,
                         decoder = decoder,
                         processor_group = processor_group,
                         losses = [spectral_loss])

    def call(self, features, training=True):
        """Run the core of the network, get predictions and loss."""
        features = self.encode(features, training=training)
        features.update(self.decoder(features, training=training))

        # Run through processor group.
        pg_out = self.processor_group(features, return_outputs_dict=True)

        # Parse outputs
        outputs = pg_out['controls']
        outputs['audio_synth'] = pg_out['signal']

        if training:
            self._update_losses_dict(
                self.loss_objs, features['audio_WB'], outputs['audio_synth'])

        return outputs
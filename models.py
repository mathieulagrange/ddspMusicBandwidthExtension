from ddsp.training import train_util, models, trainers, encoders, decoders, preprocessing
from ddsp.training.models import Autoencoder, Model
from ddsp import processors
from ddsp import synths
from ddsp import core
from ddsp import effects
from ddsp import losses
import tensorflow as tf
import logging
import time
from utils import get_checkpoint

class OriginalAutoencoder(Autoencoder):
    def __init__(self, output_nBands='WB'):
        self.output_nBands = output_nBands

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
            if self.output_nBands == 'WB':
                self._update_losses_dict(
                    self.loss_objs, features['audio_WB'], outputs['audio_synth'])
            elif self.output_nBands == 'HB':
                self._update_losses_dict(
                    self.loss_objs, features['audio_HB'], outputs['audio_synth'])

        return outputs

    def restore(self, checkpoint_path, step=None, verbose=True, restore_keys=None):
        """Restore model and optimizer from a checkpoint.
        Args:
        checkpoint_path: Path to checkpoint file or directory.
        verbose: Warn about missing variables.
        restore_keys: Optional list of strings for submodules to restore.
        Raises:
        FileNotFoundError: If no checkpoint is found.
        """
        start_time = time.time()
        if step is None:
            latest_checkpoint = train_util.get_latest_checkpoint(checkpoint_path)
        else:
            latest_checkpoint = get_checkpoint(checkpoint_path, step)

        if restore_keys is None:
            # If no keys are passed, restore the whole model.
            checkpoint = tf.train.Checkpoint(model=self)
            logging.info('Model restoring all components.')
            if verbose:
                checkpoint.restore(latest_checkpoint)
            else:
                checkpoint.restore(latest_checkpoint).expect_partial()

        else:
            # Restore only sub-modules by building a new subgraph.
            # Following https://www.tensorflow.org/guide/checkpoint#loading_mechanics.
            logging.info('Trainer restoring model subcomponents:')
            for k in restore_keys:
                to_restore = {k: getattr(self, k)}
                log_str = 'Restoring {}'.format(to_restore)
                logging.info(log_str)
                fake_model = tf.train.Checkpoint(**to_restore)
                new_root = tf.train.Checkpoint(model=fake_model)
                status = new_root.restore(latest_checkpoint)
                status.assert_existing_objects_matched()

        logging.info('Loaded checkpoint %s', latest_checkpoint)
        logging.info('Loading model took %.1f seconds', time.time() - start_time)

def conv1D_relu(filters, kernel_size, padding = 'same', activation = None):
    return tf.keras.layers.Conv1D(
        filters = filters,
        kernel_size = kernel_size,
        padding = padding,
        activation = activation
    )

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, n_hidden_ch, kernel_size,
                 batchnorm=False, dropout=0.0,
                 activation='relu', res_scale=0.1):
        super().__init__()
        self.n_hidden_ch = n_hidden_ch
        self.kernel_size = kernel_size
        self.res_scale = res_scale

        layers = []
        for i in range(2):
            if i == 0:
                layers.append(conv1D_relu(filters=self.n_hidden_ch, kernel_size=self.kernel_size, activation='relu'))
            else:
                layers.append(conv1D_relu(filters=self.n_hidden_ch, kernel_size=self.kernel_size, activation=None))

        self.block = tf.keras.Sequential(layers, name='res_block')

    def call(self, inputs):
        x = inputs
        y = self.block(x)
        y = x*self.res_scale
        y = x+y
        return y

class SulunResNet(Model):
    def __init__(self,
                 batchnorm = False,
                 dropout = 0.0):

        super().__init__()

        self.batchnorm = batchnorm
        self.bias = not self.batchnorm

        self.n_res_block = 15
        self.n_hidden_ch = 512
        self.kernel_size = 7
        self.activation = 'relu'
        self.res_scaling = 0.1
        self.n_input_ch = 1
        self.n_output_ch = 1
        self.layers = []

        self.layers.append(conv1D_relu(self.n_hidden_ch, self.kernel_size))
        for i in range(self.n_res_block):
            self.layers.append(ResBlock(self.n_hidden_ch, self.kernel_size))
        self.layers.append(conv1D_relu(self.n_output_ch, kernel_size=1))

    def call(self, features, training=True):
        for layer in self.layers:
            x = layer(features['audio'])

        outputs = {}
        outputs['audio_synth'] = x

        if training:
            self._update_losses_dict(
                self.loss_objs, features['audio'], outputs['audio_synth'])
        return outputs


from data import OrchideaSol, MedleySolosDB
from ddsp.training import train_util, models, trainers, encoders, decoders, preprocessing
from ddsp import losses
from ddsp import processors
from ddsp import synths
from ddsp import core
from ddsp import effects
import matplotlib.pyplot as plt

### load dataset ###
orchidea = OrchideaSol('test', sample_rate=16000, frame_rate=250, batch_size=1)
dataset = orchidea.get_dataset()

### model ###
ae = models.Autoencoder()

# preprocessor
preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=1000)

# encoder
encoder = encoders.MfccTimeDistributedRnnEncoder(
    rnn_channels = 512,
    rnn_type = 'gru',
    z_dims = 16,
    z_time_steps = 250
)

# Decoder
decoder = decoders.RnnFcDecoder(
    rnn_channels = 512,
    rnn_type = 'gru',
    ch = 512,
    layers_per_stack = 3,
    input_keys = ('ld_scaled', 'f0_scaled', 'z'),
    output_splits = (('amps', 1), ('harmonic_distribution', 100), ('noise_magnitudes', 65))
)

# Synths
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

# Processor group
dag = [
    (harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
    (filtered_noise, ['noise_magnitudes']),
    (harmonic_plus_fn, ['filtered_noise/signal', 'harmonic/signal']),
]
processor_group = processors.ProcessorGroup(dag = dag, name='processor_group')

# Reverb
reverb = effects.Reverb()

### Loss ###
fft_sizes = [64, 128, 256, 512, 1024, 2048]
spectral_loss = losses.SpectralLoss(
    loss_type = 'L1',
    mag_weight = 1.0,
    logmag_weight = 1.0
)

### training loop ###
batch_size = 1
save_dir = './results/ddsp/'
dataset = orchidea.get_batch(batch_size=batch_size, shuffle=False).take(1).repeat()

strategy = train_util.get_strategy()
with strategy.scope():
    model = models.Autoencoder(preprocessor=preprocessor, 
                                encoder=encoder, 
                                decoder=decoder, 
                                processor_group=processor_group,
                                losses=[spectral_loss])
    trainer = trainers.get_trainer_class()(model, strategy, learning_rate=1e-3)

batch = next(iter(dataset))
dataset = trainer.distribute_dataset(dataset)
trainer.build(next(iter(dataset)))

dataset_iter = iter(dataset)

for i in range(30):
  losses = trainer.train_step(dataset_iter)
  res_str = 'step: {}\t'.format(i)
  for k, v in losses.items():
    res_str += '{}: {:.2f}\t'.format(k, v)
  print(res_str)

# train_util.train(data_provider=orchidea,
#                     trainer=trainer,
#                     save_dir=save_dir,
#                     restore_dir=save_dir)
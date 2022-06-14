from data import OrchideaSol, MedleySolosDB
from ddsp.training import train_util, trainers, encoders, decoders, preprocessing
from ddsp import losses
from ddsp import processors
from ddsp import synths
from ddsp import core
from ddsp import effects
import ddsp
from models import OriginalAutoencoder
import matplotlib.pyplot as plt
import logging

# prepare log file
logging.basicConfig(filename='test.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


### dataset ###
orchidea = OrchideaSol('train', sample_rate=16000, frame_rate=250, batch_size=1)
dataset = orchidea.get_dataset()

### model ###
model = OriginalAutoencoder()

### training ###
batch_size = 10
save_dir = './results/ddsp/'
n_steps = 100
early_stop_loss_value = None

# create strategy
strategy = train_util.get_strategy()
with strategy.scope():
    model = OriginalAutoencoder()
    trainer = trainers.get_trainer_class()(model, strategy, learning_rate=1e-3)

# initiate dataset for training
dataset = orchidea.get_batch(batch_size=batch_size, shuffle=True, repeats=-1)
dataset = trainer.distribute_dataset(dataset)
trainer.build(next(iter(dataset)))
dataset_iter = iter(dataset)

# training loop
for i in range(n_steps):
    losses = trainer.train_step(dataset_iter)
    res_str = 'step: {}\t'.format(i)
    for k, v in losses.items():
        res_str += '{}: {:.2f}\t'.format(k, v)
    print(res_str)

# Stop the training when the loss reaches given value
    if (early_stop_loss_value is not None and
        losses['total_loss'] <= early_stop_loss_value):
        break

# save model
trainer.save(save_dir)
from data import OrchideaSol, OrchideaSolTiny, MedleySolosDB
from ddsp.training import train_util, trainers
from ddsp import losses
import tensorflow as tf
from models import OriginalAutoencoder
import matplotlib.pyplot as plt
import logging
import os
import time

def train(model_dir, params):
    # dataset
    if params['data'] == 'sol':
        dataset = OrchideaSol('train', audio_length=4, sample_rate=16000, frame_rate=250)
    elif params['data'] == 'tiny':
        dataset = OrchideaSolTiny('train', audio_length=4, sample_rate=16000, frame_rate=250)
    elif params['data'] == 'medley':
        dataset = MedleySolosDB('train', audio_length=4, sample_rate=16000, frame_rate=250)

    # model
    if params['model'] == 'original_autoencoder':
        model = OriginalAutoencoder()

    # create training strategy
    strategy = train_util.get_strategy()
    with strategy.scope():
        trainer = trainers.get_trainer_class()(model, strategy, learning_rate=1e-3)

    # initiate dataset
    dataset = dataset.get_batch(batch_size=params['batch_size'], shuffle=True, repeats=-1)
    dataset = trainer.distribute_dataset(dataset)
    trainer.build(next(iter(dataset)))
    dataset_iter = iter(dataset)

    # load latest checkpoint if existing
    try:
        trainer.restore(model_dir)
        logging.info(f'Restarting training from step {trainer.step.numpy()}')
    except FileNotFoundError:
        logging.info('No existing checkpoint, retraining from scratch')

    # summary writer
    summary_dir = os.path.join(model_dir, 'summaries', 'train')
    summary_writer = tf.summary.create_file_writer(summary_dir)

    # training loop
    with summary_writer.as_default():
        tic = time.time()
        first_step = True

        if trainer.step < params['n_steps_total']:
            for i in range(params['n_steps_per_training']):
                losses = trainer.train_step(dataset_iter)

                # if first step (starting or restarting) we create the loss metrics
                if first_step:
                    loss_names = list(losses.keys())
                    logging.info('Creating metrics for %s', loss_names)
                    avg_losses = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32) for name in loss_names}
                    first_step = False

                # update metrics
                for k, v in losses.items():
                    avg_losses[k].update_state(v)

                log_str = 'step: {}\t'.format(int(trainer.step.numpy()))
                for k, v in losses.items():
                    log_str += '{}: {:.2f}\t'.format(k, v)
                logging.info(log_str)

                # write summaries
                if trainer.step % params['n_steps_per_training'] == 0:
                    # training speed
                    steps_per_sec = params['n_steps_per_training']/ (time.time() - tic)
                    tf.summary.scalar('steps_per_sec', steps_per_sec, step=trainer.step)
                    tic = time.time()

                    # metrics
                    for k, metric in avg_losses.items():
                        tf.summary.scalar('losses/{}'.format(k), metric.result(), step=trainer.step)
                        metric.reset_states()

                # early stopping
                if (params['early_stop_loss_value'] is not None and
                    losses['total_loss'] <= params['early_stop_loss_value']):
                    logging.info('Total loss reached early stopping value of %s', params['early_stop_loss_value'])
                    break

        #save model
        print('Saving model ...')
        trainer.save(model_dir)
        print('Model saved.')
        summary_writer.flush()

    n_steps_total = params['n_steps_total']
    logging.info(f'Training done up to step {trainer.step.numpy()} of {n_steps_total}')
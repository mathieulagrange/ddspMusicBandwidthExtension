#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from metrics import sdr, lsd
import doce
import sys
import customPath
import os

# experiment definition
experiment = doce.Experiment(
    name = 'sbr_replication',
    purpose = 'evaluating sbr metrics with different parameters',
    author = 'PA Grumiaux',
)

# experiment path
experiment.set_path('output', os.path.join(customPath.results(),experiment.name, ''))

# experiment plan
experiment.add_plan('plan',
    alg = ['sbr', 'oracle', 'dumb', 'ddsp'],
    data = ['sol', 'tiny', 'medley', 'gtzan'],
    # method = ['replication', 'harmonic', 'replication+harmonic'],
    # phase = ['oracle', 'flipped', 'noise'],
    # matchingEnergy = [0.25, 0.5, 1.0],
    # nfft = [1024],
    batch_size = [16],
    model = ['original_autoencoder'],
    n_steps_total = [25],
    n_steps_per_training = [5],
    early_stop_loss_value = [None]
    )

# experiment metrics
experiment.set_metric(
    name = 'sdr',
    significance = True,
    higher_the_better = True
)

experiment.set_metric(
    name = 'lsd',
    significance = True,
    lower_the_better = True
)

experiment.set_metric(
    name = 'time',
)

# processing for each step in the plan 
def step(setting, experiment):
    from evaluate import evaluate
    # use the evaluate function with the given settings
    sdr, lsd, time = evaluate(setting, experiment)

    # store the resulting metrics for a whole dataset
    np.save(os.path.join(experiment.path.output,setting.identifier()+'_sdr.npy'), sdr)
    np.save(os.path.join(experiment.path.output,setting.identifier()+'_lsd.npy'), lsd)
    np.save(os.path.join(experiment.path.output,setting.identifier()+'_time.npy'), time)

if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func = step)
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from metrics import sdr, lsd
import doce
import sys
import customPath
import os
from utils import filter_out_inf_then_mean

# experiment definition
experiment = doce.Experiment(
    name = 'ddsp',
    purpose = 'ddsp for bwe',
    author = 'PA Grumiaux',
)

# experiment path
experiment.set_path('output', os.path.join(customPath.results(),experiment.name, ''))

# experiment plan
experiment.add_plan('sbr',
    alg = ['sbr'],#, 'oracle', 'dumb'],
    data = ['sol', 'tiny', 'medley', 'gtzan', 'synthetic'],
    method = ['replication', 'harmonic', 'replication+harmonic'],
    phase = ['oracle', 'flipped', 'noise'],
    matchingEnergy = [0.25, 0.5, 1.0],
    nfft = [1024],
    sampling_rate = [16000],
    split = ['train', 'test']
    )

experiment.add_plan('ddsp',
    alg = ['ddsp'],
    data = ['sol', 'tiny', 'medley', 'gtzan', 'synthetic'],
    model = ['original_autoencoder'],
    n_steps_total = [50000],
    n_steps_per_training = [5000],
    sampling_rate = [16000],
    split = ['train', 'test'],
    nfft = [1024]
    )

# experiment metrics
experiment.set_metric(
    name = 'sdr',
    significance = True,
    higher_the_better = True,
    func = filter_out_inf_then_mean
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
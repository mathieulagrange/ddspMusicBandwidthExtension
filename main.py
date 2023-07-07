#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from metrics import sdr, lsd
import doce
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
    data = ['synthetic', 'synthetic_poly', 'synthetic_poly_2', 'sol', 'medley', 'dsd_sources', 'dsd_mixtures', 'gtzan', 'medleyDB_stems', 'medleyDB_mixtures'],
    alg = ['sbr', 'oracle', 'dumb', 'noise'],
    method = ['replication'],#, 'harmonic', 'replication+harmonic'],
    phase = ['oracle'],#, 'flipped', 'noise'],
    matchingEnergy = [0.5], #[0.25, 0.5, 1.0],
    nfft = [1024],
    sampling_rate = [8000, 16000],
    split = ['train', 'test'],
    downsampling_factor = [2, 4]
    )

experiment.add_plan('ddsp',
    data = ['synthetic', 'synthetic_crepe', 'synthetic_poly', 'synthetic_poly_2', 'synthetic_poly_3', 'sol', 'medley', 'dsd_sources', 'dsd_mixtures', 'gtzan', 'medleyDB_stems', 'medleyDB_mixtures'],
    alg = ['ddsp'],
    model = ['ddsp_original_autoencoder', 'ddsp_non_harmo', 'resnet', 'ddsp_noise'],
    sampling_rate = [8000, 16000],
    downsampling_factor = [2, 4],
    split = ['train', 'test'],
    nfft = [1024],
    loss = ['WB', 'HB'],
    noiseTraining = [True, False],
    )

experiment.add_plan('ddsp_multi',
    data = ['synthetic', 'synthetic_poly', 'synthetic_poly_2', 'synthetic_poly_3', 'synthetic_poly_mono', 'dsd_mixtures', 'gtzan', 'medleyDB_mixtures'],
    alg = ['ddsp_multi'],
    train_data = ['synthetic', 'synthetic_poly', 'dsd_sources', 'gtzan', 'medleyDB_mixtures'],
    model = ['ddsp_original_autoencoder'],
    sampling_rate = [8000, 16000],
    downsampling_factor = [2, 4],
    split = ['train', 'test'],
    nfft = [1024],
    loss = ['WB', 'HB'],
    iteration = [0, 1, 2, 5, 10], # 0 means we use ground-truth number of sources
    loudness_gt = [True, False],
    pitch = ['gt', 'crepe', 'yin', 'bittner'],
    noise = ['all', 'last'],
    comb_filter = [True, False],
    noiseTraining = [True, False],
    )

experiment.add_plan('ddsp_poly_decoder',
    data = ['synthetic', 'synthetic_poly', 'synthetic_poly_2', 'synthetic_poly_3', 'synthetic_poly_mono', 'synthetic_poly_mono_2', 'gtzan', 'medleyDB_mixtures'],
    train_data = ['synthetic_poly', 'synthetic_poly_2', 'synthetic_poly_3', 'synthetic_poly_mono', 'synthetic_poly_mono_2', 'gtzan', 'medleyDB_mixtures'],
    alg = ['ddsp_poly_decoder'],
    model = ['ddsp_decoder_multi'],
    sampling_rate = [8000, 16000],
    downsampling_factor = [2, 4],
    split = ['train', 'test'],
    nfft = [1024],
    loss = ['WB', 'HB'],
    max_n_sources = [1, 2, 3, 5],
    pitch = ['gt', 'bittner'],
    noiseTraining = [True, False],
    )

experiment.set_metric(
    name = 'lsd',
    # significance = True,
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
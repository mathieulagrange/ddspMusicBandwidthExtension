import numpy as np
from metrics import sdr, lsd
import doce
import sys
from evaluate import evaluate
import customPath
import os

def set (args):
    # experiment definition
    experiment = doce.Experiment(
        name = 'sbr_replication',
        purpose = 'evaluating sbr metrics with different parameters',
        author = 'PA Grumiaux',
    )

    # experiment path
    experiment.setPath('output', os.path.join(customPath.results(),experiment.name, ''))

    # experiment plan
    experiment.addPlan('plan',
        alg = ['sbr'],
        data = ['orchideaSol', 'medleySolosDB', 'gtzan'],
        # data = ['medleySolosDB', 'gtzan'],
        # data = ['gtzan'],
        method = ['replication'],
        phase = ['oracle', 'flipped', 'noise'],
        matchingEnergy = [0.25, 0.5, 1.0],
        nfft = [512, 1024]
        )

    # experiment metrics
    experiment.setMetrics(
        sdr = ['mean+'],
        lsd = ['mean-']
    )

    return experiment

# processing for each step in the plan 
def step(setting, experiment):
    # use the evaluate function with the given settings
    sdr, lsd = evaluate(setting, experiment)

    # store the resulting metrics for a whole dataset
    np.save(os.path.join(experiment.path.output,setting.id()+'_sdr.npy'), sdr)
    np.save(os.path.join(experiment.path.output,setting.id()+'_lsd.npy'), lsd)

if __name__ == "__main__":
  doce.cli.main()
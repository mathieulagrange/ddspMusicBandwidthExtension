import numpy as np
from metrics import sdr, lsd
import doce
import sys
from evaluate import evaluate

def set (args):
    # experiment definition
    experiment = doce.Experiment(
        name = 'sbr_replication',
        purpose = 'evaluating sbr metrics with different parameters',
        author = 'PA Grumiaux',
    )

    # experiment path
    experiment.setPath('output', '/home/user/Documents/Python/modibe/'+experiment.name+'/')

    # experiment plan
    experiment.addPlan('plan',
        alg = ['sbr'],
        data = ['orchideaSOL'],
        method = ['replication'],
        phase = ['oracle', 'flipped', 'random'],
        matchingEnergy = [0.25, 0.5, 1.0]
        )

    # experiment metrics
    experiment.setMetrics(
        sdr = ['+'],
        lsd = ['-']
    )

    return experiment

# processing for each step in the plan 
def step(setting, experiment):
    # use the evaluate function with the given settings
    sdr, lsd = evaluate(setting)

    # store the resulting metrics for a whole dataset
    np.save(experiment.path.output+setting.id()+'sdr.npy', sdr)
    np.save(experiment.path.output+setting.id()+'lsd.npy', lsd)

if __name__ == "__main__":
  doce.cli.main()
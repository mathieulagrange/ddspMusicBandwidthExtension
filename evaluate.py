from datasets import NsynthDataset, OrchideaSOLDataset, MedleyDBSoloDataset, GtzanDataset
from metrics import sdr, lsd
from sbr import sbr
import numpy as np

def evaluate(alg, data, metrics):
    # dataset instantiation
    if data == 'nsynth':
        dataset = NsynthDataset()
    elif data == 'orchidea':
        dataset = OrchideaSOLDataset()
    elif data == 'medley':
        dataset = MedleyDBSoloDataset()
    elif data == 'gtzan':
        dataset = GtzanDataset()

    
import os

def dataset():
    return '/home/user/Documents/Datasets'

def orchideaSOL():
    return os.path.join(dataset(),'orchideaSOL')

def orchideaSOL_tiny():
    return os.path.join(dataset(),'orchideaSOL_tiny')

def medleySolosDB():
    return os.path.join(dataset(),'medley')

def gtzan():
    return os.path.join(dataset(), 'gtzan')

def nsynth():
    return os.path.join(dataset(), 'nsynth')

def synthetic():
    return os.path.join(dataset(), 'synthetic')

def dsd():
    return os.path.join(dataset(), 'dsd')

def dsd_sources():
    return os.path.join(dsd(), 'sources')

def dsd_mixtures():
    return os.path.join(dsd(), 'mixtures')

def results():
    return os.path.join(os.getcwd(),'results', '')

def models():
    return os.path.join(os.getcwd(),'models', '')
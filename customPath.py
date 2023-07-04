import os

def dataset():
    return 'Y:/Datasets/postdoc/Datasets/'

def orchideaSOL():
    return os.path.join(dataset(),'orchideaSOL')

def medleySolosDB():
    return os.path.join(dataset(),'medleySolos')

def medleyDB_mixtures():
    return os.path.join(dataset(),'medleyDB','mixtures')

def gtzan():
    return os.path.join(dataset(), 'gtzan')

def synthetic():
    return os.path.join(dataset(), 'synthetic')

def synthetic_poly():
    return os.path.join(dataset(), 'synthetic_poly')

def results():
    return os.path.join(os.getcwd(),'results', '')

def models():
    return os.path.join(os.getcwd(),'models', '')

def config():
    return os.path.join(os.getcwd(), 'config', '')

def slurm():
    return os.path.join(os.getcwd(), 'slurm', '')
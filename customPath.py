import os

def dataset():
    return 'F:/postdoc/Documents/Datasets/Datasets'

def orchideaSOL():
    return os.path.join(dataset(),'orchideaSOL')

def orchideaSOL_tiny():
    return os.path.join(dataset(),'orchideaSOL_tiny')

def medleySolosDB():
    return os.path.join(dataset(),'medleySolos')

def medleyDB_stems():
    return os.path.join(dataset(),'medleyDB','stems')

def medleyDB_mixtures():
    return os.path.join(dataset(),'medleyDB','mixtures')

def gtzan():
    return os.path.join(dataset(), 'gtzan')

def nsynth():
    return os.path.join(dataset(), 'nsynth')

def synthetic():
    return os.path.join(dataset(), 'synthetic')

def synthetic_crepe():
    return os.path.join(dataset(), 'synthetic_crepe')

def synthetic_poly():
    return os.path.join(dataset(), 'synthetic_poly')

def synthetic_poly_2():
    return os.path.join(dataset(), 'synthetic_poly_2')

def synthetic_poly_3():
    return os.path.join(dataset(), 'synthetic_poly_3')

def synthetic_poly_mono():
    return os.path.join(dataset(), 'synthetic_poly_mono')

def synthetic_poly_mono_2():
    return os.path.join(dataset(), 'synthetic_poly_mono_2')

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

def config():
    return os.path.join(os.getcwd(), 'config', '')

def slurm():
    return os.path.join(os.getcwd(), 'slurm', '')
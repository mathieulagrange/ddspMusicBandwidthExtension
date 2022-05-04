import os

def dataset():
    return '/home/user/Documents/Datasets'

def orchideaSOL():
    return os.path.join(dataset(),'orchideaSOL')

def medleySolosDB():
    return os.path.join(dataset(),'medley-solos-DB')

def gtzan():
    return os.path.join(dataset(), 'gtzan')

def results():
    return os.path.join(os.getcwd(),'results', '')
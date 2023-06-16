# Efficient bandwidth extension of musical signals using a differentiable harmonic plus noise model

This repo contains code for the paper: **Efficient bandwidth extension of musical signals using a differentiable harmonic plus noise model** [1]. It uses a DDSP [2] approach to reconstruct the missing high frequency content of a signal sampled at 16 kHz with only the low frequencies up to 2 kHz. Results of the paper can be reproduced looking at section 2. Processing low-band input signal is explained in section 3. Demonstration of different model reconstructions can be found in [the audio samples page](https://mathieulagrange.github.io/ddspMusicBandwidthExtension/)

## 1 - Setup Instructions

To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

If you want to replicate paper results, you need to download the required datasets, namely the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification), [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html), and [SONYC-UST](https://zenodo.org/record/3966543#.ZFtddpHP1kg) datasets. Preprocessing on a particular dataset is done by using

```
python3 preprocessing.py my_dataset_folder
```

## 2 - Replication of paper results

## REFERENCES

[1] link to our paper

[2] Engel, J., Hantrakul, L., Gu, C., & Roberts, A., “Differentiable digital signal processing” 2020, arXiv preprint arXiv:2001.04643.
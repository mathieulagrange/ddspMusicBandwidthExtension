# Efficient bandwidth extension of musical signals using a differentiable harmonic plus noise model

This repo contains code for the paper: **Efficient bandwidth extension of musical signals using a differentiable harmonic plus noise model** [1]. It uses a DDSP [2] approach to reconstruct the missing high frequency content of a signal sampled at 16 kHz with only the low frequencies up to 2 kHz. Results of the paper can be reproduced looking at section 2. Processing low-band input signal is explained in section 3. Demonstration of different model reconstructions can be found in [the audio samples page](https://mathieulagrange.github.io/ddspMusicBandwidthExtension/)

## 1 - Setup Instructions

To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

If you want to replicate paper results, you need to download the required datasets, namely [Medley-solos-db](https://zenodo.org/record/3464194), [OrchideaSOL](https://forum.ircam.fr/projects/detail/orchideasol/), [Gtzan](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), and [MedleyDB](https://medleydb.weebly.com/) datasets. Monophonic and synthetic datasets are obtained by using scripts `generate_synth_dataset.py` and `generate_polysynth_dataset.py`. Preprocessing on a particular dataset is done by using

```
python3 preprocessing.py --name config_file_name --split split
```

where config_file_name is the name of the config file without extension, i.e., *config* if you are using the provided default config file `config.yaml` in the config folder, and split is either *train* or *test*.

In each dataset folder, you should have two split folders labeled *train* and *test* containing the .wav files. Modify `customPath.py` to provide your own paths.

## 2 - Replication of paper results

## REFERENCES

[1] link to our paper

[2] Engel, J., Hantrakul, L., Gu, C., & Roberts, A., “Differentiable digital signal processing” 2020, arXiv preprint arXiv:2001.04643.
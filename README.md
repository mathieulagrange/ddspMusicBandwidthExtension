# Efficient bandwidth extension of musical signals using a differentiable harmonic plus noise model

This repo contains code for the paper: **Efficient bandwidth extension of musical signals using a differentiable harmonic plus noise model** [1]. It uses a DDSP [2] approach to reconstruct the missing high frequency content of a signal sampled at 16 kHz with only the low frequencies up to 2 kHz. Results of the paper can be reproduced looking at section 2. Processing low-band input signal is explained in section 3. Demonstration of different model reconstructions can be found in [the audio samples page](https://mathieulagrange.github.io/ddspMusicBandwidthExtension/)

## 1 - Setup Instructions

To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

## 2 - Preparation and preprocessing of datasets

If you want to replicate paper results, you need to download the required datasets, namely [Medley-solos-db](https://zenodo.org/record/3464194), [OrchideaSOL](https://forum.ircam.fr/projects/detail/orchideasol/), [Gtzan](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), and [MedleyDB](https://medleydb.weebly.com/) datasets. Monophonic and synthetic datasets are obtained by using scripts `generate_synth_dataset.py` and `generate_polysynth_dataset.py`. Preprocessing on a particular dataset is done by using

```
python3 preprocessing.py --name config_file_name --split split
```

where *config_file_name* is the name of the config file without extension, i.e., *config* if you are using the provided default config file `config.yaml` in the config folder, and *split* is either *train* or *test*.

In each dataset folder, you should have two split folders labeled *train* and *test* containing the .wav files. **Modify `customPath.py` to provide your own dataset paths**.

## 3 - Training models

Once your data has been preprocessed, you can train the model specified by `config.yaml` using

```
python3 train.py --name config_file_name --steps num_steps --retrain retrain
```

where *config_file_name* is the name of the config file (typically the same as with preprocess.py), *num_steps* is the number of training steps (which will be rounded to perform a whole number of epochs, depending on the dataset size), and *retrain* is a boolean specifying if the training must be continued from a previous saved model (if applicable, and with the same name) or if the model is retrained from zero.

The model are normally automatically saved in a folder *models/model_name* which contains the original config file *config.yaml*, the model state *state.pth* and a log file *training.log*.

## 4 Evaluating models

Once your model has been trained and saved, you can evaluate it using:

```
python evaluate.py --name config_file_name --dataset dataset_name --max-n-sources max_number_of_sources
```

where *config_file_name* is the name of the config file, *dataset_name* is the name of the dataset which can be the same as entry *dataset* of *data* in your config file or another one (be careful of checking your dataset paths in *customPath.py*), and *max_number_of_sources* corresponding the maximum number of f0 that are considered in the polyphonic mixture.

## 5 Config file entries

Below we detail what each entry of the config file correspond to in our experiment/models:
- **data**
    - **dataset**: dataset name (be sure to modify `customPath.py` with your own paths)
    - **extension**: sound file extension
    - **input**: `LB` if you input a low-band signal in your model, `WB` if you input a full-band signal
    - **mean_loudness**: automatically added by `preprocessed.py`, computed as the mean loudness over the whole dataset
    - **std_loudness**: automatically added by `preprocessed.py`, computed as the standard deviation of the loudness over the whole dataset
- **model**
    - **block_size**: size of blocks in the model
    - **device**: `cpu` or `cuda`, device on which computation is done
    - **hidden_size**: size of some hidden layers in the model
    - **max_sources**: max number of f0 considered in the model
    - **n_bands**: number of frequency bands in the estimated noise transfer function  
    - **n_harmonics**: number of harmonics considered by the additive synthesizer
    - **sampling_rate**: sampling rate of input and output signals
- **preprocess**
    - **block_size**: block size used in the pitch estimators
    - **downsampling_factor**: ratio between the highest frequency wanted in the output signal and the actual highest frequency in the input signal (4 in our experiments)
    - **oneshot**: `true` if we crop the preprocess signal to the first chunk of lenght *signal_length*, `false` if we just frame it with chunks of size *signal_length*
    - **sampling_rate**: sampling rate of the considered signals
    - **signal_length**: length of the chunks obtained by preprocessing, which correspond to the model input signal length
- **train**
    - **HF**: `true` if MSS loss is computed only on reconstructed high frequencies, `false` if it is computed on the whole band
    - **batch**: batch size
    - **model**: model type: `ddsp`, `ddsp_noise`, `ddsp_poly_decoder` or `resnet`
    - **overlap**: frame overlap used in MSS loss
    - **scales**: FFT sizes used in MSS loss
    - **start_lr**: start learning rate during training
    - **stop_lr**: last learning rate

## REFERENCES

[1] link to our paper

[2] Engel, J., Hantrakul, L., Gu, C., & Roberts, A., “Differentiable digital signal processing” 2020, arXiv preprint arXiv:2001.04643.
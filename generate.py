from scipy.io.wavfile import write
from models import OriginalAutoencoder
from data import OrchideaSol, OrchideaSolTiny, MedleySolosDB, Gtzan
import os

filenames_to_generate_sol_test = ['Acc-ord-C6-mf-alt2-N.wav',
                             'Vc-ord-D3-pp-4c-N.wav',
                             'Hp-ord-A#4-ff-N-N.wav',
                             'Cb-ord-C#3-pp-1c-N.wav',
                             'Gtr-ord-A#4-mf-3c-T21d.wav',
                             'ASax-ord-B3-ff-N-N.wav',
                             'Vn-ord-F#5-ff-3c-N.wav',
                             'TpC-ord-F#5-ff-N-N.wav',
                             'Tbn-ord-F#4-pp-N-T10u.wav',
                             'Fl-ord-F4-mf-N-N.wav',
                             'Ob-ord-C#6-mf-N-N.wav',
                             'Hn-ord-C3-pp-N-N.wav']

filenames_to_generate_sol_train = ['Acc-ord-A3-pp-N-N.wav',
                             'Vc-ord-E4-mf-2c-N.wav',
                             'Hp-ord-B3-pp-N-N.wav',
                             'Cb-ord-A#4-ff-1c-N.wav',
                             'Gtr-ord-A#3-mf-5c-T11d.wav',
                             'ASax-ord-D4-pp-N-N.wav',
                             'Vn-ord-D5-mf-2c-N.wav',
                             'TpC-ord-A3-pp-N-T10d.wav',
                             'Tbn-ord-G#3-mf-N-N.wav',
                             'Fl-ord-F#5-pp-N-N.wav',
                             'Ob-ord-A6-pp-N-N.wav',
                             'Hn-ord-G1-mf-N-T40d.wav']

def checkpoint_test_generation(model_dir, model, dataset, latest_checkpoint_n_steps = 0):
    if dataset == 'sol':
        dataset_test = OrchideaSol('test', 4, 16000, 250)
        ds_test = dataset_test.get_dataset()
        ds_test = ds_test.batch(batch_size=1)

    model.restore(os.path.join(model_dir, 'train_files'))

    for batch in ds_test:
        name = batch['filename'].numpy()[0].decode('utf-8')
        if name in filenames_to_generate_sol_test:
            outputs = model(batch, training=False)
            regen_audio = model.get_audio_from_outputs(outputs)

            # save audio file
            audio_dir = os.path.join(model_dir, 'audio')
            if not os.path.isdir(audio_dir):
                os.mkdir(audio_dir)
                os.mkdir(os.path.join(audio_dir, 'test'))
            write(os.path.join(audio_dir, 'test', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav'), 16000, regen_audio.numpy()[0])

def checkpoint_train_generation(model_dir, model, dataset, latest_checkpoint_n_steps = 0):
    if dataset == 'sol':
        dataset_test = OrchideaSol('train', 4, 16000, 250)
        ds_test = dataset_test.get_dataset()
        ds_test = ds_test.batch(batch_size=1)

    model.restore(os.path.join(model_dir, 'train_files'))

    for batch in ds_test:
        name = batch['filename'].numpy()[0].decode('utf-8')
        if name in filenames_to_generate_sol_train:
            outputs = model(batch, training=False)
            regen_audio = model.get_audio_from_outputs(outputs)

            # save audio file
            audio_dir = os.path.join(model_dir, 'audio')
            if not os.path.isdir(os.path.join(audio_dir, 'train')):
                os.mkdir(os.path.join(audio_dir, 'train'))
            write(os.path.join(audio_dir, 'train', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav'), 16000, regen_audio.numpy()[0])

def generate_audio(model_dir, model, batch):
    model = OriginalAutoencoder()
    model.restore(model_dir)

    outputs = model(batch, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)

    return audio_gen
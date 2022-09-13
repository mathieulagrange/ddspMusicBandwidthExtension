from scipy.io.wavfile import write
from models import OriginalAutoencoder
from data import OrchideaSol, OrchideaSolTiny, MedleySolosDB, Gtzan
import os
import logging

filenames_to_generate_sol_test = {'sol': ['Acc-ord-C6-mf-alt2-N.wav',
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
                                          'Hn-ord-C3-pp-N-N.wav'],
                                 'medley': ['Medley-solos-DB_test-0_15a17c2d-6c65-5d88-f6f2-e85359a4e115.wav',
                                            'Medley-solos-DB_test-1_04be2e68-9a7e-5ec4-f58e-ea81edd237f2.wav',
                                            'Medley-solos-DB_test-2_5ddfd525-2b8f-5a35-f62a-305da76775c2.wav',
                                            'Medley-solos-DB_test-3_1e12111e-b736-517f-f0e6-8ec6d1e6b272.wav',
                                            'Medley-solos-DB_test-4_0f9f7245-4da4-52c9-f129-b810bd1a306b.wav',
                                            'Medley-solos-DB_test-5_4fb29769-ebb2-5743-f9f4-3527048accfb.wav',
                                            'Medley-solos-DB_test-6_62697b00-3b10-5b8d-fbff-de69fbbfcc90.wav',
                                            'Medley-solos-DB_test-7_4c39a27e-344e-5089-f039-1cb0092221bc.wav',
                                            ]
                                }

filenames_to_generate_sol_train = {'sol': ['Acc-ord-A3-pp-N-N.wav',
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
                                            'Hn-ord-G1-mf-N-T40d.wav'],
                                   'medley': ['Medley-solos-DB_training-0_3f62ba94-7e44-5989-f780-952c9778b57e.wav',
                                              'Medley-solos-DB_training-1_b4ecdccd-b46d-5e94-fdcd-6f97a1069bf4.wav',
                                              'Medley-solos-DB_training-2_757d3c3a-9463-5726-fbb9-037ebd69efbb.wav',
                                              'Medley-solos-DB_training-3_35b261f8-bc8d-5eb9-f999-2bbe3caf55a2.wav',
                                              'Medley-solos-DB_training-4_1e73d239-09a7-5544-f5ca-637d2b9bb4a0.wav',
                                              'Medley-solos-DB_training-5_5795ce2a-7852-5949-f231-b8d90ad228fc.wav',
                                              'Medley-solos-DB_training-6_a6746c67-f763-5c73-f408-197acdc40e0b.wav',
                                              'Medley-solos-DB_training-7_5a5e4949-6886-507e-f49b-db979a72b371.wav']
                                }

def checkpoint_test_generation(model_dir, model, dataset, latest_checkpoint_n_steps = 0, hb = False):
    if dataset == 'sol':
        dataset_test = OrchideaSol('test', 4, 16000, 250)
        ds_test = dataset_test.get_dataset()
        ds_test = ds_test.batch(batch_size=1)
    elif dataset == 'medley':
        dataset_test = MedleySolosDB('test', 4, 16000, 250)
        ds_test = dataset_test.get_dataset()
        ds_test = ds_test.batch(batch_size=1)

    model.restore(os.path.join(model_dir, 'train_files'))

    for batch in ds_test:
        name = batch['filename'].numpy()[0].decode('utf-8')
        audio_dir = os.path.join(model_dir, 'audio')
        if name in filenames_to_generate_sol_test[dataset] and not os.path.isfile(os.path.join(audio_dir, 'test', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav')):
            path_temp = os.path.join(audio_dir, 'test', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav')
            # outputs from model
            outputs = model(batch, training=False)
            
            # reconstructed signal
            regen_audio = model.get_audio_from_outputs(outputs).numpy()[0]
            
            # if only high band
            if hb:
                audio_LB = batch['audio_LB'].numpy()[0]
                regen_audio = regen_audio + audio_LB

            # save audio file
            if not os.path.isdir(audio_dir):
                os.mkdir(audio_dir)
                os.mkdir(os.path.join(audio_dir, 'test'))
            write(os.path.join(audio_dir, 'test', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav'), 16000, regen_audio)

def checkpoint_train_generation(model_dir, model, dataset, latest_checkpoint_n_steps = 0, hb = False):
    if dataset == 'sol':
        dataset_test = OrchideaSol('train', 4, 16000, 250)
        ds_test = dataset_test.get_dataset()
        ds_test = ds_test.batch(batch_size=1)
    elif dataset == 'medley':
        dataset_test = MedleySolosDB('test', 4, 16000, 250)
        ds_test = dataset_test.get_dataset()
        ds_test = ds_test.batch(batch_size=1)

    model.restore(os.path.join(model_dir, 'train_files'))

    for batch in ds_test:
        name = batch['filename'].numpy()[0].decode('utf-8')
        audio_dir = os.path.join(model_dir, 'audio')
        if name in filenames_to_generate_sol_train[dataset] and not os.path.isfile(os.path.join(audio_dir, 'train', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav')):
            outputs = model(batch, training=False)
            regen_audio = model.get_audio_from_outputs(outputs).numpy()[0]

            # if only high band
            if hb:
                audio_LB = batch['audio_LB'].numpy()[0]
                regen_audio = regen_audio + audio_LB

            # save audio file
            audio_dir = os.path.join(model_dir, 'audio')
            if not os.path.isdir(os.path.join(audio_dir, 'train')):
                os.mkdir(os.path.join(audio_dir, 'train'))
            write(os.path.join(audio_dir, 'train', f'{name[:-4]}_regen_{latest_checkpoint_n_steps}.wav'), 16000, regen_audio)

def generate_audio(model_dir, model, batch):
    model = OriginalAutoencoder()
    model.restore(model_dir)

    outputs = model(batch, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)

    return audio_gen    
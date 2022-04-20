import os
from pickle import FALSE
import random
import shutil

random.seed(4)

dataset_path = "/home/user/Documents/Datasets/orchideaSOL/"
data_path = os.path.join(dataset_path, 'OrchideaSOL2020')
override = True

split_proportions = [0.8, 0.1, 0.1]

train_list_ids = []
valid_list_ids = []
test_list_ids = []

train_split_file = os.path.join(dataset_path, 'train_split.txt')
valid_split_file = os.path.join(dataset_path, 'valid_split.txt')
test_split_file = os.path.join(dataset_path, 'test_split.txt')

if os.path.isfile(train_split_file) and not override:
    raise ValueError("train_split.txt already exist")
else:
    # random split and save ids
    for family in os.listdir(data_path):
        for instrument in os.listdir(os.path.join(data_path, family)):
            list_ids = [f for f in os.listdir(os.path.join(data_path, family, instrument))]
            list_ids = sorted(list_ids)
            random.shuffle(list_ids)
            n_ids = len(list_ids)
            # add to train list ids
            train_list_ids += list_ids[:int(split_proportions[0]*n_ids)]
            # add to valid list ids
            valid_list_ids += list_ids[int(split_proportions[0]*n_ids):int((split_proportions[0]+split_proportions[1])*n_ids)]
            # add to test list ids
            test_list_ids += list_ids[int((split_proportions[0]+split_proportions[1])*n_ids):]

    # save ids in txt files
    with open(os.path.join(dataset_path, 'train_split.txt'), 'w') as f:
        f.writelines([f'{s}\n' for s in train_list_ids])
    with open(os.path.join(dataset_path, 'valid_split.txt'), 'w') as f:
        f.writelines([f'{s}\n' for s in valid_list_ids])
    with open(os.path.join(dataset_path, 'test_split.txt'), 'w') as f:
        f.writelines([f'{s}\n' for s in test_list_ids])

    # copy in train, valid and test folders
    for family in os.listdir(data_path):
        for instrument in os.listdir(os.path.join(data_path, family)):
            for audio_file in os.listdir(os.path.join(data_path, family, instrument)):
                if audio_file in train_list_ids:
                    shutil.copy(os.path.join(data_path, family, instrument, audio_file), os.path.join(dataset_path, 'train', audio_file))
                if audio_file in valid_list_ids:
                    shutil.copy(os.path.join(data_path, family, instrument, audio_file), os.path.join(dataset_path, 'valid', audio_file))
                if audio_file in test_list_ids:
                    shutil.copy(os.path.join(data_path, family, instrument, audio_file), os.path.join(dataset_path, 'test', audio_file))


print('Split in train, valid and test folders done:')
print(f'{len(train_list_ids)} ids in train split')
print(f'{len(valid_list_ids)} ids in valid split')
print(f'{len(test_list_ids)} ids in test split')
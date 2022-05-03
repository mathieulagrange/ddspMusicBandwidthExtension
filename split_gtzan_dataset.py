import os
from pickle import FALSE
import random
import shutil
from tqdm import tqdm

random.seed(4)

dataset_path = "/home/user/Documents/Datasets/gtzan/"
override = True
splits = ['train', 'test', 'valid']

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
    for genre in [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,d)) and d not in splits]:
        print(f'Scanning and splitting audio files of genre {genre} ...')
        list_ids = [f for f in os.listdir(os.path.join(dataset_path, genre)) if f.endswith('.wav')]
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

    print('Splits train, valid et test prepared.')

    # copy in train, valid and test folders
    for genre in [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,d)) and d not in splits]:
        print(f'Copying files from genre {genre} into splits ...')
        for audio in tqdm(os.listdir(os.path.join(dataset_path, genre))):
            if audio in train_list_ids:
                shutil.copy(os.path.join(dataset_path, genre, audio), os.path.join(dataset_path, 'train', audio))
            if audio in valid_list_ids:
                shutil.copy(os.path.join(dataset_path, genre, audio), os.path.join(dataset_path, 'valid', audio))
            if audio in test_list_ids:
                shutil.copy(os.path.join(dataset_path, genre, audio), os.path.join(dataset_path, 'test', audio))


print('Split in train, valid and test folders done:')
print(f'{len(train_list_ids)} ids in train split')
print(f'{len(valid_list_ids)} ids in valid split')
print(f'{len(test_list_ids)} ids in test split')
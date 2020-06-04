import numpy as np
import os


def split_2(mydir = '../../dataset/', 
               file_vol = 'vol.npy', 
               file_lbl = 'lbl.npy'):
    
    files = {'vol': file_vol, 'lbl': file_lbl}
    
#     for label in ('vol', 'lbl'):
#         path = os.path.join(mydir, files[label])
#         dataset = np.load(path)
#     	for split in ('test', 'train'):

    path_input_lbl = os.path.join(mydir, file_lbl)
    dataset_lbl = np.load(path_input_lbl)

    print("Volumes data size: \n", dataset_vol.shape)
    print("Labels data size: \n", dataset_lbl.shape)

    # shuffle the dataset
    n_examples = len(dataset_lbl)
    shuffle = np.random.permutation(n_examples)

    train_percent = 0.8
    n_train = int(train_percent * n_examples)

    ds_lbl_train = dataset_lbl[shuffle[:n_train], :]
    ds_lbl_test = dataset_lbl[shuffle[n_train:],:]

    ds_vol_train = dataset_vol[shuffle[:n_train],:]
    ds_vol_test = dataset_vol[shuffle[n_train:],:]

    file_train_vol = os.path.join(mydir, 'train/vol_classification_train.npy')
    file_test_vol = os.path.join(mydir, 'test/vol_classification_test.npy')

    file_train_lbl = os.path.join(mydir, 'train/dataset_lbl_classification_training.npy')
    file_test_lbl = os.path.join(mydir, 'test/dataset_lbl_classification_validation.npy')

    np.save(file_train_vol, ds_vol_train)
    np.save(file_test_vol, ds_vol_test)

    np.save(file_train_lbl, ds_lbl_train)
    np.save(file_test_lbl, ds_lbl_validation)
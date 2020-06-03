import numpy as np
import os


def split_n():
    

def split_2(mydir = '../dataset/classification/', 
               file_vol = 'dataset_vol_classification.npy', 
               file_lbl = 'dataset_lbl_classification.npy'):
    
    path_input_vol = os.path.join(mydir, file_vol)
    path_input_lbl = os.path.join(mydir, file_lbl)
    
    dataset_vol = np.load(path_input_vol)
    dataset_lbl = np.load(path_input_lbl)

    print("Volumes data size: \n", dataset_vol.shape)
    print("Labels data size: \n", dataset_lbl.shape)

    # shuffle the dataset
    n_examples = len(dataset_lbl)
    shuffle = np.random.permutation(n_examples)

    train_percent = 0.8
    n_train = int(train_percent * n_examples)

    ds_lbl_train = dataset_lbl[shuffle[:n_train], :]
    ds_lbl_validation = dataset_lbl[shuffle[n_train:],:]

    ds_vol_train = dataset_vol[shuffle[:n_train],:]
    ds_vol_validation = dataset_vol[shuffle[n_train:],:]

    file_training_vol = os.path.join(mydir, 'training/dataset_vol_classification_training.npy')
    file_validation_vol = os.path.join(mydir, 'validation/dataset_vol_classification_validation.npy')

    file_training_lbl = os.path.join(mydir, 'training/dataset_lbl_classification_training.npy')
    file_validation_lbl = os.path.join(mydir, 'validation/dataset_lbl_classification_validation.npy')

    np.save(file_training_vol, ds_vol_train)
    np.save(file_validation_vol, ds_vol_validation)

    np.save(file_training_lbl, ds_lbl_train)
    np.save(file_validation_lbl, ds_lbl_validation)
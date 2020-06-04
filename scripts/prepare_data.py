import numpy as np
import os
import sys
import argparse

def split_train_test(dataset_folder, 
                   file_vol, 
                   file_lbl,
                   train_percent=0.8):
    
#     files = {'vol': file_vol, 'lbl': file_lbl}
    
#     for label in ('vol', 'lbl'):
#         path = os.path.join(dataset_folder, files[label])
#         dataset = np.load(path)
#     	for split in ('test', 'train'):

    
    path_vol = os.path.join(dataset_folder, file_vol)
    dataset_vol = np.load(path_vol)
    
    path_lbl = os.path.join(dataset_folder, file_lbl)
    dataset_lbl = np.load(path_lbl)

    # shuffle the dataset
    n_examples = len(dataset_lbl)
    shuffle = np.random.permutation(n_examples)
    
    n_train = int(train_percent * n_examples)

    ds_lbl_train = dataset_lbl[shuffle[:n_train], :]
    ds_lbl_test = dataset_lbl[shuffle[n_train:],:]
    del dataset_lbl
    
    ds_vol_train = dataset_vol[shuffle[:n_train],:]
    ds_vol_test = dataset_vol[shuffle[n_train:],:]
    del dataset_vol
    
    file_train_vol = os.path.join(dataset_folder, 'vol_train.npy')   
    file_train_lbl = os.path.join(dataset_folder, 'lbl_train.npy')
    
    file_test_vol = os.path.join(dataset_folder, 'vol_test.npy')
    file_test_lbl = os.path.join(dataset_folder, 'lbl_test.npy')

    np.save(file_train_vol, ds_vol_train)
    np.save(file_test_vol, ds_vol_test)

    np.save(file_train_lbl, ds_lbl_train)
    np.save(file_test_lbl, ds_lbl_test)

    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='dataset')
    parser.add_argument('--file_vol', type=str, default='np_dataset_vol.npy')
    parser.add_argument('--file_lbl', type=str, default='np_dataset_lbl.npy')
    parser.add_argument('--train_percent', type=float, default=0.8)
    return parser
       
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    split_train_test(dataset_folder=args.dataset_folder,
            file_vol=args.file_vol,
            file_lbl=args.file_lbl)
    
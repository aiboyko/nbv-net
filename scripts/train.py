import numpy as np
import os
import sys
import argparse
import csv
import cnbv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim
from torch.autograd import Variable

#from tqdm import tqdm
from cnbv import validation



def check_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(dataset_folder, file_vol_train, file_lbl_train, file_vol_test, file_lbl_test):
    path_to_vol_train = os.path.join(dataset_folder, file_vol_train)
    path_to_lbl_train = os.path.join(dataset_folder, file_lbl_train)
    
    train_dataset = cnbv.Dataset_NBVC_Full_numpy(vol_file=path_to_vol_train,
                                                 lbl_file=path_to_lbl_train,
                                                 transform=transforms.Compose([cnbv.To3DGrid(), cnbv.ToTensor()])
                                                 )
                                                
    
    path_to_vol_test = os.path.join(dataset_folder, file_vol_test)
    path_to_lbl_test = os.path.join(dataset_folder, file_lbl_test)

    test_dataset = cnbv.Dataset_NBVC_Full_numpy(vol_file = path_to_vol_test,
                                            lbl_file = path_to_lbl_test,
                                            transform=transforms.Compose([cnbv.To3DGrid(), cnbv.ToTensor()])
                                            )
    megads_train = torch.utils.data.TensorDataset(torch.Tensor(train_dataset.grid_data.reshape(-1,1,32,32,32)).to(device),
                                                          torch.tensor(train_dataset.nbv_class_data, dtype=torch.long).to(device))
    train_dataloader_fullcast = DataLoader(dataset=megads_train, shuffle=True, batch_size=batch_size, drop_last=True)
    
    
    megads_test = torch.utils.data.TensorDataset(torch.Tensor(test_dataset.grid_data.reshape(-1,1,32,32,32)).to(device),
                                                 torch.tensor(test_dataset.nbv_class_data, dtype=torch.long).to(device))
    test_dataloader_fullcast = DataLoader(dataset=megads_test, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_dataloader_fullcast, test_dataloader_fullcast



def get_models():
    model_1fc = cnbv.NBV_Net_1FC(dropout_prob)
    model_2fc = cnbv.NBV_Net_2FC(dropout_prob)
    model_3fc = cnbv.NBV_Net_3FC(dropout_prob)
    model_4fc = cnbv.NBV_Net_4FC(dropout_prob)
    model_5fc = cnbv.NBV_Net_5FC(dropout_prob)
    
    
    model_1fc = model_1fc.to(device)
    model_2fc = model_2fc.to(device)
    model_3fc = model_3fc.to(device)
    model_4fc = model_4fc.to(device)
    model_5fc = model_5fc.to(device)
    return [model_1fc, model_2fc, model_3fc, model_4fc, model_5fc]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='dataset')
    parser.add_argument('--file_vol_train', type=str, default='vol_train.npy')
    parser.add_argument('--file_lbl_train', type=str, default='lbl_train.npy')
    parser.add_argument('--file_vol_test', type=str, default='vol_test.npy')
    parser.add_argument('--file_lbl_test', type=str, default='lbl_test.npy')
    return parser






if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    device = check_device()
    
    display_dataset = True
    display_fwd_pretraining = True
    load_weights = False
    reading_weights_file = 'weights/paper_param.pth'
    saving_weights_file = 'log/current_param.pth'
    epochs = 40
    batch_size = 256
    learning_rate = 0.001
    dropout_prob = 0.3
    
    
    train_dataloader_fullcast, test_dataloader_fullcast = get_data(dataset_folder=args.dataset_folder,
                                           file_vol_train=args.file_vol_train,
                                           file_lbl_train=args.file_lbl_train,
                                           file_vol_test=args.file_vol_test,
                                           file_lbl_test=args.file_lbl_test)

    models = get_models()
    criterion = torch.nn.CrossEntropyLoss()
    
    for layers, model in enumerate(models):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        cnbv.train(model,
                   optimizer,
                   train_dataloader_fullcast,
                   test_dataloader_fullcast,
                   criterion,
                   device='cpu',
                   calculate_eval=False,
                   epochs=epochs,
                   layers = str(layers))







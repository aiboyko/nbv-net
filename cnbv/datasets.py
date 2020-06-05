from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

class To3DGrid(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv_class = sample['grid'], sample['nbv_class']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        grid = np.reshape(grid, (32,32,32))
        return grid, nbv_class


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv_class = sample['grid'], sample['nbv_class']
        
        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        return torch.from_numpy(np.array([grid])), torch.tensor(nbv_class[0], dtype=torch.int64)


class Dataset_NBVC_Full_numpy(Dataset):
    """NBV dataset."""
    def __init__(self, vol_file, lbl_file, transform=None):
        """
        Args:
            poses_file (string): Path to the sensor poses (next-best-views).
            root_dir (string): Directory with all the grids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grid_data = np.load(vol_file)
        self.nbv_class_data = np.load(lbl_file)
        self.transform = transform

    def __len__(self):
        return len(self.nbv_class_data)

    def __getitem__(self, idx):
        grid = self.grid_data[idx] 
        nbv_class = self.nbv_class_data[idx]
        sample = {'grid': grid, 'nbv_class': nbv_class}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class Dataset_NBVC_Folder(Dataset):
    """Requires a folder with .npy datafiles, which contain dicts with X, y  = data['X'], data['y'].

    Has build-in transforms to torch.Tensor
    Stores only list of data paths, loads data sample dynamically on request.
    If device=None does not cast tensor on the device, allowing for (potentially faster) full-batch casting on the upper (dataloader) level.
    """
    
    def __init__(self, dataset_dir='dataset', device=None):
        self.dataset_dir = dataset_dir
        self.listfiles = os.listdir(dataset_dir)
        self.device = device
        
    def __len__(self):
        return len(self.listfiles)
    
    def __getitem__(self, index):
        path = os.path.join(self.dataset_dir, self.listfiles[index])
        data = np.load(path, allow_pickle=True)
        X, y = torch.Tensor(data.item()['X']), torch.Tensor(data.item()['y']).long()
        
        if device is not None:
            X, y = X.to(device), Y.to(device)
            
        return X, y
    
    def generate_train_test(self, train_fraction=0.8):
        full_size = len(self)
        train_size = int(full_size*train_fraction)
        validation_size = full_size - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.dataset.random_split(dataset=self, lengths=[train_size, validation_size])
        return self.train_dataset, self.test_dataset

class Dataset_NBVC_Full_torch(Dataset):
    """NBV dataset."""
    def __init__(self, vol_file, lbl_file, device='cuda:0', transform=None):
        """
        Created to load the entire dataset on GPU, to speed up learning
        """
        self.grid_data =  torch.Tensor(np.load(vol_file)).to(device)
        self.nbv_class_data = np.load(lbl_file)
        self.transform = transform

    def __len__(self):
        return len(self.nbv_class_data)

    def __getitem__(self, idx):
        grid = self.grid_data[idx] 
        nbv_class = self.nbv_class_data[idx]
        sample = {'grid': grid, 'nbv_class': nbv_class}

        if self.transform:
            sample = self.transform(sample)

        return sample    
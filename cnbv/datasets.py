from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
    
class Dataset_NBVC_Full(Dataset):
    """NBV dataset."""
    def __init__(self, grid_file, nbv_class_file, transform=None):
        """
        Args:
            poses_file (string): Path to the sensor poses (next-best-views).
            root_dir (string): Directory with all the grids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.root_dir = root_dir
        self.grid_data = np.load(grid_file)
        self.nbv_class_data = np.load(nbv_class_file)
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
    
    
class To3DGrid(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv_class = sample['grid'], sample['nbv_class']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        grid = np.reshape(grid, (32,32,32))
        return {'grid': grid,
                'nbv_class': nbv_class}
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv_class = sample['grid'], sample['nbv_class']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        return {'grid': torch.from_numpy(np.array([grid])),
                'nbv_class': torch.tensor(nbv_class[0], dtype=torch.int64)}
    
    
class Dataset_NBVC_Folder(Dataset):
    def __init__(self, dataset_dir='dataset', device='cpu'):
        self.dataset_dir = dataset_dir
        self.listfiles = os.listdir(dataset_dir)
        self.device = device
        
    def __len__(self):
        return len(self.listfiles)
    
    def __getitem__(self, index):
        path = os.path.join(self.dataset_dir, self.listfiles[index])
        data = np.load(path, allow_pickle=True)
        return torch.Tensor(data.item()['X']), torch.Tensor(data.item()['y']).long()
    
    def generate_train_test(self, train_fraction=0.8):
        N = len(self)
        tr = int(N*train_fraction)
        val = N - tr
        self.train_dataset, self.test_dataset = torch.utils.data.dataset.random_split(dataset=self, lengths=[tr, val])
        return self.train_dataset, self.test_dataset
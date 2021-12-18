import functools
import numpy as np
import torch
from sklearn import preprocessing
from torch.utils import data
from torch.utils.data import DataLoader, Subset


class EcogDataset(data.Dataset):
    # Dataset for Ecog 
    def __init__(self, config, data, i):
        self.data = np.load(data, allow_pickle=True)
        self.ecogs = self.data[0][i]
        print(self.ecogs.shape)
        #self.ecogs = np.moveaxis(self.data[0], 2, 0)
        #self.ecogs = np.moveaxis(self.ecogs, 1, 2)
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.data[1])
        self.config = config
        
    def __getitem__(self, index):
        ecog = torch.from_numpy(self.ecogs[index]).float()
        ecog = ecog.unsqueeze(0)
        label = torch.from_numpy(self.labels)[index]
        data = {
            "ecog": ecog,
            "label": label,
        }
        return data
    
    def __len__(self):
        return self.ecogs.shape[0]

def get_dataloader(config, i):
    print('Loading EcoG dataset')
    dataset = EcogDataset(config, '../data/freq_2000.npy', i)
    loader = functools.partial(
        DataLoader,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
    )
    print(dataset.__getitem__(0)['ecog'].shape)
    num_all = len(dataset)
    num_tr = int(config.ratio_tr_data * num_all)
    idx = np.random.permutation(np.arange(num_all))
    
    dataset_tr = Subset(dataset, idx[:num_tr])
    dataset_va = Subset(dataset, idx[num_tr:])
    
    loader_tr = loader(dataset = dataset_tr, shuffle=True)
    loader_va = loader(dataset = dataset_va, shuffle=False)
    
    print(f'Number of training samples: {num_tr}')
    print(f'Number of valid samples: {num_all - num_tr}')
    print(f'Batch size: {config.batch_size}')
    return loader_tr, loader_va

if __name__ == '__main__':
    from get_config import get_config
    config = get_config()
    dataloader = get_dataloader(config)

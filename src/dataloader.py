import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class HeatSupplyDataSet(Dataset):
    def __init__(self, dir, device) -> None:
        super(HeatSupplyDataSet, self).__init__()
        data = pd.read_excel(io=dir, sheet_name=0,header=0)
        nplist = data[1:].T.to_numpy()[4:8].T
        t_before = data[0:-1].T.to_numpy()[-1]
        self.data = np.float64(np.insert(nplist, 4, t_before, axis=1))
        self.target = np.float64(data[1:].T.to_numpy()[-1])

        self.data = np.array(self.data)
        self.data = torch.FloatTensor(self.data).cuda(device= device)
        self.target = np.array(self.target)
        self.target = torch.FloatTensor(self.target).cuda(device=device)

    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.target)

def data_split(data, rate):
    train_l = int(len(data) * rate)
    test_l = len(data) -  train_l
    train_set, test_set = torch.utils.data.random_split(data, [train_l, test_l])
    return train_set, test_set
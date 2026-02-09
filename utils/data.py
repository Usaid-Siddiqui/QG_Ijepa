import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.misc import pad_to_128

class QG_Dataset(Dataset):
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        self.h5 = None
        self.data = None
        self.meta = None

        if indices is None:
            self.indices = np.arange(self._get_length())
        else:
            self.indices = indices

    def _get_length(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f['images'])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5 is None:   #happens once per worker
            self.h5 = h5py.File(self.h5_path, 'r', swmr=True)
            self.data = self.h5['images']
            self.meta = self.h5['meta']

        i = self.indices[idx]
        img = self.data[i]

        x = torch.from_numpy(img).float()
        x = torch.log1p(x)

        mean = torch.tensor([0.01122842, 0.01998984, 0.08350217])
        std = torch.tensor([0.17538942, 0.21435375, 0.55736226])
        x = (x - mean) / (std + 1e-6)

        if x.ndim == 3 and x.shape[2] <= 3:
            x = x.permute(2, 0, 1)

        x = pad_to_128(x)
        y = int(self.meta[i, 2])

        return x, y
    
# The dataset returns both image data and quark/gluon labels for each sample.
# However, the pretraining loop only requires image data, so the labels are ignored there.
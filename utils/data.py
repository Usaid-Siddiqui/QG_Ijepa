import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.misc import pad_to_128

class QG_Dataset(Dataset):
    def __init__(self, h5_path, indices=None):
        self.h5 = h5py.File(h5_path, 'r')

        # Lazy load data
        self.data = self.h5['images']
        self.meta = self.h5['meta']

        if indices is None:
            self.indices = np.arange(len(self.data))
        else:
            self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Load as numpy first, then convert to tensor
        i = self.indices[idx]
        img = self.data[i] 
        
        # Convert to torch [C, H, W]
        x = torch.from_numpy(img).float()
        if x.ndim == 3 and x.shape[2] <= 3: # If HWC
            x = x.permute(2, 0, 1)
    
        # Apply pad_to_128(x) here so the batching works seamlessly
        x = pad_to_128(x)

        # Load label (quark/gluon) from metadata
        y = int(self.meta[i, 2])

        return x, y
    
# The dataset returns both image data and quark/gluon labels for each sample.
# However, the pretraining loop only requires image data, so the labels are ignored there.
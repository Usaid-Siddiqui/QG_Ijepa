import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import h5py
import random

file_path = "/Volumes/Usaids_USB/Quark_Gluon.h5"
f = h5py.File(file_path, "r")
print(list(f.keys()))        # top-level groups

dset = f['train_jet']
meta = f['train_meta']
print(dset.shape)
X = dset[0:1000]
y = meta[0:1000]
print(X.shape)
print(y.shape)

# print("Printing chunk:")
# print(chunk[0])

# print("Printing metadata:")
# print(meta_chunk[0])

i = random.randint(0, len(X) - 1)  # jet index
plt.imshow(X[i])
plt.title(f"Label: {int(meta[i,2])}")
plt.axis("off")
plt.show(block=True)

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

file_path = "/Users/usaid/Stuff/ML/QG_ijepa/data/qg_small_train.h5"
f = h5py.File(file_path, "r")
# print(list(f.keys()))        # top-level groups
X = f['images']
y = f['meta']
print(X.shape)
print(y.shape)


"""
    In image:
    Channels (0,1,2) correspond to (Tracks, ECAL, HCAL)
    Eta/Pseudorapidity is the vertical axis
    Phi/Azimuthal angle is the horizontal axis

    In metadata:
    Channels (0,1,2) correspond to (jet pT, Multiplicity, Quark/Gluon)
    NOTE: Quark = 1, Gluon = 0
"""


sample = X[:100]          # shape (100,125,125,3)

for c in range(3):
    vals = sample[..., c].ravel()
    vals = vals[vals > 0]   # ignore zeros (important)
    plt.hist(vals, bins=100, log=True)
    plt.title(f"Channel {c}")
    plt.show()
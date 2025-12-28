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

keys = list(f.keys())
for key in keys:
    print(f"{key}: {f[key].shape}")

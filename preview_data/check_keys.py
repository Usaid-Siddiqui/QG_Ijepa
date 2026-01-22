import os
import h5py

file_path = "/Volumes/Usaids_USB/Quark_Gluon.h5"
f = h5py.File(file_path, "r")
print(list(f.keys()))        # top-level groups

keys = list(f.keys())
for key in keys:
    print(f"{key}: {f[key].shape}")

import h5py
import numpy as np
import sys

IN_FILE  = "/Volumes/Usaids_USB/Quark_Gluon.h5"
OUT_FILE = "/Volumes/Usaids_USB/qg_medium_train.h5"

N = 100_000        # Total samples (must be even)
batch_size = 2000  # Number of samples to copy per update (smaller for more frequent prints)

with h5py.File(IN_FILE, "r") as fin:
    images = fin["train_jet"]
    meta   = fin["train_meta"]
    
    # 1. Balanced Index Selection
    print("Analyzing dataset for balanced split...")
    labels = meta[:, 2].astype(int)
    q_idx = np.where(labels == 0)[0]
    g_idx = np.where(labels == 1)[0]
    
    n_each = N // 2
    idx = np.concatenate([
        np.random.choice(q_idx, n_each, replace=False),
        np.random.choice(g_idx, n_each, replace=False),
    ])
    idx = np.sort(idx) # Sorting helps HDF5 read speed

    # 2. Setup Output File
    with h5py.File(OUT_FILE, "w") as fout:
        # Initialize datasets
        img_ds = fout.create_dataset("images", (N, 125, 125, 3), 
                                     dtype='f', compression="gzip", compression_opts=4)
        meta_ds = fout.create_dataset("meta", (N, meta.shape[1]), dtype=meta.dtype)

        print(f"Starting copy of {N} samples to: {OUT_FILE}")

        # 3. Chunked Copy with Progress Print
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_indices = idx[start:end]

            # Write data chunk by chunk
            img_ds[start:end] = images[batch_indices]
            meta_ds[start:end] = meta[batch_indices]

            # Update progress line in terminal
            percent = (end / N) * 100
            sys.stdout.write(f"\rProgress: [{end}/{N}] samples copied ({percent:.1f}%)")
            sys.stdout.flush()

print(f"\nSuccessfully saved balanced subset to: {OUT_FILE}")
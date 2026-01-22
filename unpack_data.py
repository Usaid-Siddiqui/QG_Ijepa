import h5py
import argparse
import os

def unpack_h5(input_path, output_path, chunk_size=500):
    if os.path.exists(output_path):
        print(f"Target file {output_path} already exists. Skipping.")
        return

    with h5py.File(input_path, 'r') as fin:
        n_samples = fin["images"].shape[0]
        with h5py.File(output_path, 'w') as fout:
            print(f"Unpacking {n_samples} samples to {output_path}...")
            
            # Create uncompressed datasets with optimized chunking
            img_ds = fout.create_dataset("images", (n_samples, 125, 125, 3), 
                                         dtype='f', chunks=(1, 125, 125, 3))
            meta_ds = fout.create_dataset("meta", (n_samples, fin["meta"].shape[1]), 
                                          dtype=fin["meta"].dtype)

            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                img_ds[i:end] = fin["images"][i:end]
                meta_ds[i:end] = fin["meta"][i:end]
                if i % 5000 == 0:
                    print(f"Progress: {i}/{n_samples}")

    print("Unpacking Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack compressed HDF5 for fast training.")
    parser.add_argument("--input", type=str, required=True, help="Path to compressed .h5")
    parser.add_argument("--output", type=str, required=True, help="Path for uncompressed .h5")
    parser.add_argument("--batch", type=int, default=500, help="RAM-safe copy batch size")
    
    args = parser.parse_args()
    unpack_h5(args.input, args.output, args.batch)
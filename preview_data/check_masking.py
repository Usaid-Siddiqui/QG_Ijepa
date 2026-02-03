import h5py
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE EXACT I-JEPA COLLATOR LOGIC ---
class QG_MaskCollator:
    def __init__(self, grid_size=16, context_scale=(0.4, 0.6), target_scale=(0.1, 0.15), num_targets=4):
        self.grid_size = grid_size
        self.num_patches = grid_size ** 2
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.num_targets = num_targets

    def _get_block_indices(self, scale_range):
        target_area = self.num_patches * torch.empty(1).uniform_(*scale_range).item()
        aspect_ratio = torch.empty(1).uniform_(0.75, 1.5).item()
        
        h = max(2, int(round(math.sqrt(target_area * aspect_ratio))))
        w = max(2, int(round(math.sqrt(target_area / aspect_ratio))))
        
        h = min(h, self.grid_size - 1)
        w = min(w, self.grid_size - 1)
        
        top = torch.randint(0, self.grid_size - h, (1,)).item()
        left = torch.randint(0, self.grid_size - w, (1,)).item()
        
        grid = torch.arange(self.num_patches).view(self.grid_size, self.grid_size)
        block_indices = grid[top:top+h, left:left+w].flatten()
        return block_indices

    def generate_single_mask_pair(self):
        # 1. Targets
        targets = []
        combined_target_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        for _ in range(self.num_targets):
            indices = self._get_block_indices(self.target_scale)
            targets.append(indices)
            combined_target_mask[indices] = True
        
        # 2. Context
        ctx_indices = self._get_block_indices(self.context_scale)
        ctx_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        ctx_mask[ctx_indices] = True
        ctx_mask[combined_target_mask] = False # Remove overlaps (I-JEPA rule)
        
        return ctx_mask, combined_target_mask

# --- 2. DATA LOADING, NORMALIZATION & PADDING ---
def load_local_sample(h5_path, index=0):
    with h5py.File(h5_path, 'r') as f:
        # X: [Tracks, ECAL, HCAL]
        img = f['images'][index] 
        # y: [jet pT, Multiplicity, Quark/Gluon]
        meta = f['meta'][index]
    
    # Extract Q/G Truth (Index 2 in metadata)
    label = int(meta[2])
    
    # Convert to Tensor and log-normalize
    # log1p is critical to prevent the 'brightest pixel' from hogging all attention
    img_tensor = torch.from_numpy(img).float()
    img_tensor = torch.log1p(img_tensor)
    
    # Sum across channels for a consolidated visualization
    # This combines Tracks + ECAL + HCAL into one "image"
    img_vis = img_tensor.sum(dim=-1) if img_tensor.ndim == 3 else img_tensor
    
    # Pad 125x125 to 128x128
    pad_total = 128 - 125
    pad_half = pad_total // 2
    img_padded = torch.nn.functional.pad(img_vis, (pad_half, pad_total-pad_half, pad_half, pad_total-pad_half))
    
    return img_padded, label

# --- 3. MAIN VISUALIZATION LOOP ---
def visualize_random(h5_path):
    with h5py.File(h5_path, 'r') as f:
        num_samples = len(f['meta'])

    # Pick a random sample
    sample_idx = np.random.randint(0, num_samples)
    img, label = load_local_sample(h5_path, sample_idx)
    
    # Masking Logic (Using your original config_used.yaml scales)
    # Try changing context_scale to (0.4, 0.6) to see "Hard Mode"
    collator = QG_MaskCollator(
        grid_size=16, 
        context_scale=(0.3, 0.5), 
        target_scale=(0.1, 0.12),
        num_targets=2
    )
    ctx_mask, trg_mask = collator.generate_single_mask_pair()
    
    # Convert 1D masks to 2D grids for plotting
    ctx_grid = ctx_mask.view(16, 16).numpy()
    trg_grid = trg_mask.view(16, 16).numpy()
    
    # Upscale masks to match image size (8x8 pixel patches)
    ctx_up = np.repeat(np.repeat(ctx_grid, 8, axis=0), 8, axis=1)
    trg_up = np.repeat(np.repeat(trg_grid, 8, axis=0), 8, axis=1)

    plt.close('all') 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    jet_name = "Quark" if label == 1 else "Gluon"
    
    # Panel 1: Original Jet (Log-Scaled & Summed Channels)
    axes[0].imshow(img.numpy(), cmap='gist_heat')
    axes[0].set_title(f"Original {jet_name} (Idx: {sample_idx})")
    
    # Panel 2: Target Blocks (What the model must predict)
    axes[1].imshow(img.numpy(), cmap='gray')
    axes[1].imshow(np.ma.masked_where(trg_up == 0, trg_up), cmap='Blues', alpha=0.6)
    axes[1].set_title("Targets (Teacher Embeddings)")
    
    # Panel 3: Context Block (What the Encoder sees)
    axes[2].imshow(img.numpy(), cmap='gray')
    axes[2].imshow(np.ma.masked_where(ctx_up == 0, ctx_up), cmap='Greens', alpha=0.6)
    axes[2].set_title("Context (Encoder Input)")
    
    for ax in axes: ax.axis('off')
    plt.suptitle("Close window for next random sample | Ctrl+C to exit", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    H5_PATH = "/Users/usaid/Stuff/ML/QG_ijepa/data/qg_small_train.h5"  # Make sure this points to your local file
    print("Starting Visualizer...")
    try:
        while True:
            visualize_random(H5_PATH)
    except KeyboardInterrupt:
        print("\nVisualizer stopped.")
import torch
import torch.nn.functional as F
import yaml
from logging import setup_logger
import os
import datetime

def pad_to_128(x):
    # Pads tensor w/ zeros to dimension 128x128
    C, H, W = x.shape
    pad_h = 128 - H
    pad_w = 128 - W
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return x

def generate_patches(x, patch_size=8):
    """
    Converts a batch of images [B, C, H, W] into a batch of flattened patches [B, N, D].
    
    N = Number of patches (e.g., 256 for 128x128 image with patch_size 8)
    D = Patch dimension (C * patch_size * patch_size)
    
    This preserves the spatial grid order (row by row), which is 
    essential for the I-JEPA Block Masker to select contiguous regions.
    """
    B, C, H, W = x.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size."

    # 1. Reshape to separate spatial dimensions into patches: [B, C, H//p, p, W//p, p]
    x = x.view(B, C, H // p, p, W // p, p)
    
    # 2. Permute to [B, H//p, W//p, C, p, p]
    # This ensures that flattening creates a sequence of patches in row-major order.
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    
    # 3. Flatten into [B, num_patches, patch_dim]
    num_patches = (H // p) * (W // p)
    patch_dim = C * p * p
    x = x.view(B, num_patches, patch_dim)
    
    return x

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def init_experiment(cfg):
    """
    Sets up experiment directories and logging.
    Returns: logger, checkpoint_dir, and device.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['model']['name']}_{timestamp}"

    # Define paths
    checkpoint_dir = os.path.join(cfg['train']['checkpoint_dir'], run_name)
    log_dir = os.path.join(cfg['train']['log_dir'], run_name)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize logger
    logger = setup_logger(log_dir=log_dir, model_name=cfg['model']['name'])
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return logger, checkpoint_dir, device
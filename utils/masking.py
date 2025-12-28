import torch

class BlockMaskGenerator:
    def __init__(self, grid_size=16, num_targets=4, target_size=(3, 5), context_size=(8, 12)):
        """
        grid_size: 16 (for 128x128 image with patch_size 8)
        num_targets: Number of target blocks to predict
        """
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.target_size = target_size
        self.context_size = context_size

    def _get_random_block(self, low, high):
        """Generates a single rectangular mask."""
        h = torch.randint(low[0], high[0], (1,)).item()
        w = torch.randint(low[1], high[1], (1,)).item()
        
        top = torch.randint(0, self.grid_size - h + 1, (1,)).item()
        left = torch.randint(0, self.grid_size - w + 1, (1,)).item()
        
        mask = torch.zeros((self.grid_size, self.grid_size), dtype=torch.bool)
        mask[top:top+h, left:left+w] = True
        return mask.view(-1)

    def generate_batch_masks(self, batch_size, device):
        """
        Returns:
            context_masks: [B, num_patches] boolean
            target_masks: [B, num_patches] boolean
        """
        all_context = []
        all_target = []

        for _ in range(batch_size):
            # 1. Create Target Mask (union of several small blocks)
            t_mask = torch.zeros(self.grid_size**2, dtype=torch.bool)
            for _ in range(self.num_targets):
                t_mask |= self._get_random_block(self.target_size, (self.target_size[1]+1, self.target_size[1]+1))
            
            # 2. Create Context Mask (one larger block)
            c_mask = self._get_random_block(self.context_size, (self.context_size[1]+1, self.context_size[1]+1))
            
            # 3. I-JEPA Rule: Context patches must NOT include Target patches
            c_mask = c_mask & ~t_mask
            
            all_context.append(c_mask)
            all_target.append(t_mask)

        return torch.stack(all_context).to(device), torch.stack(all_target).to(device)

def apply_batch_mask(x_patches, mask):
    """
    Since each image in the batch has a different number of True values in its mask
    after the 'c_mask & ~t_mask' operation, we must extract them individually.
    
    Inputs:
        x_patches: [B, N, D] (from generate_patches)
        mask: [B, N] (boolean mask)
    Returns:
        A list of tensors, each [num_active_patches, D]
    """
    # Use list comprehension to handle varying sequence lengths per batch item
    return [x_patches[i, mask[i]] for i in range(x_patches.size(0))]
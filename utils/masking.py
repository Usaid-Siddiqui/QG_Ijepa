import torch
import math

class QG_MaskCollator:
    def __init__(self, grid_size=16, context_scale=(0.4, 0.6), target_scale=(0.15, 0.2), num_targets=4):
        self.grid_size = grid_size
        self.num_patches = grid_size ** 2
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.num_targets = num_targets

    def _get_block_indices(self, scale_range):
        # 1. Determine random block dimensions
        target_area = self.num_patches * torch.empty(1).uniform_(*scale_range).item()
        aspect_ratio = torch.empty(1).uniform_(0.75, 1.5).item()
        
        h = max(2, int(round(math.sqrt(target_area * aspect_ratio))))
        w = max(2, int(round(math.sqrt(target_area / aspect_ratio))))
        
        # Ensure block fits in grid
        h = min(h, self.grid_size - 1)
        w = min(w, self.grid_size - 1)
        
        # 2. Pick random top-left corner
        top = torch.randint(0, self.grid_size - h, (1,)).item()
        left = torch.randint(0, self.grid_size - w, (1,)).item()
        
        # 3. Create grid of indices
        grid = torch.arange(self.num_patches).view(self.grid_size, self.grid_size)
        block_indices = grid[top:top+h, left:left+w].flatten()
        return block_indices

    def __call__(self, batch):
        # batch is a list of (image, label) from QG_Dataset
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        B = images.shape[0]

        all_ctx_indices = []
        all_trg_indices = []

        for _ in range(B):
            # Generate Target Blocks
            targets = []
            combined_target_mask = torch.zeros(self.num_patches, dtype=torch.bool)
            for _ in range(self.num_targets):
                indices = self._get_block_indices(self.target_scale)
                targets.append(indices)
                combined_target_mask[indices] = True
            
            # Generate Context Block and ensure it doesn't overlap
            ctx_indices = self._get_block_indices(self.context_scale)
            # Remove target patches from context
            ctx_mask = torch.zeros(self.num_patches, dtype=torch.bool)
            ctx_mask[ctx_indices] = True
            ctx_mask[combined_target_mask] = False
            
            all_ctx_indices.append(torch.where(ctx_mask)[0])
            all_trg_indices.append(torch.cat(targets))

        # Pad indices to uniform length for batching
        # This makes the tensor shape [B, Fixed_N]
        ctx_collated = torch.nn.utils.rnn.pad_sequence(all_ctx_indices, batch_first=True, padding_value=-1)
        trg_collated = torch.nn.utils.rnn.pad_sequence(all_trg_indices, batch_first=True, padding_value=-1)

        return images, labels, ctx_collated, trg_collated
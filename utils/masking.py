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

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        B = images.shape[0]

        all_ctx_indices, all_trg_indices = [], []
        
        # Initialize trackers for the smallest mask size in the batch
        min_ctx = self.num_patches
        min_trg = self.num_patches 

        for _ in range(B):
            # 1. Generate Target Blocks
            targets = []
            combined_target_mask = torch.zeros(self.num_patches, dtype=torch.bool)
            for _ in range(self.num_targets):
                indices = self._get_block_indices(self.target_scale)
                targets.append(indices)
                combined_target_mask[indices] = True
            
            # 2. Generate Context Block
            ctx_indices = self._get_block_indices(self.context_scale)
            
            # 3. Remove overlaps (Target patches are never in Context)
            ctx_mask = torch.zeros(self.num_patches, dtype=torch.bool)
            ctx_mask[ctx_indices] = True
            ctx_mask[combined_target_mask] = False
            
            curr_ctx = torch.where(ctx_mask)[0]
            curr_trg = torch.cat(targets)
            
            all_ctx_indices.append(curr_ctx)
            all_trg_indices.append(curr_trg)
            
            # Update minimum lengths for this batch
            min_ctx = min(min_ctx, len(curr_ctx))
            min_trg = min(min_trg, len(curr_trg))

        # 4. Truncate all masks to the minimum length
        # This removes the need for pad_sequence and -1 padding
        ctx_collated = torch.stack([c[:min_ctx] for c in all_ctx_indices])
        trg_collated = torch.stack([t[:min_trg] for t in all_trg_indices])

        return images, labels, ctx_collated, trg_collated
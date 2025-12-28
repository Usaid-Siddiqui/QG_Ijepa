import torch.nn.functional as F
import torch

def masked_mse_loss(preds, targets, target_mask):
    """
    preds: [B, max_target_patches, D] (padded)
    targets: [B, max_target_patches, D] (padded)
    target_mask: [B, 256] (the original boolean mask)
    """
    # 1. Create a weight mask to ignore padding
    # We need to know which patches in the padded 'preds' are real data.
    # Since target_mask is what we used to extract patches, its sum tells us 
    # the count of real patches for each item in the batch.
    
    B, max_patches, D = preds.shape
    weights = torch.zeros((B, max_patches), device=preds.device)
    
    for i in range(B):
        num_real_patches = target_mask[i].sum()
        weights[i, :num_real_patches] = 1.0
        
    # 2. Expand weights to match dimensions [B, max_patches, D]
    weights = weights.unsqueeze(-1)
    
    # 3. Compute MSE and apply weights
    loss = F.mse_loss(preds, targets, reduction='none')
    loss = (loss * weights).sum() / weights.sum() # Average over real patches only
    
    return loss
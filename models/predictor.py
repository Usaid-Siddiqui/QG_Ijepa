import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.vit import Block, get_2d_sincos_pos_embed

class MaskPredictor(nn.Module):
    def __init__(self, encoder_dim=384, predictor_dim=192, depth=6, num_heads=6):
        super().__init__()
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim)
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, predictor_dim), requires_grad=False)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(predictor_dim, 16))
        
        self.blocks = nn.ModuleList([Block(predictor_dim, num_heads) for _ in range(depth)])
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim)
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context_latents, target_indices):
        """
        context_latents: [B, N_ctx, encoder_dim]
        target_indices: [B, N_targets]
        """
        B = context_latents.shape[0]

        # 1. Map context to predictor dimension
        context_latents = self.predictor_embed(context_latents)
        
        # 2. OPTIMIZED: Get target positional embeddings using gather
        # full_pos shape: [B, 256, predictor_dim]
        full_pos = self.pos_embed.expand(B, -1, -1)
        
        # target_indices expanded to [B, N_targets, predictor_dim]
        trg_idx_expanded = target_indices.unsqueeze(-1).expand(-1, -1, full_pos.size(-1))
        pos_target = torch.gather(full_pos, 1, trg_idx_expanded)
        
        # 3. Prepare mask tokens and add spatial info
        num_target_patches = pos_target.shape[1]
        mask_tokens = self.mask_token.expand(B, num_target_patches, -1)
        target_tokens = mask_tokens + pos_target
        
        # 4. Concatenate and process
        x = torch.cat([context_latents, target_tokens], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.predictor_norm(x)
        
        # 5. Extract only the target predictions
        preds = x[:, context_latents.shape[1]:]
        preds = self.predictor_proj(preds)
        return preds
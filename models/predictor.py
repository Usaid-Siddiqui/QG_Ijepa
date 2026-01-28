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

    def forward(self, context_latents, context_indices, target_indices):
        B = context_latents.shape[0]

        # 1. Map context to predictor dimension
        x_ctx = self.predictor_embed(context_latents)
        
        # 2. Add Positional Embeddings to CONTEXT
        full_pos = self.pos_embed.expand(B, -1, -1)
        
        ctx_idx_expanded = context_indices.unsqueeze(-1).expand(-1, -1, full_pos.size(-1))
        pos_ctx = torch.gather(full_pos, 1, ctx_idx_expanded)
        x_ctx = x_ctx + pos_ctx # Student now knows WHERE the context is

        # 3. Prepare target mask tokens with Positional Embeddings
        trg_idx_expanded = target_indices.unsqueeze(-1).expand(-1, -1, full_pos.size(-1))
        pos_target = torch.gather(full_pos, 1, trg_idx_expanded)
        
        mask_tokens = self.mask_token.expand(B, target_indices.shape[1], -1)
        x_trg = mask_tokens + pos_target
        
        # 4. Concatenate and process
        x = torch.cat([x_ctx, x_trg], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.predictor_norm(x)
        
        # 5. Extract target predictions and project back to encoder_dim
        preds = x[:, x_ctx.shape[1]:]
        preds = self.predictor_proj(preds)
        
        return preds
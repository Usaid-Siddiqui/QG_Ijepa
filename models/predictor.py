import torch
import torch.nn as nn
import numpy
from models.vit import Block, get_2d_sincos_pos_embed

class MaskPredictor(nn.Module):
    def __init__(self, encoder_dim=384, predictor_dim=192, depth=6, num_heads=6):
        super().__init__()
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim)
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        
        # Fixed spatial awareness for the predictor
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, predictor_dim), requires_grad=False)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(predictor_dim, 16))
        
        self.blocks = nn.ModuleList([Block(predictor_dim, num_heads) for _ in range(depth)])
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim)
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context_latents, target_mask):
        B = context_latents.shape[0]
        x_context = self.predictor_embed(context_latents)
        
        # Identify target positions in the 2D grid
        num_target = target_mask[0].sum().item()
        pos_target = self.pos_embed.expand(B, -1, -1)[target_mask].view(B, num_target, -1)
        
        # Initialize mask tokens with spatial location
        x_target = self.mask_token.expand(B, num_target, -1) + pos_target
        
        x = torch.cat([x_context, x_target], dim=1)
        for block in self.blocks:
            x = block(x)
        
        # Predict only the target tokens
        return self.predictor_proj(self.predictor_norm(x)[:, -num_target:])
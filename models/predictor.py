import torch
import torch.nn as nn
import numpy
from torch.nn.utils.rnn import pad_sequence
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
    
        # 1. Handle non-uniform target positional embeddings
        # self.pos_embed is [1, 256, D]. Expand it to batch size.
        full_pos = self.pos_embed.expand(B, -1, -1)
        
        # Extract target positions for each image separately to handle varying counts
        pos_target_list = [full_pos[i, target_mask[i]] for i in range(B)]
        
        # Pad them so they form a valid batch tensor
        pos_target = pad_sequence(pos_target_list, batch_first=True) # [B, max_target_patches, D]
        
        # 2. Prepare mask tokens
        # self.mask_token is [1, 1, D]
        num_target_patches = pos_target.shape[1]
        mask_tokens = self.mask_token.expand(B, num_target_patches, -1)
        
        # 3. Combine mask tokens with their spatial positions
        target_tokens = mask_tokens + pos_target
        
        # 4. Concatenate context and target tokens
        # context_latents is already padded from the Encoder
        x = torch.cat([context_latents, target_tokens], dim=1)
        
        # 5. Process through Predictor Transformer blocks
        for block in self.predictor_blocks:
            x = block(x)
        
        x = self.predictor_norm(x)
        
        # 6. Extract only the target predictions (the last 'num_target_patches' tokens)
        preds = x[:, context_latents.shape[1]:]
        
        return preds
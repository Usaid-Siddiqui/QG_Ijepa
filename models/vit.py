import torch
import torch.nn as nn
import math
import numpy as np

def trunc_normal_(tensor, std=0.02, a=-2.0, b=2.0):
    """ Fills the input Tensor with values drawn from a truncated normal distribution. """
    with torch.no_grad():
        l = (1. + math.erf(a / math.sqrt(2.))) / 2.
        u = (1. + math.erf(b / math.sqrt(2.))) / 2.
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        return tensor

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv shape: [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop)
        
        # Residual connection normalization
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Creates fixed 2D spatial coordinates for the 16x16 patch grid.
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # Generate embeddings for height and width separately
    def get_1d_sincos(dim, pos):
        omega = np.arange(dim // 2, dtype=float)
        omega /= dim / 2.
        omega = 1. / 10000**omega
        out = np.einsum('m,d->md', pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos(embed_dim // 2, grid[1])
    return torch.from_numpy(np.concatenate([emb_h, emb_w], axis=1)).float().unsqueeze(0)

class VisionTransformer(nn.Module):
    def __init__(self, patch_dim=192, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        # FIX 1: Spatially Aware Fixed Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim), requires_grad=False)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(embed_dim, 16))
        
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # FIX 2: Stable Weight Initialization
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _fix_init_weight(self):
        """ Rescales weights based on layer depth to prevent gradient explosion """
        for i, block in enumerate(self.blocks):
            # Scale attention projection and MLP second layer
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))
    # FIX 3: Mask-Aware Forward Pass
    def forward(self, x_patches, mask=None):
        x = self.patch_embed(x_patches)
        x = x + self.pos_embed

        if mask is not None:
            # mask is [B, 256] boolean. Extracts only context tokens.
            B, _, D = x.shape
            x = x[mask].view(B, -1, D)

        for block in self.blocks:
            x = block(x)
        return self.norm(x)
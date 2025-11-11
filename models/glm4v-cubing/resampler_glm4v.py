# coding=utf-8
# resampler_glm4v.py

"""
GLM4V 3D Resampler Module

Based on Quicksviewer's 3D Resampler, adapted for GLM4V architecture.
Compresses video cubes to fixed number of tokens while projecting to LLM dimension.
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 3D sinusoidal position embeddings
    
    Args:
        embed_dim: int - Embedding dimension
        grid_size: tuple - (T, H, W) grid size
    
    Returns:
        pos_embed: [T, H, W, embed_dim] - 3D position embeddings
    """
    grid_t, grid_h, grid_w = grid_size
    
    # Create coordinate grids
    grid_t = torch.arange(grid_t, dtype=torch.float32)
    grid_h = torch.arange(grid_h, dtype=torch.float32)
    grid_w = torch.arange(grid_w, dtype=torch.float32)
    
    # Create 3D meshgrid with correct order (T, H, W)
    grid = torch.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # [3, T, H, W]
    
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 3D position embeddings from grid
    
    Dimension allocation: temporal 2/8, height 3/8, width 3/8
    
    Args:
        embed_dim: Embedding dimension (must be divisible by 8)
        grid: [3, T, H, W] - Coordinate grid
    
    Returns:
        emb: [T, H, W, embed_dim] - Position embeddings
    """
    assert embed_dim % 8 == 0
    
    # Temporal dimension: 2/8
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 8 * 2, grid[0])
    # Height dimension: 3/8
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 8 * 3, grid[1])
    # Width dimension: 3/8
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 8 * 3, grid[2])
    
    emb = torch.cat([emb_t, emb_h, emb_w], dim=-1)  # [T, H, W, embed_dim]
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal position embeddings
    
    Args:
        embed_dim: Embedding dimension
        pos: [T, H, W] - Position indices
    
    Returns:
        emb: [T, H, W, embed_dim] - Position embeddings
    """
    assert embed_dim % 2 == 0
    
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # [embed_dim // 2]
    
    # Outer product: position × omega
    out = torch.einsum('thw,d->thwd', pos, omega)  # [T, H, W, embed_dim // 2]
    
    emb_sin = torch.sin(out)  # [T, H, W, embed_dim // 2]
    emb_cos = torch.cos(out)  # [T, H, W, embed_dim // 2]
    
    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # [T, H, W, embed_dim]
    return emb


class Glm4vResampler(nn.Module):
    """
    GLM4V 3D Resampler
    
    A 3D perceiver-resampler network that compresses video cubes to fixed number
    of tokens while projecting from vision dimension to LLM dimension.
    
    Key features:
        - Uses learnable queries for compression
        - Applies 3D sinusoidal position embeddings
        - Single cross-attention layer
        - Projects from vision_dim (1536) to lm_dim (4096)
    """
    
    def __init__(self, config):
        """
        Args:
            config: Glm4vConfig object containing:
                - vision_config.hidden_size: Vision feature dimension (1536)
                - text_config.hidden_size: LLM feature dimension (4096)
                - resampler_num_queries: Number of output tokens (64)
                - resampler_num_heads: Number of attention heads (32)
                - resampler_max_size: Max 3D size (300, 24, 24)
        """
        super().__init__()
        
        # Configuration parameters
        self.vision_dim = config.vision_config.hidden_size  # 1536
        self.lm_dim = config.text_config.hidden_size  # 4096
        self.num_queries = config.resampler_num_queries  # 64
        self.num_heads = config.resampler_num_heads  # 32
        self.max_size = config.resampler_max_size  # (300, 24, 24)
        
        # Learnable queries in LLM dimension
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.lm_dim))
        
        # KV projection: vision_dim → lm_dim
        self.kv_proj = nn.Linear(self.vision_dim, self.lm_dim, bias=False)
        
        # Cross-Attention
        # Note: batch_first=False means (seq, batch, dim) format
        self.attn = nn.MultiheadAttention(
            self.lm_dim,
            self.num_heads,
            batch_first=False
        )
        
        # Layer Normalizations
        self.ln_q = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(self.lm_dim, eps=1e-6)
        
        # Output projection
        self.proj = nn.Parameter(
            (self.lm_dim ** -0.5) * torch.randn(self.lm_dim, self.lm_dim)
        )
        
        # Initialize 3D position embeddings cache
        self._set_3d_pos_cache(self.max_size)
        
        # Initialize weights
        # self.apply(self._init_weights)
    
    def _set_3d_pos_cache(self, max_size, device='cpu'):
        """
        Initialize 3D position embeddings cache
        
        Args:
            max_size: tuple - (max_T, max_H, max_W)
            device: Device to create embeddings on
        """
        pos_embed = get_3d_sincos_pos_embed(self.lm_dim, max_size).to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)
    
    def _adjust_pos_cache(self, tgt_size_range, device):
        """
        Dynamically adjust position embeddings cache if needed
        
        Args:
            tgt_size_range: list - [[t_start, t_end], [h_start, h_end], [w_start, w_end]]
            device: Target device
        """
        max_t = tgt_size_range[0][1]
        max_h = tgt_size_range[1][1]
        max_w = tgt_size_range[2][1]
        
        if max_t > self.max_size[0] or max_h > self.max_size[1] or max_w > self.max_size[2]:
            self.max_size = (
                max(max_t, self.max_size[0]),
                max(max_h, self.max_size[1]),
                max(max_w, self.max_size[2])
            )
            self._set_3d_pos_cache(self.max_size, device)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        cube_features: torch.Tensor,
        tgt_size_range: list,
    ):
        """
        Compress cube features
        
        Args:
            cube_features: [N_i * 576, 1536] - Flattened cube patches
                - N_i: Number of frames in this cube
                - 576: Patches per frame (24×24)
                - 1536: Vision feature dimension
            
            tgt_size_range: list - 3D range specification
                - For images: [[0, h], [0, w]]
                - For cubes: [[t_start, t_end], [0, h], [0, w]]
        
        Returns:
            output: [num_queries, lm_dim] - Compressed tokens
                - If batch input, returns [B, num_queries, lm_dim]
        """
        # ========== Step 1: Normalize tgt_size_range ==========
        # Convert to 3D format
        tgt_size_range = [
            [0, _] if isinstance(_, int) else _
            for _ in tgt_size_range
        ]
        if len(tgt_size_range) == 2:
            # Image: add temporal dimension
            tgt_size_range = [[0, 1], tgt_size_range[0], tgt_size_range[1]]
        
        # ========== Step 2: Reshape input ==========
        if cube_features.dim() == 3:
            # [B, L, D]
            pass
        else:
            # [L, D] → [1, L, D]
            cube_features = cube_features.unsqueeze(0)
        
        B, L, D = cube_features.shape
        device = cube_features.device
        dtype = cube_features.dtype
        
        # ========== Step 3: Calculate sizes ==========
        tgt_sizes = torch.tensor(
            [[r[1] - r[0] for r in tgt_size_range]],
            device=device,
            dtype=torch.int32
        ).repeat(B, 1)  # [B, 3]
        
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1] * tgt_sizes[:, 2]
        max_patch_len = torch.max(patch_len)
        
        # ========== Step 4: KV projection ==========
        x = self.kv_proj(cube_features)  # [B, L, lm_dim]
        x = self.ln_kv(x).permute(1, 0, 2)  # [L, B, lm_dim]
        
        # ========== Step 5: Position embeddings ==========
        self._adjust_pos_cache(tgt_size_range, device)
        
        pos_embed_list = []
        key_padding_mask = torch.zeros(
            (B, max_patch_len), dtype=torch.bool, device=device
        )
        
        for i in range(B):
            t_range, h_range, w_range = tgt_size_range
            
            # Extract position embeddings for this sample
            pos = self.pos_embed[
                t_range[0]:t_range[1],  # Temporal
                h_range[0]:h_range[1],  # Height
                w_range[0]:w_range[1],  # Width
                :
            ]
            pos = pos.reshape(-1, self.lm_dim).to(dtype)
            pos_embed_list.append(pos)
            
            # Mark padding positions
            key_padding_mask[i, patch_len[i]:] = True
        
        # Pad position embeddings
        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed_list, batch_first=True, padding_value=0.0
        ).permute(1, 0, 2)  # [L, B, lm_dim]
        
        # ========== Step 6: Cross-Attention ==========
        q = self.ln_q(self.query)  # [num_queries, lm_dim]
        q = q.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, lm_dim]
        
        out = self.attn(
            q,  # [num_queries, B, lm_dim]
            x + pos_embed,  # [L, B, lm_dim] - KV with positional info
            x,  # [L, B, lm_dim] - Values without position
            key_padding_mask=key_padding_mask
        )[0]  # [num_queries, B, lm_dim]
        
        # ========== Step 7: Output projection ==========
        x = out.permute(1, 0, 2)  # [B, num_queries, lm_dim]
        x = self.ln_post(x)
        x = x @ self.proj  # [B, num_queries, lm_dim]
        
        # If single sample, squeeze batch dimension
        if B == 1:
            x = x.squeeze(0)  # [num_queries, lm_dim]
        
        return x
    
    @property
    def config(self):
        """Return configuration dict (for save/load)"""
        return {
            "vision_dim": self.vision_dim,
            "lm_dim": self.lm_dim,
            "num_queries": self.num_queries,
            "num_heads": self.num_heads,
            "max_size": self.max_size,
        }
    
__all__ = ["Glm4vResampler"]

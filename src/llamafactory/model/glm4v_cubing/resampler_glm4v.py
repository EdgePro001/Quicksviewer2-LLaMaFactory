# coding=utf-8
# resampler_glm4v.py

"""
GLM4V 3D Resampler Module with RoPE

Based on Quicksviewer's 3D Resampler, adapted with RoPE for better scalability.
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class RotaryEmbedding(nn.Module):
    """
    1D Rotary Position Embedding
    
    Args:
        dim: Embedding dimension (must be even)
        max_position_embeddings: Maximum supported sequence length
        base: Base for frequency calculation
    """
    def __init__(self, dim, max_position_embeddings=10000, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seq_len, device=None):
        """
        Generate RoPE frequencies
        
        Args:
            seq_len: Sequence length (or max position value)
            device: Target device
        
        Returns:
            freqs: [seq_len, dim] - Frequency tensor
        """
        if device is None:
            device = self.inv_freq.device
        
        # Generate position indices
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Compute outer product: position × inv_freq
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        
        # Duplicate for cos and sin
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        return freqs


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embedding to input tensor
    
    Args:
        x: [*, seq_len, dim] - Input tensor
        cos: [seq_len, dim] - Cosine frequencies
        sin: [seq_len, dim] - Sine frequencies
    
    Returns:
        x_rotated: [*, seq_len, dim]
    """
    # Split into even and odd dimensions
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    cos_half = cos[..., 0::2]
    sin_half = sin[..., 0::2]
    
    # Rotation
    x1_rot = x1 * cos_half - x2 * sin_half
    x2_rot = x1 * sin_half + x2 * cos_half
    
    # Interleave back
    x_rot = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
    
    return x_rot


def interpolate_rope_freqs(freqs, positions):
    """
    Interpolate RoPE frequencies for floating point positions
    
    Args:
        freqs: [max_seq_len, dim] - Precomputed frequencies
        positions: [N] - Floating point position indices
    
    Returns:
        interpolated_freqs: [N, dim]
    """
    device = freqs.device
    
    # Floor and ceiling indices
    pos_floor = positions.floor().long().clamp(0, freqs.shape[0] - 1)
    pos_ceil = positions.ceil().long().clamp(0, freqs.shape[0] - 1)
    
    # Interpolation weight
    weight = (positions - pos_floor.float()).unsqueeze(-1)  # [N, 1]
    
    # Linear interpolation
    freqs_floor = freqs[pos_floor]  # [N, dim]
    freqs_ceil = freqs[pos_ceil]
    
    freqs_interp = freqs_floor * (1 - weight) + freqs_ceil * weight
    
    return freqs_interp


class Glm4vResampler(nn.Module):
    """
    GLM4V 3D Resampler with RoPE
    
    Compresses video cubes to fixed number of tokens using:
        - Learnable queries
        - 3D RoPE for position encoding
        - Cross-attention
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration parameters
        self.vision_dim = config.vision_config.hidden_size
        self.lm_dim = config.text_config.hidden_size
        self.num_queries = config.resampler_num_queries
        self.num_heads = config.resampler_num_heads
        
        # === Dimension allocation (2:3:3 for T:H:W) ===
        # You can change this to (1,1,1) if needed
        self.dim_ratio = (2, 3, 3)
        total_ratio = sum(self.dim_ratio)
        base_dim = self.lm_dim // total_ratio
        
        self.dim_t = base_dim * self.dim_ratio[0]
        self.dim_h = base_dim * self.dim_ratio[1]
        self.dim_w = base_dim * self.dim_ratio[2]
        
        # Adjust for exact division
        leftover = self.lm_dim - (self.dim_t + self.dim_h + self.dim_w)
        self.dim_w += leftover
        
        # === 3D RoPE modules ===
        self.rope_t = RotaryEmbedding(self.dim_t)
        self.rope_h = RotaryEmbedding(self.dim_h)
        self.rope_w = RotaryEmbedding(self.dim_w)
        
        # === Core components ===
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.lm_dim))
        self.kv_proj = nn.Linear(self.vision_dim, self.lm_dim, bias=False)
        self.attn = nn.MultiheadAttention(self.lm_dim, self.num_heads, batch_first=False)
        self.ln_q = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.proj = nn.Parameter((self.lm_dim ** -0.5) * torch.randn(self.lm_dim, self.lm_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize nn.Module subclasses"""
        if hasattr(m, 'weight') and m.weight.device.type == 'meta':
            return
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.MultiheadAttention):
            if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                if m.in_proj_weight.device.type != 'meta':
                    nn.init.xavier_uniform_(m.in_proj_weight)
            if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
                if m.in_proj_bias.device.type != 'meta':
                    nn.init.constant_(m.in_proj_bias, 0)
    
    def _post_init_parameters(self):
        """Post-initialization for nn.Parameter"""
        if self.query.device.type == 'meta':
            return
        
        if torch.isnan(self.query).any() or self.query.std() < 1e-6:
            trunc_normal_(self.query, std=0.02)
            print(f"[POST-INIT] Initialized query")
        
        if torch.isnan(self.proj).any() or self.proj.std() < 1e-6:
            trunc_normal_(self.proj, std=0.02)
            print(f"[POST-INIT] Initialized proj")
    
    def forward(
        self,
        cube_features: torch.Tensor,
        tgt_size_range: list,
    ):
        """
        Compress cube features with 3D RoPE
        
        Args:
            cube_features: [N_patches, vision_dim] or [B, N_patches, vision_dim]
            tgt_size_range: [[t_start, t_end], [h_start, h_end], [w_start, w_end]]
                or [[h_start, h_end], [w_start, w_end]] for images
        
        Returns:
            output: [num_queries, lm_dim] or [B, num_queries, lm_dim]
        """
        device = cube_features.device
        dtype = cube_features.dtype
        
        # ========== Step 1: Normalize input ==========
        print("[DEBUG RESAMPLER] step 1: normalize input")
        
        # Normalize tgt_size_range to 3D format
        tgt_size_range = [
            [0, _] if isinstance(_, int) else _
            for _ in tgt_size_range
        ]
        
        if len(tgt_size_range) == 2:
            # Image: add temporal dimension
            tgt_size_range = [[0, 1], tgt_size_range[0], tgt_size_range[1]]
        
        # Ensure batch dimension
        if cube_features.dim() == 2:
            cube_features = cube_features.unsqueeze(0)  # [1, L, D]
        
        B, L, D = cube_features.shape
        print(f"[DEBUG RESAMPLER] cube_features: {cube_features.shape}")
        
        # ========== Step 2: Parse ranges ==========
        print("[DEBUG RESAMPLER] step 2: parse ranges")
        
        t_range, h_range, w_range = tgt_size_range
        t_start, t_end = t_range
        h_start, h_end = h_range
        w_start, w_end = w_range
        
        num_t = t_end - t_start
        num_h = h_end - h_start
        num_w = w_end - w_start
        
        print(f"[DEBUG RESAMPLER] T: {t_start}~{t_end}, H: {h_start}~{h_end}, W: {w_start}~{w_end}")
        
        # ========== Step 3: Generate 3D positions ==========
        print("[DEBUG RESAMPLER] step 3: generate 3D positions")
        
        # Position indices (integer for now, will support float later)
        t_positions = torch.arange(t_start, t_end, device=device, dtype=torch.float32)
        h_positions = torch.arange(h_start, h_end, device=device, dtype=torch.float32)
        w_positions = torch.arange(w_start, w_end, device=device, dtype=torch.float32)
        
        # Create 3D meshgrid
        grid_t, grid_h, grid_w = torch.meshgrid(
            t_positions, h_positions, w_positions, indexing='ij'
        )
        # Shape: [num_t, num_h, num_w]
        
        # Flatten
        flat_t = grid_t.flatten()  # [num_t * num_h * num_w]
        flat_h = grid_h.flatten()
        flat_w = grid_w.flatten()
        
        # ========== Step 4: Compute RoPE frequencies ==========
        print("[DEBUG RESAMPLER] step 4: compute RoPE")
        
        # Get max values for precomputing frequencies
        max_t = int(flat_t.max().item()) + 1
        max_h = int(flat_h.max().item()) + 1
        max_w = int(flat_w.max().item()) + 1
        
        # Compute frequencies
        freqs_t = self.rope_t(max_t, device)  # [max_t, dim_t]
        freqs_h = self.rope_h(max_h, device)  # [max_h, dim_h]
        freqs_w = self.rope_w(max_w, device)  # [max_w, dim_w]
        
        # Index frequencies (support floating point via interpolation)
        if flat_t.dtype == torch.float32:
            freq_t = interpolate_rope_freqs(freqs_t, flat_t)
            freq_h = interpolate_rope_freqs(freqs_h, flat_h)
            freq_w = interpolate_rope_freqs(freqs_w, flat_w)
        else:
            freq_t = freqs_t[flat_t.long()]
            freq_h = freqs_h[flat_h.long()]
            freq_w = freqs_w[flat_w.long()]
        
        # Concatenate: [N_patches, lm_dim]
        freqs = torch.cat([freq_t, freq_h, freq_w], dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        
        # ========== Step 5: KV projection ==========
        print("[DEBUG RESAMPLER] step 5: KV projection")
        
        kv = self.kv_proj(cube_features)  # [B, L, lm_dim]
        kv = self.ln_kv(kv)
        
        # Apply RoPE to KV
        # kv: [B, L, lm_dim] → permute to [L, B, lm_dim] for attention
        kv = kv.permute(1, 0, 2)  # [L, B, lm_dim]
        
        # Apply RoPE: broadcast cos/sin to batch dimension
        cos_expanded = cos.unsqueeze(1)  # [L, 1, lm_dim]
        sin_expanded = sin.unsqueeze(1)
        
        kv_rot = apply_rotary_pos_emb(kv, cos_expanded, sin_expanded)  # [L, B, lm_dim]
        
        # ========== Step 6: Query preparation ==========
        print("[DEBUG RESAMPLER] step 6: query preparation")
        
        q = self.ln_q(self.query)  # [num_queries, lm_dim]
        q = q.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, lm_dim]
        
        # Query positions: set to zero (queries have no position preference)
        # Alternatively, you can assign learnable positions
        q_cos = torch.ones_like(q)
        q_sin = torch.zeros_like(q)
        q_rot = apply_rotary_pos_emb(q, q_cos, q_sin)
        
        # ========== Step 7: Cross-Attention ==========
        print("[DEBUG RESAMPLER] step 7: cross-attention")
        
        # Padding mask (optional, for variable-length inputs)
        # For now, assume all patches are valid
        key_padding_mask = None
        
        out, _ = self.attn(
            q_rot,      # [num_queries, B, lm_dim]
            kv_rot,     # [L, B, lm_dim] - Keys with RoPE
            kv,         # [L, B, lm_dim] - Values without RoPE
            key_padding_mask=key_padding_mask
        )  # [num_queries, B, lm_dim]
        
        # ========== Step 8: Output projection ==========
        print("[DEBUG RESAMPLER] step 8: output projection")
        
        out = out.permute(1, 0, 2)  # [B, num_queries, lm_dim]
        out = self.ln_post(out)
        out = out @ self.proj  # [B, num_queries, lm_dim]
        
        # Squeeze batch dimension if B=1
        if B == 1:
            out = out.squeeze(0)  # [num_queries, lm_dim]
        
        print(f"[DEBUG RESAMPLER] output shape: {out.shape}")
        
        return out
    
    @property
    def config(self):
        """Return configuration dict"""
        return {
            "vision_dim": self.vision_dim,
            "lm_dim": self.lm_dim,
            "num_queries": self.num_queries,
            "num_heads": self.num_heads,
            "dim_ratio": self.dim_ratio,
        }


__all__ = ["Glm4vResampler"]
# coding=utf-8
# resampler_glm4v.py

"""
GLM4V 3D Resampler Module with RoPE and FPS scaling
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from typing import Optional


class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding"""
    def __init__(self, dim, max_position_embeddings=10000, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute inverse frequencies (keep as float32 for stability)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seq_len, device=None, dtype=None):
        """
        Generate RoPE frequencies
        
        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Target dtype
        
        Returns:
            freqs: [seq_len, dim]
        """
        if device is None:
            device = self.inv_freq.device
        if dtype is None:
            dtype = torch.float32
        
        # Generate position indices
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute outer product
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, dim//2]
        
        # Duplicate
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        # Convert to target dtype
        return freqs.to(dtype)


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embedding
    
    Args:
        x: [*, seq_len, dim]
        cos, sin: [seq_len, dim]
    
    Returns:
        x_rotated: [*, seq_len, dim]
    """
    # Ensure cos/sin match input dtype
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    
    # Split into even and odd
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    cos_half = cos[..., 0::2]
    sin_half = sin[..., 0::2]
    
    # Rotation
    x1_rot = x1 * cos_half - x2 * sin_half
    x2_rot = x1 * sin_half + x2 * cos_half
    
    # Interleave
    x_rot = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
    
    return x_rot


def interpolate_rope_freqs(freqs, positions):
    """
    Interpolate RoPE frequencies for floating point positions
    
    Args:
        freqs: [max_seq_len, dim]
        positions: [N] - Floating point positions
    
    Returns:
        interpolated_freqs: [N, dim]
    """
    device = freqs.device
    dtype = freqs.dtype
    
    # Ensure positions is float32 for computation
    positions = positions.to(torch.float32)
    
    # Floor and ceiling
    pos_floor = positions.floor().long().clamp(0, freqs.shape[0] - 1)
    pos_ceil = positions.ceil().long().clamp(0, freqs.shape[0] - 1)
    
    # Weight
    weight = (positions - pos_floor.float()).unsqueeze(-1)  # [N, 1]
    
    # Interpolation
    freqs_floor = freqs[pos_floor]  # [N, dim]
    freqs_ceil = freqs[pos_ceil]
    
    freqs_interp = freqs_floor * (1 - weight) + freqs_ceil * weight
    
    # Return in original dtype
    return freqs_interp.to(dtype)


class Glm4vResampler(nn.Module):
    """GLM4V 3D Resampler with RoPE and FPS scaling"""
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.vision_dim = config.vision_config.hidden_size
        self.lm_dim = config.text_config.hidden_size
        self.num_queries = config.resampler_num_queries
        self.num_heads = config.resampler_num_heads
        
        # === FPS 配置（修正）===
        self.reference_fps = getattr(config, 'reference_fps', 10.0)
        
        # ✅ 读取 effective_video_fps（而不是 default_video_fps）
        self.effective_video_fps = getattr(config, 'effective_video_fps', 1.0)
        
        print(f"[Resampler Init] Reference FPS: {self.reference_fps}")
        print(f"[Resampler Init] Default Effective Video FPS: {self.effective_video_fps}")
        
        # 维度分配 (2:3:3)
        self.dim_ratio = (2, 3, 3)
        total_ratio = sum(self.dim_ratio)
        base_dim = self.lm_dim // total_ratio
        
        self.dim_t = base_dim * self.dim_ratio[0]
        self.dim_h = base_dim * self.dim_ratio[1]
        self.dim_w = base_dim * self.dim_ratio[2]
        
        leftover = self.lm_dim - (self.dim_t + self.dim_h + self.dim_w)
        self.dim_w += leftover
        
        print(f"[Resampler Init] Dimension allocation - T:{self.dim_t}, H:{self.dim_h}, W:{self.dim_w}")
        
        # 3D RoPE
        self.rope_t = RotaryEmbedding(self.dim_t)
        self.rope_h = RotaryEmbedding(self.dim_h)
        self.rope_w = RotaryEmbedding(self.dim_w)
        
        # Core components
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.lm_dim))
        self.kv_proj = nn.Linear(self.vision_dim, self.lm_dim, bias=False)
        self.attn = nn.MultiheadAttention(self.lm_dim, self.num_heads, batch_first=False)
        self.ln_q = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(self.lm_dim, eps=1e-6)
        self.proj = nn.Parameter((self.lm_dim ** -0.5) * torch.randn(self.lm_dim, self.lm_dim))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights"""
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
        """Post-init for nn.Parameter"""
        if self.query.device.type == 'meta':
            return
        
        if torch.isnan(self.query).any() or self.query.std() < 1e-6:
            trunc_normal_(self.query, std=0.02)
        
        if torch.isnan(self.proj).any() or self.proj.std() < 1e-6:
            trunc_normal_(self.proj, std=0.02)
    
    def forward(
        self,
        cube_features: torch.Tensor,
        tgt_size_range: list,
        fps: Optional[float] = None,  # ← 新增参数
    ):
        """
        Compress cube features with 3D RoPE and FPS scaling
        
        Args:
            cube_features: [N_patches, vision_dim] or [B, N_patches, vision_dim]
            tgt_size_range: [[t_start, t_end], [h_start, h_end], [w_start, w_end]]
                注意：t_start/t_end 是 temporal_patch_size 合并后的帧索引
            fps: Optional[float] - 当前视频的有效 FPS（已考虑 temporal_patch_size）
                 例如：原始 30 FPS，temporal_patch_size=2 → fps=15.0
                 如果为 None，使用 effective_video_fps
        
        Returns:
            output: [num_queries, lm_dim] or [B, num_queries, lm_dim]
        """
        device = cube_features.device
        dtype = cube_features.dtype
        
        print("[DEBUG RESAMPLER] step 1: normalize input")
        
        # Normalize tgt_size_range
        tgt_size_range = [
            [0, _] if isinstance(_, int) else _
            for _ in tgt_size_range
        ]
        
        if len(tgt_size_range) == 2:
            tgt_size_range = [[0, 1], tgt_size_range[0], tgt_size_range[1]]
        
        # Ensure batch dimension
        if cube_features.dim() == 2:
            cube_features = cube_features.unsqueeze(0)
        
        B, L, D = cube_features.shape
        print(f"[DEBUG RESAMPLER] cube_features: {cube_features.shape}, dtype: {dtype}")
        
        print("[DEBUG RESAMPLER] step 2: parse ranges")
        
        t_range, h_range, w_range = tgt_size_range
        t_start, t_end = t_range
        h_start, h_end = h_range
        w_start, w_end = w_range
        
        num_t = t_end - t_start
        num_h = h_end - h_start
        num_w = w_end - w_start
        
        print(f"[DEBUG RESAMPLER] T: {t_start}~{t_end}, H: {h_start}~{h_end}, W: {w_start}~{w_end}")
        
        # === Step 3: 生成 3D 位置（加 FPS 缩放）===
        print("[DEBUG RESAMPLER] step 3: generate 3D positions with FPS scaling")
        
        t_positions = torch.arange(t_start, t_end, device=device, dtype=torch.float32)
        
        # === FPS 缩放（关键）===
        
        if fps is None:
            fps = self.effective_video_fps  # = config.effective_video_fps
            print(f"[DEBUG RESAMPLER] Using default effective FPS: {fps}")
        
        if fps != self.reference_fps:
            fps_scale = self.reference_fps / fps
            t_positions_scaled = t_positions * fps_scale
            print(f"[DEBUG RESAMPLER] FPS scaling: {fps:.2f} → {self.reference_fps:.2f}, scale={fps_scale:.3f}")
        else:
            t_positions_scaled = t_positions
            print(f"[DEBUG RESAMPLER] No FPS scaling needed")
        
        # H/W 维度位置（不缩放）
        h_positions = torch.arange(h_start, h_end, device=device, dtype=torch.float32)
        w_positions = torch.arange(w_start, w_end, device=device, dtype=torch.float32)
        
        # 3D meshgrid
        grid_t, grid_h, grid_w = torch.meshgrid(
            t_positions_scaled, h_positions, w_positions, indexing='ij'
        )
        
        # Flatten
        flat_t = grid_t.flatten()
        flat_h = grid_h.flatten()
        flat_w = grid_w.flatten()
        
        print("[DEBUG RESAMPLER] step 4: compute RoPE")
        
        # Max values
        max_t = int(flat_t.max().item()) + 1
        max_h = int(flat_h.max().item()) + 1
        max_w = int(flat_w.max().item()) + 1
        
        # Generate frequencies with target dtype
        freqs_t = self.rope_t(max_t, device, dtype)
        freqs_h = self.rope_h(max_h, device, dtype)
        freqs_w = self.rope_w(max_w, device, dtype)
        
        # Index/Interpolate (支持浮点位置)
        if flat_t.dtype == torch.float32 and (flat_t % 1.0).any():
            # Floating point positions (FPS 缩放后可能产生)
            freq_t = interpolate_rope_freqs(freqs_t, flat_t)
            freq_h = interpolate_rope_freqs(freqs_h, flat_h)
            freq_w = interpolate_rope_freqs(freqs_w, flat_w)
            print("[DEBUG RESAMPLER] Using interpolated RoPE (floating point positions)")
        else:
            # Integer positions
            freq_t = freqs_t[flat_t.long()]
            freq_h = freqs_h[flat_h.long()]
            freq_w = freqs_w[flat_w.long()]
            print("[DEBUG RESAMPLER] Using indexed RoPE (integer positions)")
        
        # Concatenate
        freqs = torch.cat([freq_t, freq_h, freq_w], dim=-1)  # [N_patches, lm_dim]
        cos = freqs.cos()
        sin = freqs.sin()
        
        print(f"[DEBUG RESAMPLER] RoPE freqs dtype: {freqs.dtype}")
        
        print("[DEBUG RESAMPLER] step 5: KV projection")
        
        kv = self.kv_proj(cube_features)  # [B, L, lm_dim]
        kv = self.ln_kv(kv)
        kv = kv.permute(1, 0, 2)  # [L, B, lm_dim]
        
        print(f"[DEBUG RESAMPLER] KV dtype: {kv.dtype}")
        
        # Apply RoPE to KV
        cos_expanded = cos.unsqueeze(1)  # [L, 1, lm_dim]
        sin_expanded = sin.unsqueeze(1)
        
        kv_rot = apply_rotary_pos_emb(kv, cos_expanded, sin_expanded)
        
        print(f"[DEBUG RESAMPLER] KV_rot dtype: {kv_rot.dtype}")
        
        print("[DEBUG RESAMPLER] step 6: query preparation")
        
        q = self.ln_q(self.query)  # [num_queries, lm_dim]
        q = q.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, lm_dim]
        
        print(f"[DEBUG RESAMPLER] Q dtype: {q.dtype}")
        
        # Query RoPE (position = 0)
        q_cos = torch.ones_like(q)
        q_sin = torch.zeros_like(q)
        q_rot = apply_rotary_pos_emb(q, q_cos, q_sin)
        
        print(f"[DEBUG RESAMPLER] Q_rot dtype: {q_rot.dtype}")
        
        print("[DEBUG RESAMPLER] step 7: cross-attention")
        
        # Ensure all inputs have same dtype
        q_rot = q_rot.to(dtype)
        kv_rot = kv_rot.to(dtype)
        kv = kv.to(dtype)
        
        out, _ = self.attn(
            q_rot,
            kv_rot,
            kv,
            key_padding_mask=None
        )
        
        print("[DEBUG RESAMPLER] step 8: output projection")
        
        out = out.permute(1, 0, 2)  # [B, num_queries, lm_dim]
        out = self.ln_post(out)
        out = out @ self.proj
        
        if B == 1:
            out = out.squeeze(0)
        
        print(f"[DEBUG RESAMPLER] output shape: {out.shape}, dtype: {out.dtype}")
        
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
            "reference_fps": self.reference_fps,
            "effective_video_fps": self.effective_video_fps,
        }


__all__ = ["Glm4vResampler"]
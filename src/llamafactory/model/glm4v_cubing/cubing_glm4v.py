# cubing_glm4v.py
"""
GLM4V Cubing Module

Based on Quicksviewer's Cubing technique, adapted for GLM4V architecture.
Implements differentiable video adaptive segmentation via Gumbel-Softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20, dtype=torch.bfloat16):
    """
    Sample Gumbel noise
    
    Args:
        shape: Tensor shape
        eps: Numerical stability constant
        dtype: Data type
    
    Returns:
        Gumbel noise tensor
    """
    U = torch.rand(shape, dtype=dtype, device='cuda')
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, lr_gumbel):
    """
    Gumbel-Softmax sampling
    
    Args:
        logits: [*, n_class] - Unnormalized logits
        temperature: float - Temperature parameter controlling distribution smoothness
        lr_gumbel: float - Weight of Gumbel noise (for annealing)
    
    Returns:
        Sampled softmax probability distribution
    """
    y = logits + sample_gumbel(logits.size(), dtype=logits.dtype) * lr_gumbel
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, topk=1, lr_gumbel=0.1):
    """
    Straight-Through Gumbel-Softmax
    
    Uses hard mask in forward pass and soft gradients in backward pass,
    enabling differentiable discrete sampling.
    
    Args:
        logits: [B, N, 2] - Gate logits, 2 for [non-keyframe, keyframe]
        temperature: float - Gumbel temperature
        topk: int - Select top-k keyframes
        lr_gumbel: float - Gumbel noise weight
    
    Returns:
        y: [B, N] - Soft probabilities (for gradients)
        y_hard: [B, N] - Hard mask (for forward)
    """
    shape = logits.shape
    
    # Gumbel-Softmax sampling
    y = gumbel_softmax_sample(logits, temperature, lr_gumbel)  # [B, N, 2]
    
    # Take second dimension as keyframe probability
    # Note: Consistent with Quicksviewer, dimension 1 represents keyframe
    y = y[:, :, 1]  # [B, N]
    
    # Top-K selection
    _, ind = y.topk(k=topk, dim=-1)  # [B, topk]
    
    # Create hard mask
    y_hard = torch.zeros_like(y)  # [B, N]
    y_hard.scatter_(1, ind, 1)  # Only top-k positions are 1
    
    # Straight-Through: forward uses hard, backward uses soft
    y_hard = (y_hard - y).detach() + y
    
    return y, y_hard


def find_segments(mask):
    """
    Identify continuous segments (cubes) from binary mask
    
    Args:
        mask: [N] - Binary mask, 1 indicates keyframe position
    
    Returns:
        segments: List[(start, end)] - Segment boundaries, half-open [start, end)
    
    Example:
        mask = [1, 0, 0, 1, 0, 1, 0, 0]
        → segments = [(0, 3), (3, 5), (5, 8)]
    """
    segments = []
    pre, cur = 0, 0
    
    while cur < len(mask):
        # End current segment when encountering keyframe or reaching end
        if cur == len(mask) - 1 or (mask[cur + 1] != 0):
            segments.append((pre, cur + 1))
            pre = cur + 1
        cur += 1
    
    return segments


class Glm4vCubingModule(nn.Module):
    """
    GLM4V Cubing Module
    
    Identifies keyframes in video through momentum analysis and segments
    video into multiple cubes. Each cube represents a semantically coherent
    video segment.
    
    Core idea:
        1. Calculate inter-frame momentum: Δ_i = α(F_i - F_{i-1}) + (1-α)Δ_{i-1}
        2. Gate network predicts keyframes
        3. Gumbel-Softmax sampling (differentiable)
        4. Identify cube boundaries
    """
    
    def __init__(self, config):
        """
        Args:
            config: Glm4vConfig object containing:
                - vision_config.hidden_size: Vision feature dimension (1536)
                - text_config.hidden_size: LLM feature dimension (4096)
                - cubing_alpha: Momentum discount factor
                - cubing_use_thumbnail: Whether to generate thumbnail
        """
        super().__init__()
        
        # Configuration parameters
        self.vision_dim = config.vision_config.hidden_size  # 1536
        self.lm_dim = config.text_config.hidden_size  # 4096
        self.alpha = config.cubing_alpha  # 0.8
        self.use_thumbnail = config.cubing_use_thumbnail
        
        # === Frame-level aggregation network ===
        # Purpose: Aggregate 576 patches per frame into 1 vector
        self.agg_frame_fn = nn.Sequential(
            nn.Linear(self.vision_dim, self.vision_dim),
        )
        
        # === Gate network ===
        # Purpose: Predict whether each frame is a keyframe based on momentum features
        # Output: [..., 2], where [:, 0] is non-keyframe prob, [:, 1] is keyframe prob
        self.gate_network = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.vision_dim),
            nn.GELU(),
            nn.Linear(self.vision_dim, 2),
        )
        
        # === Thumbnail projection (optional) ===
        # Purpose: Compress and project keyframe features to LLM dimension as global video representation
        # QuickSViewer uses AvgPool2d(kernel_size=(9,1)) for 576→64
        # We use AdaptiveAvgPool1d for flexibility with different resolutions
        if self.use_thumbnail:
            # Store num_queries for thumbnail (typically 64)
            self.thumbnail_num_queries = getattr(config, 'thumbnail_num_queries', 64)
            
            self.thumbnail_fn = nn.Sequential(
                # Adaptive pooling: works for any number of input patches
                nn.AdaptiveAvgPool1d(self.thumbnail_num_queries),
                # Project to LLM dimension
                nn.Linear(self.vision_dim, self.lm_dim)
            )
    
    def forward(
        self,
        video_features: torch.Tensor,
        fpq: int = 14,
        temperature: float = 0.5,
        lr_gumbel: float = 0.1,
    ):
        """
        Perform Cubing on a single video
        
        Args:
            video_features: [N_temporal_tokens, 576, 1536] - Video frame features
                - N_temporal_tokens: Number of video frames
                - 576: Number of patches per frame (24*24)
                - 1536: Vision feature dimension
            fpq: int - Frames Per Query, target average frames per cube
            temperature: float - Gumbel-Softmax temperature parameter
            lr_gumbel: float - Gumbel noise weight (annealed from 1.0 to 0.001 during training)
        
        Returns:
            dict containing:
                - cube_bounds: List[(start, end)] - List of cube boundaries
                - gate_logits: [N_temporal_tokens-1, 2] - Gate network output (for auxiliary loss)
                - thumbnail: [thumbnail_num_queries, lm_dim] or None - Global video representation
                             (typically 64 tokens, but configurable)
                - z_hard: [N_temporal_tokens] - Keyframe mask (1 indicates keyframe)
        """
        N_temporal_tokens = video_features.shape[0]
        device = video_features.device
        temporal_patch_size = getattr(
            self.config, 
            'temporal_patch_size', 
            1 
        )
        print(f"[DEBUG CUBE] video_features.shape: {video_features.shape}")
        
        effective_fpq = fpq / temporal_patch_size    
        
        # ========== Step 1: Calculate momentum ==========
        print("[DEBUG CUBE] step 1: Calculate momentum")
        # Momentum : Δ_i = α(F_i - F_{i-1}) + (1-α)Δ_{i-1}
        # [N, 576, 1536] → [N-1, 576, 1536]
        vid_feats_momentum = [video_features[1] - video_features[0]]

        for i in range(2, N_temporal_tokens):
            delta = self.alpha * (video_features[i] - video_features[i-1]) + \
                    (1 - self.alpha) * vid_feats_momentum[-1]
            vid_feats_momentum.append(delta)

        vid_feats = torch.stack(vid_feats_momentum, dim=0)  # [N-1, 576, 1536]
        print(f"[DEBUG CUBE step1] vid_feats shape: {vid_feats.shape}")

        # ========== Step 2: Aggregate momentum features ==========
        # [N-1, 576, 1536] → [N-1, 576, 1536] → [N-1, 1536]
        print("[DEBUG CUBE] step 2: Aggregate momentum features")
        vid_feats = self.agg_frame_fn(vid_feats)
        print(f"[DEBUG CUBE step2] vid_feats after agg_frame_fn: {vid_feats.shape}")
        vid_feats = vid_feats.mean(dim=1)  # [N-1, 1536]

        
        # ========== Step 3: Gate network prediction ==========
        # Predict whether each position should be a keyframe based on momentum features
        print("[DEBUG CUBE] step 3: Gate network prediction")
        gate_logits = self.gate_network(vid_feats)  # [N-1, 2]
        
        # ========== Step 4: Gumbel-Softmax sampling ==========
        # Calculate number of keyframes to select
        print("[DEBUG CUBE] step 4: Gumbel-Softmax sampling")
        num_cubes = max(round(N_temporal_tokens / effective_fpq) - 1, 1)
        # Why -1? Because first frame will be forced as keyframe
        
        # Gumbel-Softmax sampling
        z, z_hard = gumbel_softmax(
            gate_logits.unsqueeze(0),  # [1, N-1, 2]
            temperature=temperature,
            topk=num_cubes,
            lr_gumbel=lr_gumbel
        )
        z = z.squeeze(0)  # [N-1]
        z_hard = z_hard.squeeze(0)  # [N-1]
        
        # ========== Step 5: Force first frame as keyframe ==========
        # Ensure video beginning is always a cube start point
        print("[DEBUG CUBE] step 5: Force first frame as keyframe")
        pad_z_hard = torch.ones(1, dtype=z_hard.dtype, device=device)
        z_hard = torch.cat([pad_z_hard, z_hard], dim=0)  # [N]
        
        # ========== Step 6: Identify cube boundaries ==========
        print("[DEBUG CUBE] step 6: Identify cube boundaries")
        cube_bounds = find_segments(z_hard)
        
        # ========== Step 7: Generate Thumbnail (QuickSViewer-style) ==========
        print("[DEBUG CUBE] step 7: Generate Thumbnail")
        thumbnail = None
        if self.use_thumbnail:
            # Following QuickSViewer's approach:
            # 1. Weighted average of keyframe features from original video_features
            # 2. Compress N patches to thumbnail_num_queries tokens (typically 64)
            # 3. Project to LLM dimension
            
            # Get number of patches from input
            num_patches = video_features.shape[1]
            print(f"[DEBUG CUBE step7] Input has {num_patches} patches per frame")
            
            # Use z_hard to weight the original video features
            # Note: z_hard has shape [N], video_features has shape [N, num_patches, 1536]
            weights = z_hard.float().unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
            weighted_feats = video_features * weights  # [N, num_patches, 1536]
            
            # Sum over frames and normalize
            thumbnail_feat = weighted_feats.sum(dim=0, keepdim=True)  # [1, num_patches, 1536]
            thumbnail_feat = thumbnail_feat / z_hard.sum()  # Normalize by number of keyframes
            
            print(f"[DEBUG CUBE step7] thumbnail_feat shape before pooling: {thumbnail_feat.shape}")
            
            # Transpose for AdaptiveAvgPool1d: [1, num_patches, 1536] → [1, 1536, num_patches]
            thumbnail_feat = thumbnail_feat.permute(0, 2, 1)  # [1, 1536, num_patches]
            
            # Compress num_patches to thumbnail_num_queries tokens using adaptive pooling
            # This works for any num_patches (576, 1024, 256, etc.)
            # Input: [1, 1536, num_patches] → Output: [1, 1536, thumbnail_num_queries]
            thumbnail_feat = F.adaptive_avg_pool1d(
                thumbnail_feat, 
                self.thumbnail_num_queries
            )  # [1, 1536, thumbnail_num_queries]
            
            # Transpose back: [1, 1536, thumbnail_num_queries] → [1, thumbnail_num_queries, 1536]
            thumbnail_feat = thumbnail_feat.permute(0, 2, 1)  # [1, thumbnail_num_queries, 1536]
            
            print(f"[DEBUG CUBE step7] thumbnail_feat shape after pooling: {thumbnail_feat.shape}")
            
            # Remove batch dimension and apply projection
            thumbnail_feat = thumbnail_feat.squeeze(0)  # [thumbnail_num_queries, 1536]
            
            # Project to LLM dimension
            thumbnail = self.thumbnail_fn[1](thumbnail_feat)  # [thumbnail_num_queries, 4096]
            # Note: thumbnail_fn[0] is AdaptiveAvgPool1d which we already applied manually
            #       thumbnail_fn[1] is the Linear layer
            
            print(f"[DEBUG CUBE step7] Final thumbnail shape: {thumbnail.shape}")
            print(f"[DEBUG CUBE step7] Compression ratio: {num_patches} patches → {self.thumbnail_num_queries} tokens")

        print(f"[DEBUG CUBE] Summary:")
        print(f"  - cube_bounds: {cube_bounds}")
        print(f"  - num_cubes: {len(cube_bounds)}")
        print(f"  - thumbnail shape: {thumbnail.shape if thumbnail is not None else None}")
        if thumbnail is not None:
            print(f"  - thumbnail tokens: {thumbnail.shape[0]}")
        print(f"  - z_hard sum (num keyframes): {z_hard.sum().item()}")
        
        return {
            'cube_bounds': cube_bounds,
            'gate_logits': gate_logits,
            'thumbnail': thumbnail,  # [thumbnail_num_queries, 4096] if use_thumbnail else None
            'z_hard': z_hard,
        }
    
    @property
    def config(self):
        """Return configuration dict (for save/load)"""
        return {
            "vision_dim": self.vision_dim,
            "lm_dim": self.lm_dim,
            "alpha": self.alpha,
            "use_thumbnail": self.use_thumbnail,
        }

__all__ = ["Glm4vCubingModule"]
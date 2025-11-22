# coding = utf-8
# modeling_glm4v_withCube.py

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.init import trunc_normal_

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs
from .configuration_glm4v_withCube import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from .cubing_glm4v import Glm4vCubingModule
from .resampler_glm4v import Glm4vResampler

from transformers import AutoModel, AutoModelForCausalLM
import numpy as np


@dataclass
class Glm4vCubingVideoMetadata:
    original_grid_thw: torch.Tensor  # [[20, 24, 24]] - 视频级别
    actual_num_tokens: Optional[int] = None  # 192 - 实际生成的 tokens
    mode: str = "native"  # "native" 或 "cubing"
    
    # Cubing 
    cube_bounds: Optional[List[List[Tuple[int, int]]]] = None  # [[(0, 7), (7, 14), ...]]
    cube_timestamps: Optional[List[List[str]]] = None  # [["0000.0", "0007.0", ...]]
    cube_timestamp_tokens: Optional[List[List[List[int]]]] = None  # ← 新增：缓存 tokenized 结果
    temporal_patch_size: int = 2
    use_thumbnail: bool = False  # 是否启用 thumbnail
    thumbnail_num_queries: int = 64  # Thumbnail token 数量（默认 64）
    
    def to_flattened(self) -> torch.Tensor:
        """转换成 temporal token 级别格式"""
        # result = []
        # for t, h, w in self.original_grid_thw:
        #     num_temporal_tokens = t.item() // self.temporal_patch_size
        #     repeated = torch.tensor(
        #         [[self.temporal_patch_size, h.item(), w.item()]] * num_temporal_tokens,
        #         device=self.original_grid_thw.device,
        #         dtype=self.original_grid_thw.dtype
        #     )
        #     result.append(repeated)
        
        # return torch.cat(result, dim=0)

        result = []
        print(f"[DEBUG flatten] original_grid_thw: {self.original_grid_thw}")
        for t, h, w in self.original_grid_thw:
            repeated = torch.tensor(
                [[1, h.item(), w.item()]] * t.item(),  # ← 注意这里是 1 和 t.item()
                device=self.original_grid_thw.device,
                dtype=self.original_grid_thw.dtype
            )
            result.append(repeated)
    
        return torch.cat(result, dim=0)

@use_kernel_forward_from_hub("RMSNorm")
class Glm4vRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Glm4vRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm4VisionMlp(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.out_hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm4vVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)
        self.use_3d = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype

        if self.use_3d:
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        else:
            N, C, H, W = hidden_states.shape
        
            # [N, C, H, W] -> [N, C, H//patch_size, patch_size, W//patch_size, patch_size]
            h_patches = H // self.patch_size
            w_patches = W // self.patch_size
            
            hidden_states = hidden_states.reshape(
                N, C,
                h_patches, self.patch_size,
                w_patches, self.patch_size
            )
            # [N, C, h_patches, patch_size, w_patches, patch_size]
            # -> [N, C, h_patches, w_patches, patch_size, patch_size]
            hidden_states = hidden_states.permute(0, 1, 2, 4, 3, 5)
            # -> [N*h_patches*w_patches, C, patch_size, patch_size]
            hidden_states = hidden_states.reshape(-1, C, self.patch_size, self.patch_size)
            
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype))  # [N*h*w, embed_dim, 1, 1]
            hidden_states = hidden_states.view(-1, self.embed_dim)  # [N*h*w, embed_dim]

        return hidden_states
            


class Glm4vVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Glm4vVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            if len(lengths) > image_shapes.shape[0]:
                h_val = image_shapes[0, 1]
                w_val = image_shapes[0, 2]
                target_h = torch.cat([h_val.repeat(lengths[i]) for i in range(len(lengths))]).to(
                    device=device, dtype=torch.float32
                )
                target_w = torch.cat([w_val.repeat(lengths[i]) for i in range(len(lengths))]).to(
                    device=device, dtype=torch.float32
                )
            else:
                target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
                    device=device, dtype=torch.float32
                )
                target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
                    device=device, dtype=torch.float32
                )

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d, grid, mode="bicubic", align_corners=False, padding_mode="border"
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(embeddings.device)

        if embeddings.shape[0] != adapted_pos_embed.shape[0]:
            num_repeats = embeddings.shape[0] // adapted_pos_embed.shape[0]
            
            # 确保是整倍数，否则会出错
            if embeddings.shape[0] % adapted_pos_embed.shape[0] != 0:
                raise ValueError(
                    f"Embeddings dimension ({embeddings.shape[0]}) is not a multiple of "
                    f"position embeddings dimension ({adapted_pos_embed.shape[0]})"
                )
                
            adapted_pos_embed = adapted_pos_embed.repeat(num_repeats, 1)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed

        return embeddings


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[0] != cos.shape[0]:
        num_repeats = q.shape[0] // cos.shape[0]
        
        # 确保是整倍数
        if q.shape[0] % cos.shape[0] != 0:
            raise ValueError(
                f"Query dimension ({q.shape[0]}) is not a multiple of "
                f"Cos/Sin dimension ({cos.shape[0]})"
            )
        
        # cos/sin 原始shape是 [10752, head_dim]
        # 将它们复制以匹配 q 的 batch 维度
        cos = cos.repeat(num_repeats, 1) # new shape [21504, head_dim]
        sin = sin.repeat(num_repeats, 1) # new shape [21504, head_dim]

    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Glm4vVisionAttention(nn.Module):
    def __init__(self, config: Glm4vVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]

            tensor_dim = query_states.shape[2]  # query_states, key_states, value_states 维度 2 应该都一样

            # ✨ 重要：确保 lengths 是 Tensor 以便使用 .repeat()
            if not isinstance(lengths, torch.Tensor):
                lengths_tensor = torch.tensor(lengths, device=query_states.device) # 假设 lengths 是 list
            else:
                lengths_tensor = lengths

            lengths_sum = torch.sum(lengths).item() # 注意：如果 lengths 已经是 python list, 直接 sum(lengths)
            # print(f"[DEBUG Attention Split] Original tensor_dim: {tensor_dim}")
            # print(f"[DEBUG Attention Split] Original lengths_tensor shape: {lengths_tensor.shape}")
            # print(f"[DEBUG Attention Split] Original lengths_sum: {lengths_sum}")

            if tensor_dim != lengths_sum:
                num_repeats = tensor_dim // lengths_sum

                # 确保是整倍数
                if tensor_dim % lengths_sum != 0:
                    print(f"[ERROR Attention Split] Tensor dim {tensor_dim} is NOT a multiple of lengths sum {lengths_sum}!")
                    raise ValueError(
                        f"Tensor dim ({tensor_dim}) is not a multiple of "
                        f"lengths sum ({lengths_sum})"
                    )
                
                # print(f"[DEBUG Attention Split] Repeating lengths_tensor {num_repeats} times.")

                # 重复 'lengths' 张量以匹配合并后的输入
                # ✨ 使用 tensor 进行 repeat
                lengths_tensor = lengths_tensor.repeat(num_repeats)

                # print(f"[DEBUG Attention Split] Repeated lengths_tensor shape: {lengths_tensor.shape}")
                # print(f"[DEBUG Attention Split] Repeated lengths_sum: {torch.sum(lengths_tensor).item()}")

            final_split_list = lengths_tensor.tolist()
            # print(f"[DEBUG Attention Split] Final split list length: {len(final_split_list)}")
            # print(f"[DEBUG Attention Split] Final split list sum: {sum(final_split_list)}")
            splits = [
                torch.split(tensor, final_split_list, dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Glm4vVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm4vVisionAttention(config)
        self.mlp = Glm4VisionMlp(config, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Glm4vTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Glm4vTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Glm4vText has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class Glm4vTextAttention(nn.Module):

    def __init__(self, config: Glm4vTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(  # diff with Llama
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Glm4vTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class Glm4vTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm4vTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4vTextAttention(config, layer_idx)
        self.mlp = Glm4vTextMLP(config)
        self.input_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Glm4vModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@auto_docstring
class Glm4vPreTrainedModel(PreTrainedModel):
    config: Glm4vConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm4vTextDecoderLayer", "Glm4vVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Glm4vTextDecoderLayer,
        "attentions": Glm4vTextAttention,
    }

    def _init_weights(self, module):
        """
        Initialize the weights
        """
        std = self.config.initializer_range
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)


class Glm4vVisionModel(Glm4vPreTrainedModel):
    config: Glm4vVisionConfig
    _no_split_modules = ["Glm4vVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = Glm4vVisionEmbeddings(config)
        self.patch_embed = Glm4vVisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Glm4vVisionBlock(config) for _ in range(config.depth)])
        self.merger = Glm4vVisionPatchMerger(
            dim=config.out_hidden_size, context_dim=config.intermediate_size, hidden_act=config.hidden_act
        )

        self.post_conv_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, return_before_merge: bool = False) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        temporal_patch_size = self.config.temporal_patch_size
        # if temporal_patch_size > 1:
        # 将 [[20, 24, 24]] 转换为 [[2, 24, 24]] × 10
        #     flattened_grid_thw = []
        #     for t, h, w in grid_thw:
        #         num_temporal_tokens = t.item() // temporal_patch_size
        #         repeated = torch.tensor(
        #             [[1, h.item(), w.item()]] * num_temporal_tokens,
        #             device=grid_thw.device,
        #             dtype=grid_thw.dtype
        #         )
        #         flattened_grid_thw.append(repeated)
        #     grid_thw_for_pos = torch.cat(flattened_grid_thw, dim=0)
        # else:
        #     grid_thw_for_pos = grid_thw

        grid_thw_for_pos = grid_thw
        print(f"[DEBUG Vision] grid_thw: {grid_thw}")
        print(f"[DEBUG Vision] hidden_states.shape[0]: {hidden_states.shape[0]}")
        print(f"[DEBUG Vision] expected tokens: {sum(t*h*w for t,h,w in grid_thw)}")

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw_for_pos)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw_for_pos[:, 1] * grid_thw_for_pos[:, 2], grid_thw_for_pos[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw_for_pos.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw_for_pos, image_type_ids[:, 0], image_type_ids[:, 1])

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)

        if return_before_merge:
            # Cubing Mode
            return hidden_states

        # Naive Mode
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
        hidden_states = self.merger(hidden_states)
        return hidden_states


@auto_docstring
class Glm4vTextModel(Glm4vPreTrainedModel):
    config: Glm4vTextConfig

    def __init__(self, config: Glm4vTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm4vTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Glm4vTextRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Glm4vModel(Glm4vPreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Glm4vConfig
    _no_split_modules = ["Glm4vTextDecoderLayer", "Glm4vVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Glm4vVisionModel._from_config(config.vision_config)
        self.language_model = Glm4vTextModel._from_config(config.text_config)
        self.rope_deltas = None

        if config.use_cubing:
            self.cubing_module = Glm4vCubingModule(config)
            self.resampler = Glm4vResampler(config)
            self._current_lr_gumbel = config.cubing_lr_gumbel_start
        
        self.post_init()

        if hasattr(self, "_init_weights"): # Check if the method exists
            print("[INFO INIT WEIGHTS] Applying custom initialization to cubing_module...")
            if hasattr(self, 'cubing_module'):
                self.cubing_module.apply(self._init_weights)
            else:
                print("[WARNING INIT WEIGHTS] cubing_module not found for initialization.")

            print("[INFO INIT WEIGHTS] Applying custom initialization to resampler...")
            if hasattr(self, 'resampler'):
                self.resampler.apply(self._init_weights)
            else:
                print("[WARNING INIT WEIGHTS] resampler not found for initialization.")
        else:
            print("[WARNING INIT WEIGHTS] _init_weights method not found in Glm4vPreTrainedModel. Skipping explicit initialization.")

    
    def initialize_cubing_modules_if_needed(self):
        """智能初始化 Cubing 和 Resampler 模块"""
        if not hasattr(self, 'cubing_module'):
            return
        
        print("\n[CHECK] Verifying Cubing module initialization...")
        
        # ===== 检查所有关键参数 =====
        needs_init = self._check_resampler_params()
        
        if not needs_init:
            print("[INFO] All Cubing modules properly initialized\n")
            return
        
        print("[INFO] Initializing/Fixing Cubing modules...\n")
        
        # ===== 强制重新初始化所有参数 =====
        self._force_init_resampler()
        self._force_init_cubing()
        
        print("[INFO] Initialization complete!\n")

    def _check_resampler_params(self):
        """检查 Resampler 的所有参数是否正确初始化"""
        issues = []
        
        for name, param in self.resampler.named_parameters():
            device = param.device.type
            
            if device == 'meta':
                issues.append(f"{name}: on meta device")
                continue
            
            has_nan = torch.isnan(param).any()
            is_zero = (param.std() < 1e-6)
            
            if has_nan:
                issues.append(f"{name}: contains NaN")
            elif is_zero and 'bias' not in name and 'ln' not in name:
                # bias 和 LayerNorm 参数可以是 0
                issues.append(f"{name}: all zeros (std={param.std():.6f})")
        
        if issues:
            print("[ISSUES FOUND]")
            for issue in issues:
                print(f"  ❌ {issue}")
            return True
        
        return False

    def _force_init_resampler(self):
        """强制初始化 Resampler 的所有参数"""
        print("[INIT] Resampler modules:")
        
        # 1. 初始化所有 nn.Module
        for name, m in self.resampler.named_modules():
            if isinstance(m, nn.Linear):
                if m.weight.device.type != 'meta':
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    print(f"  ✅ {name} (Linear)")
            
            elif isinstance(m, nn.LayerNorm):
                if m.weight.device.type != 'meta':
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
                    print(f"  ✅ {name} (LayerNorm)")
            
            elif isinstance(m, nn.MultiheadAttention):
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight.device.type != 'meta':
                    nn.init.xavier_uniform_(m.in_proj_weight)
                    print(f"  ✅ {name}.in_proj_weight")
                
                if hasattr(m, 'in_proj_bias') and m.in_proj_bias.device.type != 'meta':
                    nn.init.constant_(m.in_proj_bias, 0)
                    print(f"  ✅ {name}.in_proj_bias")
        
        # 2. 初始化 nn.Parameter
        if self.resampler.query.device.type != 'meta':
            trunc_normal_(self.resampler.query, std=0.02)
            print(f"  ✅ query")
        
        if self.resampler.proj.device.type != 'meta':
            trunc_normal_(self.resampler.proj, std=0.02)
            print(f"  ✅ proj")

    def _force_init_cubing(self):
        """强制初始化 Cubing Module"""
        print("[INIT] Cubing module:")
        
        for name, m in self.cubing_module.named_modules():
            if isinstance(m, nn.Linear):
                if m.weight.device.type != 'meta':
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    print(f"  ✅ {name}")


    def set_gumbel_noise(self, lr_gumbel: float):
        """Set current Gumbel noise level (called during training)"""
        if hasattr(self, 'cubing_module'):
            self._current_lr_gumbel = lr_gumbel

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)


    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_metadata: Optional[Glm4vCubingVideoMetadata] = None, 
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index
        """
        # import traceback
        # print(f"\n[DEBUG] get_rope_index called:")
        # print(f"  input_ids.shape: {input_ids.shape if input_ids is not None else None}")
        # print(f"  video_metadata: {video_metadata}")
        # print(f"\n[CALL STACK]:")
        # traceback.print_stack(limit=20)

        # ========== Cubing 模式：使用简化位置编码 ==========
        if video_metadata is not None and video_metadata.mode == "cubing":
            return self._get_rope_index_for_cubing(
                input_ids,
                video_metadata,
                attention_mask
            )
        
        # ========== Native 模式：原有逻辑 ==========
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            video_group_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == video_start_token_id:
                        video_check_flg = True
                    elif token == video_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                video_frame_num = 1
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_frame_num,
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )

                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()

                            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        video_group_index += 1

                        if video_group_index >= video_grid_thw[video_index][0]:
                            video_index += 1
                            video_group_index = 0

                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        video_frame_num = 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _get_rope_index_for_cubing(
        self,
        input_ids: torch.LongTensor,
        video_metadata: Glm4vCubingVideoMetadata,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cubing 模式的 RoPE 位置编码（支持时间戳和 Thumbnail）"""
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        video_token_id = self.config.video_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id
        
        # 初始化 position_ids
        position_ids = torch.ones(
            3, batch_size, seq_len,
            dtype=torch.float,
            device=device
        )
        
        mrope_position_deltas = []
        
        # === 主循环 ===
        for i in range(batch_size):
            st_idx = 0
            cube_idx = 0
            cube_video_token_idx = 0
            in_video_section = False
            cube_start_pos = None
            
            segment_records = []
            current_segment = None
            
            # ✅ Thumbnail 相关状态
            expecting_thumbnail = False
            thumbnail_token_count = 0
            thumbnail_start_pos = None
            
            # ✅ 关键修复：为当前 batch item 计算位置分配
            cube_position_counts = None
            if video_metadata is not None and video_metadata.cube_bounds:
                cube_position_counts = self._allocate_cube_position_counts(
                    video_metadata,
                    batch_idx=i  # ← 传递当前 batch 索引
                )
            
            for j in range(seq_len):
                # Padding 位置
                if attention_mask is not None and attention_mask[i, j] == 0:
                    position_ids[:, i, j] = 1
                    continue
                
                token = input_ids[i, j].item()
                
                # === 处理 <|video_start|> ===
                if token == video_start_token_id:
                    if current_segment is not None:
                        if current_segment['type'] in ['text', 'timestamp']:
                            current_segment['token_end'] = j - 1
                            current_segment['pos_end'] = st_idx - 1
                        segment_records.append(current_segment)
                    
                    in_video_section = True
                    position_ids[0, i, j] = st_idx
                    position_ids[1, i, j] = 1.0
                    position_ids[2, i, j] = 1.0
                    st_idx += 1
                    
                    segment_records.append({
                        'type': 'video_start',
                        'token_idx': j,
                        'token_id': token,
                        'position': st_idx - 1,
                    })
                    
                    cube_idx = 0
                    cube_video_token_idx = 0
                    cube_start_pos = None
                    current_segment = None
                    
                    # ✅ 新增：初始化 thumbnail 状态
                    expecting_thumbnail = (
                        video_metadata is not None and 
                        video_metadata.use_thumbnail
                    )
                    thumbnail_token_count = 0
                    thumbnail_start_pos = None
                    
                    continue
                
                # ✅ 新增：处理 Thumbnail
                elif (expecting_thumbnail and 
                      in_video_section and 
                      token == video_token_id and
                      thumbnail_token_count < video_metadata.thumbnail_num_queries):
                    
                    # 第一个 thumbnail token - 创建新 segment
                    if thumbnail_token_count == 0:
                        # 保存之前的 segment
                        if current_segment is not None:
                            if current_segment['type'] in ['text', 'timestamp']:
                                current_segment['token_end'] = j - 1
                                current_segment['pos_end'] = st_idx - 1
                            segment_records.append(current_segment)
                        
                        thumbnail_start_pos = st_idx
                        current_segment = {
                            'type': 'thumbnail',
                            'token_start': j,
                            'pos_start': thumbnail_start_pos,
                            'num_tokens': video_metadata.thumbnail_num_queries,
                        }
                    
                    # 分配连续位置
                    position_ids[0, i, j] = st_idx
                    position_ids[1, i, j] = 1.0
                    position_ids[2, i, j] = 1.0
                    st_idx += 1
                    thumbnail_token_count += 1
                    
                    # Thumbnail 完成
                    if thumbnail_token_count >= video_metadata.thumbnail_num_queries:
                        current_segment['token_end'] = j
                        current_segment['pos_end'] = st_idx - 1
                        segment_records.append(current_segment)
                        current_segment = None
                        expecting_thumbnail = False
                        
                        # 重置 cube 相关计数器（准备处理后续 cubes）
                        # cube_idx = 0
                        cube_video_token_idx = 0
                        cube_start_pos = None
                        
                        print(f"[DEBUG RoPE Thumbnail] Completed thumbnail: "
                              f"tokens [{thumbnail_start_pos}, {st_idx-1}], "
                              f"positions [{thumbnail_start_pos}, {st_idx-1}]")
                    
                    continue
                
                # === 处理 <|video_end|> ===
                elif token == video_end_token_id:
                    if current_segment is not None:
                        if current_segment['type'] == 'video_cube':
                            current_segment['token_end'] = j - 1
                            current_segment['pos_end'] = position_ids[0, i, j-1].item()
                        elif current_segment['type'] in ['text', 'timestamp']:
                            current_segment['token_end'] = j - 1
                            current_segment['pos_end'] = st_idx - 1
                        segment_records.append(current_segment)
                        current_segment = None
                    
                    in_video_section = False
                    position_ids[0, i, j] = st_idx
                    position_ids[1, i, j] = 1.0
                    position_ids[2, i, j] = 1.0
                    st_idx += 1
                    
                    segment_records.append({
                        'type': 'video_end',
                        'token_idx': j,
                        'token_id': token,
                        'position': st_idx - 1,
                    })
                    continue
                
                # === 处理 <|video|> tokens (Cubes) ===
                elif token == video_token_id and in_video_section and not expecting_thumbnail:
                    # ✅ 关键修复：先检查是否需要切换 cube
                    if cube_video_token_idx >= 64:  # num_tokens = 64
                        # 保存当前 cube 段
                        if current_segment is not None:
                            current_segment['token_end'] = j - 1
                            current_segment['pos_end'] = position_ids[0, i, j-1].item()
                            segment_records.append(current_segment)
                        
                        # 切换到下一个 cube
                        cube_idx += 1
                        cube_video_token_idx = 0
                        cube_start_pos = None
                        current_segment = None
                    
                    # 检查新 cube 是否有效
                    if cube_position_counts is not None and cube_idx < len(cube_position_counts):
                        num_positions = cube_position_counts[cube_idx]
                        num_tokens = 64
                        
                        # 初始化新 cube
                        if cube_video_token_idx == 0:
                            # 保存前一个 segment（可能是 timestamp）
                            if current_segment is not None:
                                if current_segment['type'] == 'timestamp':
                                    current_segment['token_end'] = j - 1
                                    current_segment['pos_end'] = st_idx - 1
                                segment_records.append(current_segment)
                            
                            cube_start_pos = st_idx
                            
                            current_segment = {
                                'type': 'video_cube',
                                'cube_idx': cube_idx,
                                'cube_bounds': video_metadata.cube_bounds[i][cube_idx],
                                'timestamp': video_metadata.cube_timestamps[i][cube_idx],
                                'allocated_positions': num_positions,
                                'num_tokens': num_tokens,
                                'token_start': j,
                                'pos_start': cube_start_pos,
                            }
                        
                        # 分配位置（linspace）
                        if num_tokens > 1:
                            relative_pos = cube_video_token_idx / (num_tokens - 1)
                        else:
                            relative_pos = 0
                        
                        actual_pos = cube_start_pos + relative_pos * (num_positions - 1)
                        
                        position_ids[0, i, j] = actual_pos
                        position_ids[1, i, j] = 1.0
                        position_ids[2, i, j] = 1.0
                        
                        cube_video_token_idx += 1
                        
                        # Cube 完成后更新 st_idx
                        if cube_video_token_idx >= num_tokens:
                            st_idx = cube_start_pos + num_positions
                    
                    else:
                        # 超出范围，降级处理
                        position_ids[0, i, j] = st_idx
                        position_ids[1, i, j] = 1.0
                        position_ids[2, i, j] = 1.0
                        st_idx += 1
                
                # === 处理其他 tokens（文本、时间戳等）===
                else:
                    is_timestamp = in_video_section
                    
                    if is_timestamp:
                        if current_segment is not None and current_segment['type'] == 'video_cube':
                            current_segment['token_end'] = j - 1
                            current_segment['pos_end'] = position_ids[0, i, j-1].item()
                            segment_records.append(current_segment)
                            current_segment = None
                        
                        if current_segment is None or current_segment['type'] != 'timestamp':
                            if current_segment is not None:
                                if current_segment['type'] in ['text', 'timestamp']:
                                    current_segment['token_end'] = j - 1
                                    current_segment['pos_end'] = st_idx - 1
                                segment_records.append(current_segment)
                            
                            current_segment = {
                                'type': 'timestamp',
                                'token_start': j,
                                'pos_start': st_idx,
                                'token_ids': [],
                                'positions': [],
                            }
                        
                        position_ids[0, i, j] = st_idx
                        position_ids[1, i, j] = 1.0
                        position_ids[2, i, j] = 1.0
                        current_segment['token_ids'].append(token)
                        current_segment['positions'].append(st_idx)
                        st_idx += 1
                    else:
                        # 文本 token
                        if current_segment is None or current_segment['type'] != 'text':
                            if current_segment is not None:
                                if current_segment['type'] in ['text', 'timestamp']:
                                    current_segment['token_end'] = j - 1
                                    current_segment['pos_end'] = st_idx - 1
                                segment_records.append(current_segment)
                            
                            current_segment = {
                                'type': 'text',
                                'token_start': j,
                                'pos_start': st_idx,
                                'token_ids': [],
                                'positions': [],
                            }
                        
                        position_ids[0, i, j] = st_idx
                        position_ids[1, i, j] = 1.0
                        position_ids[2, i, j] = 1.0
                        current_segment['token_ids'].append(token)
                        current_segment['positions'].append(st_idx)
                        st_idx += 1
            
            # 保存最后一个段
            if current_segment is not None:
                if current_segment['type'] in ['text', 'timestamp']:
                    current_segment['token_end'] = j
                    current_segment['pos_end'] = st_idx - 1
                elif current_segment['type'] == 'video_cube':
                    current_segment['token_end'] = j
                    current_segment['pos_end'] = position_ids[0, i, j].item()
                segment_records.append(current_segment)

            # ========================================
            # ✅ 新增：验证 Cube 分配的一致性
            # ========================================
            print(f"\n{'='*80}")
            print(f"[VALIDATION] Batch {i} - Cube Allocation Consistency Check")
            print(f"{'='*80}")
            
            # 提取所有 video_cube 段
            cube_segments = [seg for seg in segment_records if seg['type'] == 'video_cube']
            
            if len(cube_segments) > 0:
                # 计算每个 cube 的帧数
                cube_bounds = video_metadata.cube_bounds[i]
                cube_frame_counts = []
                for start_frame, end_frame in cube_bounds:
                    cube_frame_counts.append(end_frame - start_frame)
                
                total_frames = sum(cube_frame_counts)
                
                # 计算每个 cube 的 token 数量
                cube_token_counts = []
                for seg in cube_segments:
                    num_tokens = seg['token_end'] - seg['token_start'] + 1
                    cube_token_counts.append(num_tokens)
                
                # 计算每个 cube 的位置范围
                cube_position_ranges = []
                for seg in cube_segments:
                    pos_range = seg['pos_end'] - seg['pos_start']
                    cube_position_ranges.append(pos_range)
                
                print(f"\n[Frame Distribution]")
                print(f"  Cube frame counts: {cube_frame_counts}")
                print(f"  Total frames: {total_frames}")
                print(f"  Frame ratios: {[f'{count/total_frames:.4f}' for count in cube_frame_counts]}")
                
                print(f"\n[Token Distribution]")
                print(f"  Cube token counts: {cube_token_counts}")
                print(f"  Expected: all 64")
                print(f"  ✓ All correct: {all(count == 64 for count in cube_token_counts)}")
                
                print(f"\n[Position Range Distribution]")
                print(f"  Cube position ranges: {[f'{r:.4f}' for r in cube_position_ranges]}")
                print(f"  Total position range: {sum(cube_position_ranges):.4f}")
                print(f"  Position ratios: {[f'{r/sum(cube_position_ranges):.4f}' for r in cube_position_ranges]}")
                
                # ✅ 关键验证：帧数比例 vs 位置范围比例
                print(f"\n[Ratio Consistency Check]")
                print(f"  {'Cube':<6} {'Frames':<8} {'Frame%':<10} {'Pos Range':<12} {'Pos%':<10} {'Diff%':<10} {'Status':<10}")
                print(f"  {'-'*70}")
                
                max_diff = 0.0
                for idx, (frame_count, pos_range) in enumerate(zip(cube_frame_counts, cube_position_ranges)):
                    frame_ratio = frame_count / total_frames
                    pos_ratio = pos_range / sum(cube_position_ranges)
                    diff = abs(frame_ratio - pos_ratio)
                    max_diff = max(max_diff, diff)
                    
                    status = "✓ PASS" if diff < 0.01 else "✗ FAIL"
                    
                    print(f"  {idx:<6} {frame_count:<8} {frame_ratio:<10.4f} {pos_range:<12.4f} "
                        f"{pos_ratio:<10.4f} {diff:<10.4f} {status:<10}")
                
                print(f"  {'-'*70}")
                print(f"  Maximum difference: {max_diff:.6f}")
                
                if max_diff < 0.01:
                    print(f"  ✅ VALIDATION PASSED: All cubes have consistent frame/position ratios")
                else:
                    print(f"  ⚠️  VALIDATION WARNING: Some cubes have inconsistent ratios (diff > 1%)")
                
                # 额外验证：每个 cube 内部的 linspace 是否均匀
                print(f"\n[Internal Linspace Verification]")
                for idx, seg in enumerate(cube_segments):
                    token_start = seg['token_start']
                    token_end = seg['token_end']
                    num_tokens = token_end - token_start + 1
                    
                    if num_tokens > 1:
                        # 提取该 cube 的所有位置
                        cube_positions = position_ids[0, i, token_start:token_end+1].cpu().numpy()
                        
                        # 计算相邻位置的差值
                        diffs = np.diff(cube_positions)
                        mean_diff = np.mean(diffs)
                        std_diff = np.std(diffs)
                        
                        print(f"  Cube {idx}: mean_step={mean_diff:.6f}, std={std_diff:.6f}")
                        
                        if std_diff < 1e-4:
                            print(f"    ✓ Uniform spacing")
                        else:
                            print(f"    ⚠️  Non-uniform spacing detected")
            
            else:
                print(f"  No video cubes found in this batch")
            
            print(f"{'='*80}\n")
            # ========================================
            # 结束验证逻辑
            # ========================================


            print(f"\n{'='*80}")
            print(f"[DEBUG RoPE] Batch {i} - Detailed Position Assignments")
            print(f"{'='*80}")

            print(f"Tokens 3-6: {input_ids[0, 3:7]}")
            print(f"Decoded: {self._safe_decode(input_ids[0, 3:7])}")
            
            segment_num = 0
            for seg in segment_records:
                if seg['type'] == 'text':
                    decoded = self._safe_decode(seg['token_ids'])
                    print(f"\n[Segment {segment_num}] TEXT")
                    print(f"  Token range: [{seg.get('token_start', '?')}, {seg.get('token_end', '?')}]")
                    print(f"  Position IDs: {seg['positions'][:5]}...{seg['positions'][-3:] if len(seg['positions']) > 5 else ''}")
                    print(f"  Decoded: {decoded}")
                    segment_num += 1
                
                elif seg['type'] == 'video_start':
                    print(f"\n[Segment {segment_num}] <|video_start|>")
                    print(f"  Token index: {seg['token_idx']}")
                    print(f"  Position ID: {seg['position']}")
                    segment_num += 1
                
                # ✅ 新增：Thumbnail 打印
                elif seg['type'] == 'thumbnail':
                    print(f"\n[Segment {segment_num}] THUMBNAIL")
                    print(f"  Token range: [{seg['token_start']}, {seg['token_end']}]")
                    print(f"  Position range: [{seg['pos_start']:.0f}, {seg['pos_end']:.0f}]")
                    print(f"  Num tokens: {seg['num_tokens']}")
                    print(f"  Position span: {seg['pos_end'] - seg['pos_start']:.0f}")
                    segment_num += 1
                
                elif seg['type'] == 'timestamp':
                    decoded = self._safe_decode(seg['token_ids'])
                    print(f"\n[Segment {segment_num}] TIMESTAMP")
                    print(f"  Token range: [{seg.get('token_start', '?')}, {seg.get('token_end', '?')}]")
                    print(f"  Token count: {len(seg['token_ids'])} tokens")
                    print(f"  Position IDs: {seg['positions']}")
                    print(f"  Token IDs: {seg['token_ids']}")
                    print(f"  Decoded: '{decoded}'")
                    segment_num += 1
                
                elif seg['type'] == 'video_cube':
                    print(f"\n[Segment {segment_num}] VIDEO CUBE {seg['cube_idx']}")
                    print(f"  Cube bounds: {seg['cube_bounds']} (temporal tokens)")
                    print(f"  Timestamp: {seg['timestamp']}")
                    print(f"  Allocated positions: {seg['allocated_positions']}")
                    print(f"  Token range: [{seg.get('token_start', '?')}, {seg.get('token_end', '?')}] ({seg['num_tokens']} tokens)")
                    print(f"  Position range: [{seg.get('pos_start', 0):.4f}, {seg.get('pos_end', 0):.4f}]")
                    print(f"  Position span: {seg.get('pos_end', 0) - seg.get('pos_start', 0):.4f}")
                    
                    # 显示前几个和后几个位置
                    sample_positions = []
                    token_start = seg.get('token_start', 0)
                    for k in range(min(3, seg['num_tokens'])):
                        pos = position_ids[0, i, token_start + k].item()
                        sample_positions.append(f"{pos:.4f}")
                    
                    if seg['num_tokens'] > 6:
                        sample_positions.append("...")
                    
                    for k in range(max(0, seg['num_tokens'] - 3), seg['num_tokens']):
                        pos = position_ids[0, i, token_start + k].item()
                        sample_positions.append(f"{pos:.4f}")
                    
                    print(f"  Sample positions: [{', '.join(sample_positions)}]")
                    segment_num += 1
                
                elif seg['type'] == 'video_end':
                    print(f"\n[Segment {segment_num}] <|video_end|>")
                    print(f"  Token index: {seg['token_idx']}")
                    print(f"  Position ID: {seg['position']}")
                    segment_num += 1
            
            print(f"\n{'='*80}")
            print(f"[DEBUG RoPE] Batch {i} Summary")
            print(f"  Total segments: {len(segment_records)}")
            print(f"  Final st_idx: {st_idx}")
            print(f"  Max position used: {position_ids[0, i, :].max().item():.4f}")
            print(f"{'='*80}\n")
            
            
            # 计算 delta
            max_pos = position_ids[:, i, :].max()
            delta = max_pos + 1 - seq_len
            mrope_position_deltas.append(delta)
        
        # 转换并返回
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas,
            device=device,
            dtype=torch.float
        ).unsqueeze(1)
        
        return position_ids, mrope_position_deltas
    
    def _safe_decode(self, token_ids):
        """安全解码 token IDs"""
        try:
            if hasattr(self, '_debug_tokenizer') and self._debug_tokenizer is not None:
                return self._debug_tokenizer.decode(token_ids, skip_special_tokens=False)
            else:
                return f"<token_ids: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}>"
        except Exception as e:
            return f"<decode_error: {str(e)}>"
                
    # def _allocate_cube_positions(
    #     self,
    #     video_metadata: Glm4vCubingVideoMetadata,
    #     device: torch.device
    # ) -> torch.Tensor:
    #     """
    #     为每个 video token 分配位置（相对于 video_start 的偏移）
        
    #     策略：按 cube 的帧数比例分配位置范围
    #     """
    #     cube_bounds = video_metadata.cube_bounds[0]
    #     total_tokens = video_metadata.actual_num_tokens
        
    #     # 计算每个 cube 的帧数
    #     cube_frames = []
    #     for start_frame, end_frame in cube_bounds:
    #         num_frames = end_frame - start_frame
    #         cube_frames.append(num_frames)
        
    #     total_frames = sum(cube_frames)
    #     num_cubes = len(cube_frames)
        
    #     # 按比例分配位置数
    #     position_allocations = []
    #     for num_frames in cube_frames:
    #         allocated = int(round(total_tokens * num_frames / total_frames))
    #         position_allocations.append(allocated)
        
    #     # 调整确保总和 = total_tokens
    #     diff = total_tokens - sum(position_allocations)
    #     if diff != 0:
    #         position_allocations[-1] += diff
        
    #     # 计算边界
    #     boundaries = [0]
    #     for alloc in position_allocations:
    #         boundaries.append(boundaries[-1] + alloc)
        
    #     # 为每个 cube 分配 tokens
    #     tokens_per_cube = total_tokens // num_cubes
    #     remaining = total_tokens % num_cubes
        
    #     # 生成位置（从 0 开始的相对位置）
    #     all_positions = []
        
    #     for cube_idx in range(num_cubes):
    #         # 当前 cube 的 token 数
    #         if cube_idx < remaining:
    #             current_tokens = tokens_per_cube + 1
    #         else:
    #             current_tokens = tokens_per_cube
            
    #         # 位置范围
    #         start_pos = boundaries[cube_idx]
    #         end_pos = boundaries[cube_idx + 1] - 1
            
    #         # 生成位置
    #         if current_tokens > 0:
    #             cube_pos = torch.linspace(
    #                 start_pos, 
    #                 end_pos, 
    #                 current_tokens,
    #                 device=device
    #             )
    #             all_positions.append(cube_pos)
        
    #     if len(all_positions) > 0:
    #         positions = torch.cat(all_positions, dim=0)
    #     else:
    #         positions = torch.tensor([], dtype=torch.long, device=device)
        

    #     print(f"\n[DEBUG AllocatePos] ----- Start -----")
    #     print(f"  Input total_tokens: {total_tokens}")
    #     print(f"  Input cube_bounds: {cube_bounds}")
    #     print(f"  Calculated cube_frames: {cube_frames}")
    #     print(f"  Calculated total_frames: {total_frames}")
    #     print(f"  Position allocations per cube: {position_allocations}")
    #     print(f"  Calculated boundaries: {boundaries}")
    #     print(f"  Final generated relative positions (first 64): {positions[:64]}")
    #     print(f"  Final generated relative positions shape: {positions.shape}")
    #     print(f"[DEBUG AllocatePos] ----- End -----")
    #     return positions

    def _allocate_cube_position_counts(
        self,
        video_metadata: Glm4vCubingVideoMetadata,
        batch_idx: int,  # ← 新增参数：当前 batch 的索引
    ) -> List[int]:
        """
        计算指定 batch item 的每个 cube 应该占用多少个位置（按帧数比例）
        
        Args:
            video_metadata: 视频元数据
            batch_idx: 当前 batch 的索引（0, 1, 2, ...）
        
        Returns:
            List of position counts，例如: [58, 77, 57]
        """
        cube_bounds = video_metadata.cube_bounds[batch_idx]
        total_tokens = video_metadata.actual_num_tokens

        # 关键修复：扣除 thumbnail 占用的 tokens
        if video_metadata.use_thumbnail:
            cube_total_tokens = total_tokens - video_metadata.thumbnail_num_queries
        else:
            cube_total_tokens = total_tokens

        # 计算每个 cube 的帧数
        cube_frames = []
        for start_frame, end_frame in cube_bounds:
            num_frames = end_frame - start_frame
            cube_frames.append(num_frames)

        total_frames = sum(cube_frames)

        # 按帧数比例分配位置（使用 cube_total_tokens 而非 total_tokens）
        position_counts = []
        for num_frames in cube_frames:
            allocated = int(round(cube_total_tokens * num_frames / total_frames))
            position_counts.append(allocated)
        
        # 调整确保总和 = total_tokens
        diff = cube_total_tokens - sum(position_counts)
        if diff != 0:
            position_counts[-1] += diff
        
        print(f"\n[DEBUG AllocatePositionCounts] Batch {batch_idx}:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Cube total tokens: {cube_total_tokens}") 
        print(f"  Cube frames: {cube_frames}")
        print(f"  Position counts: {position_counts}")
        
        return position_counts

    def _compute_cube_timestamps(
        self,
        cube_bounds: List[Tuple[int, int]],
        video_start_token: int,
        temporal_patch_size: int,
        fps: float,
    ) -> List[str]:
        """
        计算每个 cube 的起始时间戳
        
        Args:
            cube_bounds: Cube 边界（temporal token 索引，相对于视频开始）
                例如: [(0, 3), (3, 7), (7, 10)]
            video_start_token: 该视频在全局的起始**原始帧**索引
                例如: 0（第一个视频）或 40（第二个视频从第 40 帧开始）
            temporal_patch_size: 时间维度的 patch 大小
                例如: 2
            fps: 有效帧率（已考虑 temporal_patch_size）
                例如: 1.0
        
        Returns:
            时间戳字符串列表
                例如: ["0000.0", "0007.0", "0014.0"]
        """
        timestamps = []
        
        # 重建原始 FPS
        original_fps = fps * temporal_patch_size
        
        print(f"[DEBUG _compute_timestamps] Inputs:")
        print(f"  cube_bounds: {cube_bounds}")
        print(f"  video_start_token (original frame): {video_start_token}")
        print(f"  temporal_patch_size: {temporal_patch_size}")
        print(f"  effective fps: {fps}")
        print(f"  reconstructed original_fps: {original_fps}")
        
        for cube_idx, (start_token, end_token) in enumerate(cube_bounds):
            # start_token 是相对于视频开始的 temporal token 索引
            # 转换为绝对原始帧索引
            abs_frame = video_start_token + start_token * temporal_patch_size
            
            # 转换为秒数
            seconds = abs_frame / original_fps
            
            # 格式化
            timestamp_str = self._format_timestamp(seconds)
            timestamps.append(timestamp_str)
            
            print(f"[DEBUG _compute_timestamps] Cube {cube_idx}:")
            print(f"  start_token (relative): {start_token}")
            print(f"  abs_frame: {abs_frame}")
            print(f"  seconds: {seconds:.1f}")
            print(f"  formatted: {timestamp_str}")
        
        return timestamps

    def _format_timestamp(self, seconds: float) -> str:
        """
        格式化时间戳为固定长度字符串
        
        Args:
            seconds: 秒数（浮点数）
        
        Returns:
            固定格式字符串 "0000.0"
        
        Examples:
            0.0   -> "0000.0"
            3.5   -> "0003.5"
            123.7 -> "0123.7"
        """
        return f"{seconds:06.1f}"

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        videos_bound: Optional[list] = None,
        tokenizer=None,  # ← 新增参数
    ) -> Tuple[tuple, Glm4vCubingVideoMetadata]:
        """
        Encodes videos into continuous embeddings
        
        Args:
            pixel_values_videos: Video pixel values
            video_grid_thw: Video grid dimensions
            videos_bound: Frame boundaries
            tokenizer: Tokenizer for encoding timestamps (required for cubing mode)
        
        Returns:
            video_embeds: tuple of tensors
            video_metadata: Glm4vCubingVideoMetadata object
        """
        print(f"[DEBUG] pixel_values_videos shape: {pixel_values_videos.shape}")
        print(f"[DEBUG] video_grid_thw: {video_grid_thw}")
        
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        
        video_metadata = Glm4vCubingVideoMetadata(
            original_grid_thw=video_grid_thw,
            mode="cubing" if self.config.use_cubing else "native",
            temporal_patch_size=self.config.vision_config.temporal_patch_size
        )
        
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        
        if self.config.use_cubing:
            # ✅ 传递 tokenizer
            video_embeds, video_metadata = self._get_video_features_with_cubing(
                pixel_values_videos,
                video_metadata,
                videos_bound,
                tokenizer=tokenizer  # ← 传递
            )
        else:
            video_embeds = self._get_video_features_native(
                pixel_values_videos,
                video_metadata
            )
        
        return video_embeds, video_metadata

    def _get_video_features_native(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_metadata: Glm4vCubingVideoMetadata,
    ) -> tuple:
        """Native GLM4V video processing"""
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        flattened_grid_thw = video_metadata.to_flattened()
        
        video_embeds = self.visual(
            pixel_values_videos,
            grid_thw=flattened_grid_thw,
            return_before_merge=False
        )
        
        split_sizes = (
            video_metadata.original_grid_thw.prod(-1) 
            // self.visual.spatial_merge_size ** 2 * temporal_patch_size
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        
        video_metadata.actual_num_tokens = sum(split_sizes)
        
        return video_embeds

    def _get_video_features_with_cubing(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_metadata: Glm4vCubingVideoMetadata,
        videos_bound: Optional[list] = None,
        video_fps: Optional[float] = None,
        tokenizer=None,  # ← 新增参数
    ) -> Tuple[tuple, Glm4vCubingVideoMetadata]:
        """Video processing with Cubing technique"""
        
        # === Step 0: 验证 tokenizer ===
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for cubing mode with timestamps. "
                "Please pass tokenizer to model.forward() or get_video_features()."
            )
        
        # === 获取关键参数 ===
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        effective_fps = video_fps if video_fps is not None else self.config.effective_video_fps
        
        print(f"[DEBUG Timestamps] temporal_patch_size: {temporal_patch_size}")
        print(f"[DEBUG Timestamps] effective_fps: {effective_fps}")
        
        if videos_bound is None:
            videos_bound = []
            current_frame_idx = 0
            for t, h, w in video_metadata.original_grid_thw:
                num_frames = t.item()
                videos_bound.append((current_frame_idx, current_frame_idx + num_frames))
                current_frame_idx += num_frames

        print(f"[DEBUG cube feat] videos_bound (patch): {videos_bound}")
        
        # ========== 原有逻辑 ========== 
        flattened_grid_thw = video_metadata.to_flattened()

        print("[DEBUG] get video features before visual")
        vision_features = self.visual(
            pixel_values_videos,
            grid_thw=flattened_grid_thw,
            return_before_merge=True
        )
        print("[DEBUG] get video features after visual")
        
        frame_features = self._reconstruct_frames(
            vision_features,
            video_metadata.original_grid_thw
        )

        print(f"\n[DEBUG VERIFY] ===== Frame Features Verification =====")
        print(f"  original_grid_thw: {video_metadata.original_grid_thw}")
        print(f"  len(frame_features): {len(frame_features)}")
        print(f"  Expected: {video_metadata.original_grid_thw[0, 0].item()}")
        print(f"  temporal_patch_size: {temporal_patch_size}")
        print(f"  Actual sampled frames (before temporal patch): {video_metadata.original_grid_thw[0, 0].item() * temporal_patch_size}")
        print(f"  videos_bound (input): {videos_bound}")
        
        # 验证 videos_bound 的范围
        if videos_bound is not None:
            for video_idx, (vs, ve) in enumerate(videos_bound):
                print(f"  Video {video_idx}: vs={vs}, ve={ve}, range={ve - vs}")
                if ve > len(frame_features):
                    print(f"    ❌ ERROR: ve ({ve}) > len(frame_features) ({len(frame_features)})")
                    print(f"    This suggests videos_bound is in ORIGINAL FRAME level, not temporal token level!")
                else:
                    print(f"    ✅ OK: ve ({ve}) <= len(frame_features) ({len(frame_features)})")
    
        
        video_embeds_list = []
        all_cube_bounds = []
        all_gate_logits = []
        all_cube_timestamps = []
        all_cube_timestamp_tokens = []  # ← 新增：存储 tokenized 结果
        
        for video_idx, (vs, ve) in enumerate(videos_bound):
            video_frames = torch.stack([
                frame_features[i] for i in range(vs, ve)
            ])
            
            print("[DEBUG] get cubing_result before cubing_module")
            cubing_result = self.cubing_module(
                video_frames,
                fpq=self.config.cubing_fpq,
                temperature=self.config.cubing_temperature,
                lr_gumbel=self._current_lr_gumbel,
            )
            print(f"[DEBUG cube features] cubing_result: {cubing_result}")
            
            all_gate_logits.append(cubing_result['gate_logits'])
            all_cube_bounds.append(cubing_result['cube_bounds'])
            
            # === 计算时间戳 ===
            cube_timestamps = self._compute_cube_timestamps(
                cube_bounds=cubing_result['cube_bounds'],
                video_start_token=0,
                temporal_patch_size=temporal_patch_size,
                fps=effective_fps,
            )
            all_cube_timestamps.append(cube_timestamps)
            print(f"[DEBUG Timestamps] Video {video_idx} timestamps: {cube_timestamps}")
            
            # === 立即 tokenize 时间戳 ===
            cube_timestamp_tokens = []
            for ts_text in cube_timestamps:
                ts_token_ids = tokenizer.encode(ts_text, add_special_tokens=False)
                cube_timestamp_tokens.append(ts_token_ids)
                print(f"[DEBUG Tokenize] '{ts_text}' → {ts_token_ids} ({len(ts_token_ids)} tokens)")
            
            all_cube_timestamp_tokens.append(cube_timestamp_tokens)
            
            # === Resampler 逻辑 ===
            cube_tokens = []
            h_patches = video_metadata.original_grid_thw[video_idx][1].item()
            w_patches = video_metadata.original_grid_thw[video_idx][2].item()
            
            for start, end in cubing_result['cube_bounds']:
                cube = video_frames[start:end]
                cube_flat = cube.reshape(-1, 1536)

                print("[DEBUG] get tokens before resampler")
                tokens = self.resampler(
                    cube_flat,
                    tgt_size_range=[
                        [start, end],
                        [0, h_patches],
                        [0, w_patches]
                    ],
                    fps=effective_fps,
                )
                print(f"[DEBUG] resampled tokens: {tokens}")
                cube_tokens.append(tokens)
            
            if cubing_result['thumbnail'] is not None:
                cube_tokens.insert(0, cubing_result['thumbnail'].unsqueeze(0))
            
            video_tokens = torch.cat(cube_tokens, dim=0)
            video_embeds_list.append(video_tokens)
        
        total_tokens = sum(v.shape[0] for v in video_embeds_list)
        video_metadata.actual_num_tokens = total_tokens
        video_metadata.cube_bounds = all_cube_bounds
        video_metadata.cube_timestamps = all_cube_timestamps
        video_metadata.cube_timestamp_tokens = all_cube_timestamp_tokens  # ← 存储 tokenized 结果

        video_metadata.use_thumbnail = self.config.cubing_use_thumbnail
        video_metadata.thumbnail_num_queries = getattr(
            self.config, 'thumbnail_num_queries', 64
)
        
        if self.training and all_gate_logits:
            self._last_gate_logits = torch.cat(all_gate_logits, dim=0)

        print(f"[DEBUG features cubing] video_embeds_list: {len(video_embeds_list)}, {video_embeds_list[0].shape}")
        print(f"[DEBUG features cubing] video_metadata: {video_metadata}")
        
        return tuple(video_embeds_list), video_metadata

    def _reconstruct_frames(
        self,
        flat_features: torch.Tensor,
        grid_thw: torch.Tensor 
    ) -> list:
        """
        Reconstruct flattened features into frame format
        """
        frames = []
        start_idx = 0
        
        for t, h, w in grid_thw:
            num_temporal_tokens = t 
            seq_len = num_temporal_tokens * h * w

            feat = flat_features[start_idx:start_idx + seq_len]
            feat = feat.reshape(num_temporal_tokens, h * w, -1)
            
            for frame_idx in range(num_temporal_tokens):
                frames.append(feat[frame_idx])
            
            start_idx += seq_len
        
        return frames

    def get_cubing_aux_loss(self, alpha=0.001):
        """Calculate auxiliary loss for Cubing gate network"""
        if hasattr(self, '_last_gate_logits'):
            gate_logits = self._last_gate_logits
            aux_loss = alpha * torch.norm(gate_logits, p=2)
            delattr(self, '_last_gate_logits')
            return aux_loss
        return torch.tensor(0.0, device=self.visual.patch_embed.proj.weight.device)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """Encodes images into continuous embeddings"""
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """Obtains multimodal placeholder mask"""
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask
    
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,          # Shape [B, L_orig]
        attention_mask: Optional[torch.Tensor] = None,         # Shape [B, L_orig] or dict
        position_ids: Optional[torch.LongTensor] = None,       # Shape [3, B, L_orig] - Calculated by Collator, WE WILL IGNORE THIS
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,     # Shape [B, L_orig, D]
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        videos_bound: Optional[list] = None,
        rope_deltas: Optional[torch.LongTensor] = None,       # Calculated by Collator, WE WILL RECALCULATE
        cache_position: Optional[torch.LongTensor] = None,
        tokenizer=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Glm4vModelOutputWithPast]:
        self._debug_tokenizer = tokenizer
        import torch.distributed as dist
        
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        actual_batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        print(f"[RANK {rank}/{world_size}] Device: {device}, Batch size: {actual_batch_size}")
    

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # ========== Step 1: 获取初始文本 Embeddings ==========
        print(f"===STEP 1 Getting initial inputs and embeddings===")
        if inputs_embeds is None:
            # print(f"\n[DEBUG NAN SRC] Checking expanded input_ids before embedding:")
            # print(f"  Shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            # print(f"  Min ID: {input_ids.min()}, Max ID: {input_ids.max()}") # Check for invalid IDs
            initial_inputs_embeds = self.get_input_embeddings()(input_ids) # Shape [B, L_orig, D]
        else:
            initial_inputs_embeds = inputs_embeds # Shape [B, L_orig, D]
        # print(f"initial video_grid_thw: {video_grid_thw}")

        print(f"\n[DEBUG] Initial Text Embeddings:")
        print(f"  Shape: {initial_inputs_embeds.shape}, dtype: {initial_inputs_embeds.dtype}")
        # print(f"  Has NaN: {torch.isnan(initial_inputs_embeds).any()}")
        # print(f"  Has Inf: {torch.isinf(initial_inputs_embeds).any()}")

        # ========== Step 2: 获取并插入图像特征 (如果存在) ==========
        print(f"===STEP 2 Getting image features===")
        # (假设图像特征已由 Plugin 正确插入或由 get_placeholder_mask 处理)
        # 注意：如果图像也需要动态插入，此逻辑需要合并/调整
        current_inputs_embeds = initial_inputs_embeds
        if pixel_values is not None:
            # 假设 image 特征已通过 masked_scatter 插入 initial_inputs_embeds
            # (省略 image 插入代码，假定它在前面已完成或在此处完成)
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(current_inputs_embeds.device, current_inputs_embeds.dtype)
            # print(f"\n[DEBUG NAN SRC] Image Embeddings (before scatter):")
            # print(f"  Shape: {image_embeds.shape}, dtype: {image_embeds.dtype}")
            # print(f"  Has NaN: {torch.isnan(image_embeds).any()}")
            # print(f"  Has Inf: {torch.isinf(image_embeds).any()}")

            image_mask, _ = self.get_placeholder_mask(input_ids, current_inputs_embeds, image_features=image_embeds) # Assuming get_placeholder_mask works for images
            current_inputs_embeds = current_inputs_embeds.masked_scatter(image_mask, image_embeds)
            # print(f"[DEBUG FORWARD V2] Applied image features scatter.")
            # print(f"\n[DEBUG NAN SRC] Embeddings After Image Scatter:")
            # print(f"  Shape: {current_inputs_embeds.shape}, dtype: {current_inputs_embeds.dtype}")
            # print(f"  Has NaN: {torch.isnan(current_inputs_embeds).any()}")
            # print(f"  Has Inf: {torch.isinf(current_inputs_embeds).any()}")


        # ========== Step 3: 获取视频特征 ==========
        print(f"===STEP 3 Getting video features===")
        video_embeds_list = []
        video_metadata = None
        actual_num_video_tokens = 0

        if pixel_values_videos is not None:
            # ✅ 传递 tokenizer
            video_embeds_tuple, video_metadata = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                videos_bound=videos_bound,
                # tokenizer=kwargs.get('tokenizer'),  # ← 从 kwargs 获取
                tokenizer=tokenizer
            )
            
            video_embeds_list = [v.to(current_inputs_embeds.device, current_inputs_embeds.dtype) for v in video_embeds_tuple]
            actual_num_video_tokens = video_embeds_list[0].shape[0]
            print(f"[DEBUG FORWARD V2] Got video features. Actual video tokens: {actual_num_video_tokens}")
            print(f"  Video metadata: {video_metadata}")

        # ========== Step 4: 动态构建最终 Embeddings 和 Mask ==========
        print(f"===STEP 4 Building final embeddings and masks===")

        if video_embeds_list:
            video_start_embed = self.get_input_embeddings()(
                torch.tensor(self.config.video_start_token_id, device=current_inputs_embeds.device)
            ).unsqueeze(0)
            
            video_end_embed = self.get_input_embeddings()(
                torch.tensor(self.config.video_end_token_id, device=current_inputs_embeds.device)
            ).unsqueeze(0)
            
            final_inputs_embeds_list = []
            final_attention_mask_list = []
            batch_size, original_seq_len = input_ids.shape
            video_placeholder_id = self.config.video_token_id
            device = input_ids.device
            
            for i in range(batch_size):
                video_positions = (input_ids[i] == video_placeholder_id).nonzero(as_tuple=True)[0]

                if len(video_positions) == 0:
                    final_inputs_embeds_list.append(current_inputs_embeds[i])
                    final_attention_mask_list.append(
                        attention_mask[i] if attention_mask is not None 
                        else torch.ones(original_seq_len, device=device)
                    )
                    continue
                
                if len(video_positions) > 1:
                    print(f"[WARNING STEP4] Multiple video placeholders in batch {i}, using first one")

                video_pos = video_positions[0].item()
                current_video_embeds = video_embeds_list[i]  # [num_video_tokens, D]
                
                # === 获取时间戳信息 ===
                cube_timestamps = video_metadata.cube_timestamps[i]
                cube_timestamp_tokens = video_metadata.cube_timestamp_tokens[i]  # ← 使用预计算的 tokens
                num_cubes = len(cube_timestamps)
                tokens_per_cube = current_video_embeds.shape[0] // num_cubes
                
                # === ✅ 直接使用预计算的 token IDs ===
                timestamp_embeds_list = []
                timestamp_token_counts = []
                
                for ts_token_ids in cube_timestamp_tokens:
                    ts_embeds = self.get_input_embeddings()(
                        torch.tensor(ts_token_ids, device=device)
                    )
                    timestamp_embeds_list.append(ts_embeds)
                    timestamp_token_counts.append(len(ts_token_ids))
                
                print(f"[DEBUG Step4] Timestamp token counts: {timestamp_token_counts}")
                
                # === ✅ 检查是否有 thumbnail
                if video_metadata.use_thumbnail:
                    has_thumbnail = True
                    thumbnail_num_tokens = video_metadata.thumbnail_num_queries
                    thumbnail_embeds = current_video_embeds[0:thumbnail_num_tokens]  # 前 64 个
                    cube_offset = thumbnail_num_tokens  # 后续 cubes 从这里开始
                else:
                    has_thumbnail = False
                    thumbnail_num_tokens = 0
                    cube_offset = 0

                # 计算每个 cube 的 token 数
                num_cube_tokens = current_video_embeds.shape[0] - cube_offset
                tokens_per_cube = num_cube_tokens // num_cubes

                # === 构建 embeddings 序列 ===
                parts = [
                    current_inputs_embeds[i, :video_pos],
                    video_start_embed,
                ]

                # 插入 Thumbnail（如果存在）
                if has_thumbnail:
                    parts.append(thumbnail_embeds)
                    print(f"[DEBUG Step4 Thumbnail] Added thumbnail: {thumbnail_embeds.shape}")

                # 插入 Cubes
                for cube_idx in range(num_cubes):
                    parts.append(timestamp_embeds_list[cube_idx])  # 时间戳
                    
                    cube_start = cube_offset + cube_idx * tokens_per_cube
                    cube_end = cube_start + tokens_per_cube
                    parts.append(current_video_embeds[cube_start:cube_end])  # 64 个 video tokens

                parts.extend([
                    video_end_embed,
                    current_inputs_embeds[i, video_pos + 1:]
                ])

                new_embeds = torch.cat(parts, dim=0)
                final_inputs_embeds_list.append(new_embeds)
                
                # === 同步构建 attention_mask ===
                if attention_mask is not None:
                    current_mask = attention_mask[i]
                    
                    mask_parts = [
                        current_mask[:video_pos],
                        torch.ones(1, dtype=current_mask.dtype, device=device),  # video_start
                    ]

                    # ✅ Thumbnail mask
                    if has_thumbnail:
                        mask_parts.append(
                            torch.ones(thumbnail_num_tokens, dtype=current_mask.dtype, device=device)
                        )

                    # Cubes mask
                    for cube_idx in range(num_cubes):
                        num_ts_tokens = timestamp_token_counts[cube_idx]
                        mask_parts.append(torch.ones(num_ts_tokens, dtype=current_mask.dtype, device=device))
                        mask_parts.append(torch.ones(tokens_per_cube, dtype=current_mask.dtype, device=device))
                        
                    mask_parts.extend([
                        torch.ones(1, dtype=current_mask.dtype, device=device),  # video_end
                        current_mask[video_pos + 1:]
                    ])
                    
                    new_mask = torch.cat(mask_parts, dim=0)
                    final_attention_mask_list.append(new_mask)
                    
                    assert new_mask.shape[0] == new_embeds.shape[0], \
                        f"Mask length {new_mask.shape[0]} != Embeds length {new_embeds.shape[0]}"
                    
                    print(f"[DEBUG Step4] Batch {i}: new_embeds={new_embeds.shape[0]}, new_mask={new_mask.shape[0]}")
                else:
                    new_len = new_embeds.shape[0]
                    final_attention_mask_list.append(
                        torch.ones(new_len, dtype=torch.long, device=device)
                    )
            
            # === Padding ===
            max_len_new = max(embed.shape[0] for embed in final_inputs_embeds_list)
            
            final_inputs_embeds = torch.zeros(
                batch_size, max_len_new, current_inputs_embeds.shape[2], 
                dtype=current_inputs_embeds.dtype, 
                device=device
            )
            
            final_attention_mask = torch.zeros(
                batch_size, max_len_new, 
                dtype=torch.long, 
                device=device
            )
            
            for i in range(batch_size):
                seq_len_i = final_inputs_embeds_list[i].shape[0]
                final_inputs_embeds[i, :seq_len_i] = final_inputs_embeds_list[i]
                final_attention_mask[i, :seq_len_i] = final_attention_mask_list[i]
            
            print(f"[DEBUG Step4] Final shapes:")
            print(f"  final_inputs_embeds: {final_inputs_embeds.shape}")
            print(f"  final_attention_mask: {final_attention_mask.shape}")

        else:
            final_inputs_embeds = current_inputs_embeds
            final_attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            max_len_new = final_inputs_embeds.shape[1]
            
        # ========== Step 5: 重新计算 Position IDs ==========
        print(f"===STEP 5 Recalculating Position IDs===")

        final_position_ids = None
        final_rope_deltas = None

        if video_embeds_list:
            actual_num_video_tokens_per_video_recalc = [v.shape[0] for v in video_embeds_list]
            
            temp_input_ids = torch.full(
                (batch_size, max_len_new), 
                self.config.pad_token_id or 0, 
                dtype=torch.long, 
                device=device
            )

            video_placeholder_id = self.config.video_token_id
            video_start_token_id = self.config.video_start_token_id
            video_end_token_id = self.config.video_end_token_id

            for i in range(batch_size):
                original_len_i = input_ids[i].shape[0]
                video_positions = (input_ids[i] == video_placeholder_id).nonzero(as_tuple=True)[0]
                
                if len(video_positions) > 0:
                    video_pos_orig = video_positions[0].item()
                    
                    cube_timestamps = video_metadata.cube_timestamps[i]
                    cube_timestamp_tokens = video_metadata.cube_timestamp_tokens[i]  # ← 使用预计算的 tokens
                    num_cubes = len(cube_timestamps)
                    num_video_tokens_inserted = actual_num_video_tokens_per_video_recalc[i]
                    
                    # Calc offsets
                    if video_metadata.use_thumbnail:
                        has_thumbnail = True
                        thumbnail_num_tokens = video_metadata.thumbnail_num_queries
                        cube_offset = thumbnail_num_tokens
                    else:
                        has_thumbnail = False
                        thumbnail_num_tokens = 0
                        cube_offset = 0

                    num_cube_tokens = num_video_tokens_inserted - cube_offset
                    tokens_per_cube = num_cube_tokens // num_cubes

                    # === 构建 temp_input_ids ===
                    parts = []

                    # 前面部分
                    parts.append(input_ids[i, :video_pos_orig])

                    # <|video_start|>
                    parts.append(torch.tensor([video_start_token_id], device=device))

                    # ✅ Thumbnail
                    if has_thumbnail:
                        parts.append(
                            torch.full((thumbnail_num_tokens,), video_placeholder_id, device=device)
                        )

                    # Cubes: timestamp + video tokens
                    for cube_idx in range(num_cubes):
                        # Timestamp
                        ts_token_ids = cube_timestamp_tokens[cube_idx]
                        parts.append(torch.tensor(ts_token_ids, device=device))
                        
                        print(f"[DEBUG Step5] Cube {cube_idx} timestamp '{cube_timestamps[cube_idx]}' "
                            f"→ {len(ts_token_ids)} tokens: {ts_token_ids}")
                        
                        # Video tokens
                        parts.append(torch.full((tokens_per_cube,), video_placeholder_id, device=device))

                    # <|video_end|>
                    parts.append(torch.tensor([video_end_token_id], device=device))

                    # 后面部分
                    orig_after_start = video_pos_orig + 1
                    parts.append(input_ids[i, orig_after_start:])

                    # 拼接
                    temp_seq = torch.cat(parts)
                    
                    # 验证长度
                    expected_len = final_attention_mask[i].sum().item()
                    actual_len = temp_seq.shape[0]
                    
                    print(f"[DEBUG Step5] Batch {i}: temp_seq length={actual_len}, expected={expected_len}")
                    
                    if actual_len <= max_len_new:
                        temp_input_ids[i, :actual_len] = temp_seq
                    else:
                        print(f"[ERROR Step5] temp_seq too long! Truncating from {actual_len} to {max_len_new}")
                        temp_input_ids[i, :] = temp_seq[:max_len_new]
                
                else:
                    temp_input_ids[i, :original_len_i] = input_ids[i]
            
            # 验证
            assert temp_input_ids.shape == final_attention_mask.shape, \
                f"Shape mismatch: temp_input_ids {temp_input_ids.shape} != attention_mask {final_attention_mask.shape}"
            
            # 调用 get_rope_index
            final_position_ids, final_rope_deltas = self.get_rope_index(
                input_ids=temp_input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                video_metadata=video_metadata,
                attention_mask=final_attention_mask
            )
            
            print(f"[DEBUG STEP5] Profill Stage position_ids shape: {final_position_ids.shape}")

        else:
            # Decode Stage
            seq_len = final_inputs_embeds.shape[1]
            batch_size = final_inputs_embeds.shape[0]
            
            if past_key_values is not None:
                try:
                    if len(past_key_values) > 0 and len(past_key_values[0]) > 0:
                        past_len = past_key_values[0][0].shape[2]
                except:
                    past_len = 0
                start_pos = past_len
            else:
                start_pos = 0

            final_position_ids = torch.arange(
                start_pos, start_pos + seq_len,
                device=final_inputs_embeds.device,
                dtype=torch.long
            ).view(1, 1, -1).expand(3, batch_size, -1)

            final_rope_deltas = torch.zeros(
                batch_size, 1,
                device=final_inputs_embeds.device,
                dtype=torch.long
            )
            
            print(f"[DEBUG STEP5] Decode position_ids: shape={final_position_ids.shape}, value={start_pos}")

        # ========== Step 6: 调用 language_model ==========
        print(f"===STEP 6 Calling Language Model===")
    
        # Reconstruct attention_mask for decode stage
        # 修复：从 position_ids 获取 past_len
        if past_key_values is not None and not video_embeds_list:
            # Decode 阶段
            # position_ids[0, 0, 0] 的值就是 past_length
            past_len = final_position_ids[0, 0, 0].item()
            
            correct_len = past_len + final_inputs_embeds.shape[1]
            current_len = final_attention_mask.shape[1]
            
            print(f"[FIX] past_len from position_ids: {past_len}")
            print(f"[FIX] current_len: {current_len}, correct_len: {correct_len}")
            
            if current_len != correct_len:
                print(f"[FIX] Rebuilding attention_mask: {current_len} → {correct_len}")
                
                final_attention_mask = torch.ones(
                    final_inputs_embeds.shape[0],
                    correct_len,
                    dtype=torch.long,
                    device=final_inputs_embeds.device
                )
                
                print(f"[FIX] Fixed attention_mask shape: {final_attention_mask.shape}")


        # 确保 position_ids 最终形状正确 [3, B, L_new]
        if final_position_ids is None or final_position_ids.shape[2] != final_inputs_embeds.shape[1]:
            raise RuntimeError(f"Final position_ids shape {final_position_ids.shape if final_position_ids is not None else 'None'} "
                              f"mismatched with final inputs_embeds seq length {final_inputs_embeds.shape[1]}")

        # print(f"\n[DEBUG NAN] === Before language_model ===")
        # print(f"  final_inputs_embeds shape: {final_inputs_embeds.shape}, dtype: {final_inputs_embeds.dtype}")
        # print(f"  final_inputs_embeds has NaN: {torch.isnan(final_inputs_embeds).any()}")
        # print(f"  final_inputs_embeds has Inf: {torch.isinf(final_inputs_embeds).any()}")
        # Check value range (convert to float32 for stable min/max/mean)
        if not torch.isnan(final_inputs_embeds).any() and not torch.isinf(final_inputs_embeds).any():
             embeds_float = final_inputs_embeds.float()
             print(f"  final_inputs_embeds Range: min={embeds_float.min():.4f}, max={embeds_float.max():.4f}, mean={embeds_float.mean():.4f}")

        print(f"  final_position_ids shape: {final_position_ids.shape}, dtype: {final_position_ids.dtype}")
        torch.set_printoptions(threshold=10000)
        # print(f"  final_position_ids: {final_position_ids}")
        # Position IDs are indices, usually don't cause NaN directly unless used incorrectly

        print(f"  final_attention_mask shape: {final_attention_mask.shape}, dtype: {final_attention_mask.dtype}")
        # Attention mask is usually 0/1, unlikely source of NaN

        outputs = self.language_model(
            input_ids=None,
            position_ids=final_position_ids,     # <--- 使用新计算的 position_ids
            attention_mask=final_attention_mask, # <--- 使用新计算的 attention_mask
            past_key_values=past_key_values,
            inputs_embeds=final_inputs_embeds,   # <--- 使用动态构建的 embeds
            cache_position=cache_position,
            **kwargs,
        )

        if outputs.past_key_values is not None:
            new_kv_len = outputs.past_key_values[0][0].shape[2] if len(outputs.past_key_values) > 0 else 0
            print(f"  Output KV cache length: {new_kv_len}")
        else:
            print(f"  ⚠️ Output has no KV cache!")


        # llm_output_hidden_state = outputs.last_hidden_state
        # print(f"\n[DEBUG NAN] === After language_model ===")
        # print(f"  LLM output shape: {llm_output_hidden_state.shape}, dtype: {llm_output_hidden_state.dtype}")
        # print(f"  LLM output has NaN: {torch.isnan(llm_output_hidden_state).any()}")
        # print(f"  LLM output has Inf: {torch.isinf(llm_output_hidden_state).any()}")
        # if not torch.isnan(llm_output_hidden_state).any() and not torch.isinf(llm_output_hidden_state).any():
        #      llm_output_float = llm_output_hidden_state.float()
        #      print(f"  LLM output Range: min={llm_output_float.min():.4f}, max={llm_output_float.max():.4f}, mean={llm_output_float.mean():.4f}")

        return Glm4vModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=final_rope_deltas, # <--- 使用新计算的 rope_deltas
        )

    def _expand_video_placeholders(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        video_metadata: Glm4vCubingVideoMetadata,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """
        将 input_ids 中的单个 <|video|> token 扩展为 video_metadata.actual_num_tokens 个
        video_token_id，并相应扩展 attention_mask。进行 Padding。
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 假设 video_metadata 对应整个 batch (或需要按 batch item 索引)
        # 这里简化为假设 batch size=1 或所有样本视频 token 数相同
        actual_tokens = video_metadata.actual_num_tokens
        video_token_id = self.config.video_token_id # 要重复插入的 ID
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0


        expanded_input_ids_list = []
        expanded_attention_mask_list = [] if attention_mask is not None else None

        for i in range(batch_size):
            video_positions = (input_ids[i] == video_token_id).nonzero(as_tuple=True)[0]

            if len(video_positions) == 0:
                expanded_input_ids_list.append(input_ids[i])
                if attention_mask is not None:
                    expanded_attention_mask_list.append(attention_mask[i])
                continue

            # 假设只有一个 video placeholder
            video_pos = video_positions[0].item()

            expanded_ids = torch.cat([
                input_ids[i, :video_pos],
                torch.full((actual_tokens,), video_token_id, dtype=input_ids.dtype, device=device),
                input_ids[i, video_pos+1:]
            ])
            expanded_input_ids_list.append(expanded_ids)

            if attention_mask is not None:
                expanded_mask = torch.cat([
                    attention_mask[i, :video_pos],
                    torch.ones(actual_tokens, dtype=attention_mask.dtype, device=device),
                    attention_mask[i, video_pos+1:]
                ])
                expanded_attention_mask_list.append(expanded_mask)

        # Padding
        max_len = max(ids.shape[0] for ids in expanded_input_ids_list)

        padded_input_ids = torch.stack([
            F.pad(ids, (0, max_len - ids.shape[0]), value=pad_token_id)
            for ids in expanded_input_ids_list
        ])

        padded_attention_mask = None
        if attention_mask is not None:
            padded_attention_mask = torch.stack([
                F.pad(mask, (0, max_len - mask.shape[0]), value=0) # Pad mask with 0
                for mask in expanded_attention_mask_list
            ])

        # print(f"[DEBUG EXPAND] Expanded input_ids shape: {padded_input_ids.shape}")
        # if padded_attention_mask is not None:
        #      print(f"[DEBUG EXPAND] Expanded attention_mask shape: {padded_attention_mask.shape}")

        return padded_input_ids, padded_attention_mask
        
@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Glm4v causal language model (or autoregressive) outputs.
    """
)
class Glm4vCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Glm4vForConditionalGeneration(Glm4vPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm4vModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        重写 from_pretrained 以自动初始化 Cubing 模块
        """
        # 调用父类的 from_pretrained
        model = super().from_pretrained(*args, **kwargs)
        
        # 自动初始化 Cubing 模块（如果需要）
        if hasattr(model, 'model') and hasattr(model.model, 'initialize_cubing_modules_if_needed'):
            model.model.initialize_cubing_modules_if_needed()
        
        return model
    

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        videos_bound: Optional[list] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        tokenizer=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Glm4vCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        
        videos_bound (`list`, *optional*):
            Frame boundaries for each video in the batch. Format: [(start_idx, end_idx), ...].
        
        rope_deltas (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        """
        print(f"[DEBUG generation] videos_bound: {videos_bound}")
        print(f"[DEBUG generation] video_grid_thw: {video_grid_thw}")
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            videos_bound=videos_bound,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            tokenizer=tokenizer,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # print(f"\n[DEBUG NAN] === After lm_head (Logits) ===")
        # print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
        # print(f"  logits has NaN: {torch.isnan(logits).any()}")
        # print(f"  logits has Inf: {torch.isinf(logits).any()}")
        if not torch.isnan(logits).any() and not torch.isinf(logits).any():
             logits_float = logits.float()
             print(f"  logits Range: min={logits_float.min():.4f}, max={logits_float.max():.4f}, mean={logits_float.mean():.4f}")

        loss = None
        if labels is not None:
            original_seq_len = labels.shape[1] # L_orig (e.g., 24)
            # print(f"[DEBUG LOSS]original_seq_len: {original_seq_len}")
            new_seq_len = logits.shape[1]      # L_new (e.g., 343)
            # print(f"[DEBUG LOSS]new_seq_len: {new_seq_len}")
            batch_size = labels.shape[0]
            device = labels.device
            ignore_index = -100 # Standard ignore index for cross entropy

            if original_seq_len != new_seq_len:
                # print(f"[DEBUG LOSS] Expanding labels from {original_seq_len} to {new_seq_len}")
                # We need to know where the video tokens were inserted.
                # This requires finding the video placeholder in the ORIGINAL input_ids.
                # Assuming input_ids (original) is available here, or passed through outputs
                # Let's assume original input_ids ARE available as function argument `input_ids`
                # (If not, this needs adjustment to get the original IDs)

                if input_ids is None:
                    raise RuntimeError("Original input_ids are required in Glm4vForConditionalGeneration.forward to expand labels correctly.")
                if input_ids.shape[1] != original_seq_len:
                    raise RuntimeError(f"Shape mismatch: labels length ({original_seq_len}) does not match original input_ids length ({input_ids.shape[1]})")


                video_placeholder_id = self.config.video_token_id
                num_tokens_to_insert = new_seq_len - original_seq_len + 1 # e.g., 343 - 24 + 1 = 320

                labels_expanded = torch.full((batch_size, new_seq_len), ignore_index, dtype=labels.dtype, device=device)

                for i in range(batch_size):
                    video_positions = (input_ids[i] == video_placeholder_id).nonzero(as_tuple=True)[0]
                    if len(video_positions) > 0:
                        video_pos_orig = video_positions[0].item() # Position in original sequence

                        # Copy labels before placeholder
                        labels_expanded[i, :video_pos_orig] = labels[i, :video_pos_orig]
                        # Video part is filled with ignore_index (already done by torch.full)
                        # Copy labels after placeholder
                        orig_after_start = video_pos_orig + 1
                        new_after_start = video_pos_orig + num_tokens_to_insert
                        len_after = original_seq_len - orig_after_start
                        if len_after > 0:
                                end_idx_new = min(new_after_start + len_after, new_seq_len)
                                end_idx_orig = min(orig_after_start + len_after, original_seq_len)
                                labels_expanded[i, new_after_start : end_idx_new] = labels[i, orig_after_start : end_idx_orig]
                    else: # No video placeholder found in original input_ids
                        # Copy original labels directly, padding the rest with ignore_index
                        labels_expanded[i, :original_seq_len] = labels[i]

                # Use the expanded labels for loss calculation
                labels_to_use = labels_expanded
                # print(f"[DEBUG LOSS] Using expanded labels shape: {labels_to_use.shape}")
                # print(f"\n[DEBUG NAN] === Before loss_function ===")
                # print(f"  logits shape (input to loss): {logits.shape}") # Should match above
                # print(f"  labels_to_use shape (input to loss): {labels_to_use.shape}")
                # print(f"  labels_to_use has values outside [-100, vocab_size-1]?: ",
                    # f"min={labels_to_use.min()}, max={labels_to_use.max()}, vocab_size={self.config.text_config.vocab_size}")

            else: # No expansion needed
                labels_to_use = labels
                print(f"[DEBUG LOSS] Using original labels shape: {labels_to_use.shape}")

            if labels_to_use is not None:
                # 假设 self.loss_function 处理 label shifting
                loss = self.loss_function(logits=logits, labels=labels_to_use, vocab_size=self.config.text_config.vocab_size)
                # print(f"[DEBUG NAN] === After loss_function ===")
                # print(f"  Calculated loss value: {loss.item() if loss is not None else 'None'}")
                # print(f"  Loss is NaN: {torch.isnan(loss).any() if loss is not None else 'N/A'}")
                # print(f"  Loss is Inf: {torch.isinf(loss).any() if loss is not None else 'N/A'}")
                print(f"[DEBUG LOSS] Calculated loss: {loss.item() if loss is not None else 'None'}")
            else:
                 print("[ERROR LOSS] labels_to_use is None, cannot calculate loss.")

            if self.config.use_cubing and self.training:
                aux_loss = self.model.get_cubing_aux_loss(alpha=0.001)
                loss = loss + aux_loss

        return Glm4vCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
    
    def loss_function(self, logits, labels, vocab_size, ignore_index=-100):
        # Simplified version assuming standard cross entropy loss
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # Calculate loss
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            is_image = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.image_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            is_video_start = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.video_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            is_video_end = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.video_end_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            is_image = input_ids == self.config.image_start_token_id
            is_video_start = input_ids == self.config.video_start_token_id
            is_video_end = input_ids == self.config.video_end_token_id

        video_level = torch.cumsum(is_video_start.int() - is_video_end.int(), dim=1)
        inside_video = video_level > 0

        standalone_images = is_image & (~inside_video)

        image_counts = standalone_images.sum(dim=1)
        video_counts = is_video_start.sum(dim=1)

        return image_counts, video_counts

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def get_output_embeddings(self):
        """获取输出embedding层（lm_head）"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出embedding层"""
        self.lm_head = new_embeddings

__all__ = ["Glm4vForConditionalGeneration", "Glm4vModel", "Glm4vPreTrainedModel", "Glm4vTextModel", "Glm4vVisionModel"]
AutoModel.register(Glm4vConfig, Glm4vModel, exist_ok=True)
AutoModelForCausalLM.register(Glm4vConfig, Glm4vForConditionalGeneration, exist_ok=True)
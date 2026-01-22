from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CondenserConfig:
    memory_tokens: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0


class AttentionCondenser(nn.Module):
    """
    Condenser : img token -> memory tokens (# cfg : memory_tokens)
    input : h_img_seq : [1,S,D]
    output : m_tokens : [1,M,D]
    output : m_vec : [1,D] (using mean pooling)
    """

    def __init__(self, d_model: int, cfg: CondenserConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model

        self.latents = nn.Parameter(torch.randn(cfg.memory_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=cfg.num_heads, batch_first=True
        )
        hidden = int(d_model * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h_img_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("h_img_seq_shape : ",h_img_seq.shape)
        if h_img_seq.dim() != 3:
            raise ValueError(f"h_img_seq must be [B,S,D], got {tuple(h_img_seq.shape)}")
        b, _, d = h_img_seq.shape
        if d != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {d}")

        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        # Ensure dtype matches attention weights to avoid half/float matmul errors
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        if h_img_seq.dtype != attn_dtype:
            h_img_seq = h_img_seq.to(dtype=attn_dtype)
        if q.dtype != attn_dtype:
            q = q.to(dtype=attn_dtype)
        # Cross-attn: latents attend to image-token sequence
        m_tokens, _ = self.cross_attn(query=q, key=h_img_seq, value=h_img_seq, need_weights=False)
        m_tokens = m_tokens + self.mlp(m_tokens)
        m_vec = m_tokens.mean(dim=1)
        return m_tokens, m_vec

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class LongCondenserConfig:
    # 输出固定 M 个 memory tokens
    memory_tokens: int = 8
    # 注意力头数（需要整除 d_model）
    num_heads: int = 8
    # MLP hidden ratio
    mlp_ratio: float = 2.0
    # 分块大小
    chunk_size: int = 32
    # 是否用 float32 
    use_fp32_accum: bool = True


class LongAttentionCondenser(nn.Module):
    """
    Long Condenser : img token -> memory tokens (# cfg : memory_tokens)
    input : h_img_seq : [1,S,D] (long image-token hidden states)
    output : m_tokens : [1,M,D]
    output : m_vec : [1,D] (using mean pooling)    
    """

    def __init__(self, d_model: int, cfg: LongCondenserConfig):
        super().__init__()
        if d_model % int(cfg.num_heads) != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {cfg.num_heads}")
        self.cfg = cfg
        self.d_model = int(d_model)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.latents = nn.Parameter(torch.randn(int(cfg.memory_tokens), self.d_model) * 0.02)

        # 手写 MHA 投影，便于做 streaming softmax
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        hidden = int(self.d_model * float(cfg.mlp_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_model),
        )

    def forward(self, h_img_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_img_seq.dim() != 3:
            raise ValueError(f"h_img_seq must be [B,S,D], got {tuple(h_img_seq.shape)}")
        b, s, d = h_img_seq.shape
        if d != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {d}")
        if s <= 0:
            raise ValueError("h_img_seq must have S>0")

        m_tokens = int(self.cfg.memory_tokens)
        chunk = int(max(1, self.cfg.chunk_size))

        # q: [B, H, M, Dh]
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        q = self.q_proj(q).view(b, m_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # streaming softmax accumulators
        acc_dtype = torch.float32 if self.cfg.use_fp32_accum else h_img_seq.dtype
        q_acc = q.to(acc_dtype)
        device = h_img_seq.device

        m = torch.full((b, self.num_heads, m_tokens, 1), float("-inf"), device=device, dtype=acc_dtype)
        l = torch.zeros((b, self.num_heads, m_tokens, 1), device=device, dtype=acc_dtype)
        acc = torch.zeros((b, self.num_heads, m_tokens, self.head_dim), device=device, dtype=acc_dtype)

        for start in range(0, s, chunk):
            end = min(s, start + chunk)
            x = h_img_seq[:, start:end, :]  # [B, C, D]
            c = x.shape[1]

            k = self.k_proj(x).view(b, c, self.num_heads, self.head_dim).transpose(1, 2).to(acc_dtype)  # [B,H,C,Dh]
            v = self.v_proj(x).view(b, c, self.num_heads, self.head_dim).transpose(1, 2).to(acc_dtype)  # [B,H,C,Dh]

            # logits: [B,H,M,C]
            logits = torch.matmul(q_acc, k.transpose(-2, -1)) * float(self.scale)
            chunk_max = logits.max(dim=-1, keepdim=True).values
            new_m = torch.maximum(m, chunk_max)

            exp_m = torch.exp(m - new_m)
            exp_logits = torch.exp(logits - new_m)

            l = l * exp_m + exp_logits.sum(dim=-1, keepdim=True)
            acc = acc * exp_m + torch.matmul(exp_logits, v)
            m = new_m

        out = acc / (l + 1e-9)  # [B,H,M,Dh]
        out = out.transpose(1, 2).contiguous().view(b, m_tokens, d)  # [B,M,D]
        out = self.o_proj(out.to(h_img_seq.dtype))
        out = out + self.mlp(out)
        m_vec = out.mean(dim=1)
        return out, m_vec



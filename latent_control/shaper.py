from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ShaperConfig:
    # Option A: map c to K control tokens.
    control_tokens: int = 4
    hidden_ratio: float = 2.0


def expand_to_cfg_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Expand [B, ...] into Janus CFG batch layout [2B, ...]:
      [cond0, uncond0, cond1, uncond1, ...]
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    b = x.shape[0]
    x2 = x.unsqueeze(1).expand(b, 2, *x.shape[1:]).reshape(b * 2, *x.shape[1:])
    return x2


class ControlTokenShaper(nn.Module):
    """
    Shaper (option A): Dynamic control-token injection
      E_ctrl = MLP(c_t) -> [B, K, D]
    """

    def __init__(self, d_model: int, cfg: ShaperConfig):
        super().__init__()
        self.cfg = cfg
        hidden = int(d_model * cfg.hidden_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.control_tokens * d_model),
        )

    def make_control_tokens(self, c_vec: torch.Tensor) -> torch.Tensor:
        """
        Input:  c_vec [B, D]
        Output: E_ctrl [B, K, D]
        """
        b, d = c_vec.shape
        out = self.mlp(c_vec).view(b, self.cfg.control_tokens, d)
        return out

    def make_control_tokens_for_cfg(self, c_vec: torch.Tensor) -> torch.Tensor:
        """
        Input:  c_vec [B, D]
        Output: E_ctrl [2B, K, D] (inject the same control prefix to both cond/uncond to avoid CFG conflict)
        """
        e = self.make_control_tokens(c_vec)  # [B,K,D]
        return expand_to_cfg_batch(e)        # [2B,K,D]



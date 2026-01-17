from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TranslatorConfig:
    hidden_ratio: float = 2.0
    # 输出控制向量的幅度限制（tanh gate）
    max_scale: float = 1.0


class Translator(nn.Module):
    """
    Translator: Think -> Gen 的跨模态控制桥。

    输入:
      - z_vec: [B, D]  (think mode 的 latent thought pooled vector)
      - m_vec: [B, D]  (visual memory pooled vector)
      - p_vec: [B, D]  (prompt embedding pooled vector)

    输出:
      - c_vec: [B, D]  (控制向量，交给 Shaper 做“安全注入”)
    """

    def __init__(self, d_model: int, cfg: TranslatorConfig):
        super().__init__()
        self.cfg = cfg
        hidden = int(d_model * cfg.hidden_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, z_vec: torch.Tensor, m_vec: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_vec, m_vec, p_vec], dim=-1)
        c = self.net(x)
        g = self.gate(c) * self.cfg.max_scale
        return c * g



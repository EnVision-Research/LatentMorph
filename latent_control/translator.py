from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TranslatorConfig:
    hidden_ratio: float = 2.0
    # Limit the control-vector magnitude (tanh gate).
    max_scale: float = 1.0


class Translator(nn.Module):
    """
    Translator: a cross-modal control bridge from Think -> Gen.

    Inputs:
      - z_vec: [B, D]  (think-mode latent thought pooled vector)
      - m_vec: [B, D]  (visual memory pooled vector)
      - p_vec: [B, D]  (prompt embedding pooled vector)

    Output:
      - c_vec: [B, D]  (control vector; passed to Shaper for "safe injection")
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



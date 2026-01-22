from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TriggerConfig:
    # Check every N image tokens.
    check_every: int = 32
    # Window size for volatility statistics (over s_t).
    window: int = 8
    # Heuristic trigger thresholds.
    tau_drop: float = 0.05   # Δs_t < -tau_drop
    tau_var: float = 0.002   # Var(s_{t-w:t}) > tau_var
    tau_entropy: float = 3.5 # u_t > tau_entropy
    tau_sim: float = 0.25    # s_t < tau_sim


@dataclass
class TriggerPolicyConfig:
    """
    Trainable trigger policy (for RL):
    x_t = [s_t, Δs_t, Var(s_{t-w:t}), u_t] -> logits -> p_t = sigmoid(logits)
    """

    in_dim: int = 4
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    init_bias: float = -2.0  # More conservative init: p≈sigmoid(-2)=0.119


def build_trigger_features(
    *,
    s_t: torch.Tensor,
    delta_s: torch.Tensor,
    var_s: torch.Tensor,
    u_t: torch.Tensor,
) -> torch.Tensor:
    """
    Stack scalar features into the policy input x_t: [B, 4].
    """
    return torch.stack([s_t, delta_s, var_s, u_t], dim=-1)


class PolicyTrigger(nn.Module):
    """
    A tiny network (MLP/logistic) that outputs trigger probabilities:
      p_t = sigmoid(MLP(x_t))
      a_t ~ Bernoulli(p_t)
    """

    def __init__(self, cfg: TriggerPolicyConfig):
        super().__init__()
        self.cfg = cfg

        in_dim = int(cfg.in_dim)
        hidden = int(cfg.hidden_dim)
        nl = int(cfg.num_layers)
        drop = float(cfg.dropout)

        if nl <= 0:
            raise ValueError(f"num_layers must be >=1, got {nl}")

        layers: list[nn.Module] = []
        if nl == 1:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            if drop > 0:
                layers.append(nn.Dropout(p=drop))
            for _ in range(nl - 2):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
                if drop > 0:
                    layers.append(nn.Dropout(p=drop))
            layers.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*layers)

        # Initialize the last-layer bias so the initial trigger probability is low (saves compute).
        try:
            last = None
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if last is not None and last.bias is not None:
                nn.init.constant_(last.bias, float(cfg.init_bias))
        except Exception:
            pass

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B,4]
        return: logits [B]
        """
        if x_t.dim() != 2 or int(x_t.shape[-1]) != int(self.cfg.in_dim):
            raise ValueError(f"x_t must be [B,{int(self.cfg.in_dim)}], got {tuple(x_t.shape)}")
        # Numerical stability: sanitize NaN/Inf and clamp to avoid NaN logits.
        if not x_t.is_floating_point():
            x_t = x_t.to(torch.float32)
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
        x_t = x_t.clamp(min=-10.0, max=10.0)
        logits = self.net(x_t).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits

    def probs(self, x_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_t))

    def sample(self, x_t: torch.Tensor):
        """
        Returns:
          - a_t: [B] 0/1 (float)
          - logprob: [B]
          - entropy: [B]
          - p_t: [B]
        """
        logits = self.forward(x_t)
        dist = torch.distributions.Bernoulli(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        p = dist.probs
        return a, logp, ent, p


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p = probs.clamp_min(eps)
    # print("entropy_from_probs : ",p)
    # return -(p * p.log()).sum(dim=-1)
    if torch.isnan(probs).any():
        print(f"WARNING: probs contains NaN! probs shape: {probs.shape}, probs stats: min={probs.min()}, max={probs.max()}, mean={probs.mean()}")
        probs = torch.nan_to_num(probs, nan=eps)
    
    if torch.isinf(probs).any():
        print(f"WARNING: probs contains Inf! probs shape: {probs.shape}")
        probs = torch.clamp(probs, min=eps, max=1.0)
    
    # Pick an eps large enough for the dtype (esp. fp16: 1e-8 can underflow to 0 and cause log(0)=-inf).
    finfo = torch.finfo(probs.dtype) if probs.is_floating_point() else torch.finfo(torch.float32)
    eps_eff = float(max(eps, finfo.tiny))

    # Compute entropy in float32 for better numerical stability.
    p = probs.to(torch.float32)

    # Ensure probabilities are in a valid range (clamp to [0, 1], then renormalize if needed).
    p = p.clamp_min(0.0).clamp_max(1.0)
    
    # Check normalization (allow small tolerance).
    probs_sum = p.sum(dim=-1)
    if not torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3):
        print(f"WARNING: probs not normalized! sum range: [{probs_sum.min()}, {probs_sum.max()}]")
        # Renormalize.
        p = p / (p.sum(dim=-1, keepdim=True) + eps_eff)
    
    # Compute entropy: avoid 0 * (-inf) -> NaN.
    # torch.special.entr(x) = -x*log(x), and returns 0 at x=0 (more stable).
    if hasattr(torch.special, "entr"):
        entropy = torch.special.entr(p).sum(dim=-1)
    else:
        p_pos = p > 0
        entropy = -torch.where(p_pos, p * torch.log(p.clamp_min(eps_eff)), torch.zeros_like(p)).sum(dim=-1)
    
    # Sanity-check results.
    if torch.isnan(entropy).any() or torch.isinf(entropy).any():
        print(f"ERROR: entropy contains NaN/Inf! entropy: {entropy}, p stats: min={p.min()}, max={p.max()}")
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    
    return entropy


class TriggerState:
    """
    Maintain a sliding window of s_t for each sample, and compute Var and Δs.
    """

    def __init__(self, batch_size: int, window: int, device: torch.device):
        self.window = int(window)
        self.s_hist = torch.full((batch_size, self.window), float("nan"), device=device)
        self.ptr = 0

    def update(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: s_t [B]
        Output:
          - delta: Δs_t [B] (difference to previous valid value; 0 if missing)
          - var:   Var(s_{t-w:t}) [B] (ignoring NaN)
        """
        # Numerical stability: use float32 and sanitize NaN/Inf.
        if not s_t.is_floating_point():
            s_t = s_t.to(torch.float32)
        else:
            s_t = s_t.to(torch.float32)
        s_t = torch.nan_to_num(s_t, nan=0.0, posinf=0.0, neginf=0.0)
        prev_idx = (self.ptr - 1) % self.window
        prev = self.s_hist[:, prev_idx].to(torch.float32)
        prev_valid = torch.isfinite(prev)
        delta = torch.zeros_like(s_t)
        delta[prev_valid] = s_t[prev_valid] - prev[prev_valid]

        self.s_hist[:, self.ptr] = s_t
        self.ptr = (self.ptr + 1) % self.window

        # Variance (ignore NaN).
        hist = self.s_hist
        valid = torch.isfinite(hist)
        # mean over valid
        count = valid.sum(dim=1).clamp_min(1)
        mean = torch.where(valid, hist, torch.zeros_like(hist)).sum(dim=1) / count
        var = torch.where(valid, (hist - mean.unsqueeze(1)) ** 2, torch.zeros_like(hist)).sum(dim=1) / count
        var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        return delta, var



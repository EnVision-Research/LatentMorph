from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TriggerConfig:
    # 每隔多少个 image tokens 做一次检查
    check_every: int = 32
    # 计算波动性的窗口大小（对 s_t）
    window: int = 8
    # 触发阈值
    tau_drop: float = 0.05   # Δs_t < -tau_drop
    tau_var: float = 0.002   # Var(s_{t-w:t}) > tau_var
    tau_entropy: float = 3.5 # u_t > tau_entropy
    tau_sim: float = 0.25    # s_t < tau_sim


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
    
    # 为不同 dtype 选择足够大的 eps（尤其是 fp16：1e-8 会下溢成 0，导致 log(0)=-inf）
    finfo = torch.finfo(probs.dtype) if probs.is_floating_point() else torch.finfo(torch.float32)
    eps_eff = float(max(eps, finfo.tiny))

    # 熵计算最好用 float32 做数值更稳
    p = probs.to(torch.float32)

    # 确保概率在有效范围内（先裁到 [0,1]，后续再做归一化）
    p = p.clamp_min(0.0).clamp_max(1.0)
    
    # 检查是否归一化（允许小的误差）
    probs_sum = p.sum(dim=-1)
    if not torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3):
        print(f"WARNING: probs not normalized! sum range: [{probs_sum.min()}, {probs_sum.max()}]")
        # 重新归一化
        p = p / (p.sum(dim=-1, keepdim=True) + eps_eff)
    
    # 计算熵：避免 0 * (-inf) -> NaN
    # torch.special.entr(x) = -x*log(x)，并且在 x=0 时返回 0（更稳定）
    if hasattr(torch.special, "entr"):
        entropy = torch.special.entr(p).sum(dim=-1)
    else:
        p_pos = p > 0
        entropy = -torch.where(p_pos, p * torch.log(p.clamp_min(eps_eff)), torch.zeros_like(p)).sum(dim=-1)
    
    # 检查结果
    if torch.isnan(entropy).any() or torch.isinf(entropy).any():
        print(f"ERROR: entropy contains NaN/Inf! entropy: {entropy}, p stats: min={p.min()}, max={p.max()}")
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    
    return entropy


class TriggerState:
    """
    维护每个 sample 的 s_t 滑动窗口，计算 Var 与 Δs。
    """

    def __init__(self, batch_size: int, window: int, device: torch.device):
        self.window = int(window)
        self.s_hist = torch.full((batch_size, self.window), float("nan"), device=device)
        self.ptr = 0

    def update(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: s_t [B]
        输出:
          - delta: Δs_t [B] (和上一个有效值比较；如果不存在则 0)
          - var:   Var(s_{t-w:t}) [B] (忽略 NaN)
        """
        s_t.to(torch.float16)
        prev_idx = (self.ptr - 1) % self.window
        prev = self.s_hist[:, prev_idx].to(torch.float16)
        prev_valid = torch.isfinite(prev)
        delta = torch.zeros_like(s_t)
        delta[prev_valid] = s_t[prev_valid] - prev[prev_valid]

        self.s_hist[:, self.ptr] = s_t
        self.ptr = (self.ptr + 1) % self.window

        # 方差（忽略 NaN）
        hist = self.s_hist
        valid = torch.isfinite(hist)
        # mean over valid
        count = valid.sum(dim=1).clamp_min(1)
        mean = torch.where(valid, hist, torch.zeros_like(hist)).sum(dim=1) / count
        var = torch.where(valid, (hist - mean.unsqueeze(1)) ** 2, torch.zeros_like(hist)).sum(dim=1) / count
        return delta, var


def should_trigger(
    cfg: TriggerConfig,
    s_t: torch.Tensor,
    delta_s: torch.Tensor,
    var_s: torch.Tensor,
    u_t: torch.Tensor,
) -> torch.Tensor:
    """
    返回 trigger mask: [B] bool

    触发逻辑（按你给的式子）：
      trigger = 1 当 (Δs_t < -τ1 AND Var>τ2) OR (u_t>τ3 AND s_t<τ4)
    """
    cond1 = (delta_s < -cfg.tau_drop) & (var_s > cfg.tau_var)
    cond2 = (u_t > cfg.tau_entropy) & (s_t < cfg.tau_sim)
    return cond1 | cond2



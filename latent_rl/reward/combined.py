from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .clip_reward import ClipRewardConfig, ClipTextImageScorer
from .hps_reward import HpsRewardConfig, HpsV21Scorer


@dataclass
class RewardConfig:
    # DanceGRPO style: HPS + CLIP
    clip_weight: float = 1.0
    hps_weight: float = 1.0
    clip: ClipRewardConfig = ClipRewardConfig()
    hps: HpsRewardConfig = HpsRewardConfig()


class CombinedReward:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.clip = ClipTextImageScorer(cfg.clip) if float(cfg.clip_weight) != 0.0 else None
        self.hps = HpsV21Scorer(cfg.hps) if float(cfg.hps_weight) != 0.0 else None

    @torch.no_grad()
    def score(self, *, images, prompts: List[str]) -> Tuple[torch.Tensor, dict]:
        """
        return:
          total: [B] float32 cpu
          info: dict of component scores (cpu)
        """
        b = len(prompts)
        total = torch.zeros((b,), dtype=torch.float32)
        info: dict = {}

        if self.clip is not None:
            s = self.clip.score(images=images, prompts=prompts)
            info["clip"] = s
            total = total + float(self.cfg.clip_weight) * s

        if self.hps is not None:
            s = self.hps.score(images=images, prompts=prompts)
            info["hps"] = s
            total = total + float(self.cfg.hps_weight) * s

        info["total"] = total
        return total, info



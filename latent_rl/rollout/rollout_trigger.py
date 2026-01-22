from __future__ import annotations

"""
Compatibility shim (do NOT import this file in new code):

- RL rollout (policy decisions): `latent_rl.rollout.rollout_rl`
- Baseline (SFT-style single injection): `latent_rl.rollout.rollout_sft`
- Demo triplet (base/sft/rl): `latent_rl.rollout.rollout_demo`

We keep the legacy name here to avoid breaking historical imports.
"""

from latent_rl.rollout.rollout_demo import rollout_one_demo_triplet
from latent_rl.rollout.rollout_rl import rollout_one_rl as rollout_one
from latent_rl.rollout.rollout_sft import rollout_one_sft_once
from latent_rl.rollout.rollout_utils import RolloutConfig



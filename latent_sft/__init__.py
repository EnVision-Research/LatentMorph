"""
latent_sft: Supervised fine-tuning (SFT) for LatentMorph.

This package contains:
- Data loading utilities for image/caption parquet datasets
- The LatentMorph wrapper used for teacher-forcing training
- A trainer that freezes the large Janus model and trains only the lightweight control modules
  (LatentController: condenser/trigger/translator/shaper, etc.)
"""

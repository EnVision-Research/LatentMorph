#!/usr/bin/env bash
set -euo pipefail

mkdir -p "infer_sft_out"

python -u inference_sft/infer_one.py \
  --prompt "A green apple and a red suitcase." \
  --controller_ckpt "outputs_sft/checkpoints_control/ckpt_latest.pt" \
  --out "infer_sft_out/single.png" \
  --config "latent_sft/models/config.json" \
  --model_local_files_only 1 \
  --seed 42 \
  --device cuda \
  --inj -1



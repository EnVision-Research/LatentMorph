#!/usr/bin/env bash
set -euo pipefail

mkdir -p "infer_rl_out"

python -u inference_rl/infer_one.py \
  --prompt "A green apple and a red suitcase." \
  --controller_ckpt "outputs_sft/checkpoints_control/ckpt_latest.pt" \
  --rl_ckpt "outputs/rl_result/ckpt_latest.pt" \
  --out "infer_rl_out/single.png" \
  --config "latent_rl/config.json" \
  --model_local_files_only 1 \
  --seed 42 \
  --seed_mode offset \
  --device cuda \
  --use_ema 1 \
  --image_token_num 576 \
  --cfg_weight 5.0 \
  --temperature 1.0 \
  --force_action -1 \
  --enable_trigger 1



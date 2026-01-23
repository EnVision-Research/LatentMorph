#!/usr/bin/env bash
set -euo pipefail

mkdir -p "infer_rl_out/batch"

python -u inference_rl/infer_one.py \
  --prompts_file "inference_rl/gen.txt" \
  --controller_ckpt "outputs_sft/checkpoints_control/ckpt_latest.pt" \
  --rl_ckpt "outputs/rl_result/ckpt_latest.pt" \
  --out_dir "infer_rl_out/batch" \
  --config "latent_rl/config.json" \
  --model_local_files_only 1 \
  --seed 42 \
  --seed_mode offset \
  --device cuda \
  --use_ema 1 \
  --image_token_num 576 \
  --cfg_weight 5.0 \
  --temperature 1.0 \
  --max_prompts 0



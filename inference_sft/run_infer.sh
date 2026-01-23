#!/usr/bin/env bash
set -euo pipefail

mkdir -p "infer_sft_out/batch"

python -u inference_sft/infer_one.py \
  --prompts_file "inference_sft/gen.txt" \
  --controller_ckpt "outputs_sft/checkpoints_control/ckpt_latest.pt" \
  --out_dir "infer_sft_out/batch" \
  --config "latent_sft/models/config.json" \
  --model_local_files_only 1 \
  --seed 42 \
  --seed_mode offset \
  --device cuda \
  --inj -1 \
  --max_prompts 0



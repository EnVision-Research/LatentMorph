#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HOME="data/.cache/huggingface"
export TORCH_HOME="data/.cache/torch"
export TWIG_IMAGE_CACHE_DIR="data/.cache/twig_image_cache"
export PYTHONPATH=".:./Janus-Pro"

mkdir -p "outputs_sft/checkpoints_control"
mkdir -p "outputs_sft/logs"
mkdir -p "data/.cache/huggingface"
mkdir -p "data/.cache/torch"
mkdir -p "data/.cache/twig_image_cache"

python -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  "latent_sft/train/run_control.py" \
  --out_dir "outputs_sft/checkpoints_control" \
  --batch_size 1 \
  --num_workers 4 \
  --image_key url \
  --caption_key prompt \
  --lora_control 0 \
  2>&1 | tee -a "outputs_sft/logs/sft_train.log"

#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data="${data:-data}"
outputs_sft="${outputs_sft:-outputs_sft}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export HF_HOME="${HF_HOME:-$data/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$data/.cache/torch}"
export TWIG_IMAGE_CACHE_DIR="${TWIG_IMAGE_CACHE_DIR:-$data/.cache/twig_image_cache}"
export LOG_DIR="${LOG_DIR:-$outputs_sft/logs}"

export PYTHONPATH="${PYTHONPATH:-.:./Janus-Pro}"

OUT_DIR="${OUT_DIR:-$outputs_sft/checkpoints_control}"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$HF_HOME" "$TORCH_HOME" "$TWIG_IMAGE_CACHE_DIR"

python -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "latent_sft/train/run_control.py" \
  --out_dir "$OUT_DIR" \
  --batch_size 1 \
  --num_workers 4 \
  --image_key url \
  --caption_key prompt \
  2>&1 | tee -a "$LOG_DIR/sft_train.log"

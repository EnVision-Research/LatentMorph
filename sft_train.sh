#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE="8"

export HF_HOME="/nfs/wenjie/wenjie_0104/LatentMorph/data/.cache/huggingface"
export TORCH_HOME="/nfs/wenjie/wenjie_0104/LatentMorph/data/.cache/torch"
export TWIG_IMAGE_CACHE_DIR="/nfs/wenjie/wenjie_0104/twig_image_cache"
export LOG_DIR="/nfs/wenjie/wenjie_0104/twig_logs"

export PYTHONPATH="/nfs/wenjie/wenjie_0104/LatentMorph:/nfs/wenjie/wenjie_0104/LatentMorph/Janus-Pro"

export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TWIG_SAVE_STEP_BASE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"

mkdir -p /nfs/wenjie/wenjie_0104/checkpoints_control_image_loss2 \
  /nfs/wenjie/wenjie_0104/twig_logs \
  /nfs/wenjie/wenjie_0104/LatentMorph/data/.cache/huggingface \
  /nfs/wenjie/wenjie_0104/LatentMorph/data/.cache/torch \
  /nfs/wenjie/wenjie_0104/twig_image_cache

python -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  /nfs/wenjie/wenjie_0104/LatentMorph/latent_sft/train/run_control.py \
  --out_dir /nfs/wenjie/wenjie_0104/checkpoints_control_image_loss2 \
  --batch_size 1 \
  --num_workers 4 \
  --image_key url \
  --caption_key prompt \
  2>&1 | tee -a /nfs/wenjie/wenjie_0104/twig_logs/sft_train.log

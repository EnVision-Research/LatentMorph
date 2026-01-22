#!/usr/bin/env bash
set -euo pipefail

# RL training launcher (GRPO-style policy gradient for trigger policy).

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data="${data:-data}"
outputs="${outputs:-outputs}"
outputs_sft="${outputs_sft:-outputs_sft}"

PROMPTS_FILE="${PROMPTS_FILE:-$data/T2I-CompBench/examples/dataset}"
OUT_DIR="${OUT_DIR:-$outputs/rl_result}"
mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/logs"

if [ -z "${RUN_ID:-}" ]; then
RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ID
LOG_FILE="$OUT_DIR/logs/console_rank0_${RUN_ID}.log"
echo "[log] console: $LOG_FILE"
TORCHRUN_LOG_DIR="$OUT_DIR/logs/torchrun_${RUN_ID}"
mkdir -p "$TORCHRUN_LOG_DIR"
echo "[log] torchrun logs: $TORCHRUN_LOG_DIR"

PYTHON="${PYTHON:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

export HF_HOME="${HF_HOME:-$data/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$data/.cache/torch}"
export PYTHONPATH="${PYTHONPATH:-.:./Janus-Pro}"

# Required paths (override as needed)
HPS_CKPT_DIR="${HPS_CKPT_DIR:-$data/hps_ckpt}"
CONTROLLER_CKPT="${CONTROLLER_CKPT:-$outputs_sft/checkpoints_control/ckpt_latest.pt}"

# Local-only by default for open-source reproducibility
CLIP_LOCAL_ONLY="${CLIP_LOCAL_ONLY:-1}"
MODEL_LOCAL_ONLY="${MODEL_LOCAL_ONLY:-1}"

if [ ! -f "$CONTROLLER_CKPT" ]; then
  echo "[ERROR] controller checkpoint not found: $CONTROLLER_CKPT"
  echo "Set CONTROLLER_CKPT=/path/to/controller_ckpt.pt"
  exit 1
fi

if [ ! -f "$HPS_CKPT_DIR/HPS_v2.1_compressed.pt" ] || [ ! -f "$HPS_CKPT_DIR/open_clip_pytorch_model.bin" ]; then
  echo "[ERROR] HPS ckpt files missing under: $HPS_CKPT_DIR"
  echo "Expected:"
  echo "  - $HPS_CKPT_DIR/HPS_v2.1_compressed.pt"
  echo "  - $HPS_CKPT_DIR/open_clip_pytorch_model.bin"
  echo "Please download them manually (see README.md)."
  exit 1
fi

"$TORCHRUN_BIN" --standalone --nproc_per_node=8 --log_dir "$TORCHRUN_LOG_DIR" --tee 3 --local_ranks_filter 0 -m latent_rl.train.run_trigger_grpo \
  --config "latent_rl/config.json" \
  --prompts_file "$PROMPTS_FILE" \
  --max_prompts 0 \
  --batch_size 1 \
  --num_generations 4 \
  --max_steps 0 \
  --out_dir "$OUT_DIR" \
  --save_every_steps 100 \
  --clip_weight 2.8 --hps_weight 2.0 \
  --clip_local_files_only "$CLIP_LOCAL_ONLY" \
  --model_local_files_only "$MODEL_LOCAL_ONLY" \
  --hps_ckpt_dir "$HPS_CKPT_DIR" \
  --controller_ckpt "$CONTROLLER_CKPT" \
  --penalty_lambda 0.2 \
  --lr 1e-5 --weight_decay 1e-4 --entropy_coef 0.001 \
  --ema 1 --ema_decay 0.999 \
  2>&1 | tee -a "$LOG_FILE"



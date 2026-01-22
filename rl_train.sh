#!/usr/bin/env bash
set -euo pipefail

# 8-GPU full run: outputs (ckpt/demo/logs) go to /nfs/wenjie/wenjie_0104/rl_result
#
# Usage:
#   bash /nfs/wenjie/wenjie_0104/LatentMorph/run_grpo_8gpus_rl_result.sh
#
# Override:
#   PROMPTS_FILE=... OUT_DIR=... bash run_grpo_8gpus_rl_result.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export RL_RANK0_ONLY=1

REPO="/nfs/wenjie/wenjie_0104/LatentMorph"
cd "$REPO"

PROMPTS_FILE="${PROMPTS_FILE:-/nfs/wenjie/wenjie_0104/T2I-CompBench/examples/dataset}"
OUT_DIR="${OUT_DIR:-/nfs/wenjie/wenjie_0104/rl_result}"
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

# -----------------------------
# Auto-prepare reward deps (HPSv2 + ckpts, CLIP cache)
# - Only modify this script: you can keep running it without manually installing packages / downloading weights
# - If the machine/cluster has no internet access: it will raise clear errors and tell you what to copy manually
# -----------------------------
# By default we run training with the `latent` conda env:
# - Avoid the TwiG env where transformers is too new and requires torch>=2.6
#   (your current TwiG: torch2.4.1 + transformers4.57.6 will error)
# - The latent env (torch2.5.1 + transformers4.44.2) can load Janus weights correctly
PYTHON="${PYTHON:-/nfs/wenjie/miniconda3/envs/latent/bin/python}"
# NOTE: do not treat "python -m pip" as a single executable path containing spaces.
# Using an array is safer and avoids errors like:
#   line XX: /path/python -m pip: No such file or directory
PIP=("$PYTHON" -m pip)

# Put all downloads/caches under OUT_DIR for easier reproducibility and packaging.
#
# Key point: your environment blocks outbound downloads, so we must prefer an existing HF cache.
# Otherwise you may fail to find deepseek-ai/Janus-Pro-7B (e.g., missing processor preprocessor_config.json).
SFT_HF_HOME="/nfs/wenjie/wenjie_0104/LatentMorph/data/.cache/huggingface"
GLOBAL_HF_HOME="${HOME}/.cache/huggingface"
if [ -d "${SFT_HF_HOME}/hub/models--deepseek-ai--Janus-Pro-7B" ]; then
  export HF_HOME="${HF_HOME:-$SFT_HF_HOME}"
elif [ -d "${GLOBAL_HF_HOME}/hub/models--deepseek-ai--Janus-Pro-7B" ]; then
  export HF_HOME="${HF_HOME:-$GLOBAL_HF_HOME}"
else
  export HF_HOME="${HF_HOME:-$OUT_DIR/hf_home}"
fi
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_TELEMETRY=1
# 强制联网模式（不走离线）
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

HPS_CKPT_DIR="${HPS_CKPT_DIR:-$OUT_DIR/hps_ckpt}"
mkdir -p "$HPS_CKPT_DIR"

echo "[deps] python: $PYTHON"
echo "[deps] HF_HOME: $HF_HOME"
echo "[deps] HPS_CKPT_DIR: $HPS_CKPT_DIR"

# 1) ensure hpsv2 is installed (DanceGRPO style)
if ! "$PYTHON" - <<'PY'
try:
    import hpsv2  # noqa: F401
    print("ok")
except Exception as e:
    raise SystemExit(1)
PY
then
  echo "[deps] hpsv2 not found, installing from GitHub..."
  # Prefer non-interactive & more robust installs
  "${PIP[@]}" install -U pip setuptools wheel >/dev/null 2>&1 || true
  # Install via git+https (no extra scripts/dirs); failure will show a no-internet hint below.
  if ! "${PIP[@]}" install "git+https://github.com/tgxs002/HPSv2.git" ; then
    echo "[deps][ERROR] Failed to install HPSv2. If your environment has no internet access, please install it manually:"
    echo "  git clone https://github.com/tgxs002/HPSv2.git && cd HPSv2 && $PYTHON -m pip install -e ."
    exit 1
  fi
else
  echo "[deps] hpsv2 already installed"
fi

# 2) ensure HPSv2 ckpts exist (try to download; otherwise ask user to place files)
HPS_V21_CKPT="$HPS_CKPT_DIR/HPS_v2.1_compressed.pt"
HPS_OPENCLIP_BIN="$HPS_CKPT_DIR/open_clip_pytorch_model.bin"
if [ ! -f "$HPS_V21_CKPT" ] || [ ! -f "$HPS_OPENCLIP_BIN" ]; then
  echo "[deps] HPS ckpt missing, trying to download to $HPS_CKPT_DIR ..."
  # NOTE: we need bash variable expansion here, so do NOT use <<'PY'
  "$PYTHON" - <<PY
import os, shutil, sys
from pathlib import Path

ckpt_dir = Path("${HPS_CKPT_DIR}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

need = []
for fn in ["HPS_v2.1_compressed.pt", "open_clip_pytorch_model.bin"]:
    if not (ckpt_dir / fn).exists():
        need.append(fn)

if not need:
    print("[deps] HPS ckpts already present")
    raise SystemExit(0)

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("[deps][ERROR] huggingface_hub is required to auto-download HPS ckpts:", e)
    raise SystemExit(2)

# HPS v2.1 weights come from xswu/HPSv2; the open_clip base weights come from laion's model repo (aligned with DanceGRPO).
hps_repo = os.environ.get("HPS_REPO_ID", "xswu/HPSv2")
openclip_repo = os.environ.get("OPENCLIP_REPO_ID", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
ok = True
for fn in need:
    try:
        repo_id = hps_repo if fn == "HPS_v2.1_compressed.pt" else openclip_repo
        p = hf_hub_download(repo_id=repo_id, filename=fn)
        shutil.copy2(p, ckpt_dir / fn)
        print(f"[deps] downloaded {fn} -> {ckpt_dir/fn}")
    except Exception as e:
        ok = False
        print(f"[deps][WARN] failed to download {fn} from {repo_id}: {e}")

if not ok:
    print("[deps][ERROR] HPS ckpt download failed. You can manually place the following two files into:")
    print(f"  {ckpt_dir}/HPS_v2.1_compressed.pt")
    print(f"  {ckpt_dir}/open_clip_pytorch_model.bin")
    print("Also ensure your environment can access the sources (or override HPS_REPO_ID / OPENCLIP_REPO_ID).")
    raise SystemExit(3)
PY
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-/nfs/wenjie/miniconda3/envs/latent/bin/torchrun}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_ARGS=()
if [ -n "$RESUME_CKPT" ]; then
  echo "[resume] ckpt: $RESUME_CKPT"
  RESUME_ARGS+=(--resume_ckpt "$RESUME_CKPT")
fi
"$TORCHRUN_BIN" --standalone --nproc_per_node=8 --log_dir "$TORCHRUN_LOG_DIR" --tee 3 --local_ranks_filter 0 -m latent_rl.train.run_trigger_grpo \
  --config "$REPO/latent_rl/config.json" \
  --prompts_file "$PROMPTS_FILE" \
  --max_prompts 0 \
  --batch_size 1 \
  --num_generations 4 \
  --max_steps 0 \
  --out_dir "$OUT_DIR" \
  --save_every_steps 100 \
  --clip_weight 2.8 --hps_weight 2.0 \
  --clip_local_files_only 0 \
  --model_local_files_only 0 \
  --hps_ckpt_dir "$HPS_CKPT_DIR" \
  --controller_ckpt "/nfs/wenjie/wenjie_0104/checkpoints_control_image_loss2/ckpt_step_006200.pt" \
  --penalty_lambda 0.2 \
  --lr 1e-5 --weight_decay 1e-4 --entropy_coef 0.001 \
  --ema 1 --ema_decay 0.999 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"



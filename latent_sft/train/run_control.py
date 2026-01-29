from __future__ import annotations

import argparse
import os
import sys

# Ensure we can import modules regardless of the current working directory.
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_JANUS_PRO_DIR = os.path.abspath(os.path.join(_REPO_ROOT, "..", "Janus-Pro"))
for _p in (_THIS_DIR, _REPO_ROOT, _JANUS_PRO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _maybe_silence_non_rank0():
    """
    Under torchrun multi-process, silence non-rank0 stdout at the earliest stage,
    and only keep stderr in per-rank files for easier debugging.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        return
    try:
        sys.stdout = open(os.devnull, "w")
    except Exception:
        pass
    log_dir = os.environ.get("LOG_DIR", "/tmp")
    try:
        os.makedirs(log_dir, exist_ok=True)
        err_path = os.path.join(log_dir, f"train_rank{rank}.stderr.log")
        sys.stderr = open(err_path, "a", buffering=1)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


_maybe_silence_non_rank0()

from models.config_io import load_json_config, resolve_config_path  # noqa: E402
from train.trainer_control import TwiGControlTrainer  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_key", type=str, default="url")
    ap.add_argument("--caption_key", type=str, default="prompt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.abspath(os.path.join(_REPO_ROOT, "..", "outputs_sft", "checkpoints_control")),
        help="If empty, defaults to repo_root/outputs_sft/checkpoints_control",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device selection: default cuda; automatically falls back to cpu when no GPU is available",
    )
    ap.add_argument(
        "--visible_gpus",
        type=str,
        default="",
        help="Optional: set CUDA_VISIBLE_DEVICES inside the script, e.g. '0,1,2,3'. "
        "Recommended to set it as a shell environment variable instead.",
    )

    # --- LoRA control (ULM / Janus-Pro language_model) ---
    # Default OFF to keep original behavior.
    ap.add_argument(
        "--lora_control",
        type=int,
        default=0,
        help="1=enable LoRA on ULM (Janus-Pro language_model) and train it together with the controller; 0=off",
    )
    ap.add_argument("--ulm_lora_r", type=int, default=8)
    ap.add_argument("--ulm_lora_alpha", type=int, default=16)
    ap.add_argument("--ulm_lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--ulm_lora_target_modules",
        type=str,
        default="",
        help="Optional comma-separated target module names (default uses LLaMA common modules).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    # Epoch-related args are not exposed; we run exactly one pass over the dataset.
    args.max_epochs = 1
    args.max_batches_per_epoch = 0

    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpus)

    cfg_path = resolve_config_path(os.path.join(_REPO_ROOT, "models", "config.json"))
    cfg = load_json_config(cfg_path)

    trainer = TwiGControlTrainer(cfg, args)
    trainer.train()


if __name__ == "__main__":
    main()






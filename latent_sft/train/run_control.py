from __future__ import annotations

import argparse
import os
import sys

# 保证无论从哪个工作目录启动都能找到模块
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _maybe_silence_non_rank0():
    """
    torchrun 多进程下，在最早阶段静默非 rank0 的 stdout，
    仅保留 stderr 到文件，便于排查。
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
        # 默认把 checkpoints 放到 TwiGpipline 外面（减少 repo 体积/压力）
        default=os.path.abspath(os.path.join(_REPO_ROOT, "..", "checkpoints_control_image_loss2")),
        help="为空则默认写到 repo_root/checkpoints_control_image_loss2",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备选择：默认 cuda；无 GPU 时自动回退到 cpu",
    )
    ap.add_argument(
        "--visible_gpus",
        type=str,
        default="",
        help="可选：在脚本内部设置 CUDA_VISIBLE_DEVICES，例如 '0,1,2,3'。建议直接在命令行环境变量里设置。",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    # 训练轮数相关参数不再暴露给命令行，固定为“只跑一遍数据”
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






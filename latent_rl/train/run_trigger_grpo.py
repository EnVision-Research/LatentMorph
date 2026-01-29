from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List

import numpy as np

import warnings

# Hard-coded: be quiet (rank0-only logs + no warnings/noisy transformers logs).
os.environ["RL_RANK0_ONLY"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
try:  # pragma: no cover
    from transformers.utils import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    pass

# Ensure Janus-Pro + LatentMorph modules are importable regardless of cwd.
_THIS_DIR = os.path.dirname(__file__)
_LM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # LatentMorph/
_JANUS_PRO_DIR = os.path.join(_LM_ROOT, "Janus-Pro")
for _p in (_LM_ROOT, _JANUS_PRO_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from janus.models import VLChatProcessor
from latent_control.controller import LatentController
from latent_control.trigger import PolicyTrigger, TriggerPolicyConfig
from latent_sft.models.config_io import load_json_config, build_latent_controller_config
from latent_sft.models.latent_morph import LatentMorph
from ulm_lora_control import UlmLoraConfig, enable_ulm_lora, iter_trainable_params, save_ulm_lora

from latent_rl.data.compbench_prompts import CompBenchPromptDataset, CompBenchPromptsConfig
from latent_rl.reward.clip_reward import ClipRewardConfig
from latent_rl.reward.hps_reward import HpsRewardConfig
from latent_rl.reward.combined import CombinedReward, RewardConfig
from latent_rl.rollout.rollout_utils import RolloutConfig


def _freeze_all_params(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)


def _set_trainable(m: torch.nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(bool(flag))

def _is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _dist_sum_int(x: int) -> int:
    if not _is_dist():
        return int(x)
    t = torch.tensor([int(x)], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _append_reward_log(*, path: str, step: int, reward_rl: float, reward_sft: float, reward_d: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"step {int(step):08d}: "
            f"reward_rl: {float(reward_rl):.6f}  "
            f"reward_sft: {float(reward_sft):.6f}  "
            f"reward_d: {float(reward_d):.6f}\n"
        )


def _unwrap_ddp(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = float(self.decay)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().float().clone()
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def state_dict(self):
        return {"decay": float(self.decay), "shadow": self.shadow}


def _save_ckpt(
    *,
    out_dir: str,
    filename: str,
    step: int,
    global_prompts_seen: int,
    policy_mod: torch.nn.Module,
    condenser_mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    policy_ema: "EMA | None" = None,
    condenser_ema: "EMA | None" = None,
    args_dict: dict | None = None,
    save_named: bool = True,
):
    """
    Save a training checkpoint and also overwrite/update ckpt_latest.pt.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "step": int(step),
        "global_prompts_seen": int(global_prompts_seen),
        "policy": policy_mod.state_dict(),
        "condenser": condenser_mod.state_dict(),
        "optimizer": opt.state_dict(),
        "args": dict(args_dict or {}),
        "time": time.time(),
    }
    if policy_ema is not None:
        ckpt["policy_ema"] = policy_ema.state_dict()
    if condenser_ema is not None:
        ckpt["condenser_ema"] = condenser_ema.state_dict()

    path = os.path.join(out_dir, filename) if save_named else None
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    if save_named:
        torch.save(ckpt, path)
    torch.save(ckpt, latest)
    return path, latest


def _save_ckpt_latest_only(
    *,
    out_dir: str,
    step: int,
    global_prompts_seen: int,
    policy_mod: torch.nn.Module,
    condenser_mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    policy_ema: "EMA | None" = None,
    condenser_ema: "EMA | None" = None,
    args_dict: dict | None = None,
):
    """
    Only overwrite ckpt_latest.pt (for per-step/high-frequency saving), without writing extra backups.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "step": int(step),
        "global_prompts_seen": int(global_prompts_seen),
        "policy": policy_mod.state_dict(),
        "condenser": condenser_mod.state_dict(),
        "optimizer": opt.state_dict(),
        "args": dict(args_dict or {}),
        "time": time.time(),
    }
    if policy_ema is not None:
        ckpt["policy_ema"] = policy_ema.state_dict()
    if condenser_ema is not None:
        ckpt["condenser_ema"] = condenser_ema.state_dict()
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    torch.save(ckpt, latest)
    return latest


def parse_args():
    ap = argparse.ArgumentParser()
    # RL defaults to latent_rl/config.json (aligned with SFT config.json for easy parameter reuse).
    ap.add_argument("--config", type=str, default="latent_rl/config.json")
    ap.add_argument("--out_dir", type=str, default="outputs/rl_result")

    # data
    ap.add_argument(
        "--prompts_file",
        type=str,
        default="data/T2I-CompBench/examples/dataset",
        help="T2I-CompBench txt file or directory (if directory, scan and merge all .txt files).",
    )
    ap.add_argument("--max_prompts", type=int, default=0, help="0 = no truncation; run through the entire prompts_file")
    ap.add_argument("--batch_size", type=int, default=4, help="How many prompts (rollouts) per GRPO step")
    ap.add_argument("--num_workers", type=int, default=0)

    # rollout
    ap.add_argument("--image_token_num", type=int, default=576)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    # Current implementation: per-prompt group advantage is controlled by num_generations=G
    # (no extra rollouts_per_prompt layer).
    ap.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="For each prompt, generate G samples to compute group advantage (recommended: 8)",
    )

    # grpo / penalty
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--entropy_coef", type=float, default=0.0)
    ap.add_argument("--penalty_lambda", type=float, default=0.5)
    ap.add_argument(
        "--reward_baseline",
        type=str,
        default="sft_once",
        choices=["none", "sft_once"],
        help="reward baseline: none=absolute reward; sft_once=R=R_ctrl-R_sft_once (SFT-style single injection baseline)",
    )
    ap.add_argument("--adv_clip", type=float, default=5.0, help="advantage clip range (+/-adv_clip)")
    ap.add_argument(
        "--adv_std_min",
        type=float,
        default=0.05,
        help="floor on per-group std for advantage normalization to avoid std~0 exploding A",
    )

    # reward
    ap.add_argument("--clip_weight", type=float, default=1.0)
    ap.add_argument("--hps_weight", type=float, default=1.0)
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    # Default to online download for CLIP reward (set to 1 if you must run fully offline).
    ap.add_argument("--clip_local_files_only", type=int, default=0)
    ap.add_argument("--model_local_files_only", type=int, default=1)
    ap.add_argument("--ema", type=int, default=0, help="1=enable EMA for trigger policy + condenser")
    ap.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="0=disable grad clipping")
    ap.add_argument(
        "--hps_ckpt_dir",
        type=str,
        default="",
        help="HPSv2 v2.1 weights directory (contains HPS_v2.1_compressed.pt and open_clip_pytorch_model.bin). "
        "If empty, defaults to out_dir/hps_ckpt",
    )
    ap.add_argument("--controller_ckpt", type=str, default="", help="Load pre-trained controller weights (e.g. long_condenser, shaper)")

    # --- LoRA control (ULM / Janus-Pro language_model) ---
    ap.add_argument("--lora_control", type=int, default=0, help="1=enable ULM LoRA and train it together; 0=off")
    ap.add_argument("--ulm_lora_r", type=int, default=8)
    ap.add_argument("--ulm_lora_alpha", type=int, default=16)
    ap.add_argument("--ulm_lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--ulm_lora_target_modules",
        type=str,
        default="",
        help="Optional comma-separated target module names (default uses LLaMA common modules).",
    )

    # train
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = no step limit; iterate through the whole dataloader once")
    ap.add_argument("--save_every_steps", type=int, default=100, help="Save a ckpt backup every N steps (0=disable; writes ckpt_step_XXXXXXXX.pt)")
    ap.add_argument("--save_latest_every_steps", type=int, default=1, help="Overwrite ckpt_latest.pt every N steps (default=1; 0=disable)")
    return ap.parse_args()


def _init_dist():
    if not _is_dist():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
    dist.init_process_group(backend="nccl")
    # no prints (rank0 log only later)


def _setup_stdio(*, out_dir: str):
    # Hard-coded: suppress non-rank0 stdout and redirect stderr to per-rank log.
    if not _is_main():
        os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
        err_path = os.path.join(out_dir, "logs", f"rank{_rank()}.stderr.log")
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(err_path, "a", buffering=1)


def _seed_everything(seed: int):
    seed = int(seed) + _rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{_local_rank()}")
    return torch.device("cpu")


def _load_model_and_processor(*, cfg: dict, model_local_only: bool, device: torch.device):
    model_path = cfg.get("model_path", "deepseek-ai/Janus-Pro-7B")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=bool(model_local_only))
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=bool(model_local_only),
    )
    model = model.to(device=device, dtype=torch.float16).eval()
    _freeze_all_params(model)

    # Disable checkpointing + allow cache (skip if attributes do not exist).
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "gradient_checkpointing_disable"):
        lm.gradient_checkpointing_disable()
    if lm is not None and hasattr(lm, "config") and hasattr(lm.config, "use_cache"):
        lm.config.use_cache = True
    if lm is not None and hasattr(lm, "eval"):
        lm.eval()

    return processor, tokenizer, model


def _load_controller_ckpt_dict(*, ckpt_path: str, device: torch.device):
    state_dict = torch.load(str(ckpt_path), map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "controller" in state_dict:
        state_dict = state_dict["controller"]

    ctrl_dict = {}
    for k, v in state_dict.items():
        if k.startswith("controller."):
            ctrl_dict[k[len("controller."):]] = v
        else:
            ctrl_dict[k] = v
    return ctrl_dict


def _build_controllers(
    *,
    cfg: dict,
    tokenizer,
    model,
    device: torch.device,
    controller_ckpt: str,
    reward_baseline: str,
):
    latent_cfg = build_latent_controller_config(cfg)
    latent_cfg.enabled = True
    d_model = int(model.language_model.get_input_embeddings().weight.shape[1])

    controller = LatentController(d_model=d_model, tokenizer=tokenizer, cfg=latent_cfg).to(device=device)

    ctrl_dict = None
    if str(controller_ckpt).strip():
        if _is_main():
            print(f"[init] Loading controller weights from: {controller_ckpt}")
        ctrl_dict = _load_controller_ckpt_dict(ckpt_path=str(controller_ckpt), device=device)
        msg = controller.load_state_dict(ctrl_dict, strict=False)
        if _is_main():
            print(f"[init] Controller load msg: {msg}")

    controller_base = None
    if str(reward_baseline).strip() != "none":
        controller_base = LatentController(d_model=d_model, tokenizer=tokenizer, cfg=latent_cfg).to(device=device)
        if ctrl_dict is not None:
            _ = controller_base.load_state_dict(ctrl_dict, strict=False)
        controller_base = controller_base.to(dtype=torch.float32).eval()
        _set_trainable(controller_base, False)

    controller = controller.to(dtype=torch.float32).train()
    _set_trainable(controller, False)
    _set_trainable(controller.condenser, True)
    _set_trainable(controller.long_condenser, False)
    _set_trainable(controller.translator, False)
    _set_trainable(controller.shaper, False)

    return controller, controller_base


def _build_policy(*, device: torch.device):
    return PolicyTrigger(TriggerPolicyConfig()).to(device=device, dtype=torch.float32).train()


def _maybe_wrap_ddp(*, controller, policy, device: torch.device):
    if _is_dist() and device.type == "cuda":
        controller.condenser = DDP(
            controller.condenser,
            device_ids=[_local_rank()],
            output_device=_local_rank(),
            find_unused_parameters=False,
        )
        policy = DDP(policy, device_ids=[_local_rank()], output_device=_local_rank(), find_unused_parameters=False)
    return controller, policy


def _build_reward(*, args, device: torch.device):
    return CombinedReward(
        RewardConfig(
            clip_weight=float(args.clip_weight),
            hps_weight=float(args.hps_weight),
            clip=ClipRewardConfig(
                enabled=(float(args.clip_weight) != 0.0),
                model_name_or_path=str(args.clip_model),
                local_files_only=bool(int(args.clip_local_files_only)),
                device=str(device),
            ),
            hps=HpsRewardConfig(
                enabled=(float(args.hps_weight) != 0.0),
                device=str(device),
                ckpt_dir=str(args.hps_ckpt_dir),
            ),
        )
    )


def _build_loader(*, args):
    ds = CompBenchPromptDataset(CompBenchPromptsConfig(prompts_file=args.prompts_file, max_prompts=int(args.max_prompts)))

    if _is_dist():
        sampler = DistributedSampler(ds, shuffle=False, drop_last=True)
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            sampler=sampler,
            num_workers=int(args.num_workers),
            drop_last=True,
        )
    else:
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), drop_last=True)
    return ds, loader


def _build_morph(*, cfg: dict, args, model, processor, tokenizer, controller, device: torch.device):
    return LatentMorph(
        frozen_model=model,
        processor=processor,
        tokenizer=tokenizer,
        controller=controller,
        img_size=int(cfg.get("img_size", 384)),
        patch_size=int(cfg.get("patch_size", 16)),
        image_token_num=int(args.image_token_num),
        stages=int(cfg.get("stages", 1)),
        part_template=str(cfg.get("part_template", "{i}-part")),
        use_understanding=True,
        understanding_max_tokens=int(cfg.get("stage_prompt", {}).get("understanding_max_tokens", 128)),
        cfg_weight=float(args.cfg_weight),
        temperature=float(args.temperature),
    ).to(device)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not str(getattr(args, "hps_ckpt_dir", "")).strip():
        args.hps_ckpt_dir = os.path.join(str(args.out_dir), "hps_ckpt")

    _init_dist()
    _setup_stdio(out_dir=str(args.out_dir))
    _seed_everything(int(args.seed))
    device = _get_device()

    ulm_lora_cfg = UlmLoraConfig(
        enabled=bool(int(getattr(args, "lora_control", 0))),
        r=int(getattr(args, "ulm_lora_r", 8)),
        alpha=int(getattr(args, "ulm_lora_alpha", 16)),
        dropout=float(getattr(args, "ulm_lora_dropout", 0.0)),
        target_modules=[
            x.strip()
            for x in str(getattr(args, "ulm_lora_target_modules", "")).split(",")
            if x.strip()
        ]
        or None,
    )

    cfg = load_json_config(os.path.abspath(args.config))
    processor, tokenizer, model = _load_model_and_processor(
        cfg=cfg,
        model_local_only=bool(int(args.model_local_files_only)),
        device=device,
    )

    # Optionally enable LoRA on ULM (language_model).
    if bool(ulm_lora_cfg.enabled):
        if _is_main():
            print(
                f"[init] enabling ULM LoRA: r={ulm_lora_cfg.r} alpha={ulm_lora_cfg.alpha} dropout={ulm_lora_cfg.dropout}",
                flush=True,
            )
        model.language_model = enable_ulm_lora(language_model=model.language_model, cfg=ulm_lora_cfg)
        model.language_model.eval()
        # Under DDP, wrap language_model so (if it ever receives grads) it can synchronize.
        # RL rollouts are non-differentiable in current implementation, so allow unused params safely.
        if _is_dist() and device.type == "cuda":
            model.language_model = DDP(
                model.language_model,
                device_ids=[_local_rank()],
                output_device=_local_rank(),
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        if _is_main():
            ulm_trainable = iter_trainable_params(model.language_model)
            n_params = sum(int(p.numel()) for p in ulm_trainable)
            print(f"[init] ULM LoRA trainable params: {len(ulm_trainable)} tensors, {n_params} params", flush=True)

    controller, controller_base = _build_controllers(
        cfg=cfg,
        tokenizer=tokenizer,
        model=model,
        device=device,
        controller_ckpt=str(getattr(args, "controller_ckpt", "")),
        reward_baseline=str(getattr(args, "reward_baseline", "sft_once")),
    )
    policy = _build_policy(device=device)
    controller, policy = _maybe_wrap_ddp(controller=controller, policy=policy, device=device)

    reward = _build_reward(args=args, device=device)
    ds, loader = _build_loader(args=args)

    # === optimizer ===
    # Note: policy/condenser may be wrapped by DDP, but parameters() is still accessible.
    params = [p for p in list(controller.condenser.parameters()) + list(policy.parameters()) if p.requires_grad]
    if bool(ulm_lora_cfg.enabled):
        params += iter_trainable_params(model.language_model)
        seen = set()
        uniq = []
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)
        params = uniq
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)

    # === EMA (optional) ===
    ema_enabled = int(args.ema) == 1
    policy_mod = policy.module if isinstance(policy, DDP) else policy
    condenser_mod = controller.condenser.module if isinstance(controller.condenser, DDP) else controller.condenser
    policy_ema = EMA(policy_mod, float(args.ema_decay)) if ema_enabled else None
    condenser_ema = EMA(condenser_mod, float(args.ema_decay)) if ema_enabled else None

    rollout_cfg = RolloutConfig(
        image_token_num=int(args.image_token_num),
        cfg_weight=float(args.cfg_weight),
        temperature=float(args.temperature),
        num_rollouts_per_prompt=1,
    )

    step = 0
    global_prompts_seen = 0  # Global (across DDP ranks) cumulative prompt count; rank0 tracks & saves.
    save_every_steps = int(getattr(args, "save_every_steps", 0))
    next_ckpt_step_at = int(max(1, save_every_steps)) if save_every_steps > 0 else 0

    # Use shell redirection/tee for logs (do not write metrics files from Python here).

    total_prompts = int(len(ds))
    if int(args.max_prompts) > 0:
        total_prompts = min(total_prompts, int(args.max_prompts))
    morph = _build_morph(cfg=cfg, args=args, model=model, processor=processor, tokenizer=tokenizer, controller=controller, device=device)
    run_id = os.environ.get("RUN_ID", "").strip() or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    reward_log_path = os.path.join(str(args.out_dir), f"reward_{run_id}.txt")
    if _is_main():
        print(f"[log] reward: {reward_log_path}", flush=True)
    for batch_prompts in loader:
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break

        base_prompts: List[str] = [str(x) for x in batch_prompts]
        G = int(max(1, args.num_generations))
        # Following Janus-Pro R1: sample G results per prompt for within-group comparison.
        prompts: List[str] = [p for p in base_prompts for _ in range(G)]

        # === rollouts (serial) ===
        from latent_rl.rollout.rollout_rl import rollout_one_rl
        from latent_rl.rollout.rollout_sft import rollout_one_sft_once

        ctrl_results_list = []
        base_results_list = []
        pol_mod = policy.module if isinstance(policy, DDP) else policy
        use_reward_baseline = (controller_base is not None) and (
            str(getattr(args, "reward_baseline", "sft_once")) != "none"
        )

        trig_window = int(getattr(controller.cfg.trigger, "window", 0))
        n_tokens = int(args.image_token_num)

        # Explicit nested loops: ensure each base_prompt's G samples use exactly the same text.
        for p_i, p in enumerate(base_prompts):
            for g_i in range(int(G)):
                # Fix seed: make ctrl/base share the same randomness (reduce reward variance; easier comparison).
                sample_seed = (
                    int(args.seed)
                    + int(_rank()) * 1000000
                    + int(step) * 10000
                    + int(p_i) * int(G)
                    + int(g_i)
                )

                # Ctrl branch (policy sampling)
                res_ctrl = rollout_one_rl(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            controller=controller,
                    policy=pol_mod,
                    prompt=p,
                    seed=int(sample_seed),
            cfg=rollout_cfg,
                    force_action=None,
        )
                ctrl_results_list.append(res_ctrl)

                if use_reward_baseline:
                    # Baseline injection point: randomly pick a token constrained by window (no hard-coded 64).
                    rng = random.Random(int(sample_seed) + 13579)
                    low = int(max(1, trig_window))
                    high_excl = int(max(low + 1, n_tokens - max(1, trig_window)))
                    inj = int(rng.randrange(low, high_excl))
                    res_base = rollout_one_sft_once(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
                        controller=controller_base,
                        prompt=p,
                        seed=int(sample_seed),
            cfg=rollout_cfg,
                        sft_inj_token_idx=int(inj),
                    )
                    base_results_list.append(res_base)

        avg_ps = [res["avg_p"] for res in ctrl_results_list]
        sum_logps = [res["sum_logprob"] for res in ctrl_results_list]
        sum_ents = [res["sum_entropy"] for res in ctrl_results_list]
        trig_counts = torch.tensor([int(res.get("n_triggers", 0)) for res in ctrl_results_list], device=device, dtype=torch.float32)
        trig_decisions = torch.tensor([int(res.get("n_decisions", 0)) for res in ctrl_results_list], device=device, dtype=torch.float32)
        trig_rates = trig_counts / torch.clamp(trig_decisions, min=1.0)
        
        all_ids_ctrl = torch.cat([res["image_ids"] for res in ctrl_results_list], dim=0)
        all_ids_base = None
        if use_reward_baseline:
            all_ids_base = torch.cat([res["image_ids"] for res in base_results_list], dim=0)

        # === decoding ===
        images_ctrl = morph.decode_image_tokens_to_pil(all_ids_ctrl)
        images_base = None
        if all_ids_base is not None:
            images_base = morph.decode_image_tokens_to_pil(all_ids_base)

        # === reward ===
        # Align with SFT: include a baseline comparison (default baseline = SFT-style single-injection inference).
        R_abs_ctrl, info_ctrl = reward.score(images=images_ctrl, prompts=prompts)  # [B] cpu
        if images_base is not None:
            R_abs_base, info_base = reward.score(images=images_base, prompts=prompts)  # [B] cpu
            R_abs = R_abs_ctrl - R_abs_base
            z = torch.zeros_like(R_abs_ctrl)
            info = {
                "total": R_abs,
                "clip": (info_ctrl.get("clip", z) - info_base.get("clip", z)),
                "hps": (info_ctrl.get("hps", z) - info_base.get("hps", z)),
            }
        else:
            R_abs = R_abs_ctrl
            info = info_ctrl
        R = R_abs.to(torch.float32).to(device)

        # write reward.txt (rank0 only): log rank0-local mean (avoid DDP collectives here)
        if _is_main():
            r_rl = float(R_abs_ctrl.to(torch.float32).mean().item())
            r_sft = float(R_abs_base.to(torch.float32).mean().item()) if images_base is not None else float("nan")
            r_d = float(R_abs.to(torch.float32).mean().item())
            _append_reward_log(path=reward_log_path, step=int(step), reward_rl=r_rl, reward_sft=r_sft, reward_d=r_d)

        # Hinge penalty (adaptive)
        avg_p_t = torch.stack(avg_ps, dim=0).to(device)
        B0 = int(len(base_prompts))
        avg_p_g = avg_p_t.view(B0, G).mean(dim=1, keepdim=True)  # [B0,1]
        p_ref_t = avg_p_g.repeat(1, G).view(-1)  # [B0*G]
        penalty = float(args.penalty_lambda) * torch.clamp(avg_p_t - p_ref_t, min=0.0)
        R_prime = R - penalty

        # === GRPO advantage (group-normalize by prompt) ===
        B0 = len(base_prompts)
        grp = R_prime.view(B0, G)
        mean_g = grp.mean(dim=1, keepdim=True)
        std_g = grp.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(getattr(args, "adv_std_min", 0.05)))
        A = ((grp - mean_g) / std_g).view(-1).to(torch.float32)

        # Align with DanceGRPO: add advantage clipping.
        adv_clip = float(getattr(args, "adv_clip", 5.0))
        A = torch.clamp(A, min=-adv_clip, max=adv_clip)

        # Policy gradient loss (GRPO / REINFORCE style)
        logp_sum = torch.stack(sum_logps, dim=0).to(device)  # [B0*G]
        ent_sum = torch.stack(sum_ents, dim=0).to(device)    # [B0*G]

        # Normalize by decision count.
        denom = trig_decisions.to(device).clamp(min=1.0)     # [B0*G]
        logp_avg = logp_sum / denom
        ent_avg = ent_sum / denom

        A_det = A.detach()
        loss_pg = -(A_det * logp_avg).mean()
        loss_ent = -float(args.entropy_coef) * ent_avg.mean()
        loss = loss_pg + loss_ent

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            if _is_main():
                print("[warn] loss is NaN/Inf, skip step", flush=True)
        else:
            loss.backward()
            grad_clip = float(args.grad_clip)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            for p in params:
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            if ema_enabled and policy_ema is not None and condenser_ema is not None:
                policy_ema.update(policy_mod)
                condenser_ema.update(condenser_mod)

        # === logging ===
        local_B0 = int(len(base_prompts))
        global_B0 = _dist_sum_int(local_B0)

        if _is_main():
            global_prompts_seen += int(global_B0)
            print(
                f"[step {step}] "
                f"R_mean={float(R.mean()):.4f} "
                f"clip={float(info.get('clip', torch.zeros_like(R)).mean()):.4f} "
                f"hps={float(info.get('hps', torch.zeros_like(R)).mean()):.4f} "
                f"group_trigger_cnt={float(trig_counts.mean().item()):.4f}",
                flush=True,
            )

            # === checkpoint saving ===
            cur_step = int(step) + 1
            done = int(global_prompts_seen)
            save_latest_every_steps = int(getattr(args, "save_latest_every_steps", 1))
            if save_latest_every_steps > 0 and (cur_step % int(save_latest_every_steps) == 0):
                _save_ckpt_latest_only(
                    out_dir=str(args.out_dir),
                    step=int(cur_step),
                    global_prompts_seen=int(done),
                    policy_mod=policy_mod,
                    condenser_mod=condenser_mod,
                    opt=opt,
                    policy_ema=policy_ema if ema_enabled else None,
                    condenser_ema=condenser_ema if ema_enabled else None,
                    args_dict=vars(args),
                )
                if bool(ulm_lora_cfg.enabled):
                    try:
                        save_ulm_lora(language_model=model.language_model, out_dir=os.path.join(str(args.out_dir), "ulm_lora_latest"))
                    except Exception as e:
                        print(f"[save] warning: failed to save ULM LoRA latest: {e}", flush=True)

            save_every_steps = int(getattr(args, "save_every_steps", 0))
            if save_every_steps > 0 and next_ckpt_step_at > 0 and cur_step >= int(next_ckpt_step_at):
                fname = f"ckpt_step_{int(cur_step):08d}.pt"
                path, _latest = _save_ckpt(
                    out_dir=str(args.out_dir),
                    filename=fname,
                    step=int(cur_step),
                    global_prompts_seen=int(done),
                    policy_mod=policy_mod,
                    condenser_mod=condenser_mod,
                    opt=opt,
                    policy_ema=policy_ema if ema_enabled else None,
                    condenser_ema=condenser_ema if ema_enabled else None,
                    args_dict=vars(args),
                )
                print(f"[ckpt] saved {path}", flush=True)
                if bool(ulm_lora_cfg.enabled):
                    try:
                        save_ulm_lora(
                            language_model=model.language_model,
                            out_dir=os.path.join(str(args.out_dir), f"ulm_lora_step_{int(cur_step):08d}"),
                        )
                    except Exception as e:
                        print(f"[save] warning: failed to save ULM LoRA step adapter: {e}", flush=True)
                while cur_step >= int(next_ckpt_step_at):
                    next_ckpt_step_at += int(save_every_steps)

        step += 1

    # === training finished ===
    if _is_main():
        print(f"[done] Training finished: {step} steps, {global_prompts_seen} prompts processed", flush=True)

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM

# --- ensure Janus-Pro + LatentMorph packages are importable ---
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_JANUS_PRO_DIR = os.path.join(_REPO_ROOT, "Janus-Pro")
for _p in (_REPO_ROOT, _JANUS_PRO_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from janus.models import VLChatProcessor  # noqa: E402
from latent_control.controller import LatentController  # noqa: E402
from latent_control.trigger import PolicyTrigger, TriggerPolicyConfig  # noqa: E402
from latent_rl.rollout.rollout_rl import rollout_one_rl  # noqa: E402
from latent_rl.rollout.rollout_utils import RolloutConfig  # noqa: E402
from latent_sft.models.config_io import build_latent_controller_config, load_json_config  # noqa: E402
from latent_sft.models.latent_morph import LatentMorph  # noqa: E402
from ulm_lora_control import load_ulm_lora  # noqa: E402


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    s = s.replace(".", " ")
    s = re.sub(r"[\\\\/:*?\"<>|]", "", s)
    s = re.sub(r"[^0-9a-zA-Z \-(),]+", "", s)
    s = s.strip().rstrip(" .-_")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "prompt"
    if len(s) > int(max_len):
        s = s[: int(max_len)].rstrip(" .-_")
    return s


def _load_controller_ckpt_dict(*, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    state_dict = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "controller" in state_dict:
        state_dict = state_dict["controller"]

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported controller ckpt format at {ckpt_path}")

    ctrl_dict: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("controller."):
            ctrl_dict[k[len("controller.") :]] = v
        else:
            ctrl_dict[str(k)] = v
    return ctrl_dict


def _load_rl_ckpt(path: str, *, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(str(path), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"RL ckpt must be a dict, got {type(ckpt)} at {path}")
    return ckpt


def _pick_state(ckpt: Dict[str, Any], *, name: str, prefer_ema: bool) -> Dict[str, Any]:
    if prefer_ema and f"{name}_ema" in ckpt and isinstance(ckpt[f"{name}_ema"], dict):
        sd = ckpt[f"{name}_ema"].get("shadow", None)
        if isinstance(sd, dict) and sd:
            return sd
    sd2 = ckpt.get(name, None)
    if not isinstance(sd2, dict):
        raise ValueError(f"RL ckpt missing state_dict '{name}'")
    return sd2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="", help="single prompt (mutually exclusive with --prompts_file)")
    ap.add_argument(
        "--prompts_file",
        type=str,
        default="",
        help="txt file, one prompt per line (mutually exclusive with --prompt)",
    )
    ap.add_argument("--out", type=str, default="", help="output .png path OR a directory (single-prompt mode)")
    ap.add_argument("--out_dir", type=str, default="", help="output directory (batch mode)")

    ap.add_argument("--controller_ckpt", type=str, required=True, help="base controller ckpt (from SFT)")
    ap.add_argument("--rl_ckpt", type=str, required=True, help="RL ckpt_latest.pt or ckpt_step_XXXXXXXX.pt")
    ap.add_argument("--use_ema", type=int, default=1, help="if RL ckpt contains EMA, load EMA weights (default=1)")

    ap.add_argument("--config", type=str, default="latent_rl/config.json")
    ap.add_argument("--model_path", type=str, default="", help="override cfg.model_path if set")
    ap.add_argument("--model_local_files_only", type=int, default=1)
    ap.add_argument("--lora_control", type=int, default=0, help="1=enable ULM LoRA; requires --ulm_weights")
    ap.add_argument("--ulm_weights", type=str, default="", help="ULM LoRA adapter directory (required if --lora_control=1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seed_mode",
        type=str,
        default="offset",
        choices=["fixed", "offset"],
        help="fixed: all prompts use --seed; offset: use --seed + idx for reproducibility & diversity",
    )
    ap.add_argument("--device", type=str, default="cuda")

    # rollout hyperparams (match training defaults)
    ap.add_argument("--image_token_num", type=int, default=576)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--force_action", type=int, default=-1, help="-1=sample policy; 0=never trigger; 1=always trigger")
    ap.add_argument("--enable_trigger", type=int, default=1)
    ap.add_argument("--max_prompts", type=int, default=0, help="0 = no truncation (batch mode only)")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    prompt = str(args.prompt).strip()
    prompts_file = str(args.prompts_file).strip()
    if bool(prompt) == bool(prompts_file):
        raise SystemExit("You must provide exactly one of: --prompt OR --prompts_file")

    cfg = load_json_config(os.path.abspath(args.config))
    model_path = str(args.model_path).strip() or str(cfg.get("model_path", "deepseek-ai/Janus-Pro-7B"))

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(0)

    out: Path | None = None
    out_dir: Path | None = None
    if prompt:
        if not str(args.out).strip():
            raise SystemExit("single-prompt mode requires --out (a .png path or a directory)")
        out = Path(args.out)
        if out.suffix.lower() != ".png":
            out.mkdir(parents=True, exist_ok=True)
            name = _sanitize_filename(prompt)
            out = out / f"{name}_seed{int(args.seed):d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not str(args.out_dir).strip():
            raise SystemExit("batch mode requires --out_dir")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rl][cfg] config={os.path.abspath(args.config)} model_path={model_path}", flush=True)
    print(f"[rl][io] controller_ckpt={args.controller_ckpt}", flush=True)
    print(f"[rl][io] rl_ckpt={args.rl_ckpt} use_ema={int(args.use_ema)}", flush=True)
    if out is not None:
        print(f"[rl][io] out={str(out)}", flush=True)
    else:
        print(f"[rl][io] out_dir={str(out_dir)}", flush=True)

    # Processor + frozen Janus
    model_local_only = bool(int(args.model_local_files_only))
    try:
        processor = VLChatProcessor.from_pretrained(model_path, local_files_only=model_local_only)
    except TypeError:
        processor = VLChatProcessor.from_pretrained(model_path)
    tok = processor.tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=model_local_only,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # Match model dtype with fp16 inputs_embeds used by rollout sampling loop to avoid Llama SDPA dtype mismatch.
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device)
    model.eval()

    # Optional: load ULM LoRA weights (adapter) for language_model.
    if bool(int(args.lora_control)):
        model.language_model = load_ulm_lora(language_model=model.language_model, ulm_weights=str(args.ulm_weights))
        model.language_model.eval()

    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "eval"):
        lm.eval()

    # Controller (load base SFT weights first)
    latent_cfg = build_latent_controller_config(cfg)
    latent_cfg.enabled = True
    d_model = int(model.language_model.get_input_embeddings().weight.shape[1])
    controller = LatentController(d_model=d_model, tokenizer=tok, cfg=latent_cfg).to(device=device)
    base_ctrl = _load_controller_ckpt_dict(ckpt_path=str(args.controller_ckpt), device=torch.device("cpu"))
    msg = controller.load_state_dict(base_ctrl, strict=False)
    print(f"[rl][init] base controller load msg: {msg}", flush=True)

    # RL ckpt overwrites policy + controller.condenser
    rl_ckpt = _load_rl_ckpt(str(args.rl_ckpt), device=torch.device("cpu"))
    prefer_ema = int(args.use_ema) == 1

    policy = PolicyTrigger(TriggerPolicyConfig()).to(device=device, dtype=torch.float32).eval()
    pol_sd = _pick_state(rl_ckpt, name="policy", prefer_ema=prefer_ema)
    msgp = policy.load_state_dict(pol_sd, strict=False)
    print(f"[rl][init] policy load msg: {msgp}", flush=True)

    cond_sd = _pick_state(rl_ckpt, name="condenser", prefer_ema=prefer_ema)
    msgc = controller.condenser.load_state_dict(cond_sd, strict=False)
    print(f"[rl][init] condenser load msg: {msgc}", flush=True)

    controller = controller.to(dtype=torch.float32).eval()

    # Morph (rollout uses its internals, plus controller.observe_trigger_inputs)
    stage_prompt_cfg = cfg.get("stage_prompt", {}) if isinstance(cfg.get("stage_prompt", {}), dict) else {}
    morph = LatentMorph(
        frozen_model=model,
        processor=processor,
        tokenizer=tok,
        controller=controller,
        img_size=int(cfg.get("img_size", 384)),
        patch_size=int(cfg.get("patch_size", 16)),
        image_token_num=int(args.image_token_num),
        stages=int(cfg.get("stages", 1)),
        part_template=str(cfg.get("part_template", "{i}-part")),
        use_understanding=bool(stage_prompt_cfg.get("use_understanding", True)),
        understanding_max_tokens=int(stage_prompt_cfg.get("understanding_max_tokens", 128)),
        cfg_weight=float(args.cfg_weight),
        temperature=float(args.temperature),
    ).to(device=device)
    morph.eval()

    def _force_action() -> Optional[int]:
        if int(args.force_action) < 0:
            return None
        return int(args.force_action)

    rollout_cfg = RolloutConfig(
        image_token_num=int(args.image_token_num),
        cfg_weight=float(args.cfg_weight),
        temperature=float(args.temperature),
        num_rollouts_per_prompt=1,
    )

    if prompt:
        res = rollout_one_rl(
            model=model,
            processor=processor,
            tokenizer=tok,
            controller=controller,
            policy=policy,
            prompt=str(prompt),
            seed=int(args.seed),
            cfg=rollout_cfg,
            enable_trigger=bool(int(args.enable_trigger)),
            force_action=_force_action(),
        )
        image_ids = res["image_ids"]
        pil_list = morph.decode_image_tokens_to_pil(image_ids)
        pil_list[0].save(str(out))
        print(
            f"[rl][done] n_triggers={int(res.get('n_triggers', 0))} "
            f"n_decisions={int(res.get('n_decisions', 0))} steps={res.get('trigger_steps', [])} -> {str(out)}",
            flush=True,
        )
        return

    # Batch mode
    assert prompts_file and out_dir is not None
    lines = Path(prompts_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    ps = [x.strip() for x in lines if x.strip()]
    if int(args.max_prompts) > 0:
        ps = ps[: int(args.max_prompts)]
    total = len(ps)
    print(
        f"[rl][batch] prompts={total} seed_mode={args.seed_mode} "
        f"enable_trigger={int(args.enable_trigger)} force_action={args.force_action}",
        flush=True,
    )
    for i, p in enumerate(ps):
        seed_i = int(args.seed) if str(args.seed_mode) == "fixed" else int(args.seed) + int(i)
        name = _sanitize_filename(p)
        out_i = out_dir / f"{i:06d}_{name}_seed{seed_i}.png"
        res = rollout_one_rl(
            model=model,
            processor=processor,
            tokenizer=tok,
            controller=controller,
            policy=policy,
            prompt=str(p),
            seed=int(seed_i),
            cfg=rollout_cfg,
            enable_trigger=bool(int(args.enable_trigger)),
            force_action=_force_action(),
        )
        image_ids = res["image_ids"]
        pil_list = morph.decode_image_tokens_to_pil(image_ids)
        pil_list[0].save(str(out_i))
        if (i == 0) or ((i + 1) % 10 == 0) or (i + 1 == total):
            print(
                f"[rl][batch] {i+1}/{total} n_triggers={int(res.get('n_triggers', 0))} "
                f"steps={res.get('trigger_steps', [])} -> {str(out_i)}",
                flush=True,
            )


if __name__ == "__main__":
    main()



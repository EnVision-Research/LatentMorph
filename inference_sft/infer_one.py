from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

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
    ap.add_argument("--controller_ckpt", type=str, required=True, help="SFT controller checkpoint (.pt)")
    ap.add_argument("--config", type=str, default="latent_sft/models/config.json")
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
    ap.add_argument("--inj", type=int, default=-1, help="injection token index; -1 = random (default training behavior)")
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

    # Resolve output
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

    print(f"[sft][cfg] config={os.path.abspath(args.config)} model_path={model_path}", flush=True)
    print(f"[sft][io] controller_ckpt={args.controller_ckpt}", flush=True)
    if out is not None:
        print(f"[sft][io] out={str(out)}", flush=True)
    else:
        print(f"[sft][io] out_dir={str(out_dir)}", flush=True)

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
    # Important: keep model dtype consistent with fp16 inputs_embeds in LatentMorph sampling loop.
    # Otherwise Llama SDPA may see query=float32 but attn_mask=float16 and crash.
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

    # Controller
    latent_cfg = build_latent_controller_config(cfg)
    latent_cfg.enabled = True
    d_model = int(model.language_model.get_input_embeddings().weight.shape[1])
    controller = LatentController(d_model=d_model, tokenizer=tok, cfg=latent_cfg).to(device=device)
    ctrl_dict = _load_controller_ckpt_dict(ckpt_path=str(args.controller_ckpt), device=torch.device("cpu"))
    msg = controller.load_state_dict(ctrl_dict, strict=False)
    print(f"[sft][init] controller load msg: {msg}", flush=True)
    controller = controller.to(dtype=torch.float32).eval()

    # Morph (uses cfg defaults to match training)
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), dict) else {}
    stage_prompt_cfg = cfg.get("stage_prompt", {}) if isinstance(cfg.get("stage_prompt", {}), dict) else {}
    morph = LatentMorph(
        frozen_model=model,
        processor=processor,
        tokenizer=tok,
        controller=controller,
        img_size=int(cfg.get("img_size", 384)),
        patch_size=int(cfg.get("patch_size", 16)),
        image_token_num=int(cfg.get("image_token_num", 576)),
        stages=int(cfg.get("stages", 1)),
        part_template=str(cfg.get("part_template", "{i}-part")),
        use_understanding=bool(stage_prompt_cfg.get("use_understanding", True)),
        understanding_max_tokens=int(stage_prompt_cfg.get("understanding_max_tokens", 128)),
        cfg_weight=float(generation_cfg.get("cfg_weight", 5.0)),
        temperature=float(generation_cfg.get("temperature", 1.0)),
    ).to(device=device)
    morph.eval()

    inj = None if int(args.inj) < 0 else int(args.inj)
    if prompt:
        image_ids, inj_used = morph.generate_image_tokens(base_prompt=str(prompt), seed=int(args.seed), inj=inj)
        pil_list = morph.decode_image_tokens_to_pil(image_ids)
        pil_list[0].save(str(out))
        print(f"[sft][done] inj={int(inj_used)} -> {str(out)}", flush=True)
        return

    # Batch mode
    assert prompts_file and out_dir is not None
    lines = Path(prompts_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    ps = [x.strip() for x in lines if x.strip()]
    if int(args.max_prompts) > 0:
        ps = ps[: int(args.max_prompts)]
    total = len(ps)
    print(f"[sft][batch] prompts={total} seed_mode={args.seed_mode} inj={inj if inj is not None else 'random'}", flush=True)
    for i, p in enumerate(ps):
        seed_i = int(args.seed) if str(args.seed_mode) == "fixed" else int(args.seed) + int(i)
        name = _sanitize_filename(p)
        out_i = out_dir / f"{i:06d}_{name}_seed{seed_i}.png"
        image_ids, inj_used = morph.generate_image_tokens(base_prompt=str(p), seed=int(seed_i), inj=inj)
        pil_list = morph.decode_image_tokens_to_pil(image_ids)
        pil_list[0].save(str(out_i))
        if (i == 0) or ((i + 1) % 10 == 0) or (i + 1 == total):
            print(f"[sft][batch] {i+1}/{total} inj={int(inj_used)} -> {str(out_i)}", flush=True)


if __name__ == "__main__":
    main()



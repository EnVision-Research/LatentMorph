from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch


@dataclass
class UlmLoraConfig:
    """
    LoRA config for ULM (Janus-Pro language_model).

    Defaults are intentionally conservative to avoid changing behavior unless explicitly enabled.
    """

    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    # LLaMA-style common target modules.
    target_modules: Optional[List[str]] = None


def _default_target_modules() -> List[str]:
    # Works for LLaMA-like architectures.
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def unwrap_module(m):
    """
    Unwrap common wrappers (e.g., DDP/FSDP) to get the underlying module.
    """
    cur = m
    # DDP/FSDP expose `.module`; unwrap repeatedly (defensive).
    for _ in range(4):
        if hasattr(cur, "module"):
            try:
                cur = cur.module
                continue
            except Exception:
                break
        break
    return cur


def _is_peft_model(m) -> bool:
    # Avoid importing peft at import time; detect by attribute.
    mm = unwrap_module(m)
    return hasattr(mm, "peft_config") and hasattr(mm, "base_model")


def get_lm_backbone(language_model):
    """
    Return the module used for embedding-forward:
      - Janus-Pro default: LlamaForCausalLM has `.model` (backbone)
      - PEFT-wrapped: try to unwrap to original model, then return `.model`

    The returned object must accept `inputs_embeds=...` with optional
    `past_key_values`, `use_cache`, and `position_ids` (as used in LatentMorph code).
    """
    lm = unwrap_module(language_model)

    # PeftModelForCausalLM: base_model.model is typically the original *ForCausalLM model.
    try:
        if _is_peft_model(lm):
            base = unwrap_module(getattr(lm, "base_model", lm))
            if hasattr(base, "model"):
                lm = base.model
    except Exception:
        pass

    # LlamaForCausalLM exposes `.model` as the backbone.
    if hasattr(lm, "model"):
        return lm.model
    return lm


def enable_ulm_lora(*, language_model, cfg: UlmLoraConfig):
    """
    Enable LoRA on the ULM language model (Janus-Pro language_model).
    Returns the possibly wrapped language_model.

    Notes:
    - This function is a no-op when cfg.enabled is False.
    - Requires `peft` only when enabled.
    """
    if not bool(cfg.enabled):
        return language_model

    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "You enabled --lora_control but `peft` is not installed in the current environment. "
            "Please install `peft` (or disable --lora_control) and try again."
        ) from e

    target_modules = list(cfg.target_modules) if cfg.target_modules else _default_target_modules()
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.r),
        lora_alpha=int(cfg.alpha),
        lora_dropout=float(cfg.dropout),
        target_modules=target_modules,
        bias="none",
    )
    # Wrap the *ForCausalLM* so embeddings/get_input_embeddings remain available.
    return get_peft_model(language_model, peft_cfg)


def iter_trainable_params(module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if getattr(p, "requires_grad", False)]


def save_ulm_lora(*, language_model, out_dir: str):
    """
    Save PEFT adapter to a directory. No-op if language_model is not a PEFT model.
    """
    lm = unwrap_module(language_model)
    if not _is_peft_model(lm):
        return
    os.makedirs(out_dir, exist_ok=True)
    # PeftModel.save_pretrained writes adapter_config.json + adapter_model.safetensors/bin
    lm.save_pretrained(out_dir)


def load_ulm_lora(*, language_model, ulm_weights: str):
    """
    Load LoRA/PEFT adapter weights into language_model.

    `ulm_weights` supports:
    - a PEFT adapter directory saved by `save_pretrained()`
    """
    if not str(ulm_weights).strip():
        raise ValueError("When --lora_control is enabled, you must provide --ulm_weights (a PEFT adapter directory path).")

    try:
        from peft import PeftModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "You enabled --lora_control but `peft` is not installed, so --ulm_weights cannot be loaded."
        ) from e

    if not os.path.isdir(str(ulm_weights)):
        raise ValueError(f"--ulm_weights must be a directory (a PEFT adapter), but got: {ulm_weights}")

    lm = unwrap_module(language_model)
    return PeftModel.from_pretrained(lm, str(ulm_weights), is_trainable=False)


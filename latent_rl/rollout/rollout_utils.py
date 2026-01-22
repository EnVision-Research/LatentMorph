from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from latent_sft.models.prompt import build_cfg_prompt_embeds


@dataclass
class RolloutConfig:
    image_token_num: int = 576
    cfg_weight: float = 5.0
    temperature: float = 1.0
    # How many rollouts to generate per prompt.
    num_rollouts_per_prompt: int = 1


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_kv_cache(past_key_values):
    # transformers>=4.36: may return a legacy tuple cache; normalize to DynamicCache for consistent access.
    from transformers.cache_utils import DynamicCache  # type: ignore

    if isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


def lm_forward(lm, *, inputs_embeds, past_key_values, use_cache: bool, position_ids=None):
    return lm(
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


@torch.no_grad()
def compute_ctrl_delta_sft_like(
    *,
    controller,
    model,
    prompt_text_for_think: str,
    prompt_vec: torch.Tensor,  # [B,D]
    long_img: torch.Tensor,  # [B,S,D]
) -> torch.Tensor:
    """
    Compute the injection signal to strictly match the SFT path:
    long_condenser -> think_latent(no_grad) -> translator -> shaper -> ctrl_tokens.mean -> ctrl_delta
    Returns ctrl_delta: [2B,1,D] fp16
    """
    ctrl_dtype = next(controller.long_condenser.parameters()).dtype
    long_m_tokens, long_m_vec = controller.long_condenser(long_img.to(dtype=ctrl_dtype))
    z_vec = controller._think_latent(
        model=model,
        prompt_text=str(prompt_text_for_think),
        m_tokens=long_m_tokens.detach(),
    )
    c_vec = controller.translator(
        z_vec=z_vec.to(dtype=ctrl_dtype),
        m_vec=long_m_vec.to(dtype=ctrl_dtype),
        p_vec=prompt_vec.to(dtype=ctrl_dtype),
    )
    ctrl_tokens = controller.shaper.make_control_tokens_for_cfg(c_vec).to(torch.float16)  # [2B,K,D]
    ctrl_delta = ctrl_tokens.mean(dim=1, keepdim=True).to(torch.float16)  # [2B,1,D]
    return ctrl_delta


def build_prompt(
    *,
    model,
    processor,
    tokenizer,
    prompt: str,
    bsz: int,
    device,
):
    gen_prompt_ids: List[int] = tokenizer.encode(str(prompt))
    prompt_embeds, prompt_vec = build_cfg_prompt_embeds(model, processor, tokenizer, gen_prompt_ids, bsz, device)
    prompt_len = int(prompt_embeds.shape[1])
    return prompt_embeds, prompt_vec, prompt_len



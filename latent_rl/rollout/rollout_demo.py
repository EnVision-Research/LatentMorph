from __future__ import annotations

from typing import Dict

import torch

from latent_control.trigger import PolicyTrigger
from ulm_lora_control import get_lm_backbone

from .rollout_rl import rollout_one_rl
from .rollout_sft import rollout_one_sft_once
from .rollout_utils import RolloutConfig, build_prompt, ensure_kv_cache, lm_forward, set_seed


@torch.no_grad()
def rollout_one_base(
    *,
    model,
    processor,
    tokenizer,
    prompt: str,
    seed: int,
    cfg: RolloutConfig,
) -> Dict:
    """
    Base branch: no injection, no observation (no controller); only generates image tokens.
    """
    set_seed(int(seed))

    device = next(model.parameters()).device
    bsz = 1
    n_tokens = int(cfg.image_token_num)

    prompt_embeds, _prompt_vec, prompt_len = build_prompt(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        prompt=str(prompt),
        bsz=bsz,
        device=device,
    )

    lm = get_lm_backbone(model.language_model)
    inputs_embeds = prompt_embeds.to(torch.float16)
    gen_ids = torch.empty((bsz, n_tokens), device=device, dtype=torch.long)

    # token0
    out = lm_forward(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
    pkv = ensure_kv_cache(out.past_key_values)
    logits2 = model.gen_head(out.last_hidden_state[:, -1, :])  # [2,V]
    logit_cond = logits2[0::2]
    logit_uncond = logits2[1::2]
    logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)  # [1,V]
    probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1).to(torch.long)  # [1,1]
    gen_ids[:, 0] = next_tok.squeeze(-1)

    next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
    inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

    for t in range(1, n_tokens):
        pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
        out = lm_forward(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos)
        pkv = ensure_kv_cache(out.past_key_values)
        hidden_last = out.last_hidden_state[:, -1, :]

        logits2 = model.gen_head(hidden_last)
        logit_cond = logits2[0::2]
        logit_uncond = logits2[1::2]
        logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)

        next_tok = torch.multinomial(probs, num_samples=1).to(torch.long)
        gen_ids[:, t] = next_tok.squeeze(-1)
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

    return {"image_ids": gen_ids.to(torch.long)}


@torch.no_grad()
def rollout_one_demo_triplet(
    *,
    model,
    processor,
    tokenizer,
    controller,
    policy: PolicyTrigger,
    prompt: str,
    seed: int,
    cfg: RolloutConfig,
    sft_inj_token_idx: int,
) -> dict:
    """
    Demo: generate three images with the same seed for easy comparison:
    - base: no injection
    - sft: a single forced injection at a fixed token_idx
    - rl: sample actions from the policy at check steps; may inject multiple times
    """
    base = rollout_one_base(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        prompt=str(prompt),
        seed=int(seed),
        cfg=cfg,
    )
    sft = rollout_one_sft_once(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        controller=controller,
        prompt=str(prompt),
        seed=int(seed),
        cfg=cfg,
        sft_inj_token_idx=int(sft_inj_token_idx),
    )
    rl = rollout_one_rl(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        controller=controller,
        policy=policy,
        prompt=str(prompt),
        seed=int(seed),
        cfg=cfg,
        enable_trigger=True,
        force_action=None,
    )
    return {"base": base, "sft": sft, "rl": rl}



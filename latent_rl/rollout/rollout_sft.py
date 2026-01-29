from __future__ import annotations

from typing import Optional

import torch

from ulm_lora_control import get_lm_backbone

from .rollout_utils import (
    RolloutConfig,
    build_prompt,
    compute_ctrl_delta_sft_like,
    ensure_kv_cache,
    lm_forward,
    set_seed,
)


@torch.no_grad()
def rollout_one_sft_once(
    *,
    model,
    processor,
    tokenizer,
    controller,
    prompt: str,
    seed: int,
    cfg: RolloutConfig,
    sft_inj_token_idx: int,
) -> dict:
    """
    SFT-style baseline rollout (single forced injection; no trigger/policy).
    Used as the RL reward baseline: R = R_ctrl - R_base
    """
    set_seed(int(seed))

    device = next(model.parameters()).device
    bsz = 1
    n_tokens = int(cfg.image_token_num)

    controller.reset(batch_size=bsz, device=device)

    prompt_embeds, prompt_vec, prompt_len = build_prompt(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        prompt=str(prompt),
        bsz=bsz,
        device=device,
    )

    lm = get_lm_backbone(model.language_model)
    pkv = None
    inputs_embeds = prompt_embeds.to(torch.float16)
    gen_ids = torch.empty((bsz, n_tokens), device=device, dtype=torch.long)

    did_inject = False

    # token0
    out = lm_forward(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
    pkv = ensure_kv_cache(out.past_key_values)
    logits2 = model.gen_head(out.last_hidden_state[:, -1, :])
    logit_cond = logits2[0::2]
    logit_uncond = logits2[1::2]
    logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)
    probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1).to(torch.long)
    gen_ids[:, 0] = next_tok.squeeze(-1)

    next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
    inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

    for t in range(1, n_tokens):
        pkv_prev = pkv
        pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
        out = lm_forward(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos)
        pkv = ensure_kv_cache(out.past_key_values)
        hidden_last = out.last_hidden_state[:, -1, :]

        logits2 = model.gen_head(hidden_last)
        logit_cond = logits2[0::2]
        logit_uncond = logits2[1::2]
        logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)

        # Baseline: always advance the long buffer so we can inject at any t.
        long_img = controller._push_img_long_hidden(hidden_last[0::2])

        if (not did_inject) and (t == int(sft_inj_token_idx)):
            ctrl_delta = compute_ctrl_delta_sft_like(
                controller=controller,
                model=model,
                prompt_text_for_think=str(prompt),
                prompt_vec=prompt_vec,
                long_img=long_img,
            )
            out2 = lm_forward(
                lm,
                inputs_embeds=(inputs_embeds + ctrl_delta.to(dtype=inputs_embeds.dtype)),
                use_cache=True,
                past_key_values=pkv_prev,
                position_ids=pos,
            )
            pkv = ensure_kv_cache(out2.past_key_values)
            hidden_last = out2.last_hidden_state[:, -1, :]
            logits2 = model.gen_head(hidden_last)
            logit_cond = logits2[0::2]
            logit_uncond = logits2[1::2]
            logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)
            controller._triggers_used += 1
            did_inject = True

        next_tok = torch.multinomial(probs, num_samples=1).to(torch.long)
        gen_ids[:, t] = next_tok.squeeze(-1)
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

    return {
        "image_ids": gen_ids.to(torch.long),
        "did_inject": bool(did_inject),
        "sft_inj_token_idx": int(sft_inj_token_idx),
    }



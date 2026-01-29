from __future__ import annotations

from typing import List, Optional

import torch

from latent_control.trigger import PolicyTrigger
from ulm_lora_control import get_lm_backbone

from .rollout_utils import (
    RolloutConfig,
    build_prompt,
    compute_ctrl_delta_sft_like,
    ensure_kv_cache,
    lm_forward,
    set_seed,
)


def rollout_one_rl(
    *,
    model,
    processor,
    tokenizer,
    controller,
    policy: PolicyTrigger,
    prompt: str,
    seed: Optional[int] = None,
    cfg: RolloutConfig,
    enable_trigger: bool = True,
    force_action: Optional[int] = None,  # None=sample, 0=never trigger, 1=always trigger (at check steps)
) -> dict:
    """
    One RL rollout:
    - The controller provides observations (check-step filtering + features x_t + long_img)
    - The policy decides the action a_t (whether to inject)
    - If a_t=1, we compute ctrl_delta in SFT style and re-run the forward pass for this step to
      override the KV cache, so the injection takes effect immediately.
    """
    set_seed(seed)

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
    inputs_embeds = prompt_embeds.to(torch.float16)  # [2,T,D]
    gen_ids = torch.empty((bsz, n_tokens), device=device, dtype=torch.long)

    sum_p = torch.zeros((), device=device, dtype=torch.float32)
    sum_logprob = torch.zeros((), device=device, dtype=torch.float32)
    sum_entropy = torch.zeros((), device=device, dtype=torch.float32)
    n_decisions = 0
    n_triggers = 0
    trigger_steps: list[int] = []

    # token0
    with torch.no_grad():
        out = lm_forward(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
    pkv = ensure_kv_cache(out.past_key_values)
    with torch.no_grad():
        logits2 = model.gen_head(out.last_hidden_state[:, -1, :])  # [2,V]
        logit_cond = logits2[0::2]
        logit_uncond = logits2[1::2]
        logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)  # [1,V]
        probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).to(torch.long)  # [1,1]
    gen_ids[:, 0] = next_tok.squeeze(-1)

    next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)  # [2]
    with torch.no_grad():
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)  # [2,1,D]

    # token1..N-1
    trig_check_every = int(getattr(controller.cfg.trigger, "check_every", 1))
    for t in range(1, n_tokens):
        pkv_prev = pkv
        pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
        with torch.no_grad():
            out = lm_forward(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos)
        pkv = ensure_kv_cache(out.past_key_values)

        hidden_last = out.last_hidden_state[:, -1, :]  # [2,D]
        with torch.no_grad():
            logits2 = model.gen_head(hidden_last)  # [2,V]
            logit_cond = logits2[0::2]
            logit_uncond = logits2[1::2]
            logits = logit_uncond + float(cfg.cfg_weight) * (logit_cond - logit_uncond)  # [1,V]
            probs = torch.softmax(logits / float(max(1e-6, cfg.temperature)), dim=-1)

        # Do not check/trigger in the last check interval
        # (e.g., if check_every=64, skip observing/triggering for the last 64 tokens).
        if int(t) > int(n_tokens - trig_check_every):
            obs = None
        elif bool(enable_trigger):
            obs = controller.observe_trigger_inputs(
                step_idx=int(t - 1),
                prompt_vec=prompt_vec,
                h_img_last_cond=hidden_last[0::2],
                next_token_probs=probs,
            )
        else:
            obs = None

        if obs is not None:
            x_t = obs["x_t"].to(dtype=torch.float32)
            logits_pol = policy.forward(x_t)
            dist = torch.distributions.Bernoulli(logits=logits_pol)
            p_t = dist.probs
            ent = dist.entropy()

            if force_action is None:
                a = dist.sample()
                logp = dist.log_prob(a)
            else:
                a = torch.full_like(p_t, float(int(force_action)))
                logp = dist.log_prob(a).detach() * 0.0

            sum_p = sum_p + p_t.mean().to(dtype=torch.float32)
            sum_logprob = sum_logprob + logp.mean().to(dtype=torch.float32)
            sum_entropy = sum_entropy + ent.mean().to(dtype=torch.float32)
            n_decisions += 1

            if bool((a > 0.5).any()):
                ctrl_delta = compute_ctrl_delta_sft_like(
                    controller=controller,
                    model=model,
                    prompt_text_for_think=str(prompt),
                    prompt_vec=prompt_vec,
                    long_img=obs["long_img"],
                )
                with torch.no_grad():
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
                n_triggers += 1
                trigger_steps.append(int(t - 1))
                controller.mark_injected(mask=torch.ones((1,), device=device, dtype=torch.bool))

        with torch.no_grad():
            next_tok = torch.multinomial(probs, num_samples=1).to(torch.long)
        gen_ids[:, t] = next_tok.squeeze(-1)
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        with torch.no_grad():
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

    avg_p = (sum_p / max(1, n_decisions)) if n_decisions > 0 else torch.zeros((), device=device, dtype=torch.float32)
    return {
        "image_ids": gen_ids.to(torch.long),
        "avg_p": avg_p,
        "sum_logprob": sum_logprob,
        "sum_entropy": sum_entropy,
        "n_decisions": int(n_decisions),
        "n_triggers": int(n_triggers),
        "trigger_steps": trigger_steps,
    }


def rollout_batch_rl(
    *,
    model,
    processor,
    tokenizer,
    controller,
    policy: PolicyTrigger,
    prompts: List[str],
    seeds: List[int],
    cfg: RolloutConfig,
    force_actions: Optional[torch.Tensor] = None,  # [B] or None
) -> dict:
    """
    Batched RL rollout (kept for future optimization; current training primarily uses serial `rollout_one_rl`).
    """
    raise NotImplementedError("rollout_batch_rl not used in current training; keep rollout_one_rl for stability")



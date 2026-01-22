from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from janus.models import MultiModalityCausalLM, VLChatProcessor


def stage_part_name(part_template: str, stage_idx: int) -> str:
    return str(part_template).format(i=stage_idx + 1, idx=stage_idx, stage=stage_idx + 1)


def truncate_ids(ids: List[int], max_tokens: int) -> List[int]:
    if int(max_tokens) <= 0:
        return []
    return ids[: int(max_tokens)]


def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    for j in range(0, len(haystack) - n + 1):
        if haystack[j : j + n] == needle:
            return j
    return -1


@torch.no_grad()
def vec_to_token_ids(model: MultiModalityCausalLM, vec: torch.Tensor, k: int) -> List[int]:
    if int(k) <= 0:
        return []
    emb_w = model.language_model.get_input_embeddings().weight.detach()  # [V,D]
    v = F.normalize(vec.to(emb_w.dtype), dim=-1)  # [D]
    w = F.normalize(emb_w, dim=-1)  # [V,D]
    sims = torch.matmul(w, v)  # [V]
    topk = torch.topk(sims, k=min(int(k), int(sims.numel())), dim=0).indices
    return topk.to(torch.long).tolist()


@torch.no_grad()
def _sample_text_ids_with_kv(
    *,
    model: MultiModalityCausalLM,
    inputs_embeds: torch.Tensor,  # [1, T, D]
    attention_mask: Optional[torch.Tensor],  # [1, T] or None
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
) -> List[int]:
    """
    Replacement for `.generate()`: call the internal forward directly and explicitly maintain
    past_key_values (preserve KV cache).
    Only used for the understanding prompt (no_grad); no training backprop.
    """
    lm = model.language_model  # LlamaForCausalLM

    # HF: gradient checkpointing + training may force use_cache=False, which breaks our manual KV sampling loop.
    # Temporarily switch to eval() to ensure use_cache=True takes effect; restore the original training state after.
    prev_training = bool(getattr(lm, "training", False))
    if prev_training:
        lm.eval()

    past_key_values = None
    cur_embeds = inputs_embeds
    cur_attn = attention_mask
    out_ids: List[int] = []

    try:
        for _ in range(int(max_new_tokens)):
            out = lm(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]  # [1,V]
            if float(temperature) != 1.0:
                logits = logits / float(temperature)
            probs = torch.softmax(logits, dim=-1)  # [1,V]
            next_id = int(torch.multinomial(probs[0], num_samples=1).item())
            out_ids.append(next_id)
            if next_id == int(eos_token_id):
                break

            next_id_t = torch.tensor([[next_id]], device=inputs_embeds.device, dtype=torch.long)
            cur_embeds = lm.get_input_embeddings()(next_id_t)  # [1,1,D]
            if cur_attn is not None:
                cur_attn = torch.cat([cur_attn, torch.ones_like(cur_attn[:, :1])], dim=1)
    finally:
        if prev_training:
            lm.train()

    return out_ids


@torch.no_grad()
def understanding_prompt_ids(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    tokenizer,
    base_prompt: str,
    stages: int,
    stage_idx: int,
    pos: str,
    stage_images: Optional[List[Image.Image]],
    understanding_max_tokens: int,
    max_new_tokens: int = 300,
) -> List[int]:
    conversation = [
        {
            "role": "User",
            "content": (
                f"You are a professional artist. We are drawing << {base_prompt} >> in {stages} stages; "
                f"we have finished {stage_idx} stages.\n"
                f"Task: write an optimized, generation-ready prompt for Stage {stage_idx + 1} focusing on the {pos}.\n"
                f"Rules: output ONLY the optimized prompt. No explanation, no bullet points."
            ),
        },
        {"role": "Assistant", "content": ""},
    ]

    if stage_images:
        prepare_inputs = processor(conversations=conversation, images=stage_images, force_batchify=True).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs).to(torch.float16)
        attention_mask = prepare_inputs.attention_mask
    else:
        input_ids_text = torch.LongTensor(tokenizer.encode(conversation[0]["content"])).unsqueeze(0).to(model.device)
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids_text).to(torch.float16)
        attention_mask = None

    ids = _sample_text_ids_with_kv(
        model=model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=int(tokenizer.eos_token_id),
        temperature=1.0,
    )
    return truncate_ids(ids, understanding_max_tokens)


def build_gen_prefix_ids(processor: VLChatProcessor, tokenizer, gen_prompt_ids: List[int]) -> List[int]:
    placeholder = "<<<PROMPT_PLACEHOLDER>>>"
    gen_conv = [
        {"role": "<|User|>", "content": placeholder},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_with_ph = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=gen_conv,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    sft_ids = tokenizer.encode(sft_with_ph)
    ph_ids = tokenizer.encode(placeholder)
    k = find_subsequence(sft_ids, ph_ids)
    if k < 0:
        prefix_ids = sft_ids
        suffix_ids: List[int] = []
    else:
        prefix_ids = sft_ids[:k]
        suffix_ids = sft_ids[k + len(ph_ids) :]
    img_tag_ids = tokenizer.encode(processor.image_start_tag)
    return prefix_ids + list(gen_prompt_ids) + suffix_ids + img_tag_ids


def build_cfg_prompt_embeds(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    tokenizer,
    gen_prompt_ids: List[int],
    bsz: int,
    device: torch.device,
):
    ids = build_gen_prefix_ids(processor, tokenizer, gen_prompt_ids)
    input_ids = torch.tensor(ids, device=device, dtype=torch.long)
    tokens = input_ids.unsqueeze(0).expand(bsz * 2, -1).clone()
    if tokens.shape[1] > 2:
        tokens[1::2, 1:-1] = processor.pad_id
    prompt_embeds = model.language_model.get_input_embeddings()(tokens).to(torch.float16)  # [2B,T,D]
    prompt_vec = prompt_embeds[0::2].mean(dim=1).to(prompt_embeds.dtype)  # [B,D]
    return prompt_embeds, prompt_vec



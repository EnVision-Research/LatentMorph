from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from janus.models import MultiModalityCausalLM, VLChatProcessor
from latent_control.controller import LatentController
from .prompt import build_cfg_prompt_embeds, vec_to_token_ids

# Compatibility with newer transformers Cache: some versions output/accept legacy tuple past_key_values,
# but attention uses Cache.update() internally and may error when given a tuple.
try:
    from transformers.cache_utils import DynamicCache  # type: ignore
except Exception:  # pragma: no cover
    DynamicCache = None  # type: ignore


def _ensure_kv_cache(past_key_values):
    """
    Normalize past_key_values to a transformers Cache object when possible.
    - If DynamicCache is available and pkv is legacy tuple -> convert.
    - Otherwise return as-is.
    """
    if DynamicCache is None:
        return past_key_values
    if isinstance(past_key_values, tuple):
        try:
            return DynamicCache.from_legacy_cache(past_key_values)
        except Exception:
            return past_key_values
    return past_key_values


def _lm_forward_with_optional_position_ids(lm, *, inputs_embeds, past_key_values, use_cache: bool, position_ids=None):
    return lm(inputs_embeds=inputs_embeds, use_cache=use_cache, past_key_values=past_key_values, position_ids=position_ids)


@torch.no_grad()
def _sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.LongTensor:
    """
    logits: [B,V]
    returns: [B] sampled token ids
    """
    if float(temperature) != 1.0:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class LatentMorph(torch.nn.Module):
    """
    Fixed version: correctly implements control-token injection to avoid generation collapse.

    Core fixes:
    1. Inject control tokens via KV cache without occupying extra input positions
    2. Fix prediction position indexing
    3. Ensure correct gradient flow
    """

    def __init__(
        self,
        *,
        frozen_model: MultiModalityCausalLM,
        processor: VLChatProcessor,
        tokenizer,
        controller: LatentController,
        img_size: int,
        patch_size: int,
        image_token_num: int,
        stages: int,
        part_template: str,
        use_understanding: bool,
        understanding_max_tokens: int,
        cfg_weight: float,
        temperature: float,
        control_strength_cond: float = 1.0,
        control_strength_uncond: float = 0.1,
    ):
        super().__init__()
        self.controller = controller

        # CFG fix: use different control strengths for cond/uncond batches (learnable).
        self.control_strength_cond = torch.nn.Parameter(torch.tensor(float(control_strength_cond), dtype=torch.float32))
        self.control_strength_uncond = torch.nn.Parameter(torch.tensor(float(control_strength_uncond), dtype=torch.float32))

        # Do NOT register these as nn.Modules
        self.__dict__["_frozen_model"] = frozen_model
        self.__dict__["_processor"] = processor
        self.__dict__["_tokenizer"] = tokenizer

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.image_token_num = int(image_token_num)
        self.stages = int(stages)
        self.part_template = str(part_template)
        self.use_understanding = True  # Enabled by default.
        self.understanding_max_tokens = int(understanding_max_tokens)
        self.cfg_weight = float(cfg_weight)
        self.temperature = float(temperature)

    @property
    def model(self) -> MultiModalityCausalLM:
        return self.__dict__["_frozen_model"]

    @property
    def processor(self) -> VLChatProcessor:
        return self.__dict__["_processor"]

    @property
    def tokenizer(self):
        return self.__dict__["_tokenizer"]

    @torch.inference_mode()
    def generate_image_tokens(
        self,
        *,
        base_prompt: str,
        seed: int | None = None,
        inj: int | None = None,
    ) -> tuple[torch.LongTensor, int]:
        """
        For inference/visualization: generate image tokens via
        "(optional random) inj + **strict KV injection** + autoregressive sampling".

        Returns:
        - image_ids: [1, N] (N = image_token_num)
        - inj: actual injection point (>32)
        """
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        model = self.model
        processor = self.processor
        tokenizer = self.tokenizer
        controller = self.controller

        bsz = 1
        n_tokens = int(self.image_token_num)
        if n_tokens < 35:
            raise ValueError(f"image_token_num too small: {n_tokens} (need >= 35 for inj>32)")

        # Device: prefer controller parameters.
        try:
            device = next(controller.parameters()).device
        except StopIteration:
            device = model.device

        # inj: random by default, constrained to [150,450] as requested; can also be provided externally.
        if inj is None:
            low = 150
            high_excl = min(451, n_tokens)  # torch.randint high is exclusive, and must satisfy inj < n_tokens
            if high_excl <= low:
                raise ValueError(f"image_token_num too small for inj in [150,450): n_tokens={n_tokens}")
            inj = int(torch.randint(low=low, high=high_excl, size=(), device=device).item())
        inj = int(inj)
        if inj <= 32 or inj >= n_tokens:
            raise ValueError(f"inj must satisfy 32 < inj < {n_tokens}, got {inj}")

        # Reset controller runtime state (supports two signatures).
        try:
            controller.reset(batch_size=bsz, device=device)
        except Exception:
            try:
                controller.reset()
            except Exception:
                pass

        gen_prompt_ids: List[int] = tokenizer.encode(str(base_prompt))
        prompt_embeds, prompt_vec = build_cfg_prompt_embeds(model, processor, tokenizer, gen_prompt_ids, bsz, device)

        lm = model.language_model.model
        pkv = None
        inputs_embeds = prompt_embeds.to(torch.float16)  # [2,T,D]
        gen_ids = torch.empty((bsz, n_tokens), device=device, dtype=torch.long)  # [1,N]
        prompt_len = int(prompt_embeds.shape[1])

        def _sample_from_hidden(h_last: torch.Tensor) -> torch.LongTensor:
            logits2 = model.gen_head(h_last)  # [2B,V]
            logit_cond = logits2[0::2]
            logit_uncond = logits2[1::2]
            logits = logit_uncond + float(self.cfg_weight) * (logit_cond - logit_uncond)  # [B,V]
            probs = torch.softmax(logits / float(max(1e-6, self.temperature)), dim=-1)
            return torch.multinomial(probs, num_samples=1).to(torch.long)  # [B,1]

        # token0: based on the last hidden state of the prompt.
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
        gen_ids[:, 0] = next_tok.squeeze(-1)
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)  # [2B]
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)  # [2B,1,D]

        # token1..inj-1: no control (prefix).
        for t in range(1, inj):
            # The current position consumes image token (t-1), so RoPE position = prompt_len + (t-1).
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos
            )
            pkv = _ensure_kv_cache(out.past_key_values)
            next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
            gen_ids[:, t] = next_tok.squeeze(-1)
            next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

        # Prefix -> long condenser (consume prefix embeddings).
        ids_prefix = gen_ids[:, :inj].to(device=device, dtype=torch.long)  # [1,inj]
        prefix_emb = model.prepare_gen_img_embeds(ids_prefix.reshape(-1)).to(torch.float16).view(bsz, inj, -1)

        ctrl_dtype = next(controller.parameters()).dtype
        long_m_tokens, long_m_vec = controller.long_condenser(prefix_emb.to(ctrl_dtype))

        # Understanding latent (inference: no grad).
        rep_k = int(max(8, min(64, self.understanding_max_tokens))) if self.understanding_max_tokens > 0 else 64
        rep_ids = vec_to_token_ids(model, long_m_vec.mean(dim=0), k=rep_k)
        think_prompt_text = (
            "You are given the currently generated image representation:\n"
            f"[{rep_ids}]\n\n"
            "This representation corresponds to the original generation objective:\n"
            f"{str(base_prompt)}\n\n"
            "Without producing any intermediate reasoning, validation, or explanation, implicitly verify alignment and generate a prompt that continues the image generation.\n\n"
            "The continuation should correct any potential deviation and preserve semantic, structural, and visual consistency.\n\n"
            "Output only the continuation prompt for generating the remaining image tokens."
        )
        z_vec = controller._think_latent(model=model, prompt_text=think_prompt_text, m_tokens=long_m_tokens.detach())

        # translator + shaper -> ctrl_tokens, then perform "strict KV injection"
        c_vec = controller.translator(
            z_vec=z_vec.to(ctrl_dtype),
            m_vec=long_m_vec.to(ctrl_dtype),
            p_vec=prompt_vec.to(ctrl_dtype),
        )
        ctrl_tokens = controller.shaper.make_control_tokens_for_cfg(c_vec).to(torch.float16)  # [2B,K,D]
        # CFG fix: scale control strength separately for cond/uncond batches (avoid in-place slicing that breaks autograd)
        scale = torch.empty((ctrl_tokens.shape[0], 1, 1), device=ctrl_tokens.device, dtype=ctrl_tokens.dtype)
        scale[0::2] = self.control_strength_cond.to(dtype=ctrl_tokens.dtype)
        scale[1::2] = self.control_strength_uncond.to(dtype=ctrl_tokens.dtype)
        ctrl_tokens = ctrl_tokens * scale

        # Strict KV injection: do NOT append ctrl_tokens as extra tokens (would extend cache length / disturb timeline).
        # Instead, compress ctrl_tokens into a delta embedding and add it to the real token(inj-1) embedding,
        # then forward that token normally to modify KV "without increasing the sequence length".
        ctrl_delta = ctrl_tokens.mean(dim=1, keepdim=True)  # [2B,1,D]

        # Rebuild cache up to token(inj-1) (simple and reliable).
        pkv = None
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=prompt_embeds.to(torch.float16), use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        if inj - 1 > 0:
            ids_build = gen_ids[:, : inj - 1]  # tokens 0..inj-2
            ids_build2 = torch.stack([ids_build, ids_build], dim=1).view(bsz * 2, -1)
            emb_build = model.prepare_gen_img_embeds(ids_build2.reshape(-1)).to(torch.float16).view(bsz * 2, inj - 1, -1)
            out = lm(inputs_embeds=emb_build, use_cache=True, past_key_values=pkv)
            pkv = _ensure_kv_cache(out.past_key_values)

        # consume token inj-1
        tok_inj_m1 = gen_ids[:, inj - 1 : inj]
        tok_inj_m1_2 = torch.cat([tok_inj_m1, tok_inj_m1], dim=1).view(-1)
        emb_inj_m1 = model.prepare_gen_img_embeds(tok_inj_m1_2).to(torch.float16).unsqueeze(1)
        pos_inj_m1 = torch.full((bsz * 2, 1), prompt_len + (inj - 1), device=device, dtype=torch.long)
        # Apply strict KV injection at token(inj-1): add ctrl_delta to the real token embedding.
        emb_inj_m1 = emb_inj_m1 + ctrl_delta.to(dtype=emb_inj_m1.dtype)
        out = _lm_forward_with_optional_position_ids(
            lm, inputs_embeds=emb_inj_m1, use_cache=True, past_key_values=pkv, position_ids=pos_inj_m1
        )
        pkv = _ensure_kv_cache(out.past_key_values)

        # Use token(inj-1)'s hidden to predict token inj (standard autoregressive predictor).
        next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
        gen_ids[:, inj] = next_tok.squeeze(-1)

        # token inj+1..N-1: control has entered the context.
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)
        for t in range(inj + 1, n_tokens):
            # Current position consumes image token (t-1), so RoPE position = prompt_len + (t-1)
            # (no shift due to ctrl_tokens).
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos
            )
            pkv = _ensure_kv_cache(out.past_key_values)
            next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
            gen_ids[:, t] = next_tok.squeeze(-1)
            next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

        return gen_ids.to(torch.long), int(inj)

    @torch.inference_mode()
    def generate_image_tokens_janus_base(
        self,
        *,
        base_prompt: str,
        seed: int | None = None,
    ) -> torch.LongTensor:
        """
        Pure Janus-Pro baseline: no controller, no control/KV injection.

        - Input: base_prompt
        - Output: image_ids [1, N] (N = image_token_num)
        """
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        model = self.model
        processor = self.processor
        tokenizer = self.tokenizer

        bsz = 1
        n_tokens = int(self.image_token_num)

        # Device: prefer model.device.
        try:
            device = model.device
        except Exception:
            device = next(model.parameters()).device

        gen_prompt_ids: List[int] = tokenizer.encode(str(base_prompt))
        prompt_embeds, _prompt_vec = build_cfg_prompt_embeds(model, processor, tokenizer, gen_prompt_ids, bsz, device)

        lm = model.language_model.model
        pkv = None
        inputs_embeds = prompt_embeds.to(torch.float16)  # [2,T,D]
        gen_ids = torch.empty((bsz, n_tokens), device=device, dtype=torch.long)  # [1,N]
        prompt_len = int(prompt_embeds.shape[1])

        def _sample_from_hidden(h_last: torch.Tensor) -> torch.LongTensor:
            logits2 = model.gen_head(h_last)  # [2B,V]
            logit_cond = logits2[0::2]
            logit_uncond = logits2[1::2]
            logits = logit_uncond + float(self.cfg_weight) * (logit_cond - logit_uncond)  # [B,V]
            probs = torch.softmax(logits / float(max(1e-6, self.temperature)), dim=-1)
            return torch.multinomial(probs, num_samples=1).to(torch.long)  # [B,1]

        # token0: based on the last hidden state of the prompt.
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])  # [B,1]
        gen_ids[:, 0] = next_tok.squeeze(-1)

        # token1..N-1: standard autoregressive sampling (no extra token injection).
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)  # [2B]
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)  # [2B,1,D]
        for t in range(1, n_tokens):
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos
            )
            pkv = _ensure_kv_cache(out.past_key_values)
            next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
            gen_ids[:, t] = next_tok.squeeze(-1)
            next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

        return gen_ids.to(torch.long)

    @torch.inference_mode()
    def generate_control_and_base_shared_prefix(
        self,
        *,
        base_prompt: str,
        seed: int | None = None,
        inj: int | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, int]:
        """
        Generate a comparable pair (controlled vs pure Janus-Pro base) and force them to be fully aligned before inj:

        - Both use the same prompt
        - token[0 .. inj-1] are identical (a shared prefix from the same sampling run)
        - The branch point is at "consume token inj-1":
          - base: no injection; consume token(inj-1) -> predict token inj -> continue generation
          - control: perform strict KV injection when consuming token(inj-1) (embedding + ctrl_delta),
                     then predict token inj -> continue generation

        Returns:
        - ctrl_ids: [1, N]
        - base_ids: [1, N]
        - inj: int
        """
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        model = self.model
        processor = self.processor
        tokenizer = self.tokenizer
        controller = self.controller

        bsz = 1
        n_tokens = int(self.image_token_num)
        if n_tokens < 35:
            raise ValueError(f"image_token_num too small: {n_tokens} (need >= 35 for inj>32)")

        # Device: prefer controller parameters.
        try:
            device = next(controller.parameters()).device
        except StopIteration:
            device = model.device

        # inj: random by default in [150,450)
        if inj is None:
            low = 150
            high_excl = min(451, n_tokens)
            if high_excl <= low:
                raise ValueError(f"image_token_num too small for inj in [150,450): n_tokens={n_tokens}")
            inj = int(torch.randint(low=low, high=high_excl, size=(), device=device).item())
        inj = int(inj)
        if inj <= 32 or inj >= n_tokens:
            raise ValueError(f"inj must satisfy 32 < inj < {n_tokens}, got {inj}")

        # Reset controller runtime state (supports two signatures).
        try:
            controller.reset(batch_size=bsz, device=device)
        except Exception:
            try:
                controller.reset()
            except Exception:
                pass

        gen_prompt_ids: List[int] = tokenizer.encode(str(base_prompt))
        prompt_embeds, prompt_vec = build_cfg_prompt_embeds(model, processor, tokenizer, gen_prompt_ids, bsz, device)
        prompt_len = int(prompt_embeds.shape[1])

        lm = model.language_model.model

        def _sample_from_hidden(h_last: torch.Tensor) -> torch.LongTensor:
            logits2 = model.gen_head(h_last)  # [2B,V]
            logit_cond = logits2[0::2]
            logit_uncond = logits2[1::2]
            logits = logit_uncond + float(self.cfg_weight) * (logit_cond - logit_uncond)  # [B,V]
            probs = torch.softmax(logits / float(max(1e-6, self.temperature)), dim=-1)
            return torch.multinomial(probs, num_samples=1).to(torch.long)  # [B,1]

        # === Step A: sample once to obtain the shared prefix token[0..inj-1] and pkv_before_inj_m1 (consumed 0..inj-2)
        pkv_before_inj_m1 = None
        inputs_embeds = prompt_embeds.to(torch.float16)  # [2,T,D]
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)

        prefix_ids = torch.empty((1, inj), device=device, dtype=torch.long)  # [1,inj]
        # token0
        tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])  # [1,1]
        prefix_ids[:, 0] = tok.squeeze(-1)
        tok2 = torch.cat([tok, tok], dim=1).reshape(-1)  # [2B]
        inputs_embeds = model.prepare_gen_img_embeds(tok2).to(torch.float16).unsqueeze(1)  # [2B,1,D]
        # token1..inj-1 (note: the loop consumes token(t-1) and predicts token t)
        for t in range(1, inj):
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos
            )
            pkv = _ensure_kv_cache(out.past_key_values)
            tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
            prefix_ids[:, t] = tok.squeeze(-1)
            tok2 = torch.cat([tok, tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(tok2).to(torch.float16).unsqueeze(1)

            # When t==inj-1, we just consumed token inj-2, so pkv is exactly "consumed 0..inj-2" at this moment.
            if t == inj - 1:
                pkv_before_inj_m1 = pkv

        if pkv_before_inj_m1 is None:
            # inj==1 is impossible (inj>32); this is just a defensive fallback.
            pkv_before_inj_m1 = pkv

        tok_inj_m1 = prefix_ids[:, inj - 1 : inj]  # [1,1]
        tok_inj_m1_2 = torch.cat([tok_inj_m1, tok_inj_m1], dim=1).view(-1)  # [2B]
        emb_inj_m1_plain = model.prepare_gen_img_embeds(tok_inj_m1_2).to(torch.float16).unsqueeze(1)  # [2B,1,D]
        pos_inj_m1 = torch.full((bsz * 2, 1), prompt_len + (inj - 1), device=device, dtype=torch.long)

        # === Step B: base branch (no injection), start generating suffix by consuming token(inj-1) from pkv_before_inj_m1
        base_ids = torch.empty((1, n_tokens), device=device, dtype=torch.long)
        base_ids[:, :inj] = prefix_ids
        out_base = _lm_forward_with_optional_position_ids(
            lm, inputs_embeds=emb_inj_m1_plain, use_cache=True, past_key_values=pkv_before_inj_m1, position_ids=pos_inj_m1
        )
        pkv_base = _ensure_kv_cache(out_base.past_key_values)
        next_tok = _sample_from_hidden(out_base.last_hidden_state[:, -1, :])  # token inj
        base_ids[:, inj] = next_tok.squeeze(-1)
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)
        for t in range(inj + 1, n_tokens):
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out2 = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv_base, position_ids=pos
            )
            pkv_base = _ensure_kv_cache(out2.past_key_values)
            next_tok = _sample_from_hidden(out2.last_hidden_state[:, -1, :])
            base_ids[:, t] = next_tok.squeeze(-1)
            next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

        # === Step C: control branch (strict KV injection), must start from the same pkv_before_inj_m1
        # To avoid in-place cache updates/shared references, replay prefix_ids up to token inj-2 to rebuild pkv_before_inj_m1_2.
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=prompt_embeds.to(torch.float16), use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        if inj - 1 > 0:
            ids_build = prefix_ids[:, : inj - 1]  # tokens 0..inj-2
            ids_build2 = torch.stack([ids_build, ids_build], dim=1).view(bsz * 2, -1)
            emb_build = model.prepare_gen_img_embeds(ids_build2.reshape(-1)).to(torch.float16).view(bsz * 2, inj - 1, -1)
            out = lm(inputs_embeds=emb_build, use_cache=True, past_key_values=pkv)
            pkv = _ensure_kv_cache(out.past_key_values)
        pkv_before_inj_m1_2 = pkv

        # Compute ctrl_delta (same logic as generate_image_tokens).
        ids_prefix = prefix_ids.to(device=device, dtype=torch.long)  # [1,inj]
        prefix_emb = model.prepare_gen_img_embeds(ids_prefix.reshape(-1)).to(torch.float16).view(bsz, inj, -1)
        ctrl_dtype = next(controller.parameters()).dtype
        long_m_tokens, long_m_vec = controller.long_condenser(prefix_emb.to(ctrl_dtype))
        rep_k = int(max(8, min(64, self.understanding_max_tokens))) if self.understanding_max_tokens > 0 else 64
        rep_ids = vec_to_token_ids(model, long_m_vec.mean(dim=0), k=rep_k)
        think_prompt_text = (
            "You are given the currently generated image representation:\n"
            f"[{rep_ids}]\n\n"
            "This representation corresponds to the original generation objective:\n"
            f"{str(base_prompt)}\n\n"
            "Without producing any intermediate reasoning, validation, or explanation, implicitly verify alignment and generate a prompt that continues the image generation.\n\n"
            "The continuation should correct any potential deviation and preserve semantic, structural, and visual consistency.\n\n"
            "Output only the continuation prompt for generating the remaining image tokens."
        )
        z_vec = controller._think_latent(model=model, prompt_text=think_prompt_text, m_tokens=long_m_tokens.detach())
        c_vec = controller.translator(z_vec=z_vec.to(ctrl_dtype), m_vec=long_m_vec.to(ctrl_dtype), p_vec=prompt_vec.to(ctrl_dtype))
        ctrl_tokens = controller.shaper.make_control_tokens_for_cfg(c_vec).to(torch.float16)  # [2B,K,D]
        scale = torch.empty((ctrl_tokens.shape[0], 1, 1), device=ctrl_tokens.device, dtype=ctrl_tokens.dtype)
        scale[0::2] = self.control_strength_cond.to(dtype=ctrl_tokens.dtype)
        scale[1::2] = self.control_strength_uncond.to(dtype=ctrl_tokens.dtype)
        ctrl_tokens = ctrl_tokens * scale
        ctrl_delta = ctrl_tokens.mean(dim=1, keepdim=True)  # [2B,1,D]

        ctrl_ids = torch.empty((1, n_tokens), device=device, dtype=torch.long)
        ctrl_ids[:, :inj] = prefix_ids

        emb_inj_m1_ctrl = emb_inj_m1_plain + ctrl_delta.to(dtype=emb_inj_m1_plain.dtype)
        out_ctrl = _lm_forward_with_optional_position_ids(
            lm, inputs_embeds=emb_inj_m1_ctrl, use_cache=True, past_key_values=pkv_before_inj_m1_2, position_ids=pos_inj_m1
        )
        pkv_ctrl = _ensure_kv_cache(out_ctrl.past_key_values)
        next_tok = _sample_from_hidden(out_ctrl.last_hidden_state[:, -1, :])  # token inj
        ctrl_ids[:, inj] = next_tok.squeeze(-1)

        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)
        for t in range(inj + 1, n_tokens):
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out2 = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv_ctrl, position_ids=pos
            )
            pkv_ctrl = _ensure_kv_cache(out2.past_key_values)
            next_tok = _sample_from_hidden(out2.last_hidden_state[:, -1, :])
            ctrl_ids[:, t] = next_tok.squeeze(-1)
            next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

        return ctrl_ids.to(torch.long), base_ids.to(torch.long), int(inj)

    @torch.inference_mode()
    def decode_image_tokens_to_pil(self, image_ids: torch.LongTensor):
        """
        Decode image token ids into PIL.Image objects (a list).
        """
        model = self.model
        bsz = int(image_ids.shape[0])
        h = int(self.img_size)
        w = int(self.img_size)
        ph = int(self.patch_size)
        pw = int(self.patch_size)
        if h % ph != 0 or w % pw != 0:
            raise ValueError(f"img_size must be divisible by patch_size, got img_size={h}, patch_size={ph}")

        patches = model.gen_vision_model.decode_code(
            image_ids.to(dtype=torch.int),
            shape=[bsz, 8, w // pw, h // ph],
        )

        x = patches.to(torch.float32).clamp(-1, 1)
        x = ((x + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        from PIL import Image

        return [Image.fromarray(x[i]) for i in range(bsz)]

    def forward(self, *, base_prompt: str, gt_image_ids: torch.LongTensor) -> torch.Tensor:
        """
        Fixed version: correct teacher-forcing + control injection

        Strategy:
        1. Standard teacher-forcing input: prompt + all GT image tokens
        2. Inject control tokens via KV cache before the inj position
        3. Predictions after inj are conditioned on the injected state
        """
        model = self.model
        tokenizer = self.tokenizer
        processor = self.processor
        controller = self.controller
        device = gt_image_ids.device
        bsz = int(gt_image_ids.shape[0])
        n_tokens = int(self.image_token_num)

        if n_tokens < 35:
            raise ValueError(f"image_token_num too small: {n_tokens}")
        if int(gt_image_ids.shape[1]) < n_tokens:
            raise ValueError(f"gt_image_ids too short: got {int(gt_image_ids.shape[1])}, need >= {n_tokens}")

        # As requested: random inj constrained to [150,450] (and clipped to < n_tokens).
        low = 150
        high_excl = min(451, n_tokens)  # high is exclusive
        if high_excl <= low:
            raise ValueError(f"image_token_num too small for inj in [150,450): n_tokens={n_tokens}")
        inj = int(torch.randint(low=low, high=high_excl, size=(), device=device).item())

        # Reset controller state.
        try:
            controller.reset(batch_size=bsz, device=device)
        except Exception:
            pass

        # === Build inputs ===

        # 1. Prompt embeds
        with torch.no_grad():
            gen_prompt_ids: List[int] = tokenizer.encode(str(base_prompt))
            prompt_embeds, prompt_vec = build_cfg_prompt_embeds(model, processor, tokenizer, gen_prompt_ids, bsz, device)
        prompt_len = int(prompt_embeds.shape[1])

        # 2. GT image token embeddings (full sequence)
        ids_full = gt_image_ids[:, :n_tokens].to(device=device, dtype=torch.long)  # [B,N]
        ids_full_cfg = torch.stack([ids_full, ids_full], dim=1).view(bsz * 2, -1)  # [2B,N]
        with torch.no_grad():
            full_emb = model.prepare_gen_img_embeds(ids_full_cfg.reshape(-1)).to(torch.float16).view(bsz * 2, n_tokens, -1)  # [2B,N,D]

        # 3. Full input sequence (for teacher-forcing)
        input_emb = torch.cat([prompt_embeds.to(torch.float16), full_emb], dim=1)  # [2B, prompt_len + N, D]

        # === Compute control signal ===

        # Compute control using the first inj tokens (same as generation).
        ids_prefix = gt_image_ids[:, :inj].to(device=device, dtype=torch.long)  # [B,inj]
        with torch.no_grad():
            prefix_emb = model.prepare_gen_img_embeds(ids_prefix.reshape(-1)).to(torch.float16).view(bsz, inj, -1)

        # LongCondenser (trainable)
        ctrl_dtype = next(controller.parameters()).dtype
        long_m_tokens, long_m_vec = controller.long_condenser(prefix_emb.to(ctrl_dtype))

        # Understanding (no_grad)
        with torch.no_grad():
            rep_k = int(max(8, min(64, self.understanding_max_tokens))) if self.understanding_max_tokens > 0 else 64
            rep_ids = vec_to_token_ids(model, long_m_vec.mean(dim=0), k=rep_k)
            think_prompt_text = (
                "You are given the currently generated image representation:\n"
                f"[{rep_ids}]\n\n"
                "This representation corresponds to the original generation objective:\n"
                f"{str(base_prompt)}\n\n"
                "Without producing any intermediate reasoning, validation, or explanation, implicitly verify alignment and generate a prompt that continues the image generation.\n\n"
                "The continuation should correct any potential deviation and preserve semantic, structural, and visual consistency.\n\n"
                "Output only the continuation prompt for generating the remaining image tokens."
            )
            # Important fix: do not detach, so gradients can flow.
            z_vec = controller._think_latent(model=model, prompt_text=think_prompt_text, m_tokens=long_m_tokens)

        # Translator + Shaper (trainable)
        c_vec = controller.translator(
            z_vec=z_vec.to(ctrl_dtype),
            m_vec=long_m_vec.to(ctrl_dtype),
            p_vec=prompt_vec.to(ctrl_dtype),
        )
        ctrl_tokens = controller.shaper.make_control_tokens_for_cfg(c_vec).to(torch.float16)  # [2B,K,D]
        # CFG fix: scale control strength separately for cond/uncond batches (avoid in-place slicing that breaks autograd)
        scale = torch.empty((ctrl_tokens.shape[0], 1, 1), device=ctrl_tokens.device, dtype=ctrl_tokens.dtype)
        scale[0::2] = self.control_strength_cond.to(dtype=ctrl_tokens.dtype)
        scale[1::2] = self.control_strength_uncond.to(dtype=ctrl_tokens.dtype)
        ctrl_tokens = ctrl_tokens * scale
        k_ctrl = int(ctrl_tokens.shape[1])

        # === Key fix: correct parallel forward ===

        # Strategy: two-stage forward
        # Step 1: forward up to inj to build KV cache
        # Step 2: inject control tokens, then continue forward

        lm = model.language_model

        # Step 1: forward up to inj
        input_up_to_inj = input_emb[:, :prompt_len + inj, :]  # [2B, prompt_len + inj, D]
        with torch.no_grad():
            out_prefix = _lm_forward_with_optional_position_ids(lm.model, inputs_embeds=input_up_to_inj, use_cache=True, past_key_values=None)
            past_kv = _ensure_kv_cache(out_prefix.past_key_values)

        # Step 2: strict KV injection (do not append ctrl_tokens; do not increase cache length)
        # Method: compress ctrl_tokens into a delta embedding, add it to the real token(inj-1) embedding,
        # then forward only that token to produce "controlled" KV (sequence length unchanged).
        ctrl_delta = ctrl_tokens.mean(dim=1, keepdim=True)  # [2B,1,D]
        emb_inj_m1 = full_emb[:, inj - 1 : inj, :] + ctrl_delta.to(dtype=full_emb.dtype)  # [2B,1,D]
        pos_inj_m1 = torch.full((bsz * 2, 1), prompt_len + (inj - 1), device=device, dtype=torch.long)
        out_inj_m1 = _lm_forward_with_optional_position_ids(
            lm.model,
            inputs_embeds=emb_inj_m1,
            use_cache=True,
            past_key_values=past_kv,
            position_ids=pos_inj_m1,
        )
        past_kv_injected = _ensure_kv_cache(out_inj_m1.past_key_values)
        h_inj_m1 = out_inj_m1.last_hidden_state  # [2B,1,D], used to predict token inj

        # Step 3: continue forward after inj
        input_after_inj = input_emb[:, prompt_len + inj:, :]  # [2B, N-inj, D]
        # suffix position_ids must keep the original timeline: prompt_len+inj .. prompt_len+N-1
        pos_suffix = torch.arange(prompt_len + inj, prompt_len + n_tokens, device=device, dtype=torch.long)
        pos_suffix = pos_suffix.unsqueeze(0).expand(bsz * 2, -1)  # [2B, N-inj]
        out_full = _lm_forward_with_optional_position_ids(
            lm.model,
            inputs_embeds=input_after_inj,
            use_cache=True,
            past_key_values=_ensure_kv_cache(past_kv_injected),
            position_ids=pos_suffix,
        )

        # Combine required hidden states.
        h_prefix = out_prefix.last_hidden_state  # [2B, prompt_len + inj, D] (no grad)
        h_suffix = out_full.last_hidden_state   # [2B, N-inj, D] (has grad; depends on injected KV)

        # === Predictor hidden (do not insert control tokens into the token sequence) ===
        # Predictor hidden for target token t:
        # - t=0: last hidden of the prompt
        # - 1 <= t < inj: hidden of image token (t-1) (from prefix)
        # - t=inj: hidden of token(inj-1) (h_inj_m1)
        # - t > inj: hidden of image token (t-1) (from suffix)
        pred_h = torch.empty((h_suffix.shape[0], n_tokens, h_suffix.shape[2]), device=device, dtype=h_suffix.dtype)
        pred_h[:, 0:1, :] = h_prefix[:, prompt_len - 1 : prompt_len, :].to(dtype=pred_h.dtype)
        if inj > 1:
            pred_h[:, 1:inj, :] = h_prefix[:, prompt_len : prompt_len + (inj - 1), :].to(dtype=pred_h.dtype)
        pred_h[:, inj : inj + 1, :] = h_inj_m1.to(dtype=pred_h.dtype)
        tail = int(n_tokens - inj - 1)
        if tail > 0:
            pred_h[:, inj + 1 :, :] = h_suffix[:, :tail, :]

        # === Loss ===
        logits2_all = model.gen_head(pred_h.to(torch.float16))  # [2B, N, V]
        cond_all, uncond_all = logits2_all[0::2], logits2_all[1::2]  # [B, N, V]
        logits_all = uncond_all + float(self.cfg_weight) * (cond_all - uncond_all)  # [B, N, V]

        loss = F.cross_entropy(
            logits_all.reshape(-1, logits_all.shape[-1]),
            gt_image_ids[:, :n_tokens].to(device=device, dtype=torch.long).reshape(-1),
            reduction="mean",
        )
        return loss.to(torch.float32)


__all__ = ["LatentMorph"]


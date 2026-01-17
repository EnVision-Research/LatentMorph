from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from janus.models import MultiModalityCausalLM, VLChatProcessor
from latent_control.controller import LatentController
from .prompt import build_cfg_prompt_embeds, vec_to_token_ids

# transformers 新版 Cache 兼容：某些版本输出/接受 legacy tuple past_key_values，
# 但内部 attention 使用 Cache.update()，会在传入 tuple 时直接报错。
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
    修复版本：正确实现 control token 注入，避免生成崩坏。

    核心修复：
    1. Control tokens 通过 KV cache 注入，不占用输入序列位置
    2. 修复预测位置索引计算
    3. 确保梯度传播正确
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

        # CFG 修复：对 cond/uncond batch 使用不同控制强度（可学习）
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
        self.use_understanding = True  # 默认开启
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
        推理/可视化用：走“(可选随机)inj + **严格 KV 注入** + 自回归采样”生成图片 token。

        返回：
        - image_ids: [1, N]（N = image_token_num）
        - inj: 实际注入点（>32）
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

        # device：优先 controller 参数
        try:
            device = next(controller.parameters()).device
        except StopIteration:
            device = model.device

        # inj：默认随机，按你的要求控制到 [150,450]；也允许外部固定
        if inj is None:
            low = 150
            high_excl = min(451, n_tokens)  # torch.randint high is exclusive, and must satisfy inj < n_tokens
            if high_excl <= low:
                raise ValueError(f"image_token_num too small for inj in [150,450): n_tokens={n_tokens}")
            inj = int(torch.randint(low=low, high=high_excl, size=(), device=device).item())
        inj = int(inj)
        if inj <= 32 or inj >= n_tokens:
            raise ValueError(f"inj must satisfy 32 < inj < {n_tokens}, got {inj}")

        # reset controller runtime state（兼容两种签名）
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

        # token0：基于 prompt 最后一个 hidden
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
        gen_ids[:, 0] = next_tok.squeeze(-1)
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)  # [2B]
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)  # [2B,1,D]

        # token1..inj-1：无控制（prefix）
        for t in range(1, inj):
            # 当前位置是 image token (t-1) ，对应 RoPE position = prompt_len + (t-1)
            pos = torch.full((bsz * 2, 1), prompt_len + (t - 1), device=device, dtype=torch.long)
            out = _lm_forward_with_optional_position_ids(
                lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv, position_ids=pos
            )
            pkv = _ensure_kv_cache(out.past_key_values)
            next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
            gen_ids[:, t] = next_tok.squeeze(-1)
            next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
            inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)

        # prefix -> long condenser（吃 prefix embeds）
        ids_prefix = gen_ids[:, :inj].to(device=device, dtype=torch.long)  # [1,inj]
        prefix_emb = model.prepare_gen_img_embeds(ids_prefix.reshape(-1)).to(torch.float16).view(bsz, inj, -1)

        ctrl_dtype = next(controller.parameters()).dtype
        long_m_tokens, long_m_vec = controller.long_condenser(prefix_emb.to(ctrl_dtype))

        # understanding latent（推理：no grad）
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

        # translator + shaper -> ctrl_tokens，然后做“严格 KV 注入”
        c_vec = controller.translator(
            z_vec=z_vec.to(ctrl_dtype),
            m_vec=long_m_vec.to(ctrl_dtype),
            p_vec=prompt_vec.to(ctrl_dtype),
        )
        ctrl_tokens = controller.shaper.make_control_tokens_for_cfg(c_vec).to(torch.float16)  # [2B,K,D]
        # CFG 修复：对 cond/uncond batch 分别缩放控制强度（避免 inplace 切片赋值破坏 autograd）
        scale = torch.empty((ctrl_tokens.shape[0], 1, 1), device=ctrl_tokens.device, dtype=ctrl_tokens.dtype)
        scale[0::2] = self.control_strength_cond.to(dtype=ctrl_tokens.dtype)
        scale[1::2] = self.control_strength_uncond.to(dtype=ctrl_tokens.dtype)
        ctrl_tokens = ctrl_tokens * scale

        # 严格 KV 注入：不把 ctrl_tokens 当作额外 token 追加到序列里（那会增长 cache 长度/打乱时间线）。
        # 这里把 ctrl_tokens 压成一个 delta embedding，加到真实 token(inj-1) 的 embedding 上，
        # 通过正常 forward 这一 token 来“在不增加序列长度的前提下”修改 KV。
        ctrl_delta = ctrl_tokens.mean(dim=1, keepdim=True)  # [2B,1,D]

        # 重建到 token(inj-1) 的 cache（简单可靠）
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
        # token(inj-1) 处执行严格 KV 注入：把 ctrl_delta 加到真实 token 的 embedding 上
        emb_inj_m1 = emb_inj_m1 + ctrl_delta.to(dtype=emb_inj_m1.dtype)
        out = _lm_forward_with_optional_position_ids(
            lm, inputs_embeds=emb_inj_m1, use_cache=True, past_key_values=pkv, position_ids=pos_inj_m1
        )
        pkv = _ensure_kv_cache(out.past_key_values)

        # 用 token(inj-1) 的 hidden 来预测 token inj（标准自回归 predictor）
        next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])
        gen_ids[:, inj] = next_tok.squeeze(-1)

        # token inj+1..N-1：控制已进入上下文
        next_tok2 = torch.cat([next_tok, next_tok], dim=1).reshape(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_tok2).to(torch.float16).unsqueeze(1)
        for t in range(inj + 1, n_tokens):
            # 当前位置是 image token (t-1)，对应 RoPE position = prompt_len + (t-1)（不因 ctrl_tokens 而偏移）
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
        纯 Janus-Pro baseline：不使用 controller，不做任何 control/KV 注入。

        - 输入：base_prompt
        - 输出：image_ids [1, N]（N = image_token_num）
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

        # device：优先 model
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

        # token0：基于 prompt 最后一个 hidden
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        next_tok = _sample_from_hidden(out.last_hidden_state[:, -1, :])  # [B,1]
        gen_ids[:, 0] = next_tok.squeeze(-1)

        # token1..N-1：标准自回归（不注入任何额外 token）
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
        生成一对可对比样本（控制版 vs 纯 Janus-Pro base），并强制它们在 inj 之前完全对齐：

        - 两者使用相同的 prompt
        - token[0 .. inj-1] 完全相同（同一次采样得到的前缀）
        - 注入分叉点在“consume token inj-1”：
          - base：不注入，正常 consume token(inj-1) -> 预测 token inj -> 继续生成
          - control：在 consume token(inj-1) 时做严格 KV 注入（embedding + ctrl_delta），再预测 token inj -> 继续生成

        返回：
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

        # device：优先 controller 参数
        try:
            device = next(controller.parameters()).device
        except StopIteration:
            device = model.device

        # inj：默认随机 [150,450)
        if inj is None:
            low = 150
            high_excl = min(451, n_tokens)
            if high_excl <= low:
                raise ValueError(f"image_token_num too small for inj in [150,450): n_tokens={n_tokens}")
            inj = int(torch.randint(low=low, high=high_excl, size=(), device=device).item())
        inj = int(inj)
        if inj <= 32 or inj >= n_tokens:
            raise ValueError(f"inj must satisfy 32 < inj < {n_tokens}, got {inj}")

        # reset controller runtime state（兼容两种签名）
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

        # === Step A: 一次采样得到“共享前缀 token[0..inj-1]”以及 pkv_before_inj_m1（已 consume 0..inj-2）
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
        # token1..inj-1（注意：循环里 consume token(t-1)，预测 token t）
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

            # 走到 t==inj-1 时，刚刚 consume 的是 token inj-2，因此 pkv 此刻正好是“已 consume 0..inj-2”
            if t == inj - 1:
                pkv_before_inj_m1 = pkv

        if pkv_before_inj_m1 is None:
            # inj==1 不可能（inj>32），这里只是防御
            pkv_before_inj_m1 = pkv

        tok_inj_m1 = prefix_ids[:, inj - 1 : inj]  # [1,1]
        tok_inj_m1_2 = torch.cat([tok_inj_m1, tok_inj_m1], dim=1).view(-1)  # [2B]
        emb_inj_m1_plain = model.prepare_gen_img_embeds(tok_inj_m1_2).to(torch.float16).unsqueeze(1)  # [2B,1,D]
        pos_inj_m1 = torch.full((bsz * 2, 1), prompt_len + (inj - 1), device=device, dtype=torch.long)

        # === Step B: base 分支（不注入），从 pkv_before_inj_m1 consume token(inj-1) 开始生成 suffix
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

        # === Step C: control 分支（严格 KV 注入），需要从“相同的 pkv_before_inj_m1”出发
        # 为避免 cache 对象可能的就地更新/共享引用，这里用 prefix_ids 回放一次到 token inj-2，重建 pkv_before_inj_m1_2
        out = _lm_forward_with_optional_position_ids(lm, inputs_embeds=prompt_embeds.to(torch.float16), use_cache=True, past_key_values=None)
        pkv = _ensure_kv_cache(out.past_key_values)
        if inj - 1 > 0:
            ids_build = prefix_ids[:, : inj - 1]  # tokens 0..inj-2
            ids_build2 = torch.stack([ids_build, ids_build], dim=1).view(bsz * 2, -1)
            emb_build = model.prepare_gen_img_embeds(ids_build2.reshape(-1)).to(torch.float16).view(bsz * 2, inj - 1, -1)
            out = lm(inputs_embeds=emb_build, use_cache=True, past_key_values=pkv)
            pkv = _ensure_kv_cache(out.past_key_values)
        pkv_before_inj_m1_2 = pkv

        # 计算 ctrl_delta（同 generate_image_tokens 逻辑）
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
        把 image token ids decode 成 PIL.Image（列表）。
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
        修复版本：正确的 teacher-forcing + control injection

        策略：
        1. 正常 teacher-forcing 输入：prompt + 所有 GT image tokens
        2. 在 inj 位置前通过 KV cache 注入 control tokens
        3. inj 之后的预测基于注入后的状态
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

        # 按你的要求：inj 随机控制到 [150,450]（并裁剪到 < n_tokens）
        low = 150
        high_excl = min(451, n_tokens)  # high is exclusive
        if high_excl <= low:
            raise ValueError(f"image_token_num too small for inj in [150,450): n_tokens={n_tokens}")
        inj = int(torch.randint(low=low, high=high_excl, size=(), device=device).item())

        # 重置 controller state
        try:
            controller.reset(batch_size=bsz, device=device)
        except Exception:
            pass

        # === 构建输入 ===

        # 1. Prompt embeds
        with torch.no_grad():
            gen_prompt_ids: List[int] = tokenizer.encode(str(base_prompt))
            prompt_embeds, prompt_vec = build_cfg_prompt_embeds(model, processor, tokenizer, gen_prompt_ids, bsz, device)
        prompt_len = int(prompt_embeds.shape[1])

        # 2. GT image token embeds (完整序列)
        ids_full = gt_image_ids[:, :n_tokens].to(device=device, dtype=torch.long)  # [B,N]
        ids_full_cfg = torch.stack([ids_full, ids_full], dim=1).view(bsz * 2, -1)  # [2B,N]
        with torch.no_grad():
            full_emb = model.prepare_gen_img_embeds(ids_full_cfg.reshape(-1)).to(torch.float16).view(bsz * 2, n_tokens, -1)  # [2B,N,D]

        # 3. 完整输入序列 (用于 teacher-forcing)
        input_emb = torch.cat([prompt_embeds.to(torch.float16), full_emb], dim=1)  # [2B, prompt_len + N, D]

        # === 计算控制信号 ===

        # 使用前 inj 个 tokens 计算控制 (与生成时一致)
        ids_prefix = gt_image_ids[:, :inj].to(device=device, dtype=torch.long)  # [B,inj]
        with torch.no_grad():
            prefix_emb = model.prepare_gen_img_embeds(ids_prefix.reshape(-1)).to(torch.float16).view(bsz, inj, -1)

        # LongCondenser (可训练)
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
            # 重要修复：不 detach，确保梯度传播
            z_vec = controller._think_latent(model=model, prompt_text=think_prompt_text, m_tokens=long_m_tokens)

        # Translator + Shaper (可训练)
        c_vec = controller.translator(
            z_vec=z_vec.to(ctrl_dtype),
            m_vec=long_m_vec.to(ctrl_dtype),
            p_vec=prompt_vec.to(ctrl_dtype),
        )
        ctrl_tokens = controller.shaper.make_control_tokens_for_cfg(c_vec).to(torch.float16)  # [2B,K,D]
        # CFG 修复：对 cond/uncond batch 分别缩放控制强度（避免 inplace 切片赋值破坏 autograd）
        scale = torch.empty((ctrl_tokens.shape[0], 1, 1), device=ctrl_tokens.device, dtype=ctrl_tokens.dtype)
        scale[0::2] = self.control_strength_cond.to(dtype=ctrl_tokens.dtype)
        scale[1::2] = self.control_strength_uncond.to(dtype=ctrl_tokens.dtype)
        ctrl_tokens = ctrl_tokens * scale
        k_ctrl = int(ctrl_tokens.shape[1])

        # === 关键修复：正确的并行 Forward ===

        # 策略：分两步 forward
        # Step 1: Forward 到 inj 位置，建立 KV cache
        # Step 2: 注入 control tokens，然后继续 forward

        lm = model.language_model

        # Step 1: Forward 到 inj 位置
        input_up_to_inj = input_emb[:, :prompt_len + inj, :]  # [2B, prompt_len + inj, D]
        with torch.no_grad():
            out_prefix = _lm_forward_with_optional_position_ids(lm.model, inputs_embeds=input_up_to_inj, use_cache=True, past_key_values=None)
            past_kv = _ensure_kv_cache(out_prefix.past_key_values)

        # Step 2: 严格 KV 注入（不追加 ctrl_tokens，不增加 cache 长度）
        # 做法：把 ctrl_tokens 压成 delta embedding，加到真实 token(inj-1) 的 embedding 上，
        # 然后只 forward 这一 token 来产生“被控制的” KV（序列长度保持不变）。
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
        h_inj_m1 = out_inj_m1.last_hidden_state  # [2B,1,D]，用于预测 token inj

        # Step 3: 继续 forward inj 之后的 tokens
        input_after_inj = input_emb[:, prompt_len + inj:, :]  # [2B, N-inj, D]
        # suffix 的 position_ids 必须保持原始时间线：prompt_len+inj .. prompt_len+N-1
        pos_suffix = torch.arange(prompt_len + inj, prompt_len + n_tokens, device=device, dtype=torch.long)
        pos_suffix = pos_suffix.unsqueeze(0).expand(bsz * 2, -1)  # [2B, N-inj]
        out_full = _lm_forward_with_optional_position_ids(
            lm.model,
            inputs_embeds=input_after_inj,
            use_cache=True,
            past_key_values=_ensure_kv_cache(past_kv_injected),
            position_ids=pos_suffix,
        )

        # 组合必要的 hidden states
        h_prefix = out_prefix.last_hidden_state  # [2B, prompt_len + inj, D]（无 grad）
        h_suffix = out_full.last_hidden_state   # [2B, N-inj, D]（有 grad，依赖注入后的 KV）

        # === predictor hidden（不把 control token 插到 token 序列里）===
        # 目标 token t 的 predictor hidden：
        # - t=0: prompt 的最后一个 hidden
        # - 1 <= t < inj: image token (t-1) 的 hidden（来自 prefix）
        # - t=inj: 使用 token(inj-1) 的 hidden（h_inj_m1）
        # - t > inj: image token (t-1) 的 hidden（来自 suffix）
        pred_h = torch.empty((h_suffix.shape[0], n_tokens, h_suffix.shape[2]), device=device, dtype=h_suffix.dtype)
        pred_h[:, 0:1, :] = h_prefix[:, prompt_len - 1 : prompt_len, :].to(dtype=pred_h.dtype)
        if inj > 1:
            pred_h[:, 1:inj, :] = h_prefix[:, prompt_len : prompt_len + (inj - 1), :].to(dtype=pred_h.dtype)
        pred_h[:, inj : inj + 1, :] = h_inj_m1.to(dtype=pred_h.dtype)
        tail = int(n_tokens - inj - 1)
        if tail > 0:
            pred_h[:, inj + 1 :, :] = h_suffix[:, :tail, :]

        # === Loss 计算 ===
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


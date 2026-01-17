from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .condenser import AttentionCondenser, CondenserConfig
from .trigger import TriggerConfig, TriggerState, cosine_sim, entropy_from_probs, should_trigger
from .translator import Translator, TranslatorConfig
from .shaper import ControlTokenShaper, ShaperConfig
from .long_condenser import LongAttentionCondenser, LongCondenserConfig


@dataclass
class LatentControllerConfig:
    enabled: bool = True

    # buffer/window
    img_hidden_window: int = 32  # 保存最近多少步 image-token hidden states 给 Condenser 用

    # trigger
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    max_triggers_per_image: int = 3

    # condenser/translator/shaper configs
    condenser: CondenserConfig = field(default_factory=CondenserConfig)
    long_condenser: LongCondenserConfig = field(default_factory=LongCondenserConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    shaper: ShaperConfig = field(default_factory=ShaperConfig)

    # think context
    think_prompt_max_tokens: int = 96


class LatentController(nn.Module):
    """
    把 Condenser / Trigger / Translator / Shaper 串起来，并负责把 control tokens
    “插入”到 KV cache（无回滚：仅额外 forward 一次 control prefix）。
    """

    def __init__(self, d_model: int, tokenizer, cfg: LatentControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.d_model = d_model

        # 子模块 dtype 由外部 trainer 决定（训练通常用 fp32 params + fp16 embeds）
        self.condenser = AttentionCondenser(d_model, cfg.condenser)
        self.long_condenser = LongAttentionCondenser(d_model, cfg.long_condenser)
        self.translator = Translator(d_model, cfg.translator)
        self.shaper = ControlTokenShaper(d_model, cfg.shaper)

        # 运行态 state（每张图/每个 stage reset）
        self._trigger_state: Optional[TriggerState] = None
        self._img_h_buf: Optional[torch.Tensor] = None  # [B, W, D]
        self._img_h_ptr: int = 0
        self._triggers_used: int = 0

    def reset(self, batch_size: int, device: torch.device):
        w = int(self.cfg.img_hidden_window)
        self._img_h_buf = torch.zeros((batch_size, w, self.d_model), device=device, dtype=torch.float16)
        self._img_h_ptr = 0
        self._trigger_state = TriggerState(batch_size, self.cfg.trigger.window, device=device)
        self._triggers_used = 0
        # 重要：长序列缓存必须随每张图/每个 batch reset，否则会把上一轮计算图带进来，
        # 触发 “Trying to backward through the graph a second time”.
        self._img_h_long_list = []
        self._img_h_long_b = int(batch_size)
        self._img_h_long_d = int(self.d_model)
        self._img_h_long_device = device

    def _push_img_hidden(self, h_img_last: torch.Tensor) -> torch.Tensor:
        """
        h_img_last: [B, D]
        返回用于 Condenser 的序列: [B, S, D] (S<=W)
        """
        assert self._img_h_buf is not None
        b, w, d = self._img_h_buf.shape
        # 训练时 LM 冻结：不需要把 h_img_last 的图跨步/跨 batch 保留下来，detach 以免图被意外复用并省显存
        self._img_h_buf[:, self._img_h_ptr] = h_img_last.detach().to(dtype=torch.float16)
        self._img_h_ptr = (self._img_h_ptr + 1) % w

        # 以时间顺序展开 buffer（最近的在最后）
        # idx: [ptr, ptr+1, ..., w-1, 0, 1, ..., ptr-1]
        idx = torch.arange(w, device=h_img_last.device)
        idx = (idx + self._img_h_ptr) % w
        seq = self._img_h_buf[:, idx]
        return seq

    def _push_img_long_hidden(self, h_img_last: torch.Tensor) -> torch.Tensor:
        
        if h_img_last.dim() != 2:
            raise ValueError(f"h_img_last must be [B,D], got {tuple(h_img_last.shape)}")

        # lazy init / safety re-init (batch size/device 变化时重置历史)
        lst = getattr(self, "_img_h_long_list", None)
        if lst is None:
            self._img_h_long_list = []
            self._img_h_long_b = int(h_img_last.shape[0])
            self._img_h_long_d = int(h_img_last.shape[1])
            self._img_h_long_device = h_img_last.device
        else:
            if (
                int(h_img_last.shape[0]) != getattr(self, "_img_h_long_b", int(h_img_last.shape[0]))
                or int(h_img_last.shape[1]) != getattr(self, "_img_h_long_d", int(h_img_last.shape[1]))
                or h_img_last.device != getattr(self, "_img_h_long_device", h_img_last.device)
            ):
                # batch size/device 变化时重置历史
                self._img_h_long_list = []
                self._img_h_long_b = int(h_img_last.shape[0])
                self._img_h_long_d = int(h_img_last.shape[1])
                self._img_h_long_device = h_img_last.device

        # 同理：长期缓存 detach，避免跨步/跨 batch 把旧计算图拼回来
        self._img_h_long_list.append(h_img_last.detach().to(dtype=torch.float16).unsqueeze(1))  # [B,1,D]
        return torch.cat(self._img_h_long_list, dim=1)

    def _think_latent(
        self,
        model,  # MultiModalityCausalLM
        prompt_text: str,
        m_tokens: torch.Tensor,  # [B, M, D]
    ) -> torch.Tensor:
        """
        使用 language model 做一次短 forward,拿 pooled hidden 作为 z_vec。
        不输出可读 CoT,只保留 hidden。
        """
        # tokenize prompt summary
        ids = self.tokenizer.encode(prompt_text)
        if len(ids) > self.cfg.think_prompt_max_tokens:
            ids = ids[-self.cfg.think_prompt_max_tokens :]
        input_ids = torch.tensor(ids, device=m_tokens.device, dtype=torch.long).unsqueeze(0)
        # embed text -> [1, T, D] then expand to [B, T, D]
        text_emb = model.language_model.get_input_embeddings()(input_ids).expand(
            m_tokens.shape[0], -1, -1
        )
        inputs_embeds = torch.cat([text_emb, m_tokens], dim=1).to(torch.float16)  # [B, T+M, D]
        out = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=False)
        h = out.last_hidden_state  # [B, T+M, D]
        z_vec = h.mean(dim=1)
        return z_vec

    def maybe_inject(
        self,
        model,  # MultiModalityCausalLM
        past_key_values,
        step_idx: int,
        prompt_vec: torch.Tensor,  # [B, D]
        h_img_last_cond: torch.Tensor,  # [B, D]
        next_token_probs: torch.Tensor,  # [B, V] (通常是 CFG 后的 probs)
        prompt_text_for_think: str,
    ):
        """
        在生成循环中调用。
        - 更新视觉记忆 (Condenser)
        - 计算 trigger
        - 触发则：think -> translate -> shape -> prefix injection (update KV)
        返回: (new_past_key_values, did_inject: bool)
        """
        if not self.cfg.enabled:
            return past_key_values, False

        if self._img_h_buf is None or self._trigger_state is None:
            self.reset(batch_size=h_img_last_cond.shape[0], device=h_img_last_cond.device)

        img_seq = self._push_img_hidden(h_img_last_cond)  # [B,W,D]
        long_img = self._push_img_long_hidden(h_img_last_cond)

        # 预算 & 频率控制
        if self._triggers_used >= self.cfg.max_triggers_per_image:
            return past_key_values, False
        if (step_idx + 1) % self.cfg.trigger.check_every != 0:
            return past_key_values, False
        # print("pass mod : ",step_idx)
        # print("h len : ",h_img_last_cond.shape)

        # import pdb; pdb.set_trace()

        # Condenser
        m_tokens, m_vec = self.condenser(img_seq)
        
        # import pdb; pdb.set_trace()

        # Trigger features
        s_t = cosine_sim(m_vec, prompt_vec)  # [B]
        u_t = entropy_from_probs(next_token_probs)  # [B]
        delta_s, var_s = self._trigger_state.update(s_t)
        # print(s_t,u_t,delta_s,var_s)
        trig = should_trigger(self.cfg.trigger, s_t=s_t, delta_s=delta_s, var_s=var_s, u_t=u_t)

        # import pdb; pdb.set_trace()

        # if not bool(trig.any()):
        #     return past_key_values, False
        # train time : trigger always True

        # BEGINING THINKING ---------
        # LONG CONDENSER
        # print("LONG ",long_img.shape)
        long_m_tokens, long_m_vec = self.long_condenser(long_img)

        # Think -> Translator -> Shaper
        # 训练时：Janus-Pro 通常冻结，我们不需要沿着 LM 的 think 路径反传到 m_tokens；
        # 否则会让显存暴涨（需要保存一整段 LLM activations 用于输入梯度）。
        with torch.no_grad():
            z_vec = self._think_latent(
                model=model,
                prompt_text=prompt_text_for_think,
                m_tokens=long_m_tokens.detach(),
            )
        c_vec = self.translator(z_vec=z_vec, m_vec=long_m_vec, p_vec=prompt_vec)  # [B,D]
        ctrl_tokens = self.shaper.make_control_tokens_for_cfg(c_vec)  # [2B,K,D]


        # import pdb; pdb.set_trace()

        # Prefix injection: 额外 forward 一次，把 control tokens 写进 KV
        inj_out = model.language_model.model(
            inputs_embeds=ctrl_tokens.to(torch.float16),
            use_cache=True,
            past_key_values=past_key_values,
        )
        self._triggers_used += 1
        return inj_out.past_key_values, True

    def maybe_control_tokens(
        self,
        *,
        model,  # MultiModalityCausalLM
        step_idx: int,
        prompt_vec: torch.Tensor,  # [B,D]
        h_img_last_cond: torch.Tensor,  # [B,D]
        next_token_probs: torch.Tensor,  # [B,V]
        prompt_text_for_think: str,
    ):
        """
        teacher-forcing 并行 forward 用：
        - 不修改 KV cache
        - 只根据当前 step 的信息，决定是否触发，并返回需要插入的 control token embeddings（CFG batch: [2B,K,D]）
        """
        if not self.cfg.enabled:
            return None, False

        if self._img_h_buf is None or self._trigger_state is None:
            self.reset(batch_size=h_img_last_cond.shape[0], device=h_img_last_cond.device)

        _ = self._push_img_hidden(h_img_last_cond)  # 更新短期 buffer
        long_img = self._push_img_long_hidden(h_img_last_cond)

        if self._triggers_used >= self.cfg.max_triggers_per_image:
            return None, False
        if (step_idx + 1) % self.cfg.trigger.check_every != 0:
            return None, False

        # Condenser / LongCondenser
        #（训练对齐当前实现：trigger 判定目前等价于“按频率触发”，不额外 gate）
        _, m_vec = self.condenser(self._img_h_buf)  # [B,D]（仅用于对齐接口；不用也可以）
        long_m_tokens, long_m_vec = self.long_condenser(long_img)

        with torch.no_grad():
            z_vec = self._think_latent(
                model=model,
                prompt_text=prompt_text_for_think,
                m_tokens=long_m_tokens.detach(),
            )

        c_vec = self.translator(z_vec=z_vec, m_vec=long_m_vec, p_vec=prompt_vec)
        ctrl_tokens = self.shaper.make_control_tokens_for_cfg(c_vec)  # [2B,K,D]
        self._triggers_used += 1
        return ctrl_tokens, True



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .condenser import AttentionCondenser, CondenserConfig
from .trigger import (
    TriggerConfig,
    TriggerState,
    build_trigger_features,
    cosine_sim,
    entropy_from_probs,
)
from .translator import Translator, TranslatorConfig
from .shaper import ControlTokenShaper, ShaperConfig
from .long_condenser import LongAttentionCondenser, LongCondenserConfig


@dataclass
class LatentControllerConfig:
    enabled: bool = True

    # buffer/window
    img_hidden_window: int = 32  # Keep the last N image-token hidden states for the Condenser.

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
    Wire up Condenser / Trigger / Translator / Shaper, and inject control tokens into the KV cache.
    This injection is non-reversible: we only do one extra forward pass for the control prefix.
    """

    def __init__(self, d_model: int, tokenizer, cfg: LatentControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.d_model = d_model

        # Sub-module dtype is decided by the external trainer
        # (training often uses fp32 params + fp16 embeddings).
        self.condenser = AttentionCondenser(d_model, cfg.condenser)
        self.long_condenser = LongAttentionCondenser(d_model, cfg.long_condenser)
        self.translator = Translator(d_model, cfg.translator)
        self.shaper = ControlTokenShaper(d_model, cfg.shaper)

        # Runtime state (reset per image / per stage).
        self._trigger_state: Optional[TriggerState] = None
        self._img_h_buf: Optional[torch.Tensor] = None  # [B, W, D]
        self._img_h_ptr: int = 0
        self._triggers_used: int = 0
        # Logic: if an injection happens at a check step, we skip the next check step
        # (but still update buffers every step). In batched mode this is tracked per-sample.
        self._skip_next_check: Optional[torch.Tensor] = None  # [B] int32 (0/1)

    def reset(self, batch_size: int, device: torch.device):
        w = int(self.cfg.img_hidden_window)
        self._img_h_buf = torch.zeros((batch_size, w, self.d_model), device=device, dtype=torch.float16)
        self._img_h_ptr = 0
        self._trigger_state = TriggerState(batch_size, self.cfg.trigger.window, device=device)
        self._triggers_used = 0
        self._skip_next_check = torch.zeros((batch_size,), device=device, dtype=torch.int32)
        # Important: long-sequence caches must be reset per image / per batch, otherwise we may
        # carry over a previous computation graph and trigger:
        # "Trying to backward through the graph a second time".
        self._img_h_long_list = []
        self._img_h_long_b = int(batch_size)
        self._img_h_long_d = int(self.d_model)
        self._img_h_long_device = device

    def mark_injected(self, *, mask: torch.Tensor):
        """
        Called by rollout after an injection is actually applied:
        - For the masked samples, we will not return obs at the next check step (skip one check).
        mask: [B] bool/0-1
        """
        if self._skip_next_check is None:
            raise RuntimeError("LatentController not reset before mark_injected()")
        if mask.dim() != 1:
            raise ValueError(f"mask must be [B], got {tuple(mask.shape)}")
        m = (mask > 0.5) if mask.is_floating_point() else mask.to(torch.bool)
        if bool(m.any()):
            self._skip_next_check[m] = 1

    def _push_img_hidden(self, h_img_last: torch.Tensor) -> torch.Tensor:
        """
        h_img_last: [B, D]
        Returns the sequence for the Condenser: [B, S, D] (S<=W)
        """
        assert self._img_h_buf is not None
        b, w, d = self._img_h_buf.shape
        # LM is frozen during training: detach to avoid accidentally reusing graphs across steps/batches,
        # and to save memory.
        self._img_h_buf[:, self._img_h_ptr] = h_img_last.detach().to(dtype=torch.float16)
        self._img_h_ptr = (self._img_h_ptr + 1) % w

        # Unroll the circular buffer in chronological order (latest at the end).
        # idx: [ptr, ptr+1, ..., w-1, 0, 1, ..., ptr-1]
        idx = torch.arange(w, device=h_img_last.device)
        idx = (idx + self._img_h_ptr) % w
        seq = self._img_h_buf[:, idx]
        return seq

    def _push_img_long_hidden(self, h_img_last: torch.Tensor) -> torch.Tensor:
        
        if h_img_last.dim() != 2:
            raise ValueError(f"h_img_last must be [B,D], got {tuple(h_img_last.shape)}")

        # Lazy init / safety re-init (reset history when batch size/device changes).
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
                # Reset history when batch size/device changes.
                self._img_h_long_list = []
                self._img_h_long_b = int(h_img_last.shape[0])
                self._img_h_long_d = int(h_img_last.shape[1])
                self._img_h_long_device = h_img_last.device

        # Likewise: detach long-term cache to avoid stitching old graphs across steps/batches.
        self._img_h_long_list.append(h_img_last.detach().to(dtype=torch.float16).unsqueeze(1))  # [B,1,D]
        return torch.cat(self._img_h_long_list, dim=1)

    def _think_latent(
        self,
        model,  # MultiModalityCausalLM
        prompt_text: str,
        m_tokens: torch.Tensor,  # [B, M, D]
    ) -> torch.Tensor:
        """
        Run a short language-model forward pass and use pooled hidden states as z_vec.
        We do not output readable CoT; we only keep hidden states.
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

    def observe_trigger_inputs(
        self,
        *,
        step_idx: int,
        prompt_vec: torch.Tensor,  # [B,D]
        h_img_last_cond: torch.Tensor,  # [B,D]
        next_token_probs: torch.Tensor,  # [B,V]
    ) -> Optional[dict]:
        """
        For RL rollouts: observation only (no injection; does not modify KV cache).

        Returning None means:
        - control disabled
        - trigger budget reached
        - not a check step (check_every)

        Otherwise returns:
          {
            "x_t": [B,4],
            "s_t": [B], "delta_s":[B], "var_s":[B], "u_t":[B],
            "m_vec": [B,D],
            "long_img": [B,S,D]  (for later inject_from_long_img)
          }
        """
        if not self.cfg.enabled:
            return None

        if self._img_h_buf is None or self._trigger_state is None:
            self.reset(batch_size=h_img_last_cond.shape[0], device=h_img_last_cond.device)

        # Update buffers every step (both short/long), but only return obs on check steps.
        img_seq = self._push_img_hidden(h_img_last_cond)  # [B,W,D]
        long_img = self._push_img_long_hidden(h_img_last_cond)  # [B,S,D]

        if self._triggers_used >= self.cfg.max_triggers_per_image:
            return None
        # Note: step_idx here is t-1 (the position consuming token t-1);
        # token_idx=t=step_idx+1.
        token_idx = int(step_idx) + 1
        check_every = int(self.cfg.trigger.check_every)
        if check_every <= 0:
            raise ValueError(f"trigger.check_every must be > 0, got {check_every}")

        # Keep the window constraint: for the first `window` tokens we only accumulate buffers,
        # and do not trigger (stats are too short otherwise).
        if token_idx <= int(self.cfg.trigger.window):
            return None
        # Only check at check steps (e.g., check_every=64 -> 64/128/192...).
        if (token_idx % check_every) != 0:
            return None
        # Key: skip the first check step (e.g., check_every=64 -> do not check at token_idx=64; start from 128).
        # Nothing is hard-coded; everything depends on check_every.
        if token_idx < (2 * check_every):
            return None
        # If injection happened at the previous check step, skip the next check step (per-sample; only once).
        check_mask = torch.ones((h_img_last_cond.shape[0],), device=h_img_last_cond.device, dtype=torch.bool)
        if self._skip_next_check is not None:
            skip = self._skip_next_check.to(torch.bool)
            if bool(skip.any()):
                check_mask = check_mask & (~skip)
                # Reset: only skip once.
                self._skip_next_check[skip] = 0
        if not bool(check_mask.any()):
            return None

        _m_tokens, m_vec = self.condenser(img_seq)
        s_t = cosine_sim(m_vec, prompt_vec)
        u_t = entropy_from_probs(next_token_probs)
        delta_s, var_s = self._trigger_state.update(s_t)
        x_t = build_trigger_features(s_t=s_t, delta_s=delta_s, var_s=var_s, u_t=u_t)
        return {
            "x_t": x_t,
            "s_t": s_t,
            "delta_s": delta_s,
            "var_s": var_s,
            "u_t": u_t,
            "m_vec": m_vec,
            "long_img": long_img,
            "check_mask": check_mask,
        }



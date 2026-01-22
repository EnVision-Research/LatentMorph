from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch


@dataclass
class HpsRewardConfig:
    enabled: bool = True
    device: str = "cuda"
    # DanceGRPO style: HPSv2 v2.1 needs two files:
    # - HPS_v2.1_compressed.pt
    # - open_clip_pytorch_model.bin
    ckpt_dir: str = ""
    version: str = "v2.1"


class HpsV21Scorer:
    """
    HPS-v2.1 scorer (commonly used in DanceGRPO).

    Notes:
    - This is an optional dependency: if `hpsv2` is not installed, we raise a clear error.
    - We align with the DanceGRPO implementation: use `hpsv2.src.open_clip` + 2 ckpt files.
    """

    def __init__(self, cfg: HpsRewardConfig):
        self.cfg = cfg
        self._hps = None
        if not bool(cfg.enabled):
            return

        try:
            import hpsv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HPS-v2.1 reward requires the `hpsv2` pip package. "
                "If you don't want to use HPS, set `--hps_weight 0`."
            ) from e

        self._hps = hpsv2
        self.device = torch.device(str(cfg.device))

        # lazy init
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        self._ckpt_dir = Path(str(cfg.ckpt_dir)).expanduser() if str(cfg.ckpt_dir).strip() else None
        self._open_clip_bin = None
        self._hps_ckpt = None

        if self._ckpt_dir is not None:
            self._open_clip_bin = self._ckpt_dir / "open_clip_pytorch_model.bin"
            self._hps_ckpt = self._ckpt_dir / "HPS_v2.1_compressed.pt"

        if self._ckpt_dir is None or self._open_clip_bin is None or self._hps_ckpt is None:
            raise ValueError(
                "HPS reward requires `ckpt_dir` containing "
                "`open_clip_pytorch_model.bin` and `HPS_v2.1_compressed.pt`."
            )
        if not self._open_clip_bin.exists() or not self._hps_ckpt.exists():
            raise FileNotFoundError(
                "Missing HPS ckpt files. Please make sure the following exist:\n"
                f"- {self._open_clip_bin}\n"
                f"- {self._hps_ckpt}\n"
                "Please download them manually and place them under the directory passed via `--hps_ckpt_dir`."
            )

    def _maybe_init_model(self):
        if self._model is not None:
            return
        assert self._hps is not None

        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer  # type: ignore
        except Exception as e:
            raise ImportError(
                "hpsv2 is installed but `hpsv2.src.open_clip` is missing "
                "(wrong version or incomplete install). Please install HPSv2 following DanceGRPO's recommended method."
            ) from e

        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            str(self._open_clip_bin),
            precision="amp",
            device=str(self.device),
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        ckpt = torch.load(str(self._hps_ckpt), map_location=str(self.device))
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        try:
            model.load_state_dict(ckpt)
        except Exception:
            # Some checkpoints might be {'state_dict': ...}.
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                raise

        self._model = model.to(self.device).eval()
        self._preprocess = preprocess_val
        self._tokenizer = get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def score(self, *, images, prompts: List[str]) -> torch.Tensor:
        """
        return: [B] float32 (cpu)
        """
        if not self.cfg.enabled:
            return torch.zeros((len(prompts),), dtype=torch.float32)
        if self._hps is None:
            raise RuntimeError("hpsv2 not initialized")
        self._maybe_init_model()
        assert self._model is not None and self._preprocess is not None and self._tokenizer is not None

        # preprocess images
        img_t = torch.stack([self._preprocess(im) for im in images], dim=0).to(self.device)
        txt_t = self._tokenizer(prompts).to(self.device)

        # open_clip-style outputs: try to be compatible with multiple versions.
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = self._model(img_t, txt_t)

        logits_per_image = None
        if isinstance(out, dict):
            logits_per_image = out.get("logits_per_image", None)
            if logits_per_image is None:
                img_f = out.get("image_features", None)
                txt_f = out.get("text_features", None)
                logit_scale = out.get("logit_scale", None)
                if img_f is not None and txt_f is not None:
                    # Align with DanceGRPO: do not use logit_scale by default (or treat it as 1.0)
                    # to obtain raw cosine similarity. This keeps scores around ~0.2, similar to CLIP.
                    logits_per_image = (img_f @ txt_f.t())
        if logits_per_image is None and isinstance(out, (tuple, list)):
            if len(out) >= 2 and out[0] is not None and out[1] is not None:
                img_f, txt_f = out[0], out[1]
                logits_per_image = (img_f @ txt_f.t())
            elif len(out) >= 1:
                logits_per_image = out[0]
        if logits_per_image is None and hasattr(out, "logits_per_image"):
            logits_per_image = getattr(out, "logits_per_image")
        if logits_per_image is None:
            raise RuntimeError("Unexpected HPSv2 output format: cannot obtain `logits_per_image`.")

        # Take paired scores (diagonal).
        if logits_per_image.ndim == 2:
            score = torch.diagonal(logits_per_image)
        else:
            score = logits_per_image
        return score.to(torch.float32).detach().cpu()



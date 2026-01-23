from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class ClipRewardConfig:
    enabled: bool = True
    # Prefer transformers' CLIP. If you have open_clip locally, you can extend this later.
    model_name_or_path: str = "openai/clip-vit-large-patch14"
    # Default to allow online download. Set True for fully offline runs.
    local_files_only: bool = False
    device: str = "cuda"


class ClipTextImageScorer:
    """
    Compute CLIP text-image cosine similarity, returning one score per image.
    Dependencies: transformers + PIL
    """

    def __init__(self, cfg: ClipRewardConfig):
        self.cfg = cfg
        if not bool(cfg.enabled):
            self.model = None
            self.processor = None
            return

        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "CLIP reward requires `transformers`. Please install it and ensure it can be imported."
            ) from e

        # Keep this simple for open-source usage: honor local_files_only; allow online download when False.
        self.processor = CLIPProcessor.from_pretrained(
            str(cfg.model_name_or_path), local_files_only=bool(cfg.local_files_only)
        )
        self.model = CLIPModel.from_pretrained(
            str(cfg.model_name_or_path), local_files_only=bool(cfg.local_files_only)
        )
        self.model = self.model.to(str(cfg.device)).eval()

    @torch.no_grad()
    def score(self, *, images, prompts: List[str]) -> torch.Tensor:
        """
        images: List[PIL.Image]
        prompts: List[str]
        return: [B] float32
        """
        if not self.cfg.enabled:
            return torch.zeros((len(prompts),), dtype=torch.float32)
        assert self.model is not None and self.processor is not None
        device = next(self.model.parameters()).device
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model(**inputs)
        img = out.image_embeds
        txt = out.text_embeds
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (img * txt).sum(dim=-1)
        return sim.to(torch.float32).detach().cpu()



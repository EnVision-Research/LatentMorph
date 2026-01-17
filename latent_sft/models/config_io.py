from __future__ import annotations

import json
import os
from typing import Any, Dict

from latent_control.controller import LatentControllerConfig
from latent_control.condenser import CondenserConfig
from latent_control.long_condenser import LongCondenserConfig
from latent_control.trigger import TriggerConfig
from latent_control.translator import TranslatorConfig
from latent_control.shaper import ShaperConfig


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_latent_controller_config(data: Dict[str, Any]) -> LatentControllerConfig:
    """
    把 dict 配置映射成 LatentControllerConfig（含子模块 config）。
    只解析 latent_control 下的字段；缺省字段沿用 dataclass 默认值。
    """
    lc = data.get("latent_control", {}) if "latent_control" in data else data

    cfg = LatentControllerConfig()
    for k in ("enabled", "img_hidden_window", "max_triggers_per_image", "think_prompt_max_tokens", "condenser_mode"):
        if k in lc:
            setattr(cfg, k, lc[k])

    if "trigger" in lc:
        cfg.trigger = TriggerConfig(**lc["trigger"])
    if "condenser" in lc:
        cfg.condenser = CondenserConfig(**lc["condenser"])
    if "long_condenser" in lc:
        cfg.long_condenser = LongCondenserConfig(**lc["long_condenser"])
    if "translator" in lc:
        cfg.translator = TranslatorConfig(**lc["translator"])
    if "shaper" in lc:
        cfg.shaper = ShaperConfig(**lc["shaper"])
    return cfg


def resolve_config_path(default_path: str) -> str:
    """
    支持用环境变量覆盖配置路径：
      TWIG_CONFIG=/abs/path/to/models/config.json
    """
    return os.environ.get("TWIG_CONFIG", default_path)



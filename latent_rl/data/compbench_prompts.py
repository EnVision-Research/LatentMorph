from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class CompBenchPromptsConfig:
    """
    Read T2I-CompBench prompt text files (one prompt per line).

    If `prompts_file` is a directory, we scan all `.txt` files inside and merge them.
    If `prompts_file` is a file, we only read that file.

    Examples:
      prompts_file="data/T2I-CompBench/examples/dataset"  # directory: scan all .txt
      prompts_file="data/T2I-CompBench/examples/dataset/color.txt"  # file: read only this file
    """

    prompts_file: str
    max_prompts: int = 0  # 0 means no truncation
    shuffle: bool = True
    seed: int = 42


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.read().splitlines()]
    # drop empty
    lines = [x for x in lines if x]
    return lines


def _collect_prompt_files(prompts_file: str) -> List[str]:
    """
    Collect prompt file paths.
    If `prompts_file` is a directory, scan all `.txt` files; if it is a file, return that file.
    """
    p = os.path.abspath(prompts_file)
    if not os.path.exists(p):
        raise FileNotFoundError(f"prompts_file not found: {p}")
    
    if os.path.isdir(p):
        # Directory: scan all `.txt` files.
        pattern = os.path.join(p, "*.txt")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            raise ValueError(f"no .txt files found in directory: {p}")
        return files
    elif os.path.isfile(p):
        # File: return the file itself.
        return [p]
    else:
        raise ValueError(f"prompts_file must be a file or directory: {p}")


class CompBenchPromptDataset(Dataset):
    def __init__(self, cfg: CompBenchPromptsConfig):
        self.cfg = cfg
        files = _collect_prompt_files(cfg.prompts_file)
        
        # Merge prompts from all files.
        prompts = []
        for f in files:
            try:
                lines = _read_lines(f)
                prompts.extend(lines)
            except Exception as e:
                print(f"[warn] failed to read {f}: {e}", flush=True)
                continue
        
        # Deduplicate while preserving order.
        seen = set()
        unique_prompts = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)
        prompts = unique_prompts
        
        if int(cfg.max_prompts) > 0:
            prompts = prompts[: int(cfg.max_prompts)]

        if bool(cfg.shuffle):
            g = torch.Generator()
            g.manual_seed(int(cfg.seed))
            idx = torch.randperm(len(prompts), generator=g).tolist()
            prompts = [prompts[i] for i in idx]

        if len(prompts) == 0:
            raise ValueError(f"no prompts loaded from: {cfg.prompts_file} (checked {len(files)} files)")
        
        self.prompts = prompts
        self.source_files = files  # Keep source file paths for debugging.

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return str(self.prompts[int(idx)])



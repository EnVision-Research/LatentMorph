from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple


def _read_nonempty_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.read().splitlines()]
    return [x for x in lines if x]


def count_prompts(pattern: str) -> Tuple[Dict[str, int], int, int]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matched pattern: {pattern}")

    per_file: Dict[str, int] = {}
    all_prompts: List[str] = []
    for fp in files:
        lines = _read_nonempty_lines(fp)
        per_file[fp] = len(lines)
        all_prompts.extend(lines)

    total = len(all_prompts)
    unique_total = len(set(all_prompts))
    return per_file, total, unique_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        type=str,
        default="data/T2I-CompBench/examples/dataset/*.txt",
        help="glob pattern, e.g. /path/to/examples/dataset/*.txt or .../color_*.txt",
    )
    ap.add_argument("--show_files", type=int, default=1, help="1=print per-file line counts, 0=print totals only")
    args = ap.parse_args()

    per_file, total, unique_total = count_prompts(str(args.pattern))

    if int(args.show_files) == 1:
        print("Per-file counts (non-empty lines):")
        for fp, n in per_file.items():
            print(f"- {fp}: {n}")
        print()

    print(f"TOTAL prompts (non-empty lines): {total}")
    print(f"UNIQUE prompts (dedup across files): {unique_total}")


if __name__ == "__main__":
    main()



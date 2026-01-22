from __future__ import annotations

import io
import os
import hashlib
import time
import urllib.parse
import urllib.request
from typing import List

import pyarrow.dataset as ds
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

# By default, place the cache directory outside LatentMorph (sibling directory); can be overridden via env var.
_DEFAULT_CACHE_DIR = os.environ.get(
    "TWIG_IMAGE_CACHE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "twig_image_cache")),
)


def _cache_path_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    ext = os.path.splitext(parsed.path)[1]
    ext = ext if ext and len(ext) <= 8 else ".img"
    name = hashlib.sha1(url.encode("utf-8")).hexdigest() + ext
    os.makedirs(_DEFAULT_CACHE_DIR, exist_ok=True)
    return os.path.join(_DEFAULT_CACHE_DIR, name)


def _download_image_to_cache(url: str, timeout: float = 20.0) -> str:
    cache_path = _cache_path_from_url(url)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    tmp = cache_path + f".part-{os.getpid()}-{int(time.time() * 1000)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, cache_path)
    return cache_path


def decode_parquet_image(cell) -> Image.Image:
    if cell is None:
        raise ValueError("image cell is None")
    if isinstance(cell, dict):
        b = cell.get("bytes", None)
        p = cell.get("path", None)
        u = cell.get("url", None)
        if b is not None and len(b) > 0:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if p:
            return Image.open(p).convert("RGB")
        if u:
            return decode_parquet_image(u)
        raise ValueError(f"image dict has no bytes/path/url: keys={list(cell.keys())}")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("http://") or s.startswith("https://"):
            cached = _download_image_to_cache(s)
            return Image.open(cached).convert("RGB")
        return Image.open(s).convert("RGB")
    if hasattr(cell, "as_py"):
        return decode_parquet_image(cell.as_py())
    raise TypeError(f"Unsupported image cell type: {type(cell)}")


class ParquetImageCaptionIterable(IterableDataset):
    def __init__(
        self,
        parquet_files: List[str],
        image_key: str = "image",
        caption_key: str = "caption_composition",
        batch_rows: int = 256,
    ):
        super().__init__()
        self.files = list(parquet_files)
        if not self.files:
            raise FileNotFoundError("No parquet files provided.")
        self.image_key = image_key
        self.caption_key = caption_key
        self.batch_rows = int(max(1, batch_rows))

    def __iter__(self):
        dataset = ds.dataset(self.files, format="parquet")
        cols = [self.image_key, self.caption_key]
        scanner = dataset.scanner(columns=cols, batch_size=self.batch_rows)
        for rb in scanner.to_batches():
            data = rb.to_pydict()
            imgs = data.get(self.image_key)
            caps = data.get(self.caption_key)
            if imgs is None or caps is None:
                raise KeyError(f"Missing columns in parquet batch. keys={list(data.keys())}")
            for img_cell, cap in zip(imgs, caps):
                if cap is None:
                    continue
                yield img_cell, str(cap)


class ParquetImageCaptionInMemory(Dataset):
    """
    Map-style dataset:
    - Load selected columns from all parquet files into memory (do not decode images upfront)
    - Decode + transforms happen in __getitem__
    This works well with DistributedSampler for "even sharding + each sample processed once".
    """

    def __init__(
        self,
        parquet_files: List[str],
        *,
        image_key: str = "image",
        caption_key: str = "caption_composition",
        img_size: int = 384,
        batch_rows: int = 2048,
        max_samples: int = 0,
    ):
        super().__init__()
        self.files = list(parquet_files)
        if not self.files:
            raise FileNotFoundError("No parquet files provided.")
        self.image_key = str(image_key)
        self.caption_key = str(caption_key)
        self.batch_rows = int(max(1, batch_rows))
        self._img_size = int(img_size)
        self._max_samples = int(max(0, max_samples))

        # transforms on-the-fly
        self.tfm = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # materialize required columns into python lists
        # NOTE: do not validate/decode here (each rank would scan the full dataset on startup; very slow).
        # If decoding fails, __getitem__ returns a dummy image so batches never become None,
        # avoiding DDP collective mismatches.
        dataset = ds.dataset(self.files, format="parquet")
        cols = [self.image_key, self.caption_key]
        scanner = dataset.scanner(columns=cols, batch_size=self.batch_rows)

        imgs_all = []
        caps_all = []
        for rb in scanner.to_batches():
            data = rb.to_pydict()
            imgs = data.get(self.image_key)
            caps = data.get(self.caption_key)
            if imgs is None or caps is None:
                raise KeyError(f"Missing columns in parquet batch. keys={list(data.keys())}")
            for img_cell, cap in zip(imgs, caps):
                if cap is None:
                    continue
                imgs_all.append(img_cell)
                caps_all.append(str(cap))
                if self._max_samples > 0 and len(caps_all) >= self._max_samples:
                    break
            if self._max_samples > 0 and len(caps_all) >= self._max_samples:
                break

        if not imgs_all:
            raise RuntimeError("Loaded 0 samples from parquet files (all captions empty?)")
        self._imgs = imgs_all
        self._caps = caps_all

    def __len__(self) -> int:
        return len(self._caps)

    def __getitem__(self, idx: int):
        img_cell = self._imgs[int(idx)]
        cap = self._caps[int(idx)]
        try:
            pil = decode_parquet_image(img_cell)
            return self.tfm(pil), cap
        except Exception:
            # Decode failure: return a dummy image (matches the Normalize output range: -1).
            dummy = torch.full((3, self._img_size, self._img_size), -1.0, dtype=torch.float32)
            return dummy, cap

def resolve_train_files(cfg: dict) -> List[str]:
    td = cfg.get("train_data", None)
    if not isinstance(td, dict):
        raise ValueError("Missing `train_data` in config.json.")

    root_dir = str(td.get("root_dir", "")).strip()
    sources = td.get("sources", None)
    if not root_dir or not isinstance(sources, list) or not sources:
        raise ValueError("config.json train_data must include `root_dir` and a non-empty `sources` list.")

    out: List[str] = []
    missing: List[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        subdir = str(s.get("subdir", "")).strip()
        template = str(s.get("template", "")).strip()
        start = int(s.get("start", 0))
        end = int(s.get("end", -1))
        if not subdir or not template or end < start:
            continue
        for idx in range(start, end + 1):
            p = os.path.join(root_dir, subdir, template.format(idx=idx))
            if os.path.exists(p):
                out.append(p)
            else:
                missing.append(p)

    if not out:
        raise FileNotFoundError("train_data did not resolve any parquet files (check root_dir/template/range).")
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Some files in train_data do not exist (first 20):\n{preview}\n... total missing={len(missing)}"
        )
    return out


def build_dataloader(
    train_files: List[str],
    *,
    img_size: int,
    image_key: str,
    caption_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    in_memory: bool = True,
    max_samples: int = 0,
) -> DataLoader:
    # In-memory map-style dataset for strict sharding.
    if in_memory:
        ds_map = ParquetImageCaptionInMemory(
            parquet_files=train_files,
            image_key=image_key,
            caption_key=caption_key,
            img_size=img_size,
            batch_rows=2048,
            max_samples=int(max_samples),
        )

        # Under DDP, each rank must have the same number of iterations; otherwise collectives (all_reduce) may time out.
        # We use DistributedSampler (shuffle=False) for even sharding; to align lengths it may pad up to <world_size samples.
        sampler = None
        if ddp and int(world_size) > 1:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                ds_map,
                num_replicas=int(world_size),
                rank=int(rank),
                shuffle=False,
                drop_last=False,
            )

        def collate_map(batch):
            imgs = []
            caps = []
            for item in batch:
                img_t, cap = item
                imgs.append(img_t)
                caps.append(cap)
            if not imgs:
                return None
            return torch.stack(imgs, dim=0), caps

        return DataLoader(
            ds_map,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_map,
        )

    # Fallback: streaming iterable (kept for compatibility)
    tfm = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    ds_iter = ParquetImageCaptionIterable(
        parquet_files=train_files,
        image_key=image_key,
        caption_key=caption_key,
        batch_rows=256,
    )

    def collate_iter(batch):
        imgs = []
        caps = []
        for img_cell, cap in batch:
            try:
                pil = decode_parquet_image(img_cell)
                imgs.append(tfm(pil))
                caps.append(cap)
            except Exception:
                continue
        if not imgs:
            return None
        return torch.stack(imgs, dim=0), caps

    return DataLoader(
        ds_iter,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_iter,
    )



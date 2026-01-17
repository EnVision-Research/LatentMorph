from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistInfo:
    ddp: bool
    fsdp: bool
    world_size: int
    rank: int
    local_rank: int
    device: torch.device


def init_distributed(device_arg: str = "cuda") -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    fsdp = ddp and (os.environ.get("USE_FSDP", "1") == "1")

    if ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_arg if torch.cuda.is_available() else "cpu")

    return DistInfo(ddp=ddp, fsdp=fsdp, world_size=world_size, rank=rank, local_rank=local_rank, device=device)


def is_main_process() -> bool:
    try:
        return (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def ddp_wrap(module: torch.nn.Module, local_rank: int) -> DDP:
    return DDP(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def fsdp_wrap(module: torch.nn.Module) -> torch.nn.Module:
    """
    Minimal FSDP wrapper (align janus-sft). This is intended for controller-sized modules.
    We do NOT wrap the frozen 7B Janus model.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

    mp = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    return FSDP(
        module,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
    )


def ddp_cleanup(ddp: bool):
    if ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



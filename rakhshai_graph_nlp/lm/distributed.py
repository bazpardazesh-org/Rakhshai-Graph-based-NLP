"""PyTorch-native distributed helpers for independent LM training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch


@dataclass
class DistributedTrainingInfo:
    backend: str = "none"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    initialized: bool = False
    fsdp_available: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_distributed_info(preferred_backend: str = "none") -> DistributedTrainingInfo:
    """Inspect PyTorch distributed state without forcing initialization."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized = bool(
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    backend = preferred_backend if world_size > 1 else "none"
    return DistributedTrainingInfo(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        initialized=initialized,
        fsdp_available=hasattr(torch.distributed, "fsdp"),
    )


def maybe_wrap_distributed(model: torch.nn.Module, backend: str = "none") -> torch.nn.Module:
    """Wrap a model only when torch.distributed is already initialized.

    This keeps normal single-process training unchanged and avoids side effects
    from initializing process groups inside library code.
    """

    if (
        backend == "ddp"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return torch.nn.parallel.DistributedDataParallel(model)
    if backend == "fsdp" and torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel
        except Exception:
            return model
        return FullyShardedDataParallel(model)
    return model


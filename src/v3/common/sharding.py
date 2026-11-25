"""Deterministic shard assignment and Slurm array helpers."""
from __future__ import annotations

import hashlib
import os
from typing import Sequence, TypeVar

T = TypeVar("T")


def stable_bucket(key: str, buckets: int) -> int:
    """Bucket from a SHA-256 of the key: independent of Python hash seeds."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) % buckets


def shard_items(items: Sequence[T], keys: Sequence[str], shard: int, shards: int) -> list[T]:
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} outside [0, {shards})")
    return [item for item, key in zip(items, keys) if stable_bucket(key, shards) == shard]


def array_task(default: int | None = None) -> int:
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        if default is None:
            raise RuntimeError("SLURM_ARRAY_TASK_ID is not set; pass --shard explicitly")
        return default
    return int(value)


def cpu_count() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value:
        return max(1, int(value))
    return max(1, len(os.sched_getaffinity(0)))

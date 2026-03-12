"""
Deterministic shard split utilities for train/val/test workflows.
"""

import json
import os
import random
from datetime import datetime


def parse_split_ratios(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"split_ratios must have 3 values, got: {raw}")
    ratios = [float(p) for p in parts]
    if any(r <= 0 for r in ratios):
        raise ValueError(f"split_ratios values must be > 0, got: {raw}")
    total = sum(ratios)
    if total <= 0:
        raise ValueError(f"split_ratios sum must be > 0, got: {raw}")
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


def _count_split(
    num_items: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    raw_counts = [num_items * r for r in ratios]
    floor_counts = [int(c) for c in raw_counts]
    remainder = num_items - sum(floor_counts)

    fractional = [
        (raw_counts[i] - floor_counts[i], i) for i in range(len(floor_counts))
    ]
    fractional.sort(reverse=True)
    for i in range(remainder):
        floor_counts[fractional[i % len(floor_counts)][1]] += 1

    if num_items >= 3:
        for idx in range(3):
            if floor_counts[idx] == 0:
                donor = max(range(3), key=lambda j: floor_counts[j])
                if floor_counts[donor] > 1:
                    floor_counts[donor] -= 1
                    floor_counts[idx] += 1

    return floor_counts[0], floor_counts[1], floor_counts[2]


def build_split_lists(
    shard_paths: list[str],
    ratios: tuple[float, float, float],
    split_seed: int = 42,
    split_shuffle: bool = False,
) -> dict[str, list[str]]:
    shards = list(shard_paths)
    if split_shuffle:
        rng = random.Random(split_seed)
        rng.shuffle(shards)

    train_n, val_n, test_n = _count_split(len(shards), ratios)
    train = shards[:train_n]
    val = shards[train_n : train_n + val_n]
    test = shards[train_n + val_n : train_n + val_n + test_n]

    return {"train": train, "val": val, "test": test}


def write_split_manifest(
    split_file: str,
    data_dir: str,
    split_lists: dict[str, list[str]],
    ratios: tuple[float, float, float],
    split_seed: int,
    split_shuffle: bool,
) -> dict:
    manifest = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_dir": data_dir,
        "split_ratios": list(ratios),
        "split_seed": split_seed,
        "split_shuffle": split_shuffle,
        "shards_total": (
            len(split_lists["train"]) + len(split_lists["val"]) + len(split_lists["test"])
        ),
        "train": [os.path.basename(s) for s in split_lists["train"]],
        "val": [os.path.basename(s) for s in split_lists["val"]],
        "test": [os.path.basename(s) for s in split_lists["test"]],
    }
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_split_manifest(split_file: str) -> dict:
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_split_shards(data_dir: str, manifest: dict) -> dict[str, list[str]]:
    resolved = {}
    for split in ("train", "val", "test"):
        names = manifest.get(split, [])
        resolved[split] = [os.path.join(data_dir, n) for n in names]
    return resolved

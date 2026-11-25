"""Crash-safe filesystem primitives shared by every v3 job.

Rules enforced here:
* Every artifact is written to a temporary sibling, fsynced, then renamed, so a
  killed job never leaves a truncated file under the final name.
* A unit of work (shard, song, stage) is finished only when its done-marker
  exists *and* the hashes recorded in the marker still match the files on disk.
  Re-running a job therefore skips verified work and redoes partial work.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def raise_open_file_limit() -> int:
    """Raise the soft open-file limit to the hard limit; returns the new soft limit.

    Every live np.memmap holds a duplicated descriptor. A placement loader worker maps 17 fields x 64
    shards (~1,088), over Amarel's default soft limit of 1,024 (hard: 524,288). Call before forking
    loader workers; they inherit the limit.
    """
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or soft < hard:
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 1 << 20)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


@contextlib.contextmanager
def atomic_path(path: str | os.PathLike) -> Iterator[Path]:
    """Yield a temp path in the destination directory; rename on success only."""
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        with open(tmp, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, final)
        dir_fd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_bytes_atomic(path: str | os.PathLike, data: bytes) -> str:
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)
    return sha256_bytes(data)


def write_json_atomic(path: str | os.PathLike, value: Any) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return write_bytes_atomic(path, data)


def write_jsonl_atomic(path: str | os.PathLike, rows: Iterable[Any]) -> str:
    data = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    return write_bytes_atomic(path, data)


def write_npy_atomic(path: str | os.PathLike, array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return write_bytes_atomic(path, buffer.getvalue())


def write_npz_atomic(path: str | os.PathLike, **arrays: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    return write_bytes_atomic(path, buffer.getvalue())


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | os.PathLike) -> list[Any]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mark_done(marker: str | os.PathLike, files: Iterable[str | os.PathLike], **extra: Any) -> None:
    """Record completion with hashes of every produced file (relative names)."""
    marker = Path(marker)
    hashes = {Path(f).name: sha256_file(f) for f in files}
    write_json_atomic(marker, {"complete": True, "files": hashes, "host": socket.gethostname(),
                               "finished_unix": time.time(), **extra})


def is_done(marker: str | os.PathLike, *, verify: bool = True, expect: dict[str, Any] | None = None) -> bool:
    """True only if the marker exists, its bindings match ``expect``, and files verify."""
    marker = Path(marker)
    if not marker.is_file():
        return False
    try:
        record = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if not record.get("complete"):
        return False
    for key, value in (expect or {}).items():
        if record.get(key) != value:
            return False
    if verify:
        for name, digest in record.get("files", {}).items():
            path = marker.parent / name
            if not path.is_file() or sha256_file(path) != digest:
                return False
    return True


def write_status(path: str | os.PathLike, state: str, **fields: Any) -> None:
    """Stage status for dependents and for scripts/amarel/v3/status.sh."""
    if state not in {"running", "incomplete", "complete", "failed"}:
        raise ValueError(f"unknown status {state}")
    write_json_atomic(path, {"state": state, "updated_unix": time.time(),
                             "host": socket.gethostname(),
                             "slurm_job": os.environ.get("SLURM_JOB_ID"),
                             "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID"), **fields})


def require_complete(path: str | os.PathLike, what: str) -> dict:
    """Refuse to consume an unfinished parent stage."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{what}: missing status {path}")
    record = read_json(path)
    if record.get("state") != "complete":
        raise RuntimeError(f"{what}: parent stage is {record.get('state')!r}, not complete ({path})")
    return record

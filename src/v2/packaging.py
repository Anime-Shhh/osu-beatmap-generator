"""Strict, atomic output packaging for validated v2 generation results."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Callable, Optional

from .evaluation import Candidate
from .generation import GenerationResult


class PackagingError(RuntimeError):
    """Raised when a result is not safe to expose as a generated map package."""


@dataclass(frozen=True)
class GeneratedPackage:
    """Relative artifact facts after a successful atomic package write."""

    output_dir: Path
    osu_path: Path
    manifest_path: Path
    osu_sha256: str


def _safe_output_name(value: str, *, suffix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackagingError("output filename must be non-empty")
    normalized = value.replace("\\", "/")
    path = PurePath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise PackagingError("output filename must be a basename")
    if "\x00" in value:
        raise PackagingError("output filename must not contain NUL")
    if path.suffix.lower() != suffix.lower():
        raise PackagingError(f"output filename must end in {suffix}")
    return path.name


def _compile(map_spec: Any, compiler: Optional[Callable[[Any], str]]) -> str:
    if compiler is not None:
        content = compiler(map_spec)
    else:
        try:
            from .mapspec import compile_osu
        except ImportError as exc:
            raise PackagingError("MapSpec compiler is unavailable") from exc
        content = compile_osu(map_spec, validate=True)
    if not isinstance(content, str) or not content.strip():
        raise PackagingError("compiler returned no .osu content")
    if "[TimingPoints]" not in content or "[HitObjects]" not in content:
        raise PackagingError("compiler output is missing required osu! sections")
    return content if content.endswith("\n") else f"{content}\n"


def _round_trip_validate(content: str) -> None:
    """Use the canonical parser/validator when present; otherwise compiler is gate."""

    try:
        from .mapspec import mapspec_from_osu, validate_map_spec
    except ImportError:
        return
    parsed = mapspec_from_osu(content)
    report = validate_map_spec(parsed)
    if not bool(getattr(report, "valid", False)):
        errors = getattr(report, "errors", ())
        raise PackagingError(f"compiled .osu failed round-trip validation: {', '.join(map(str, errors))}")


def _write_atomic(path: Path, content: bytes, *, overwrite: bool) -> None:
    """Write a complete sibling temp file before exposing its final name."""

    if path.exists() and not overwrite:
        raise PackagingError(f"refusing to overwrite existing artifact: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # A hard-link create is atomic within the output directory and
            # avoids a check-then-replace race that could overwrite another
            # process's artifact between ``exists`` and ``replace``.
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise PackagingError(f"refusing to overwrite existing artifact: {path.name}") from exc
            temporary_path.unlink()
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def write_generation_package(
    result: GenerationResult,
    selected_candidate: Candidate,
    output_dir: str | Path,
    *,
    osu_filename: str = "ranked-like-draft.osu",
    manifest_filename: str = "ranked-like-draft.manifest.json",
    compiler: Optional[Callable[[Any], str]] = None,
    overwrite: bool = False,
) -> GeneratedPackage:
    """Write exactly one validated map and its audit manifest.

    The manifest is written last, so its presence indicates that the map exists
    and both artifacts were fully flushed.  This function refuses to serialize
    invalid or unselected candidates; callers should retain rejected candidates
    only in the manifest for auditability, never as playable output files.
    """

    selected = result.selected_candidate
    if selected is None:
        raise PackagingError("cannot package a result without a valid selected candidate")
    if selected_candidate.candidate_id != selected.candidate_id:
        raise PackagingError("selected candidate does not match the result manifest")
    if not selected.valid:
        raise PackagingError("cannot package an invalid candidate")

    osu_filename = _safe_output_name(osu_filename, suffix=".osu")
    manifest_filename = _safe_output_name(manifest_filename, suffix=".json")
    if osu_filename == manifest_filename:
        raise PackagingError("map and manifest filenames must differ")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise PackagingError("output_dir is not a directory")
    osu_path = destination / osu_filename
    manifest_path = destination / manifest_filename
    content = _compile(selected_candidate.map_spec, compiler)
    _round_trip_validate(content)
    content_bytes = content.encode("utf-8")
    digest = hashlib.sha256(content_bytes).hexdigest()
    artifacts = {
        "osu_file": osu_filename,
        "osu_sha256": digest,
        "manifest_file": manifest_filename,
        "format": "osu file format v14",
    }
    manifest_bytes = result.to_json(artifacts=artifacts).encode("utf-8")

    # The map goes first and the manifest is the completion marker.  If a
    # failure occurs after the map write, callers can safely remove the orphan
    # map or retry with overwrite=True; no false "completed" manifest exists.
    _write_atomic(osu_path, content_bytes, overwrite=overwrite)
    try:
        _write_atomic(manifest_path, manifest_bytes, overwrite=overwrite)
    except BaseException:
        # Preserve a pre-existing map when overwriting was requested.  For new
        # packages remove only the file we just created, never broad directories.
        if not overwrite:
            osu_path.unlink(missing_ok=True)
        raise
    return GeneratedPackage(
        output_dir=destination,
        osu_path=osu_path,
        manifest_path=manifest_path,
        osu_sha256=digest,
    )

"""Strict, atomic output packaging for validated v2 generation results."""

from __future__ import annotations

import hashlib
import os
import shutil
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
    audio_path: Path | None = None
    audio_sha256: str | None = None


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


def _safe_asset_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackagingError("audio filename must be non-empty")
    normalized = value.replace("\\", "/")
    path = PurePath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise PackagingError("audio filename must be a basename")
    if "\x00" in value:
        raise PackagingError("audio filename must not contain NUL")
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Copy one asset without exposing partial bytes at the destination."""
    if destination.exists() and not overwrite:
        raise PackagingError(f"refusing to overwrite existing artifact: {destination.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise PackagingError(
                    f"refusing to overwrite existing artifact: {destination.name}"
                ) from exc
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
    audio_source_path: str | Path | None = None,
    compiler: Optional[Callable[[Any], str]] = None,
    overwrite: bool = False,
) -> GeneratedPackage:
    """Write one validated map, optional source audio, and audit manifest.

    The manifest is written last, so its presence indicates that the map and
    any packaged audio were fully flushed. This function refuses to serialize
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
    audio_source: Path | None = None
    audio_path: Path | None = None
    audio_digest: str | None = None
    audio_already_present = False
    if audio_source_path is not None:
        audio_source = Path(audio_source_path)
        if not audio_source.is_file():
            raise PackagingError(f"audio source is not a file: {audio_source}")
        audio_filename = _safe_asset_name(result.request.audio_filename)
        if audio_filename in {osu_filename, manifest_filename}:
            raise PackagingError("audio, map, and manifest filenames must differ")
        audio_path = destination / audio_filename
        if audio_path.exists():
            try:
                audio_already_present = os.path.samefile(audio_source, audio_path)
            except OSError:
                audio_already_present = False

    if not overwrite:
        for artifact_path in (osu_path, manifest_path):
            if artifact_path.exists():
                raise PackagingError(f"refusing to overwrite existing artifact: {artifact_path.name}")
        if audio_path is not None and audio_path.exists() and not audio_already_present:
            raise PackagingError(f"refusing to overwrite existing artifact: {audio_path.name}")

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
    # Audio and map go first; the manifest is the completion marker.  Therefore
    # any visible manifest describes a package whose relative audio asset exists.
    try:
        if overwrite:
            # An old completion marker must never survive a failed overwrite.
            manifest_path.unlink(missing_ok=True)
        if audio_source is not None and audio_path is not None and not audio_already_present:
            _copy_atomic(audio_source, audio_path, overwrite=overwrite)
        if audio_path is not None:
            audio_digest = _file_sha256(audio_path)
            artifacts.update(
                {
                    "audio_file": audio_path.name,
                    "audio_sha256": audio_digest,
                }
            )
        manifest_bytes = result.to_json(artifacts=artifacts).encode("utf-8")
        _write_atomic(osu_path, content_bytes, overwrite=overwrite)
        _write_atomic(manifest_path, manifest_bytes, overwrite=overwrite)
    except BaseException:
        # Published paths may have been replaced by another process after our
        # atomic create. Never unlink them during rollback: without an atomic
        # compare-and-unlink primitive, doing so can delete another writer's
        # data. The absent completion manifest identifies the retained map or
        # audio as an incomplete package and makes retry behavior explicit.
        raise
    return GeneratedPackage(
        output_dir=destination,
        osu_path=osu_path,
        manifest_path=manifest_path,
        osu_sha256=digest,
        audio_path=audio_path,
        audio_sha256=audio_digest,
    )

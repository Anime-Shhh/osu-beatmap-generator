"""Group-safe split helpers for the v2 preprocessing pipeline.

Windows, difficulty variants, and alternate maps of one audio track are highly
correlated.  Splitting already-packed WebDataset shards therefore leaks songs
into validation and test.  These helpers assign a split from an immutable audio
group *before* window extraction and make overlap checks cheap to run in CI.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SPLIT_NAMES = ("train", "val", "test")
SOURCE_SPLIT_MANIFEST_SCHEMA = "osu-beatmap-generator/v2-source-split-manifest/1"


@dataclass(frozen=True)
class SourceIdentity:
    """Stable metadata written alongside every cached v2 example.

    ``audio_sha256`` identifies the *encoded* source blob.  It catches exact
    copies but cannot detect an MP3 and OGG that decode to the same song.
    ``canonical_audio_sha256`` is the digest of the canonical decoded PCM and
    is the preferred exact split key when it is available.  A verified
    ``acoustic_fingerprint`` can link remaining re-encodes; it must be a
    cluster identity produced by a documented algorithm, not an unverified
    nearest-neighbour score.

    The remaining identifiers are retained for auditing and for future style
    analysis.  They are deliberately not used as generation conditioning:
    keeping a mapper ID in provenance must not turn into mapper imitation.
    """

    audio_sha256: str
    canonical_audio_sha256: str | None = None
    acoustic_fingerprint: str | None = None
    fingerprint_algorithm: str | None = None
    audio_duration_ms: float | None = None
    beatmap_id: str | None = None
    beatmap_set_id: str | None = None
    mapper_id: str | None = None
    source_dataset: str = "project-riz/osu-beatmaps"
    source_record_id: str | None = None
    source_revision: str | None = None
    beatmap_status: str | None = None
    rights_status: str | None = None
    rights_url: str | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.audio_sha256, "audio_sha256")
        if self.canonical_audio_sha256 is not None:
            _validate_sha256(self.canonical_audio_sha256, "canonical_audio_sha256")
        if bool(self.acoustic_fingerprint) != bool(self.fingerprint_algorithm):
            raise ValueError(
                "acoustic_fingerprint and fingerprint_algorithm must be supplied together"
            )
        if self.acoustic_fingerprint is not None and not self.acoustic_fingerprint.strip():
            raise ValueError("acoustic_fingerprint must be non-empty when supplied")
        if self.fingerprint_algorithm is not None and not self.fingerprint_algorithm.strip():
            raise ValueError("fingerprint_algorithm must be non-empty when supplied")
        if self.audio_duration_ms is not None:
            if not math.isfinite(self.audio_duration_ms) or self.audio_duration_ms <= 0:
                raise ValueError("audio_duration_ms must be finite and positive when supplied")
        if not self.source_dataset.strip():
            raise ValueError("source_dataset must be non-empty")
        for field_name in (
            "source_record_id",
            "source_revision",
            "beatmap_status",
            "rights_status",
            "rights_url",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when supplied")

    @property
    def group_id(self) -> str:
        """Best available single split key for lightweight callers.

        Full dataset construction should use :func:`source_component_groups`,
        which also handles transitive links (for example encoded A -> canonical
        A and canonical A -> acoustic cluster A).  This property is retained
        for simple one-record streaming code only.
        """

        if self.acoustic_fingerprint is not None:
            return f"fingerprint:{self.fingerprint_algorithm}:{self.acoustic_fingerprint}"
        if self.canonical_audio_sha256 is not None:
            return f"pcm:{self.canonical_audio_sha256}"
        return f"audio:{self.audio_sha256}"

    @property
    def split_keys(self) -> tuple[str, ...]:
        """All immutable evidence keys used to create leakage-safe components."""

        keys = [f"encoded:{self.audio_sha256}"]
        if self.canonical_audio_sha256 is not None:
            keys.append(f"pcm:{self.canonical_audio_sha256}")
        if self.acoustic_fingerprint is not None:
            keys.append(
                f"fingerprint:{self.fingerprint_algorithm}:{self.acoustic_fingerprint}"
            )
        return tuple(sorted(set(keys)))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance; no local paths or credentials are stored."""

        return {
            "audio_sha256": self.audio_sha256,
            "canonical_audio_sha256": self.canonical_audio_sha256,
            "acoustic_fingerprint": self.acoustic_fingerprint,
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "audio_duration_ms": self.audio_duration_ms,
            "beatmap_id": self.beatmap_id,
            "beatmap_set_id": self.beatmap_set_id,
            "mapper_id": self.mapper_id,
            "source_dataset": self.source_dataset,
            "source_record_id": self.source_record_id,
            "source_revision": self.source_revision,
            "beatmap_status": self.beatmap_status,
            "rights_status": self.rights_status,
            "rights_url": self.rights_url,
        }

    def validate_training_provenance(self) -> None:
        """Require the source/status/rights fields needed by a training audit."""

        required = {
            "source_record_id": self.source_record_id,
            "source_revision": self.source_revision,
            "beatmap_status": self.beatmap_status,
            "rights_status": self.rights_status,
            "rights_url": self.rights_url,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(
                "training source provenance is incomplete: " + ", ".join(missing)
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceIdentity":
        """Restore a source identity from an immutable cache manifest."""

        return cls(
            audio_sha256=str(payload["audio_sha256"]),
            canonical_audio_sha256=_optional_str(payload.get("canonical_audio_sha256")),
            acoustic_fingerprint=_optional_str(payload.get("acoustic_fingerprint")),
            fingerprint_algorithm=_optional_str(payload.get("fingerprint_algorithm")),
            audio_duration_ms=_optional_float(payload.get("audio_duration_ms")),
            beatmap_id=_optional_str(payload.get("beatmap_id")),
            beatmap_set_id=_optional_str(payload.get("beatmap_set_id")),
            mapper_id=_optional_str(payload.get("mapper_id")),
            source_dataset=str(payload.get("source_dataset", "project-riz/osu-beatmaps")),
            source_record_id=_optional_str(payload.get("source_record_id")),
            source_revision=_optional_str(payload.get("source_revision")),
            beatmap_status=_optional_str(payload.get("beatmap_status")),
            rights_status=_optional_str(payload.get("rights_status")),
            rights_url=_optional_str(payload.get("rights_url")),
        )


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def audio_sha256(audio_bytes: bytes | bytearray | memoryview) -> str:
    """Return a content-addressed ID for an encoded source-audio blob.

    This is intentionally computed before generating map windows.  It catches
    all exact duplicate audio in a source dataset without relying on fragile
    input ordering or WebDataset shard boundaries.  Cross-encoding duplicates
    need a later acoustic-fingerprint pass as well.
    """

    return hashlib.sha256(bytes(audio_bytes)).hexdigest()


def _normalized_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split ratios must have exactly train, val, and test values")
    values = tuple(float(value) for value in ratios)
    if any(value <= 0.0 for value in values):
        raise ValueError("all split ratios must be positive")
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def assign_group_split(
    group_id: str,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> str:
    """Assign one group deterministically without depending on stream order."""

    if not group_id:
        raise ValueError("group_id must be non-empty")
    train_ratio, val_ratio, _ = _normalized_ratios(ratios)
    digest = hashlib.blake2b(
        f"v2-split:{seed}:{group_id}".encode("utf-8"), digest_size=8
    ).digest()
    unit_interval = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    if unit_interval < train_ratio:
        return "train"
    if unit_interval < train_ratio + val_ratio:
        return "val"
    return "test"


def build_group_split_manifest(
    group_ids: Iterable[str],
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Make an auditable split manifest with each source group exactly once."""

    manifest = {name: [] for name in SPLIT_NAMES}
    for group_id in sorted(set(group_ids)):
        manifest[assign_group_split(group_id, seed=seed, ratios=ratios)].append(group_id)
    assert_group_disjoint(manifest)
    return manifest


def assert_group_disjoint(split_groups: Mapping[str, Iterable[str]]) -> None:
    """Raise a clear error when any song/audio group crosses split boundaries."""

    normalized = {name: set(split_groups.get(name, ())) for name in SPLIT_NAMES}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:3])
                raise ValueError(
                    f"group leakage between {left} and {right}: {examples}"
                )


@dataclass(frozen=True)
class SourceSplitAssignment:
    """The persisted assignment of one source record to a split component.

    ``component_id`` represents all exact and verified-acoustic links known at
    preprocessing time.  Persist it beside every chunk so later appends do not
    accidentally re-split a song after window extraction.
    """

    source: SourceIdentity
    component_id: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {self.split!r}")
        if not self.component_id.startswith("source:"):
            raise ValueError("component_id must be a canonical source component ID")


class _DisjointSet:
    """Tiny deterministic union-find used before any source is chunked."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self._parent[value]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        root = parent
        parent = self._parent[value]
        while parent != root:
            next_parent = self._parent[parent]
            self._parent[parent] = root
            parent = next_parent
        self._parent[value] = root
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # A deterministic root makes results independent of stream order.
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def _canonical_component_id(keys: Iterable[str]) -> str:
    """Return an opaque stable component ID without exposing source identifiers."""

    payload = json.dumps(
        sorted(set(keys)), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(b"osu-v2-source-component\0" + payload).hexdigest()
    return f"source:{digest}"


def source_component_groups(
    sources: Iterable[SourceIdentity],
) -> dict[SourceIdentity, str]:
    """Resolve source identities into transitive no-leakage components.

    Two maps are linked when they share an encoded-audio digest, canonical PCM
    digest, or verified acoustic fingerprint.  The relation is transitive, so
    a chain of alternate encodes cannot put its endpoints in different splits.
    The returned IDs are content-addressed and deterministic regardless of
    source iteration order.
    """

    unique_sources = tuple(
        sorted(
            set(sources),
            key=lambda source: (
                source.split_keys,
                json.dumps(source.to_dict(), sort_keys=True, separators=(",", ":")),
            ),
        )
    )
    union_find = _DisjointSet()
    for source in unique_sources:
        keys = source.split_keys
        for key in keys:
            union_find.add(key)
        for key in keys[1:]:
            union_find.union(keys[0], key)

    component_keys: dict[str, set[str]] = {}
    for source in unique_sources:
        for key in source.split_keys:
            component_keys.setdefault(union_find.find(key), set()).add(key)

    root_ids = {
        root: _canonical_component_id(keys)
        for root, keys in component_keys.items()
    }
    return {
        source: root_ids[union_find.find(source.split_keys[0])]
        for source in unique_sources
    }


def build_source_split_assignments(
    sources: Iterable[SourceIdentity],
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> tuple[SourceSplitAssignment, ...]:
    """Assign every source to a split after resolving linked audio components."""

    components = source_component_groups(sources)
    assignments = tuple(
        SourceSplitAssignment(
            source=source,
            component_id=component_id,
            split=assign_group_split(component_id, seed=seed, ratios=ratios),
        )
        for source, component_id in sorted(
            components.items(),
            key=lambda item: (
                item[1],
                item[0].split_keys,
                json.dumps(item[0].to_dict(), sort_keys=True, separators=(",", ":")),
            ),
        )
    )
    assert_source_split_disjoint(assignments)
    return assignments


def build_source_split_manifest(
    sources: Iterable[SourceIdentity],
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Build a split manifest from source records, not already-made windows.

    The manifest stores component IDs rather than raw audio or fingerprint IDs,
    making it safe to log and cheap to compare in CI.  Keep this immutable once
    training begins; append processing must reuse the saved manifest instead of
    recomputing it from a different corpus snapshot.
    """

    assignments = build_source_split_assignments(sources, seed=seed, ratios=ratios)
    manifest = {name: [] for name in SPLIT_NAMES}
    for assignment in assignments:
        if assignment.component_id not in manifest[assignment.split]:
            manifest[assignment.split].append(assignment.component_id)
    for values in manifest.values():
        values.sort()
    assert_group_disjoint(manifest)
    return manifest


def build_immutable_source_split_manifest(
    sources: Iterable[SourceIdentity],
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Build the revision-pinned source artifact bound to a training run."""

    normalized_ratios = _normalized_ratios(ratios)
    unique_sources = tuple(set(sources))
    for source in unique_sources:
        source.validate_training_provenance()
    assignments = build_source_split_assignments(
        unique_sources, seed=seed, ratios=normalized_ratios
    )
    splits = {name: [] for name in SPLIT_NAMES}
    records: list[dict[str, Any]] = []
    for assignment in assignments:
        if assignment.component_id not in splits[assignment.split]:
            splits[assignment.split].append(assignment.component_id)
        records.append(
            {
                "component_id": assignment.component_id,
                "split": assignment.split,
                "source": assignment.source.to_dict(),
            }
        )
    for values in splits.values():
        values.sort()
    manifest: dict[str, Any] = {
        "schema_version": SOURCE_SPLIT_MANIFEST_SCHEMA,
        "seed": int(seed),
        "ratios": list(normalized_ratios),
        "splits": splits,
        "records": records,
    }
    validate_source_split_manifest(manifest)
    return manifest


def assert_source_split_disjoint(
    assignments: Iterable[SourceSplitAssignment],
) -> None:
    """Verify that all linked source records share a single split.

    This accepts assignments rather than only component IDs so a cache audit
    detects a stale or manually edited manifest even when two records became
    connected by a newly calculated PCM/fingerprint key.
    """

    values = tuple(assignments)
    components = source_component_groups(assignment.source for assignment in values)
    owners: dict[str, str] = {}
    for assignment in values:
        actual_component = components[assignment.source]
        if assignment.component_id != actual_component:
            raise ValueError(
                "source assignment component is stale or invalid for "
                f"{assignment.source.audio_sha256[:12]}"
            )
        previous = owners.setdefault(actual_component, assignment.split)
        if previous != assignment.split:
            raise ValueError(
                f"source leakage between {previous} and {assignment.split}: {actual_component}"
            )


def validate_source_split_manifest(payload: Mapping[str, Any]) -> None:
    """Reject stale ownership and identifier overlap in an immutable manifest."""

    if payload.get("schema_version") != SOURCE_SPLIT_MANIFEST_SCHEMA:
        raise ValueError("unsupported or missing v2 source split manifest schema")
    raw_splits = payload.get("splits")
    raw_records = payload.get("records")
    if not isinstance(raw_splits, Mapping) or not isinstance(raw_records, list):
        raise ValueError("source split manifest requires splits and records")
    splits: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        values = raw_splits.get(name)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"source split manifest {name} must be a string list")
        if values != sorted(set(values)):
            raise ValueError(f"source split manifest {name} must be sorted and unique")
        splits[name] = list(values)
    assert_group_disjoint(splits)

    assignments: list[SourceSplitAssignment] = []
    seen_sources: set[SourceIdentity] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping) or not isinstance(raw_record.get("source"), Mapping):
            raise ValueError(f"source split manifest record {index} is malformed")
        source = SourceIdentity.from_dict(raw_record["source"])
        source.validate_training_provenance()
        if source in seen_sources:
            raise ValueError(f"source split manifest record {index} is duplicated")
        seen_sources.add(source)
        assignment = SourceSplitAssignment(
            source=source,
            component_id=str(raw_record.get("component_id", "")),
            split=str(raw_record.get("split", "")),
        )
        if assignment.component_id not in splits[assignment.split]:
            raise ValueError(
                f"source split manifest record {index} is absent from split ownership"
            )
        assignments.append(assignment)
    if not assignments:
        raise ValueError("source split manifest must contain at least one source record")
    assert_source_split_disjoint(assignments)
    expected = {name: set() for name in SPLIT_NAMES}
    for assignment in assignments:
        expected[assignment.split].add(assignment.component_id)
    if any(expected[name] != set(splits[name]) for name in SPLIT_NAMES):
        raise ValueError("source split manifest contains unowned or missing components")


def source_split_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return a canonical digest after fully validating the split artifact."""

    validate_source_split_manifest(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

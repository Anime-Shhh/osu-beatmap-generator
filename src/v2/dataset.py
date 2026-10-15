"""Safe, bar-targeted datasets for the v2 sparse planner.

Each ``.pt`` file is one *song* record, not a shuffled 6-second window.  The
dataset expands its meter-aware bar targets lazily, retaining whole-song audio
context while teaching the decoder one grammar sequence at a time.  This keeps
the source/audio split intact and fixes the legacy shard-level leakage issue.

Records must be created by the explicit v2 preprocessing command.  Loading
uses ``weights_only=True`` so a record is restricted to tensors and ordinary
containers; do not use arbitrary pickle payloads as a training dataset.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on login nodes.
    raise RuntimeError("src.v2.dataset requires PyTorch in the training environment") from exc

from .events import IGNORE_INDEX
from .grammar import EventVocabulary, MAX_ANCHORS
from .data import SourceIdentity, SourceSplitAssignment, assert_source_split_disjoint


V2_SONG_RECORD_SCHEMA = "osu-beatmap-generator/v2-song-record/3"
V2_SONG_RECORD_INDEX_SCHEMA = "osu-beatmap-generator/v2-song-record-index/3"


class RecordSchemaError(ValueError):
    """A local record does not match the documented v2 tensor contract."""


def _tensor(value: Any, name: str, *, dim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RecordSchemaError(f"{name} must be a torch Tensor")
    if dim is not None and value.dim() != dim:
        raise RecordSchemaError(f"{name} must be {dim}D, got {value.dim()}D")
    return value


def _as_cpu_contiguous(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = value.detach().to(device="cpu", dtype=dtype or value.dtype).contiguous()
    if result.is_floating_point() and not torch.isfinite(result).all():
        raise RecordSchemaError("record contains non-finite floating point values")
    return result


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordSchemaError(f"{name} must be an object mapping")
    return value


def _read_record(path: Path) -> Mapping[str, Any]:
    try:
        # ``weights_only`` deliberately rejects executable pickle globals.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise RecordSchemaError(f"unable to load safe v2 record {path.name}: {exc}") from exc
    return _require_mapping(payload, f"record {path.name}")


def song_record_index_path(path: str | Path) -> Path:
    """Return the small JSON sidecar used to index one song record.

    A training corpus can contain many gigabytes of feature tensors.  Loading
    every ``.pt`` file merely to discover its split and bar count turns worker
    startup into a long, RAM-heavy serial scan.  The sidecar is deliberately
    boring JSON and contains only index/provenance fields; the full tensor
    record is still safely loaded and validated before it produces a sample.
    """

    source = Path(path)
    if source.suffix != ".pt":
        raise ValueError("v2 song record paths must end in .pt")
    return source.with_name(f"{source.name}.index.json")


def song_record_index(
    payload: Mapping[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    """Build the immutable, tensor-free discovery record for a song payload.

    ``validate=False`` is reserved for callers that have just completed
    :func:`validate_song_record`. It avoids walking every large feature tensor
    twice during record writing or first-cache loads; it never skips the
    validation boundary itself.
    """

    if validate:
        validate_song_record(payload)
    metadata = _require_mapping(payload["metadata"], "metadata")
    source = _require_mapping(metadata.get("source"), "metadata.source")
    return {
        "schema_version": V2_SONG_RECORD_INDEX_SCHEMA,
        "record_schema_version": V2_SONG_RECORD_SCHEMA,
        "metadata": {
            **{
                name: str(metadata[name])
                for name in (
                    "cache_key",
                    "feature_pipeline_key",
                    "source_component_id",
                    "split",
                    "architecture_id",
                )
            },
            "source": dict(source),
        },
        "bar_target_count": len(payload["bar_targets"]),
        "rhythm_dim": int(_tensor(payload["rhythm_features"], "rhythm_features", dim=2).shape[1]),
        "semantic_dim": int(_tensor(payload["semantic_features"], "semantic_features", dim=2).shape[1]),
        "clock_variant_count": len(payload.get("clock_variants", ())),
    }


def validate_song_record_index(payload: Mapping[str, Any]) -> None:
    """Validate a sidecar without loading its potentially large tensor file."""

    if payload.get("schema_version") != V2_SONG_RECORD_INDEX_SCHEMA:
        raise RecordSchemaError("unsupported or missing v2 song-record index schema")
    if payload.get("record_schema_version") != V2_SONG_RECORD_SCHEMA:
        raise RecordSchemaError("song-record index refers to an unsupported record schema")
    metadata = _require_mapping(payload.get("metadata"), "record index metadata")
    for name in ("cache_key", "feature_pipeline_key", "source_component_id", "split", "architecture_id"):
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RecordSchemaError(f"record index metadata.{name} must be a non-empty string")
    if metadata["split"] not in {"train", "val", "test"}:
        raise RecordSchemaError("record index metadata.split must be train, val, or test")
    source = _require_mapping(metadata.get("source"), "record index metadata.source")
    try:
        SourceIdentity.from_dict(source).validate_training_provenance()
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordSchemaError(f"record index source provenance is invalid: {exc}") from exc
    for name in ("bar_target_count", "rhythm_dim", "semantic_dim", "clock_variant_count"):
        value = payload.get(name)
        if not isinstance(value, int) or value < 0:
            raise RecordSchemaError(f"record index {name} must be a non-negative integer")
    if payload["bar_target_count"] < 1:
        raise RecordSchemaError("record index must describe at least one bar target")
    if payload["rhythm_dim"] < 1 or payload["semantic_dim"] < 1:
        raise RecordSchemaError("record index feature dimensions must be positive")


def _read_song_record_index(path: Path) -> Mapping[str, Any] | None:
    """Read a JSON sidecar if present; malformed sidecars are hard failures.

    Falling back when a sidecar is absent preserves compatibility with an
    already-built small corpus.  Falling back when it is malformed would hide
    a stale/corrupt split index and could silently train on an unintended
    split, so that case is rejected explicitly.
    """

    sidecar = song_record_index_path(path)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordSchemaError(f"unable to read v2 record index {sidecar.name}: {exc}") from exc
    index = _require_mapping(payload, f"record index {sidecar.name}")
    validate_song_record_index(index)
    return index


def validate_song_record(payload: Mapping[str, Any]) -> None:
    """Validate a record before it can become a training sample.

    This checks shape and provenance constraints rather than attempting to
    correct them.  Training on a cache generated from a different feature
    projection, tokenizer, or split assignment is a reproducibility bug, not a
    minor warning.
    """

    if payload.get("schema_version") != V2_SONG_RECORD_SCHEMA:
        raise RecordSchemaError("unsupported or missing v2 song record schema")
    metadata = _require_mapping(payload.get("metadata"), "metadata")
    for name in ("cache_key", "feature_pipeline_key", "source_component_id", "split", "architecture_id"):
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RecordSchemaError(f"metadata.{name} must be a non-empty string")
    if metadata["split"] not in {"train", "val", "test"}:
        raise RecordSchemaError("metadata.split must be train, val, or test")
    source = _require_mapping(metadata.get("source"), "metadata.source")
    try:
        SourceIdentity.from_dict(source)
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordSchemaError(f"metadata.source is invalid: {exc}") from exc

    rhythm = _tensor(payload.get("rhythm_features"), "rhythm_features", dim=2)
    rhythm_times = _tensor(payload.get("rhythm_times_ms"), "rhythm_times_ms", dim=1)
    semantic = _tensor(payload.get("semantic_features"), "semantic_features", dim=2)
    semantic_times = _tensor(payload.get("semantic_times_ms"), "semantic_times_ms", dim=1)
    bar_mid = _tensor(payload.get("bar_mid_times_ms"), "bar_mid_times_ms", dim=1)
    clock = _tensor(payload.get("clock_features"), "clock_features", dim=2)
    profile = _tensor(payload.get("profile"), "profile", dim=1)
    if rhythm.shape[0] != rhythm_times.numel() or semantic.shape[0] != semantic_times.numel():
        raise RecordSchemaError("feature rows and timestamp counts must match")
    if rhythm.shape[0] < 1 or semantic.shape[0] < 1 or bar_mid.numel() < 1:
        raise RecordSchemaError("feature streams and bars must be non-empty")
    if clock.shape[0] != bar_mid.numel() or clock.shape[1] != 4:
        raise RecordSchemaError("clock_features must be [bars,4]")
    if profile.numel() != 6:
        raise RecordSchemaError("profile must have six normalized controls")
    for name, values in (("bar_mid_times_ms", bar_mid), ("clock_features", clock), ("profile", profile)):
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise RecordSchemaError(f"{name} must contain finite floating values")
    if bar_mid.numel() > 1 and not torch.all(bar_mid[1:] > bar_mid[:-1]):
        raise RecordSchemaError("bar_mid_times_ms must be strictly increasing")
    for name, timestamps in (("rhythm_times_ms", rhythm_times), ("semantic_times_ms", semantic_times)):
        if not timestamps.is_floating_point() or not torch.isfinite(timestamps).all():
            raise RecordSchemaError(f"{name} must be finite floating timestamps")
        if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
            raise RecordSchemaError(f"{name} must be strictly increasing")

    variants = payload.get("clock_variants", ())
    if variants:
        if not isinstance(variants, (list, tuple)):
            raise RecordSchemaError("clock_variants must be a list when supplied")
        variant_ids: set[str] = set()
        for variant_index, raw_variant in enumerate(variants):
            variant = _require_mapping(raw_variant, f"clock_variants[{variant_index}]")
            hypothesis_id = variant.get("hypothesis_id")
            if not isinstance(hypothesis_id, str) or not hypothesis_id.strip() or hypothesis_id in variant_ids:
                raise RecordSchemaError("clock variant IDs must be unique non-empty strings")
            variant_ids.add(hypothesis_id)
            variant_times = _tensor(
                variant.get("bar_mid_times_ms"), f"clock_variants[{variant_index}].bar_mid_times_ms", dim=1
            )
            variant_clock = _tensor(
                variant.get("clock_features"), f"clock_variants[{variant_index}].clock_features", dim=2
            )
            if variant_times.shape != bar_mid.shape or variant_clock.shape != clock.shape:
                raise RecordSchemaError("clock variant tensor shapes must match canonical bar tensors")
            if not torch.isfinite(variant_times).all() or not torch.isfinite(variant_clock).all():
                raise RecordSchemaError("clock variants must be finite")

    targets = payload.get("bar_targets")
    if not isinstance(targets, (list, tuple)) or not targets:
        raise RecordSchemaError("bar_targets must be a non-empty list")
    vocabulary = EventVocabulary()
    observed_bar_indices: list[int] = []
    for target_index, raw_target in enumerate(targets):
        target = _require_mapping(raw_target, f"bar_targets[{target_index}]")
        bar_index = target.get("bar_index")
        if not isinstance(bar_index, int) or not 0 <= bar_index < bar_mid.numel():
            raise RecordSchemaError(f"bar_targets[{target_index}].bar_index is invalid")
        observed_bar_indices.append(bar_index)
        tokens = _tensor(target.get("tokens"), f"bar_targets[{target_index}].tokens", dim=1)
        if tokens.numel() < 2 or int(tokens[0]) != vocabulary.bos_id or int(tokens[-1]) != vocabulary.eos_id:
            raise RecordSchemaError(f"bar_targets[{target_index}] must begin BOS and end EOS")
        if tokens.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise RecordSchemaError(f"bar_targets[{target_index}].tokens must be integral")
        if (tokens < 0).any() or (tokens >= len(vocabulary)).any():
            raise RecordSchemaError(f"bar_targets[{target_index}].tokens contain unknown IDs")
        geometry = _tensor(target.get("geometry"), f"bar_targets[{target_index}].geometry", dim=3)
        kinds = _tensor(target.get("event_kinds"), f"bar_targets[{target_index}].event_kinds", dim=1)
        anchors = _tensor(target.get("anchor_counts"), f"bar_targets[{target_index}].anchor_counts", dim=1)
        token_indices = _tensor(
            target.get("event_token_indices"), f"bar_targets[{target_index}].event_token_indices", dim=1
        )
        lengths = _tensor(target.get("target_lengths_px"), f"bar_targets[{target_index}].target_lengths_px", dim=1)
        event_features = _tensor(target.get("event_features"), f"bar_targets[{target_index}].event_features", dim=2)
        event_positions = _tensor(target.get("event_positions"), f"bar_targets[{target_index}].event_positions", dim=1)
        event_count = geometry.shape[0]
        if geometry.shape[1:] != (MAX_ANCHORS + 1, 2):
            raise RecordSchemaError("geometry must be [events, MAX_ANCHORS + 1, 2]")
        if any(value.numel() != event_count for value in (kinds, anchors, token_indices, lengths, event_positions)):
            raise RecordSchemaError("event target fields must share event count")
        if event_features.shape != (event_count, 4):
            raise RecordSchemaError("event_features must be [events,4]")
        if any(value.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8) for value in (kinds, anchors, token_indices, event_positions)):
            raise RecordSchemaError("event type/topology/token-index fields must be integral")
        if not torch.isfinite(geometry).all() or (geometry[:, 0] < 0).any() or (geometry[:, 0] > 1).any():
            raise RecordSchemaError("geometry must be finite with heads in [0,1]; slider controls may lie outside")
        if (kinds < 0).any() or (kinds > 2).any():
            raise RecordSchemaError("event_kinds must use native EventKind IDs")
        if (anchors < 0).any() or (anchors > MAX_ANCHORS).any():
            raise RecordSchemaError("anchor_counts exceeds geometry topology bound")
        if (event_positions < 0).any():
            raise RecordSchemaError("event_positions must be non-negative")
        if not torch.isfinite(event_features).all():
            raise RecordSchemaError("event_features must be finite")
        if (token_indices < 0).any() or (token_indices >= tokens.numel() - 1).any():
            raise RecordSchemaError("event_token_indices must select completed-event decoder input positions")
        if event_count != int(tokens.eq(vocabulary.event_id).sum().item()):
            raise RecordSchemaError("every event target must correspond to exactly one EVENT token")
    if observed_bar_indices != list(range(bar_mid.numel())):
        raise RecordSchemaError(
            "bar_targets must contain every meter-aware bar exactly once in ascending order "
            "for deterministic boundary conditioning"
        )


def _copy_target(target: Mapping[str, Any]) -> dict[str, torch.Tensor | int]:
    return {
        "bar_index": int(target["bar_index"]),
        "tokens": _as_cpu_contiguous(_tensor(target["tokens"], "tokens", dim=1), dtype=torch.long),
        "geometry": _as_cpu_contiguous(_tensor(target["geometry"], "geometry", dim=3), dtype=torch.float32),
        "event_kinds": _as_cpu_contiguous(_tensor(target["event_kinds"], "event_kinds", dim=1), dtype=torch.long),
        "anchor_counts": _as_cpu_contiguous(_tensor(target["anchor_counts"], "anchor_counts", dim=1), dtype=torch.long),
        "event_token_indices": _as_cpu_contiguous(
            _tensor(target["event_token_indices"], "event_token_indices", dim=1), dtype=torch.long
        ),
        "target_lengths_px": _as_cpu_contiguous(
            _tensor(target["target_lengths_px"], "target_lengths_px", dim=1), dtype=torch.float32
        ),
        "event_features": _as_cpu_contiguous(
            _tensor(target["event_features"], "event_features", dim=2), dtype=torch.float32
        ),
        "event_positions": _as_cpu_contiguous(
            _tensor(target["event_positions"], "event_positions", dim=1), dtype=torch.long
        ),
    }


def validate_record_split_preflight(indexes: Sequence[Mapping[str, Any]]) -> None:
    """Reject source/component leakage before any training sample is emitted."""

    assignments: list[SourceSplitAssignment] = []
    for index, payload in enumerate(indexes):
        validate_song_record_index(payload)
        metadata = _require_mapping(payload["metadata"], f"record index {index} metadata")
        source_payload = _require_mapping(
            metadata.get("source"), f"record index {index} metadata.source"
        )
        try:
            source = SourceIdentity.from_dict(source_payload)
            source.validate_training_provenance()
            assignments.append(
                SourceSplitAssignment(
                    source=source,
                    component_id=str(metadata["source_component_id"]),
                    split=str(metadata["split"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordSchemaError(f"record index {index} source assignment is invalid: {exc}") from exc
    try:
        assert_source_split_disjoint(assignments)
    except ValueError as exc:
        raise RecordSchemaError(f"v2 record split preflight failed: {exc}") from exc


class SparseSongDataset(Dataset[dict[str, Any]]):
    """Expand safe per-song records into bar-targeted training examples.

    The tiny bounded LRU avoids rereading a full song record for adjacent bars,
    while never caching an unbounded corpus in a worker process. ``song``
    sampling (the default) emits one deterministically rotating bar from one
    source component per epoch; use ``bar`` only for exhaustive evaluation.
    """

    def __init__(
        self,
        records_dir: str | Path,
        *,
        split: str | None = None,
        expected_pipeline_key: str | None = None,
        cache_size: int = 2,
        clock_mode: str = "mixed",
        sampling_mode: str = "song",
        bars_per_song: int = 1,
        alignment_frames: int = 0,
    ):
        if split is not None and split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, test, or None")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        if clock_mode not in {"oracle", "mixed", "non_oracle"}:
            raise ValueError("clock_mode must be oracle, mixed, or non_oracle")
        if sampling_mode not in {"song", "bar"}:
            raise ValueError("sampling_mode must be song or bar")
        self.records_dir = Path(records_dir)
        if not self.records_dir.is_dir():
            raise FileNotFoundError(f"v2 records directory does not exist: {self.records_dir}")
        self.expected_pipeline_key = expected_pipeline_key
        self.cache_size = cache_size
        self.clock_mode = clock_mode
        self.sampling_mode = sampling_mode
        if bars_per_song < 1:
            raise ValueError("bars_per_song must be positive")
        self.bars_per_song = bars_per_song
        if alignment_frames != 0 and alignment_frames < 2:
            raise ValueError("alignment_frames must be zero or at least two")
        self.alignment_frames = alignment_frames
        self._epoch = 0
        self._records: list[Path] = []
        self._record_indexes: list[Mapping[str, Any]] = []
        self._index: list[tuple[int, int]] = []
        self._song_groups: list[tuple[int, ...]] = []
        self._cache: OrderedDict[int, Mapping[str, Any]] = OrderedDict()
        groups: dict[str, list[int]] = {}
        discovered: list[tuple[Path, Mapping[str, Any]]] = []
        for path in sorted(self.records_dir.glob("*.pt")):
            index_payload = _read_song_record_index(path)
            if index_payload is None:
                # Old records remain usable, but are intentionally slower to
                # discover. Re-write them with ``write_song_record`` to add a
                # sidecar before a large multi-worker run.
                payload = _read_record(path)
                validate_song_record(payload)
                index_payload = song_record_index(payload, validate=False)
            discovered.append((path, index_payload))
        validate_record_split_preflight([index for _, index in discovered])
        for path, index_payload in discovered:
            metadata = _require_mapping(index_payload["metadata"], "record index metadata")
            if expected_pipeline_key is not None and metadata["feature_pipeline_key"] != expected_pipeline_key:
                raise RecordSchemaError(
                    f"{path.name} feature pipeline differs from requested training revision"
                )
            if split is not None and metadata["split"] != split:
                continue
            record_index = len(self._records)
            self._records.append(path)
            self._record_indexes.append(index_payload)
            groups.setdefault(str(metadata["source_component_id"]), []).append(record_index)
            if sampling_mode == "bar":
                self._index.extend(
                    (record_index, target_index)
                    for target_index in range(int(index_payload["bar_target_count"]))
                )
        if sampling_mode == "song":
            self._song_groups = [tuple(indices) for _, indices in sorted(groups.items())]
            self._index = [(group_index, sample) for group_index in range(len(self._song_groups))
                           for sample in range(bars_per_song)]
        if not self._index:
            qualifier = f" for split {split!r}" if split else ""
            raise RecordSchemaError(f"no v2 bar targets found in {self.records_dir}{qualifier}")

    def __len__(self) -> int:
        return len(self._index)

    def set_epoch(self, epoch: int) -> None:
        """Choose mixed clock variants deterministically for the next epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

    def _record(self, record_index: int) -> Mapping[str, Any]:
        cached = self._cache.pop(record_index, None)
        if cached is None:
            cached = _read_record(self._records[record_index])
            validate_song_record(cached)
            # The JSON index only controls efficient discovery. Verify it once
            # the tensor payload is opened so a manually copied/stale sidecar
            # can never alter split membership or bar indexing silently.
            if song_record_index(cached, validate=False) != dict(self._record_indexes[record_index]):
                raise RecordSchemaError(
                    f"{self._records[record_index].name} does not match its v2 song-record index sidecar"
                )
        if self.alignment_frames and '_alignment_events' not in cached:
            from .alignment import reference_events
            cached = dict(cached)
            cached['_alignment_events'] = reference_events(cached)
        self._cache[record_index] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, target_index = self._index[index]
        if self.sampling_mode == "song":
            choices = self._song_groups[record_index]
            draw = self._epoch * self.bars_per_song + target_index
            record_index = choices[draw % len(choices)]
            # Uniformly sample a source component, then one difficulty/map
            # record, then one bar. This prevents popular songs/mapsets from
            # dominating merely because they have more cached bar targets.
            preview = self._record(record_index)
            target_index = (draw // len(choices) + record_index) % len(preview["bar_targets"])
        payload = self._record(record_index)
        target = _require_mapping(payload["bar_targets"][target_index], "bar target")
        previous_tokens = torch.empty((0,), dtype=torch.long)
        if target_index > 0:
            previous = _require_mapping(payload["bar_targets"][target_index - 1], "previous bar target")
            # Records retain even empty bars, so a gap here is evidence of a
            # malformed/reordered target list rather than a harmless missing
            # predecessor. Do not silently borrow another bar's style.
            if int(previous["bar_index"]) != int(target["bar_index"]) - 1:
                raise RecordSchemaError("bar targets must remain contiguous for boundary conditioning")
            previous_tokens = _as_cpu_contiguous(
                _tensor(previous["tokens"], "previous bar tokens", dim=1), dtype=torch.long
            )
        metadata = dict(_require_mapping(payload["metadata"], "metadata"))
        variants = tuple(payload.get("clock_variants", ()))
        selected_variant: Mapping[str, Any] | None = None
        if variants:
            oracle_variants = [
                _require_mapping(variant, "clock variant")
                for variant in variants
                if bool(_require_mapping(variant, "clock variant").get("is_oracle", False))
            ]
            non_oracle_variants = [
                _require_mapping(variant, "clock variant")
                for variant in variants
                if not bool(_require_mapping(variant, "clock variant").get("is_oracle", False))
            ]
            if self.clock_mode == "oracle":
                selected_variant = oracle_variants[0] if oracle_variants else None
            elif self.clock_mode == "non_oracle":
                if not non_oracle_variants:
                    raise RecordSchemaError("non_oracle clock_mode requires a non-oracle clock variant")
                selected_variant = non_oracle_variants[(index + self._epoch) % len(non_oracle_variants)]
            else:
                # Alternate deterministically rather than sampling from worker
                # RNG state, so resuming an epoch retains its clock exposure.
                ordered = [*oracle_variants, *non_oracle_variants]
                selected_variant = ordered[(index + self._epoch) % len(ordered)] if ordered else None
        bar_mid_source = payload["bar_mid_times_ms"] if selected_variant is None else selected_variant["bar_mid_times_ms"]
        clock_source = payload["clock_features"] if selected_variant is None else selected_variant["clock_features"]
        if selected_variant is not None:
            metadata["clock_hypothesis_id"] = str(selected_variant["hypothesis_id"])
        alignment = {}
        if self.alignment_frames:
            from .alignment import query_grid, chart_targets
            from .clock import TimingHypothesis
            hypothesis = payload['oracle_timing_hypothesis'] if selected_variant is None else selected_variant['hypothesis']
            timing = TimingHypothesis.from_dict(hypothesis).timing_map
            times = query_grid(timing, target['bar_start_beat'], target['bar_length_beats'], self.alignment_frames)
            alignment = {'local_query_times_ms': times,
                         'alignment_targets': chart_targets(times, payload['_alignment_events'])}
        return {
            **alignment,
            "rhythm_features": _as_cpu_contiguous(_tensor(payload["rhythm_features"], "rhythm_features", dim=2), dtype=torch.float32),
            "rhythm_times_ms": _as_cpu_contiguous(_tensor(payload["rhythm_times_ms"], "rhythm_times_ms", dim=1), dtype=torch.float32),
            "semantic_features": _as_cpu_contiguous(_tensor(payload["semantic_features"], "semantic_features", dim=2), dtype=torch.float32),
            "semantic_times_ms": _as_cpu_contiguous(_tensor(payload["semantic_times_ms"], "semantic_times_ms", dim=1), dtype=torch.float32),
            "bar_mid_times_ms": _as_cpu_contiguous(_tensor(bar_mid_source, "bar_mid_times_ms", dim=1), dtype=torch.float32),
            "clock_features": _as_cpu_contiguous(_tensor(clock_source, "clock_features", dim=2), dtype=torch.float32),
            "profile": _as_cpu_contiguous(_tensor(payload["profile"], "profile", dim=1), dtype=torch.float32),
            "previous_bar_tokens": previous_tokens,
            "target": _copy_target(target),
            "metadata": metadata,
        }


def _pad_time_series(values: Sequence[torch.Tensor], *, pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(value.shape[0] for value in values)
    shape = (len(values), max_length, *values[0].shape[1:])
    output = values[0].new_full(shape, pad_value)
    mask = torch.zeros((len(values), max_length), dtype=torch.bool)
    for index, value in enumerate(values):
        output[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return output, mask


def collate_sparse_examples(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad a bar-target batch without allowing padding to become supervision."""

    if not examples:
        raise ValueError("cannot collate an empty sparse batch")
    rhythm_dim = examples[0]["rhythm_features"].shape[-1]
    semantic_dim = examples[0]["semantic_features"].shape[-1]
    if any(example["rhythm_features"].shape[-1] != rhythm_dim for example in examples):
        raise RecordSchemaError("batch mixes rhythm feature dimensions")
    if any(example["semantic_features"].shape[-1] != semantic_dim for example in examples):
        raise RecordSchemaError("batch mixes semantic feature dimensions")

    rhythm, rhythm_mask = _pad_time_series([example["rhythm_features"] for example in examples])
    rhythm_times, _ = _pad_time_series([example["rhythm_times_ms"].unsqueeze(-1) for example in examples])
    semantic, semantic_mask = _pad_time_series([example["semantic_features"] for example in examples])
    semantic_times, _ = _pad_time_series([example["semantic_times_ms"].unsqueeze(-1) for example in examples])
    bar_mid, bar_mask = _pad_time_series([example["bar_mid_times_ms"].unsqueeze(-1) for example in examples])
    clock, _ = _pad_time_series([example["clock_features"] for example in examples])

    targets = [example["target"] for example in examples]
    tokens, token_mask = _pad_time_series([target["tokens"] for target in targets], pad_value=0)
    # Trainer predicts every token after BOS.  The EOS label stays valid; all
    # sequence padding becomes IGNORE_INDEX rather than a real vocabulary ID.
    decoder_tokens = tokens[:, :-1]
    decoder_mask = token_mask[:, :-1]
    labels = tokens[:, 1:].clone()
    labels[~token_mask[:, 1:]] = IGNORE_INDEX

    previous_bar_tokens, previous_bar_mask = _pad_time_series(
        [example["previous_bar_tokens"] for example in examples], pad_value=0
    )

    geometry, event_mask = _pad_time_series([target["geometry"] for target in targets])
    event_kinds, _ = _pad_time_series([target["event_kinds"].unsqueeze(-1) for target in targets], pad_value=0)
    anchor_counts, _ = _pad_time_series([target["anchor_counts"].unsqueeze(-1) for target in targets], pad_value=0)
    event_token_indices, _ = _pad_time_series([target["event_token_indices"].unsqueeze(-1) for target in targets], pad_value=0)
    target_lengths, _ = _pad_time_series([target["target_lengths_px"].unsqueeze(-1) for target in targets], pad_value=0)
    event_features, _ = _pad_time_series([target["event_features"] for target in targets], pad_value=0)
    event_positions, _ = _pad_time_series([target["event_positions"].unsqueeze(-1) for target in targets], pad_value=0)

    alignment = {}
    if any('local_query_times_ms' in example for example in examples):
        if not all('local_query_times_ms' in example for example in examples):
            raise RecordSchemaError('cannot mix alignment-enabled and legacy examples')
        queries, query_mask = _pad_time_series([e['local_query_times_ms'] for e in examples])
        targets_alignment, _ = _pad_time_series([e['alignment_targets'] for e in examples])
        alignment = {'local_query_times_ms': queries, 'local_query_mask': query_mask,
                     'alignment_targets': targets_alignment}
    return {
        **alignment,
        "rhythm_features": rhythm,
        "rhythm_times_ms": rhythm_times.squeeze(-1),
        "rhythm_mask": rhythm_mask,
        "semantic_features": semantic,
        "semantic_times_ms": semantic_times.squeeze(-1),
        "semantic_mask": semantic_mask,
        "bar_mid_times_ms": bar_mid.squeeze(-1),
        "bar_mask": bar_mask,
        "clock_features": clock,
        "profile": torch.stack([example["profile"] for example in examples]),
        "target_bar_index": torch.tensor([target["bar_index"] for target in targets], dtype=torch.long),
        "decoder_tokens": decoder_tokens.long(),
        "decoder_mask": decoder_mask,
        "labels": labels.long(),
        "previous_bar_tokens": previous_bar_tokens.long(),
        "previous_bar_mask": previous_bar_mask,
        "geometry": geometry,
        "event_mask": event_mask,
        "event_kinds": event_kinds.squeeze(-1).long(),
        "anchor_counts": anchor_counts.squeeze(-1).long(),
        "event_token_indices": event_token_indices.squeeze(-1).long(),
        "target_lengths_px": target_lengths.squeeze(-1),
        "event_features": event_features,
        "event_positions": event_positions.squeeze(-1).long(),
        "metadata": tuple(dict(example["metadata"]) for example in examples),
    }


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move only tensors; provenance stays on CPU and out of accelerator RAM."""

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


__all__ = [
    "RecordSchemaError",
    "SparseSongDataset",
    "V2_SONG_RECORD_SCHEMA",
    "V2_SONG_RECORD_INDEX_SCHEMA",
    "collate_sparse_examples",
    "move_batch_to_device",
    "song_record_index",
    "song_record_index_path",
    "validate_record_split_preflight",
    "validate_song_record",
    "validate_song_record_index",
]

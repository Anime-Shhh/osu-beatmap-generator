import unittest
import copy
import hashlib

from src.v2.data import (
    SourceIdentity,
    SourceSplitAssignment,
    assign_group_split,
    assert_group_disjoint,
    assert_source_split_disjoint,
    audio_sha256,
    build_group_split_manifest,
    build_immutable_source_split_manifest,
    source_component_groups,
    source_split_manifest_sha256,
    validate_source_split_manifest,
)


class GroupSplitTests(unittest.TestCase):
    @staticmethod
    def _source(label: str, *, fingerprint: str | None = None) -> SourceIdentity:
        digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        return SourceIdentity(
            audio_sha256=digest(f"encoded-{label}"),
            canonical_audio_sha256=digest(f"pcm-{label}"),
            acoustic_fingerprint=fingerprint,
            fingerprint_algorithm="chromaprint-v1" if fingerprint else None,
            beatmap_id=label,
            beatmap_set_id=f"set-{label}",
            mapper_id=f"mapper-{label}",
            source_record_id=f"record-{label}",
            source_revision="b82117a5de13b7df5290482aed7cffc49cb76e6d",
            beatmap_status="ranked",
            rights_status="research-only; audio rights remain with owners",
            rights_url="https://osu.ppy.sh/legal/terms",
        )

    def test_audio_hash_and_split_are_stable(self):
        digest = audio_sha256(b"same encoded audio")
        self.assertEqual(digest, audio_sha256(memoryview(b"same encoded audio")))
        self.assertNotEqual(digest, audio_sha256(b"different encoded audio"))
        self.assertEqual(
            assign_group_split(f"audio:{digest}", seed=99),
            assign_group_split(f"audio:{digest}", seed=99),
        )

    def test_manifest_is_disjoint(self):
        groups = [f"audio:{index}" for index in range(200)]
        manifest = build_group_split_manifest(groups, seed=7)
        assert_group_disjoint(manifest)
        self.assertEqual(sum(len(values) for values in manifest.values()), len(groups))
        with self.assertRaises(ValueError):
            assert_group_disjoint({"train": ["audio:collision"], "val": ["audio:collision"]})

    def test_immutable_manifest_retains_provenance_and_rejects_fingerprint_evasion(self):
        first = self._source("map-a", fingerprint="same-audio-family")
        second = self._source("map-b", fingerprint="same-audio-family")
        manifest = build_immutable_source_split_manifest((first, second), seed=7)
        validate_source_split_manifest(manifest)
        self.assertEqual(
            manifest,
            build_immutable_source_split_manifest((second, first), seed=7),
        )
        self.assertEqual(len(manifest["records"]), 2)
        self.assertEqual(len(source_split_manifest_sha256(manifest)), 64)
        self.assertEqual(
            {record["source"]["source_revision"] for record in manifest["records"]},
            {"b82117a5de13b7df5290482aed7cffc49cb76e6d"},
        )

        component = source_component_groups((first, second))[first]
        with self.assertRaisesRegex(ValueError, "stale|leakage"):
            assert_source_split_disjoint(
                (
                    SourceSplitAssignment(first, component, "train"),
                    SourceSplitAssignment(second, "source:" + "f" * 64, "val"),
                )
            )

        tampered = copy.deepcopy(manifest)
        tampered["records"][1]["component_id"] = "source:" + "f" * 64
        tampered["records"][1]["split"] = "val"
        tampered["splits"][next(
            name for name, values in tampered["splits"].items() if component in values
        )].remove(component)
        tampered["splits"]["train"] = sorted(set(tampered["splits"]["train"]) | {component})
        tampered["splits"]["val"] = sorted(set(tampered["splits"]["val"]) | {"source:" + "f" * 64})
        with self.assertRaises(ValueError):
            validate_source_split_manifest(tampered)


if __name__ == "__main__":
    unittest.main()

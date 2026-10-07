import unittest

from src.v2.data import (
    assign_group_split,
    assert_group_disjoint,
    audio_sha256,
    build_group_split_manifest,
)


class GroupSplitTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

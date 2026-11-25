import unittest

import numpy as np

from src.v3.eval.harness import _boot_groups, _mapper, aggregate
from src.v3.eval.metrics import pooled


def _dist(seed):
    rng = np.random.default_rng(seed)
    return {"divisors": list(rng.choice([1, 2, 4], 50)), "trigrams": [tuple(rng.integers(0, 4, 3)) for _ in range(40)],
            "spacing_per_ms": rng.random(60), "turn": rng.normal(size=50), "stack": [False] * 9 + [True],
            "curves": [2] * 5 + [3] * 5, "slider_len": list(rng.random(10) * 200), "density_per_min": 150.0 + seed,
            "sv_changes_per_min": 1.0, "nc_on_downbeat": 0.6}


def _row(chart, group, status, f1=0.8, short=(3, 4)):
    r = {"chart": chart, "group": group, "stars_ref": 4.2, "status": status, "ref_valid": 1.0}
    if status in ("ok", "failed"):
        ok = status == "ok"
        r.update({"f1_10": f1 if ok else 0.0, "f1_25": f1 if ok else 0.0, "f1_50": f1 if ok else 0.0,
                  "typed_f1_25": f1 if ok else 0.0, "valid": 1.0 if ok else 0.0,
                  "abs_star_err": 0.1 if ok else float("nan"), "matched_25": 8 if ok else 0, "n_gen": 10 if ok else 0,
                  "n_ref": 10, "short_pairs_got": short[0] if ok else 0, "short_pairs_total": short[1],
                  "ref_dist": _dist(1)})
        if ok:
            r.update({"dist": _dist(2), "selected": {"median_displacement_px": 0.0, "max_displacement_px": 3.0}})
    return r


class HarnessTest(unittest.TestCase):
    def test_guest_mapper_heuristic(self):
        self.assertEqual(_mapper({"diff_name": "Next's Normal", "creator": "Lulu-"}), "next")
        self.assertEqual(_mapper({"diff_name": "Kroytz' Insane", "creator": "A"}), "kroytz")
        self.assertEqual(_mapper({"diff_name": "Insane", "creator": "Lulu-"}), "lulu-")

    def test_failures_count_and_na_is_excluded(self):
        rows = [_row("a", "g1", "ok"), _row("b", "g1", "failed"), _row("c", "g2", "ok"), _row("d", "g3", "na")]
        s = aggregate(rows, pooled([_dist(3)]), reps=50, dist_reps=10)
        self.assertEqual(s["n"], {"ok": 2, "failed": 1, "na": 1})
        self.assertAlmostEqual(s["macro"]["f1_25"]["mean"], 0.8 * 2 / 3)        # failure scored as 0, na dropped
        self.assertAlmostEqual(s["macro"]["valid"]["mean"], 2 / 3)
        self.assertAlmostEqual(s["macro"]["abs_star_err"]["mean"], 0.1)          # star error only where a map exists
        self.assertAlmostEqual(s["pooled"]["short_pair_recall"]["value"], 6 / 12)  # the failed chart's pairs count
        self.assertAlmostEqual(s["pooled"]["f1_25_pooled"]["value"], 2 * 16 / (20 + 30))
        for name in ("system_vs_band", "human_ref_vs_band", "system_vs_refs"):
            self.assertIn("trigram_js", s["distribution"][name])
        ex = s["distribution"]["excess_over_human"]["trigram_js"]
        self.assertAlmostEqual(ex["value"], s["distribution"]["system_vs_band"]["trigram_js"]
                               - s["distribution"]["human_ref_vs_band"]["trigram_js"])
        self.assertLessEqual(ex["lo"], ex["hi"])
        self.assertEqual(s["distribution"]["projection"]["p99_of_max_px"], 3.0)

    def test_bootstrap_resamples_whole_songs(self):
        groups = ["s1", "s1", "s1", "s2", "s3"]
        for idx in _boot_groups(groups, 30):
            picked = collections_counter(groups[i] for i in idx)
            self.assertTrue(all(picked[g] % groups.count(g) == 0 for g in picked))


def collections_counter(it):
    import collections
    return collections.Counter(it)


if __name__ == "__main__":
    unittest.main()

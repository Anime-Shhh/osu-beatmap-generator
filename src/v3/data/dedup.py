"""Cross-split acoustic duplicate search with binary fingerprints (GPU).

Leakage only matters across splits, so queries are every non-train song and the
index is every song. Candidates come from *exact* 31-bit code hits that agree on
one time offset (robust to trimming: TV size vs full, different intros); each
candidate is verified by the bit error rate (BER) over the aligned overlap.
Queries are also searched as tempo/pitch-warped copies so nightcore (faster +
higher), DT-style sped-up and slowed versions are caught.

Recall is calibrated on *known* positives: different audio files the index
already grouped as the same song (alternate encodings/uploads).
"""
from __future__ import annotations

import collections
import json
import random
import time
from pathlib import Path

import numpy as np

from src.v3.common import io

STOP_COUNT = 64           # codes this common in the index carry no identity (silence etc.)
DT_BIN = 4                # offset histogram bin, in 10 ms frames
MIN_HITS = 3              # agreeing exact hits needed to verify a candidate (BER verifies)
VERIFY_TOP = 24
BER_MAX = 0.35
MIN_OVERLAP_FRAMES = 1000           # 10 s
MIN_OVERLAP_FRACTION = 0.2          # of the shorter song
WARPS = [(1.0, 1.0),
         (1.25, 1 / 1.25), (1.5, 1 / 1.5), (1 / 1.25, 1.25), (1 / 1.5, 1.5),   # nightcore-style
         (1.25, 1.0), (1 / 1.25, 1.0)]                                         # tempo only


def _song_table(root: Path):
    from .manifest import load_chart_metas
    split_of, group_of = {}, {}
    for m in load_chart_metas(root):
        if m["synthetic"]:
            continue
        split_of[m["audio_sha256"]] = m["split"]
        group_of[m["audio_sha256"]] = m["group"]
    return split_of, group_of


class Index:
    def __init__(self, root: Path, shas: list[str], device: str):
        import torch
        vals, songs, frames, self.fps = [], [], [], {}
        for i, sha in enumerate(shas):
            path = root / "songpass" / sha[:2] / f"{sha}.fp.npy"
            if not path.is_file():
                continue
            fp = np.load(path).astype(np.int32)          # 31-bit codes fit in int32
            if len(fp) < 100:
                continue
            self.fps[i] = fp
            vals.append(fp)
            songs.append(np.full(len(fp), i, np.int32))
            frames.append(np.arange(len(fp), dtype=np.int32))
        # Sort and drop stop-codes on the CPU (radix sort), then ship compact int32 arrays to
        # the GPU: the full corpus is ~690M codes, too large for torch.unique on a 40 GB card.
        v = np.concatenate(vals)
        order = np.argsort(v, kind="stable")
        v = v[order]
        s = np.concatenate(songs)[order]
        f = np.concatenate(frames)[order]
        del order
        starts = np.r_[0, np.flatnonzero(np.diff(v)) + 1]
        lengths = np.diff(np.r_[starts, len(v)])
        keep = np.repeat(lengths <= STOP_COUNT, lengths)
        self.dropped_fraction = float(1 - keep.mean())
        self.vals = torch.as_tensor(v[keep], device=device)
        self.song = torch.as_tensor(s[keep], device=device)
        self.frame = torch.as_tensor(f[keep], device=device)
        self.device = device

    def candidates(self, q, own: int):
        """{song: (best_offset_frames, hits)} for songs with >= MIN_HITS agreeing hits."""
        import torch
        q32 = q.to(torch.int32)
        lo = torch.searchsorted(self.vals, q32, side="left")
        hi = torch.searchsorted(self.vals, q32, side="right")
        n = hi - lo
        if int(n.sum()) == 0:
            return {}
        qi = torch.repeat_interleave(torch.arange(len(q), device=q.device), n)
        start = torch.repeat_interleave(lo, n)
        within = torch.arange(int(n.sum()), device=q.device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        idx = start + within
        song = self.song[idx].to(torch.int64)
        dt = self.frame[idx].to(torch.int64) - qi
        key = song * (1 << 24) + torch.div(dt, DT_BIN, rounding_mode="floor") + (1 << 22)
        keys, counts = torch.unique(key, return_counts=True)
        good = counts >= MIN_HITS
        out: dict[int, tuple[int, int]] = {}
        for k, c in zip(keys[good].tolist(), counts[good].tolist()):
            sg = k >> 24
            if sg == own:
                continue
            if sg not in out or c > out[sg][1]:
                out[sg] = (((k & ((1 << 24) - 1)) - (1 << 22)) * DT_BIN, c)
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1])[:VERIFY_TOP])

    def verify(self, q, song: int, offset: int):
        """Best BER over offsets offset..offset+DT_BIN-1; returns (ber, overlap_frames)."""
        import torch
        from .songpass import bit_error_rate
        ref = torch.as_tensor(self.fps[song], device=q.device).to(q.dtype)
        best = (1.0, 0)
        for d in range(offset - 1, offset + DT_BIN + 1):
            q0, r0 = (0, d) if d >= 0 else (-d, 0)
            n = min(len(q) - q0, len(ref) - r0)
            if n <= 0:
                continue
            ber = bit_error_rate(q[q0:q0 + n], ref[r0:r0 + n])
            if ber < best[0]:
                best = (ber, n)
        return best


def run_dedup(root: Path, device: str = "cuda") -> None:
    from .songpass import fingerprint
    out = root / "dedup"
    t0 = time.time()
    split_of, group_of = _song_table(root)
    shas = sorted(split_of)
    index = Index(root, shas, device)
    present = {shas[i] for i in index.fps}
    print(f"index: {len(index.vals)} codes from {len(present)}/{len(shas)} songs "
          f"(stop-code fraction {index.dropped_fraction:.3f}, {time.time() - t0:.0f}s)", flush=True)
    sha_pos = {s: i for i, s in enumerate(shas)}

    def query(sha, warps):
        feats = np.load(root / "audio" / sha[:2] / f"{sha}.npy", mmap_mode="r")
        own = sha_pos[sha]
        found = {}
        for ts, fs in warps:
            q = fingerprint(feats, device, freq_scale=fs, time_scale=ts)
            for sg, (offset, hits) in index.candidates(q, own).items():
                ber, overlap = index.verify(q, sg, offset)
                shorter = min(len(q), len(index.fps[sg]))
                if ber <= BER_MAX and overlap >= min(MIN_OVERLAP_FRAMES, shorter) and overlap >= MIN_OVERLAP_FRACTION * shorter:
                    if sg not in found or ber < found[sg]["ber"]:
                        found[sg] = {"ber": ber, "overlap_s": overlap / 100.0, "hits": hits,
                                     "offset_s": offset / 100.0, "time_scale": ts, "freq_scale": fs}
        return found

    # Calibration on known positives (alternate audio files already grouped as one song).
    by_group = collections.defaultdict(list)
    for s in shas:
        if s in present:
            by_group[group_of[s]].append(s)
    multi = sorted(g for g, v in by_group.items() if len(v) >= 2)
    calib = random.Random(0).sample(multi, min(400, len(multi)))
    found_pairs = sum(sha_pos[by_group[g][1]] in query(by_group[g][0], WARPS[:1]) for g in calib)
    recall = found_pairs / max(1, len(calib))
    print(f"calibration: {found_pairs}/{len(calib)} known same-song audio pairs found (recall {recall:.3f})", flush=True)

    queries = [s for s in shas if split_of[s] != "train" and s in present]
    unchecked = sum(1 for s in shas if split_of[s] != "train" and s not in present)
    pairs = []
    for n, sha in enumerate(queries):
        for sg, info in query(sha, WARPS).items():
            other = shas[sg]
            if group_of[other] == group_of[sha]:
                continue  # already grouped together; not a leak
            pairs.append({"query": sha, "match": other, "query_split": split_of[sha], "match_split": split_of[other],
                          "query_group": group_of[sha], "match_group": group_of[other], **info})
        if (n + 1) % 250 == 0:
            print(f"  queries {n + 1}/{len(queries)}, pairs {len(pairs)}, {time.time() - t0:.0f}s", flush=True)
    leaks = [p for p in pairs if p["query_split"] != p["match_split"]]
    io.write_jsonl_atomic(out / "pairs.jsonl", pairs)
    summary = {"queries": len(queries), "unchecked_without_features": unchecked, "index_songs": len(present),
               "pairs": len(pairs), "cross_split_pairs": len(leaks),
               "cross_split_query_groups": len({p["query_group"] for p in leaks}),
               "by_warp": dict(collections.Counter(f'{p["time_scale"]:.2f}/{p["freq_scale"]:.2f}' for p in leaks)),
               "calibration": {"pairs": len(calib), "found": found_pairs, "recall": recall},
               "thresholds": {"ber_max": BER_MAX, "min_hits": MIN_HITS, "stop_count": STOP_COUNT,
                              "min_overlap_s": MIN_OVERLAP_FRAMES / 100, "min_overlap_fraction": MIN_OVERLAP_FRACTION},
               "seconds": time.time() - t0}
    io.write_json_atomic(out / "summary.json", summary)
    io.mark_done(out / "done.json", [out / "pairs.jsonl", out / "summary.json"])
    print(json.dumps(summary, indent=1), flush=True)

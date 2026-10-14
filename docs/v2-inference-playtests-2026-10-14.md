# V2 full-song inference and playtest archives — October 14, 2026

Four self-contained `.osz` archives were generated with the trained
geometry-stage best checkpoint, revision `checkpoint-43200-a991a335083180d0`.
The weights were unchanged. The Kaguya run completed as Slurm job `506340`
(1:36); the three held-out songs completed as job `506322` (12:51), both exit
code `0:0`.

All archives passed ZIP CRC checks. Every packaged map was parsed and validated,
its `AudioFilename` exists in the archive, and packaged audio hashes match the
original Kaguya MP3 or the exact encoded audio identities in the test corpus.
This checks importable file contents; no osu! client playtest was performed.

## Archives and difficulty results

Remote output folder:

```text
/common/users/asj102/osu_project/v2_runs/run_20261014_real/inference_20261014_v2
```

| Song | Archive | Requested stars | Audio-only AI | Human-clock AI |
| --- | --- | ---: | ---: | ---: |
| Kaguya opening | `kaguya_v2_trained.osz` | 4.50 | 6.45 | — |
| Neon Rainbow (ft. Anna Yvette) | `heldout_000002.osz` | 4.70 | 5.58 | 9.75 |
| Star!! | `heldout_000033.osz` | 4.66 | 5.17 | 9.25 |
| Platinum Groove | `heldout_000057.osz` | 4.35 | 5.32 | 7.47 |

Stars use `rosu-pp-py/4.0.2`, stable/no-mods; they are not a promise of identical
numbers in every client/calculator version. Each held-out archive contains
`audio.mp3`, `audio_only.osu`, `known_timing.osu`, and `human-reference.osu`.
The reference is re-exported from the genuine selected human chart for the
same audio. The Kaguya archive contains the audio and one AI map.

Held-out source keys 000002, 000033, and 000057 were verified to belong to the
test split. Source audio and chart content were recovered from the immutable
dataset revision used for training and checked against the recorded hashes.
The human-clock variant supplies timing/SV information and profile settings;
it does not supply the human event sequence or geometry to the generator.

## Findings

Four candidates were sampled for each of seven song/mode combinations:
**28 candidates total, zero passing the full ±0.5-star target gate**. All
delivered AI maps are labelled **uncalibrated drafts** with their actual stars.
Only the target-star filter was waived to select diagnostic playtest outputs;
structural validation and the minimum-gap check remained enabled. Nothing was
presented as a successful 4.5-star result.

The generation test is considerably less encouraging than teacher-forced
accuracy: difficulty control fails throughout this small batch, and correct
human timing alone does not resolve it. Some very rapid, large jumps remain.
Map syntax validity and a 10 ms minimum gap are not evidence of comfortable
playability. These artifacts support inspection and diagnosis, not a claim
of human-quality mapping.

Against the selected human chart, one-to-one onset matching produced:

| Song | Audio-only F1 at 20 ms | Human-clock F1 at 20 ms | Audio-only F1 at 50 ms | Human-clock F1 at 50 ms |
| --- | ---: | ---: | ---: | ---: |
| Neon Rainbow | 0.512 | 0.646 | 0.521 | 0.646 |
| Star!! | 0.121 | 0.486 | 0.304 | 0.615 |
| Platinum Groove | 0.165 | 0.663 | 0.381 | 0.663 |

These are reference-onset diagnostics, not human judgments or audio-onset
detector scores. Alternative valid rhythms can differ from a reference chart.
The differences indicate that timing estimation matters, while the inflated
stars of the human-clock variants indicate another problem beyond the clock.

Kaguya's selected map contains 218 objects and 43 sliders (22 P, six B, 15 L),
with zero exact-corner heads, an 11 ms minimum start gap, and maximum slider
length 140 px. Its maximum head-to-head speed is approximately 30,944 px/s,
which is a readability diagnostic, not osu!'s aim-strain formula.

## Inference fixes required by the first pass

The initial Kaguya candidates and first held-out audio-only candidates failed
the 10 ms start-gap gate. Some also hit a floating-point comparison error at
the exact allowed ±1/192-beat residual boundary.

- The beat-snap comparison now allows a `1e-9`-beat numerical epsilon.
- Decoding masks timing choices that cannot satisfy the existing minimum gap,
  including across bar boundaries. A 1 ms internal margin covers native
  integer timestamp rounding. No sampled objects were deleted afterward.
- Trained weights and token vocabulary identities were preserved.
- The generation CLI supports `--osz-output`, with an atomic archive helper
  that requires the actual audio asset.

The first-pass audits remain under `inference_20261014`; the corrected outputs
are in `inference_20261014_v2`. The full local V2 suite passed 96 tests and
28 subtests after the changes. Regression tests cover the timing-boundary
comparison and minimum-gap continuation across bar boundaries.

## Copy to the Mac

Kaguya archive:

```text
/common/users/asj102/osu_project/v2_runs/run_20261014_real/inference_20261014_v2/kaguya_v2_trained.osz
```

Run in the Mac terminal:

```bash
scp -F /Users/anime_shh/personalCS/projects/ilab-access/ssh-config \
  rutgers-ilab:/common/users/asj102/osu_project/v2_runs/run_20261014_real/inference_20261014_v2/kaguya_v2_trained.osz \
  ~/Downloads/
```

To copy all four archives:

```bash
scp -F /Users/anime_shh/personalCS/projects/ilab-access/ssh-config \
  'rutgers-ilab:/common/users/asj102/osu_project/v2_runs/run_20261014_real/inference_20261014_v2/*.osz' \
  ~/Downloads/
```

The local `v2-inference-playtests-2026-10-14.json` retains candidate audits,
selected-map metrics, onset comparisons, archive checksums, and remote paths.

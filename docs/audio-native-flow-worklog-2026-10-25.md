# Audio/native flow work log — October 25, 2026

## Authorized goal and operational context

User requested inspection of each branch's `architecture_explained.md`, critique,
implementation and Slurm training of diffusion toward MP3 + requested stars ->
human-quality osu!standard `.osz`. The goal remains active; training completion
alone does not meet the requested map-quality target. Do not mark it complete
without generated-map evidence and candid quality assessment.

Use the established Mac OpenSSH ControlMaster via
`/Users/anime_shh/personalCS/projects/ilab-access/ilab`. Sandbox needs elevated
execution to use the socket. No SSH keys or Hermes. iLab rejects explicit CPU
counts in sbatch. GPU work belongs in Slurm. Preserve historical deployments,
data, runs, and user job 450738 (not in queue during this work).

Worktree: `/Users/anime_shh/personalCS/projects/osu-diffusion-lab`, branch
`codex/audio-native-flow`, from local main `a355755`.

## Review conclusions

Read the architecture descriptions of main, hierarchical beatgrid,
coverage/rhythm-flow, both auxiliary-distance branches, and geometry rearrangement,
using common text plus their branch differences. Verified current implementations
and prior October 18 timing measurements. These contain separate model families;
neither the rhythm-flow checkpoint nor the geometry-rearrangement ranker is a
complete audio-to-native-map model.

* Old rhythm flow: 602k parameters, three competing onset channels plus two hold
  occupancies; generated circle recall was 8.45%, duration MAE ~315ms. Its total
  loss hid these distinct failures. Target-dependent flow weighting changes the
  regression objective. Temporal GroupNorm also lets padding affect valid frames.
* V2: geometry depends on frozen autoregressive planner states and oracle event
  plans during training. Millisecond flow outputs cannot simply be substituted
  for these contexts. Independent bars and estimated clocks add mismatch.
* Latent AE path adds lossy native-curve reconstruction and another training stage.
  No established evidence favors restoring that path as the immediate solution.
* Distance-only geometry supervision misses turns, repeat-aware slider exits and
  local strain. Existing geometry-lab results support treating it as an auxiliary
  signal, not a sufficient geometry representation.
* 260 independent training source groups remain a pilot corpus. Increasing epochs
  cannot establish professional mapping or remove source diversity limitations.
* Current tempo inference is audio autocorrelation, not a trained beat/downbeat
  tracker. Whole-song musical form, fine hitsound choices, repeated motifs, and
  calibration beyond candidate selection remain limitations.

Reference implementations/papers inspected:
[Flow Matching](https://arxiv.org/abs/2210.02747),
[osu-diffusion](https://github.com/OliBomby/osu-diffusion),
[Mapperatorinator](https://github.com/OliBomby/Mapperatorinator).
Ranked-like is a quality aspiration, not official Ranked eligibility; current
[ranking criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria) separately govern
AI-generated gameplay content.

## Implemented experiment

`src/beatmap_flow/rhythm.py`: one onset flow, frame-wise channel LayerNorm,
20.48-second windows, cached local audio encoding, overlapping global-noise
sampling. Separate audio/onset-conditioned mark heads learn type CE and explicit
log-duration regression. Auxiliary audio-only onset BCE encourages alignment.
Flow velocity MSE itself is unweighted. Small onset-context shifts train mark
heads to tolerate sampled peaks. No oracle clock or reference chart prefix.

`src/beatmap_flow/geometry.py`: audio + millisecond event interval + profile
context Transformer predicts native curve family, anchor count, repeats, length,
and combo; a conditional flow learns absolute logit heads and relative native
controls. Head and anchor flow losses are separately normalized. Optional
spacing/turn/control-shape auxiliaries use the existing physical-aspect losses.
64-event windows cross bar boundaries, with known-overlap inpainting at sampling.

`src/beatmap_flow/generate.py`: MP3 and stars, two trained checkpoints,
validation-bound threshold, optional train-only settings guide, native SV solution,
native compiler/parser round trip, rendered paths, coverage and measured stars,
candidate selection, original audio bytes in OSZ. No candidate passing hard gates
means a diagnostic failure (exit 2), not an invalid exported map.

Training scripts save EMA best and resumable last checkpoints with optimizer/RNG,
architecture and dataset hash contracts. Resume is at completed epoch boundaries.
Checkpoint selection currently uses validation composite loss; generated validation
quality must be inspected before adopting the model.

## Data and Slurm record

Remote root: `/common/users/asj102/osu_project`
Python: `env/bin/python` (PyTorch + rosu-pp-py 4.0.2).
New run root: `flow_runs/run_20261025`.
Parent shared-audio cache: `v2_runs/run_20261017_coverage/expanded-cache-v2`.
Authentic native maps: `geometry_runs/run_20261024/native`.

Cache job **806300** completed. New `flow_runs/run_20261025/cache` hardlinks all
330 verified audio files; exact native millisecond labels replace older codec
times without changing any source split. 1,376 charts (1,367 Ranked, 9 Approved),
1,100 train / 153 val / 123 test charts, 260 / 36 / 34 source groups. Geometry
admits 1,315 charts (1,064 / 147 / 104); 61 charts exceed head/anchor/repeat limits
and are explicitly listed in `audit.json`. No silent clipping of training geometry.

* Parent manifest: `c90ed4044a6d4b053f5ffb4a73276fd023ac38e4f6ae2c305fd37de20b3bbb32`.
* New rhythm manifest: `fbda8a8f8fa14f17710457490a46f3f99e93e6b5de71290e3a292dbdea127f39`.
* Geometry manifest: `88579b88838a338955c2a7f03aa1181d2fe434525d60062496fb54bf39b65446`.

Jobs:

* **806195**: 5 onset tests passed; trainer caught bare CUDA device selection bug.
* **806239**: corrected onset smoke, 64 updates, completed in ~14 training seconds.
* **806316**: onset full training, **completed**, 60 epochs / 15,600 updates,
  ~1,926 training seconds. Source `codex_audio_flow_20261025_data`; outputs `onset/`.
* **806317**: 3 geometry tests passed, 64-step geometry smoke completed (~20s).
* **806431**: geometry training **completed**, 60 epochs / 15,360 updates,
  ~1,022 training seconds. Source `codex_audio_flow_20261025_geometry`; outputs
  `geometry/`. Both full runs used one 5000 Blackwell GPU.
* **812800**: completed three-seed matched evaluation, 4m56s on A4000. Source
  `codex_audio_flow_20261025_eval`; output `evaluation/`. Test onset F1 at 50ms
  improved from 42.86% to 59.68%; source-bootstrap gain CI 14.38–19.15 points.
  This compares complete training systems, not an isolated architecture ablation.
* **812895**: export tests passed. **813008**: first full Kaguya export exposed an
  inherited floating-point SV-boundary lookup bug. Fixed with an eight-ULP boundary
  tolerance, with regressions preserving genuinely earlier timestamps.
* **813017**: completed 16-candidate export with corrected timing code. Selected
  candidate 7, 4.79 stars, 247 objects. Local delivered OSZ:
  `/Users/anime_shh/personalCS/projects/codex-mds/osu/2026-10-25-audio-native-flow/kaguya/generated.osz`.
  User playtest found missing rhythmic detail and timing/sound-selection problems.

Do not edit source directories underlying historical or running jobs. Deploy a
new named source snapshot after fixes. The active local worktree contains changes
newer than each stage's remote snapshot.

## Latest user steering and work in progress

The user explicitly requested fresh inference on Kaguya and comparison against
the original OSZ in the remote project root. Read
[the detailed rhythm follow-up](rhythm-followup-2026-10-25.md) before continuing.
The original archive is `1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat.
Suzuki Airi (TV Size) [no video].osz`. Its MP3 is 92.4s; the previously used MP3
is 90.149625s and shifted 254.1667ms earlier. Do not directly compare their raw
map times. Regenerated using the exact original audio, then compared all five
native difficulties. Nearest requested difficulty is Confession, 4.487 stars.

* **814797**: original-audio inference/comparison and Beat This! diagnostic finished.
  Selected fixed seed 1737 from the preceding 16-candidate scan by closest stars,
  independent of reference-event matching. It measures 5.014 stars; diagnostic
  tolerance was explicitly widened from .5 to .6. This is not a 4.5-star success.
  242 generated vs 324 native objects; 195 starts matched at 50ms; only 6/36 short
  consecutive circle pairs reproduced as circles. Source
  `codex_audio_flow_20261025_kaguya_diagnose4`; outputs
  `kaguya-original-inference-v2`, `kaguya-original-comparison-v2`.
* **814768**: corpus expansion RUNNING at last check. Next 1,000 source records
  after first 500, bounded at 4GiB additional features. Source
  `codex_audio_flow_20261025_expand3`; output `expanded-corpus`. Retains native
  charts and exact original audio for 12 frozen validation sources. Preserves
  previous split assignments; rejects any cross-split source bridge.
* **814815**: revised phrase-attention/density model smoke passed: 25 onset/export/
  timing tests, 64 training updates. Separate pure-Python grid tests also passed.
  Source `codex_audio_flow_20261025_phrase_smoke`; output `phrase-smoke`.
* **814839**: baseline full-song validation COMPLETE: 12 sources, two seeds,
  fixed threshold .5. F1 59.51%, short-circle-pair recovery 621/3318 (18.72%).
  Output `original-full-song-validation`. Aggregate density slightly exceeds
  reference, showing that globally increasing note count is insufficient.
* **814915**: 74 regression tests passed, including native map/clock/quality/
  grammar and historical split checks. Pytest is in the isolated dependency
  overlay. Earlier broad check 814897 stopped on its missing pytest dependency.
* **814837 / 814838**: queued with `afterok:814768`, respectively old architecture
  control and new phrase attention/density architecture. Both initialize from
  the same prior best, train 20 epochs at 1e-4 on the expanded frozen-split corpus,
  then evaluate validation windows and the same 12 full validation songs.
  Immutable source `codex_audio_flow_20261025_expanded_train`; outputs
  `expanded-onset-control` and `expanded-onset-phrase`. Do not claim these finished
  until inspecting Slurm and their outputs.

Failed setup jobs retained for audit: 813039/814747 used unavailable pytest before
acquisition; corrected to unittest in 814768. 814769 correctly rejected different
encoded audio, 814786 had a CLI flag typo, 814788 completed inference but found no
candidate inside the original star tolerance. None of these trained a replacement.

Next: monitor expansion and dependent training; investigate failures promptly.
Compare new models against each other and the original checkpoint on identical
validation references/seeds. Report pair recall, precise timing, per-phrase counts,
object types and hold durations, not just total F1. Re-run Kaguya only after model
selection on validation. Kaguya is a development example, not untouched test data.
Do not assume the experimental beat tracker is integrated into export: currently
it is diagnostic only. Geometry scaling/difficulty control and multimodal mark
durations remain unresolved; the new rhythm checkpoint cannot silently bypass
the generator's dataset compatibility checks when paired with older geometry.

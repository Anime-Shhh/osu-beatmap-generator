# Joint latent / conditional VAE pilot

Current status (November6): cleanup and expanded GPU smoke completed; gated
pilot job **1062088** is submitted, beginning with baseline evaluation.
No adapted-model quality result is available yet.

Implementation follows the user-approved 48 GPU-hour plan. This is a running
experiment, not evidence that a new model is better or that a VAE is useful.
The pinned external osu-dreamer v8 source and weights remain unchanged. No V2
training, RL, generated-map positive targets, or human-review request is used.

## Interfaces

`python -m src.dreamer_adapter --help` exposes `prepare`, `compact`, `cleanup`,
`train-codec`, `train-generator`, `evaluate`, `diagnostics`, `decide`, and `generate`.
GPU commands take a JSON `--config` and require a Slurm allocation plus a shared
`budget_ledger` and one predeclared `budget_slot`. Immutable configs are written
by the sequential coordinator, `python -m src.dreamer_adapter.pilot`.

The generation configuration contains `audio`, `stars`, `upstream`, `output`,
optional `manifest`/`codec_path`/`generator_path`, `seed`, `ar`, `od`, `cs`, `hp`,
`steps`, and optional `clock_json`. Outputs are map.osu, map.osz and audit.json.
The upstream style model predicts settings; requested and predicted settings
are separately recorded. Actual stars are calculated after export, displayed
in the difficulty name, and checked against the requested ±0.5 tolerance.
The archive is a draft even when its difficulty passes.

Checkpoints bind the exact upstream weight hash, source version, feature
pipeline, data manifest, codec hash and normalization statistics. All upstream
Python source hashes are checked before loading. Initialization and exact resume
are distinct. Resume restores optimizer and RNG state and validates config/data.

## Frozen data and evaluation

Only verified ranked standard charts from the recovered source manifest are
eligible. Historical source splits are retained; all Kaguya sources are excluded.
Exactly 24 validation and 24 test source groups are fixed by sorted source hashes
before acquisition. The reference closest to 4.5 stars is chosen from the original eligible native
charts and its exact hash frozen before training. Failed conversion cannot
silently substitute another difficulty. Requests use that reference's difficulty labels;
no reference events or timing map enter generation. Published upstream training
membership is unknown, so these are only held out from our adaptation.

Original audio hashes are checked against the pinned dataset revision. Existing
mel features are never substituted for v8 resonator features. Source audio is
transient. New caches are losslessly compressed and equality-checked; the redundant
new uncompressed cache is removed only after verification. Native source maps
and all historical caches remain intact.

Evaluation separately measures native signal round-trip, pretrained/adapted
reconstruction, and full generation. It records 10/25/50ms onset and type metrics,
short pairs, raw repeated positions, slider duration, coverage, measured stars,
and rendered-body validity. Four generation seeds are paired across aligned,
zero, ±500ms zero-padded shifts, and duration-matched wrong-song feature streams.
Wrong-song resizing is a diagnostic intervention, not a claim of natural audio.
Full intermediate maps/tensors are not retained by routine evaluations.

## Training and decisions

Both codec variants freeze chart features, audio features, style extraction and
style generation. They train identical temporal/decoder paths with an identity
mean adapter. VAE adds a diagonal Gaussian posterior, log variance [-10,2], and
KL ramp over 2,000 steps, with beta .001 or .01. Both retain paired-half style
swapping, upstream latent masking, and hit/position/velocity/acceleration losses.
Style regularization and label losses are constant with the frozen heads and
are not misrepresented as trainable objectives. AE has the original .2 temporal
training noise; deterministic reconstruction uses the mean. VAE reconstruction
is checked both at its mean and with posterior samples.

Codec training: AdamW 1e-4, effective batch32, 2052-frame crops, BF16, clip1,
max10k updates, evaluation every500, four non-improving evaluations to stop.
Checkpoint score averages onset/type F1 at10ms and pair recall, with a bounded
slider-duration penalty. No test data chooses weights.

The VAE gates require onset F1≥.97, pair recall≥.95, median matched-slider
error≤12ms, at most one-point regression, and measurable latent-shuffle damage.
Sampling must also preserve these details. A failed VAE consumes no generator
allocation. A passing VAE is not yet a selected generator.

Generator adaptation freezes the codec, caches posterior moments and audio
features, computes normalization only from training latents, and fine-tunes the
upstream distance/direction objective. AdamW3e-5, effective batch32, 152-latent
crops, clip1, EMA.99, max20k updates, full-song validation every1000. VAE targets
are sampled from their cached posterior; deterministic targets use their means.
The generator operates in normalized coordinates; the decoder receives the
inverse transform. Incomplete padded final chunks are excluded from training
and normalization. Separate codec identities cannot share latent caches. Validation
restores training RNG states even on failure. A common-completed-update comparison
is retained alongside each arm’s best-checkpoint results.

Final promotion requires +3pp onset F1 at25ms with a positive paired bootstrap
interval, no >1pp fine-timing/pair regression, no worse stars/invalid rate, and
significant aligned-versus-wrong-audio advantage. A VAE must additionally beat
the AE generator under its non-regression rules. A qualifying model receives a
second generator training seed with the same frozen codec; failure to reproduce
the improvement prevents promotion. Otherwise unchanged v8 remains selected.
Only the preselected configuration is evaluated on the locked test list.

Timing replacement requires an audio-only constant grid with ≥16 beats,
median residual≤10ms, p90≤25ms and evidence of 4/4 downbeat spacing. It preserves
object timestamps and recomputes slider SV, rejecting unsupported velocities.
A native reparse checks starts/endpoints within2ms. No reference phase correction.

Reward diagnostics use real maps and explicit corruptions. Reference-relative
components are explicitly not deployable musicality rewards. Wrong-audio attack
alignment, difficulty changes, timing, deletion, duplication, short-pattern,
movement and duration corruptions are reported individually; ≥90% detection on
at least12 eligible sources is required. Failures are retained. No scalar reward
or reward-based generator update is enabled.

## Resources, retention, and status

Budgets: baseline4h (including smoke), three codecs4h each, two generator arms6h
each, confirmation8h, diagnostics4h, final evaluation8h. Slots share one ledger;
ordinary failures count against it. An interrupted job leaves an active record
requiring explicit reconciliation against Slurm accounting. The coordinator
allocation is capped at47h50m, leaving room for the preliminary smoke job.

Before every training command, require8GiB quota headroom and record that check
against the manifest. Failure stops training and writes an exact storage inventory;
the threshold is not relaxed for subsequent arms. Keep one best weight artifact and one resumable last checkpoint
per arm. Source/data checks fail closed; no historic dataset deletion fallback.

Remote root: `/common/home/asj102/osu_flow_runs/pilot_20261105`.
- CPU preparation1046611 failed on audio decoding. Retry1046631 completed with
  2,957 ranked charts /681 audio sources and zero rejected charts. A logged FFmpeg
  fallback verifies decoded duration against the immutable source within100ms.
- Compression1047507 completed:1,513,630,458 cache bytes; every array verified equal.
  Only the new redundant uncompressed cache was removed; its manifests and journal
  remain. Native inputs and historical caches were preserved.
- GPU smoke job1046622 passed real AE/VAE backward/optimizer/reconstruction checks.
- Thirteen focused tests passed, including source leakage, contracts, masking, audio
  interventions, native duration round-trip, collapse gates, budget failures and
  protected cleanup, frozen reference identities and lossless compression.
  The RNG-resume test also passed. The expanded end-to-end GPU sampler smoke
  remains pending behind the storage gate.
  Python compilation and Slurm shell syntax checks passed.
- Pip download-cache cleanup completed; NFS quota reporting lagged deletion and
  then showed approximately6.45GiB reclaimed. Installed environments preserved.
- Fifteen unselected v8 Kaguya .osu files were deleted with retained metric/hash/
  seed/source proofs; the selected map, archive and all metrics remain. These
  small files were not the cause of tens of GB of storage usage.

November6: SSH was re-established. Home quota reports8,438,752,256 free bytes
(7.86GiB), below the required8GiB. No full training or new baseline evaluation
has been submitted. `storage-inventory-20261106.json` records the exact inventory.
The separately discovered uv cache is untouched; additional deletion is not
inferred from the approved pip-cache cleanup.

Previous audio/native-flow checkpoints remain unchanged as historical baselines;
see `inference-check-2026-11-04.md`. Their12-source scores are not a paired
24-source pilot comparison. Promotion uses fresh unchanged-v8 evaluation.

No training-quality, model-promotion or final evaluation result is claimed yet.

Local evidence is retained in `docs/evidence/dreamer-pilot-2026-11-06/`: storage
inventory, lossless-compression audit, initial GPU smoke and cleanup ledgers. The
pip ledger's immediate post-delete quota reading was stale; it is preserved as
recorded rather than rewritten to imply a contemporaneous measurement.

The uv audit found13,830,438,912 allocated bytes in package archives, of which
4,360,668,160 bytes are shared through hardlinks. Potentially reclaimable bytes
are9,469,770,752 (8.82GiB). No symlink consumers outside the uv cache were found
under the user's home/project allocations. Explicit cleanup approval is pending.

Next execution steps: resolve the8GiB gate, run the expanded GPU smoke from a new
immutable source snapshot, then submit `dreamer_pilot_full.sbatch` with `PILOT`,
`SOURCE` and `BEAT_CHECKPOINT` set. The checkpoint is
`/common/users/asj102/osu_project/flow_runs/run_20261025/beat-tracker-weights/hub/checkpoints/beat_this-final0.ckpt`.
Its SHA256 is
`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`.
Do not submit training while the gate fails or overwrite existing stage outputs.

## Project-folder cleanup follow-up

The user authorized removing obsolete `codex_*` source snapshots inside
`/common/users/asj102/osu_project`, preserving relevant records. All34 folders
were removed after a recovery archive on the Mac was verified against6,130 file
hashes. No assumption that every deployed edit existed on GitHub was necessary.

584 original record/artifact paths are represented by129 unique retained files
in `legacy_snapshot_records/`;105 readable report/config/log files are exposed
in its `reports/` directory. `index.json` maps old paths to retained objects.
Unique smoke-test tensors were retained; historical datasets, run directories,
checkpoints, original audio and both installed environments were untouched.

Settled project quota reclaimed62,270,464 bytes (59.39MiB). The34 folders occupied
about274MiB, but most space was retained unique tensors, not obsolete source code.
Project and home are separate allocations: this cleanup did not clear the8GiB
home gate. Home remains about7.86GiB free, and the uv cache remains untouched.
No new GPU jobs were submitted.

Recovery archive and full per-file inventory:
`/Users/anime_shh/personalCS/projects/codex-mds/osu/2026-11-06-cleanup/`.
Archive SHA256: `2802843e92d49abcdf64ed69ec7aa5f9607a2c2f93f17ce1cbc0efb04804a29c`.
The cleanup ledger and archive verification are also retained in this repository's
pilot evidence directory.

## Authorized cache/checkpoint cleanup

The user explicitly approved `uv cache clean` and eleven exact numbered V2/legacy
checkpoint paths. Cleanup completed in that order. The uv executable was found at
`/common/home/asj102/personalCS/osu_beatmap_generator/venv/bin/uv` (version0.11.8);
it ran with `--cache-dir /common/home/asj102/.cache/uv`. The command's logical
17.3GiB removal report is not treated as physically reclaimed quota. Home quota
subsequently showed34,923,802KiB used of52,428,800KiB, clearing the8GiB gate.

The eleven approved numbered checkpoints were hashed and removed. All six
best/latest files in their four parent directories were hash-verified before and
after deletion. Other best/latest/last files, datasets, recent caches, Python
environments and synthetic-run checkpoints were outside the deletion list.
The remote ledger is `approved-cleanup/checkpoints.json`; cache evidence is
`approved-cleanup/uv-clean.json`. Project-quota reclamation awaits a settled read.

All13 focused tests passed after cleanup using the installed iLab environments.
The deployed immutable source snapshot is `source_d846f29`. Submission of the
expanded GPU smoke was blocked by SSH expiration; reconnection is pending.
No expanded smoke or full pilot job has been submitted as of this update.

Settled cleanup evidence is now local: `uv-cleanup.json` and
`intermediate-checkpoint-cleanup.json` in the pilot evidence directory. Recorded
quota reductions are8.84GiB in home and10.12GiB in project (minor concurrent
metadata writes are included). uv's17.3GiB logical-file report is not the freed
quota measurement. GPU smoke1062086 was submitted after SSH reconnection; its
expanded checks are running before the full gated pilot can be submitted.

## Gated pilot launch

Smoke1062086 stopped because the harness incorrectly required measurable stars
under every intervention. The evaluator passed a valid native fixture and the
unchanged v8 Kaguya output (4.0323217 stars). The harness now records invalid
exports explicitly, uses a fixed seed, and logs every intervention.14 focused
tests passed after this fix. No malformed candidate is dropped from evaluation.

Smoke1062087 passed on NVIDIA A100-PCIE-40GB, torch2.10.0+cu128. Correct audio and
both shifts yielded measurable native maps on the12-second training crop; zero
and reversed-audio smoke conditions produced non-finite aim attributes and were
recorded invalid. This is a runtime check, not the full wrong-song benchmark.
AE/VAE gradients, frozen weights, identity initialization and sampling passed.

Full coordinator job1062088 uses immutable source `source_afdcdea`. It starts
with the frozen24-source, four-seed baseline and all five audio conditions, then
advances to codecs and gated generator adaptation. No stage bypasses a failing
contract or storage check. The48-hour GPU budget includes the failed smoke.
Completed preliminary allocations total121 seconds; the full job is capped at
47h50m, keeping the maximum combined allocation below48 hours.

Monitor `pilot-status.json`, `gpu-budget.json`, and `pilot-1062088.log` under
`/common/home/asj102/osu_flow_runs/pilot_20261105`. Do not overwrite deployed
source or repeat locked-test evaluation. A submitted job is not a completed
experiment or evidence that AE/VAE adaptation improves map quality.

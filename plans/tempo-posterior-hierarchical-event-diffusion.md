# Tempo-Posterior Hierarchical Event Diffusion

## Product contract

Generate one playable, editor-ready, **ranked-like** osu!standard draft from
an audio file. The required input is audio. Optional controls are target star
rating, AR/CS/OD/HP/slider-setting overrides, a coarse style-cluster seed,
random seed, and candidate count.

The default request targets 4.5 stars with AR 9, CS 4, OD 8, HP 6, slider
multiplier 1.4, and slider tick rate 1 as fallback settings. The system makes
four internal candidates, validates and
reranks them, and returns one `.osu` plus an audit manifest. It emits timing,
settings, circles, sliders, spinners, combos, and default osu! hitsounds; it
does not create a background, custom assets, storyboards, or a complete mapset.

This is a research/editor-assistance tool. It must never claim that its output
is eligible for osu! ranking: the current osu! ranking criteria require hit
objects, hitsounds, and timing for Ranked maps to be created by direct human
input without generative tooling.

## Corrected architecture

```text
audio
  -> tempo / meter / downbeat posterior and confidence
  -> canonical TimingMap candidates
  -> timestamp-aligned multi-rate audio features
  -> whole-song section and bar context
  -> grammar-constrained sparse event planner
  -> active-event coordinate and slider-path diffusion
  -> constraint projection and .osu compiler
  -> strict validator, difficulty calculator, and candidate reranker
  -> audited playable draft
```

### Musical clock

The model must infer timing from audio; target `.osu` timing data is never
available at inference. A whole-song clock resolver consumes beat/downbeat
posteriors, onset evidence, and learned corrections, then emits ranked timing
hypotheses with red timing points, offset, meter, confidence, and inherited
slider-velocity (SV) segments. Kiai is preserved when parsing source maps for
audit, but v1 deliberately neither predicts it nor supplies it as a planner
feature. The downstream planner trains on a
mixture of oracle, jittered, and predicted clocks so it is robust to its own
clock error.

Each beat candidate has a real `time_ms`, continuous beat coordinate, bar ID,
meter, and confidence. Audio is sampled at those timestamps. Never stretch
frames to slots with generic interpolation: that corrupts variable-tempo maps.
Use rational snap tokens `{1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16, 1/24,
1/32, 1/48}` (with an internal rare 1/96 class) plus a bounded
`±1/192`-beat timing residual rather than a fixed two-lane grid. Its final
quantization error is bounded to half a 1/1536-beat unit; corpus audit must
report rare-1/96 and nonzero-residual rates so the model does not use them as
an uncontrolled escape hatch.

### Audio features

Canonicalize to 24 kHz mono PCM while preserving the input filename for the
compiled map. The first feature stack is:

* 96-bin log-mel at 100 Hz, log-RMS, multi-band flux, and transient/harmonic
  summaries for local rhythm;
* Beat This!-compatible beat/downbeat posterior and confidence as soft input,
  not an infallible rule;
* one frozen MERT-v1 layer compressed with train-only PCA to 128 fp16 features
  at 25 Hz for musical semantics; and
* whole-song pooled semantic/novelty features at 1 Hz for section form.

Store only compressed semantic features with timestamps, source audio hash,
codec/sample-rate details, model revision, PCA hash, schema version, and cache
hash. One full 75 Hz x 768 fp16 MERT layer over 2,526 hours would be about
1.05 TB; all hidden layers would be about 13.6 TB. High-rate features remain
on-demand unless a profile proves their cache is affordable.

Do not include speech envelope, raw waveform encoders, stems, CQT/chroma,
lyrics, MERT-v2/MuQ, or full-song audio diffusion in v1. Add them only as
isolated post-baseline ablations.

### Symbolic planning and geometry

Use a roughly 75-90M parameter single-GPU architecture: local
timestamp-aware feature encoder, full-song bar/section encoder, and an
7-layer, 768-dimensional causal sparse event decoder. Each bar receives an
explicit recurrent summary of the preceding generated bar, trained with
boundary-condition dropout; full-song context supplies form while the summary
prevents independent bar rollouts. Geometry runs over overlapping active-event
windows with deterministic overlap reconciliation. A future planner-window
ablation may add 8-16-bar symbolic rollout windows, but it must retain this
explicit boundary contract rather than silently resetting at cuts.

The canonical `MapSpec` includes timing/meter/SV/Kiai, map settings,
combo/hitsound fields, circles, spinners, and sliders with curve family,
duration, repeats, pixel length, and up to eight masked anchors. Use a
grammar-constrained event sequence rather than independent occupancy/type
heads. The decoder emits only legal ordered event structures and uses a
per-bar EOS.

Autoregressive planner inference must use a pre-norm decoder self-attention
cache. Re-evaluating a full prefix after every sampled token changes one bar
from quadratic to effectively cubic work; cache normalized self-attention
history per decoder layer and retain full bar-memory cross-attention. A CPU
equivalence test must compare cached evaluation decoding to teacher-forced
evaluation decoding before any checkpoint is trained.

Diffusion is limited to multimodal geometry. It operates over the ordered
active event sequence and predicts bounded relative circle coordinates and
variable slider anchors conditional on the sampled event plan, prior/next
cursor state, object type, timing gap, profile, and slider duration. Every
sample is projected to the playfield, valid curve topology, requested path
length, duration, and compatible SV before compilation.

### Compilation, validation, and selection

Generate four internally diverse candidates. For each candidate: compile it,
strictly validate parser and interval rules, calculate realized difficulty with
a pinned compatible calculator, score timing support/density/flow/star
calibration/novelty, then return the highest-scoring valid map and its audit.
If no candidate passes hard validation, return a structured failure report
rather than an invalid map.

The strict validator checks parser serializability, unique ticks, monotonic
interval occupancy, circle and post-slider gaps, snap accuracy, playfield and
slider-path bounds, spinner duration, settings precision, and seam conflicts.
Mapset Verifier remains an external human-facing check rather than a required
training dependency.

## Data and training

Use `project-riz/osu-beatmaps` only for v1. Filter to osu!standard Ranked maps;
retain Approved/Loved/Pending material as explicit separate tiers, not silent
training data. Preserve upstream `audio_hash`, map/mapset/mapper IDs, status,
year, map settings, timing complexity, and parser/feature versions.

Build an immutable source manifest before chunking. Split audio/acoustic-
fingerprint clusters 80/10/10 and audit distributions. Sample songs uniformly
before their difficulties/maps. Fit PCA, normalization, style clusters, and
threshold calibration only on train groups. Never expose raw mapper identity
as an imitation control; learn coarse structural style clusters instead.

Train in stages:

1. corpus audit, MapSpec round trip, strict compiler, and clock resolver;
2. grammar-constrained event planner with profile/style condition dropout;
3. planner rollout under imperfect clocks;
4. geometry/path diffusion on ground-truth and sampled plans; and
5. reranker and star/density calibration.

Run only on the approved iLab hardware when the user starts it. The primary
target is one RTX A6000 with BF16, checkpointed/resumable three-day Slurm jobs.
The 4070 Super is for smoke tests and short ablations. Do not use the old
full-song AE/flow path or its unsafe multi-GPU launcher.

The trainer records a base seed, uses an epoch-derived DataLoader generator,
restarts workers each epoch so they receive the current rotating-bar state,
and restores Torch CPU/CUDA RNG state on resume. This makes data order and
dropout/diffusion randomness reproducible for a fixed software/GPU setup; it
does not promise bitwise identity across different CUDA/PyTorch builds.
Planner-to-geometry handoff is deliberately different from a same-stage
resume: it restores compatible model weights, then starts a fresh geometry
optimizer/scaler/RNG timeline. Optimizer moments from planner parameters must
never be loaded into the geometry optimizer.

## Current-v2 corrections

Keep timing parsing, source grouping, structured target ideas, and field-aware
losses. Replace the current fixed two-lane grid, fixed 4/4 `section_slots`,
generic frame-to-slot interpolation, oracle timing/settings dependence,
single-coordinate slider diffusion, padding-sensitive convolutions, and
start-time-only validator. Parsed fields must be either modeled by `MapSpec`
or intentionally canonicalized; none may silently disappear.

## Acceptance gates

* Audit timing changes, meter changes, SV, Kiai, triplets, 1/48 snaps, long
  sliders, high repeats, dense patterns, and chunk-boundary events.
* Prove no audio/fingerprint cluster crosses train, validation, or test.
* Verify a bare audio request works without a reference map or oracle settings.
* Round-trip generated maps through an independent parser and strict validator.
* Test clock corruption, padding invariance, boundary stitching, and no-output
  candidate handling.
* Evaluate rollout-only held-out-song results: parser validity, timing phase,
  object mix, slider legality, density/section correspondence, realized star,
  strain, diversity, and nearest-training-map distance.
* Validate metrics with controlled timing shifts, type swaps, density spikes,
  invalid sliders, and repetition collapse.
* Require blinded experienced-player/mapper review before describing results as
  ranked-like.

## Implementation boundaries and failure-prevention rules

The repository implementation is intentionally strict about a few tempting
shortcuts that would invalidate this plan:

* A per-song cache content hash must **not** be checkpoint compatibility
  metadata. The checkpoint binds to a shared feature-pipeline signature;
  records retain their own canonical-PCM/cache hashes for provenance.
* Sparse grammar changes are schema changes. The current residual-aware,
  prior-bar-conditioned implementation uses song-record/checkpoint schema 2;
  never load schema-1 token targets or weights as though the vocabulary and
  decoder cache semantics were unchanged.
* Training samples are selected song/fingerprint-component first, then one map
  and one rotating bar. Do not flatten all bars/maps and shuffle them equally:
  that would make prolific mapsets and long songs dominate both optimization
  and validation.
* One song record has a small JSON sidecar containing only split, component,
  feature-pipeline, dimensions, and bar count. Dataset discovery must use that
  index rather than deserializing every full feature tensor; the first actual
  record load recomputes and checks it so the sidecar never becomes an
  unverified source of truth.
* Padding must be excluded inside temporal normalization and attention, not
  merely masked in the loss. All-empty geometry rows require a dummy masked
  attention token to avoid NaNs.
* The planner must receive an explicit target-bar ID. Whole-song cross
  attention alone cannot tell a decoder which bar it is supposed to emit.
  It must also receive a preceding-bar symbolic summary (or a learned
  no-boundary state), because audio context alone does not guarantee a
  continuous combo/spacing transition across independently decoded bars.
* Slider duration must be compiled through native BPM/SV math. Do not use a
  beat-duration token as a pixel length directly across a BPM change; derive
  the requested endpoint in milliseconds, calculate length using its starting
  BPM/SV, then round-trip validate it.
* The first release intentionally canonicalizes presentation: no generated
  backgrounds, custom samples, storyboard, Kiai predictor, or learned style
  cluster. A requested style cluster must fail explicitly until train-only
  structural clustering and nearest-training-map checks exist; never pretend a
  control works when the model was not trained on it.
* The feature-bundle/clock-resolver adapter is explicit and pinned. Do not
  make inference download a changed MERT or Beat This! model, and never fall
  back to a source `.osu` timing map when an audio clock is missing.

These are not optional polish items. They are tests or hard runtime failures
because each one otherwise produces deceptively good reconstruction metrics
while breaking the audio-only generation claim.

## Research basis

* osu! Ranking Criteria: <https://osu.ppy.sh/wiki/en/Ranking_criteria>
* Mapperatorinator: <https://github.com/OliBomby/Mapperatorinator>
* osu-diffusion: <https://github.com/OliBomby/osu-diffusion>
* MERT-v1-95M: <https://huggingface.co/m-a-p/MERT-v1-95M>
* Beat This!: <https://github.com/CPJKU/beat_this>
* ChartGenEval: <https://arxiv.org/abs/2607.12857>

# V2 model assessment — October 14, 2026

Follow-up: a [full-song generation benchmark](v2-inference-playtests-2026-10-14.md)
has now produced four validated playtest archives. None of its 28 candidates
passed the requested ±0.5-star difficulty gate, including human-clock variants.
This is evidence of poor difficulty control in the tested batch. The results
below describe the earlier teacher-forced evaluation; human playability remains
untested.

**Assessment: a promising, data-limited research prototype with useful learned
structure. Full-song beatmap quality remains unverified.** This assessment uses
the completed training history and a new read-only evaluation of the saved
geometry-stage best checkpoint on the untouched test split. No weights changed.

## What was measured

Evaluation job `505829` completed with exit code `0:0` in 2 minutes 59 seconds.
It evaluated 6,818 bars, 225,451 target tokens, and 18 independent test source
components using the checkpoint revision `checkpoint-43200-a991a335083180d0`.
The loader accepted the trained geometry checkpoint and its provenance.

The protocol supplies the human chart's preceding tokens, previous-bar tokens,
and timing grid. Geometry is conditioned on the human event sequence and the
trained planner's hidden states. It tests prediction/denoising on unseen source
components, not sampled end-to-end generation or musical playability.

Baseline job `505881` completed with exit code `0:0`. For each known token
family, it always predicts the most common token in the **training** corpus;
the resulting predictions are scored on test labels. The model uses raw
argmax logits, without grammar masking. This is a simple descriptive baseline,
not a competitive rhythm mapper or an audio-conditioned baseline.

| Test metric | Result |
| --- | ---: |
| Overall token accuracy | 71.72% |
| Frequency baseline token accuracy | 51.28% |
| Accuracy averaged equally across audio components | 71.34% |
| Median audio-component accuracy | 72.11% |
| Lowest / highest audio-component accuracy | 56.00% / 78.27% |
| Token-weighted cross-entropy | 0.806962 |
| Token perplexity | 2.2411 |
| Batch-mean cross-entropy, matching training's averaging style | 0.783924 |
| Geometry weighted loss | 0.359225 |
| Geometry noise-prediction loss | 0.346498 |
| Spacing / turn / slider-shape auxiliary losses | 0.002396 / 0.067898 / 0.090924 |

Perplexity is `exp(token-weighted cross-entropy)`. It summarizes uncertainty
about the next token, not the number of plausible maps or object accuracy.
Tokens from the same song are correlated; the large token count does not make
this an evaluation on thousands of independent songs.

| Token family | Model | Training-frequency baseline | Meaning |
| --- | ---: | ---: | --- |
| Control | 96.43% | 78.51% | Continue an event sequence or end the bar |
| Beat offset | 65.49% | 11.58% | Exact next event position in the beat grid |
| Snap | 94.89% | 46.37% | Rhythmic subdivision |
| Microtiming residual | 23.74% | 11.94% | Exact small correction to the beat-grid position |
| Object kind | 64.71% | 55.08% | Circle, slider, or spinner |
| New combo | 84.98% | 76.03% | Combo boundary |
| Hitsound | 69.06% | 63.96% | Hitsound flags |
| Duration | 56.36% | 42.55% | Slider/spinner duration |
| Curve | 55.30% | 41.87% | Slider curve family |
| Repeats | 89.31% | 86.65% | Slider repeat count |
| Anchors | 77.02% | 52.35% | Slider control-point count |

Beat offsets and subdivision show strong gains over a common-default
predictor. Object kinds and slider attributes are less dependable. Repeat
accuracy is high largely because one repeat is common. Microtiming residuals
are bounded to ±1/192 beat, so a residual-class error can be much smaller than
an incorrect main beat offset. Exact matching against one human chart also
penalizes alternative valid mapping decisions.

## Training evidence

| Selected checkpoint metric | Planner | Geometry |
| --- | ---: | ---: |
| Training loss | 0.505064 | 0.400511 |
| Validation loss | 0.862975 | 0.375872 |
| Selected epoch, one-based | 40 | 72 |
| Total epochs completed | 55 | 87 |

Planner validation loss improved 26.3% from the first completed epoch, and
geometry validation loss improved 29.6%. These are improvements from an
already-trained first epoch, not comparisons with an untrained network.

The planner's persistent train/validation gap and validation plateau suggest
limited generalization and possible overfitting. The gap is not a pure
overfitting statistic: training uses source-balanced rotating samples and
condition dropout, while validation evaluates all bars without dropout.

Geometry validation being lower than training is compatible with geometry
reflections, condition dropout, model dropout, fixed validation noise, and
different sampled content. It is not itself evidence of leakage or a superior
test distribution.

Geometry's selected best loss is 0.375872 at epoch 72. A slightly lower value,
0.375342, occurred at epoch 76, but the difference was below the configured
`min_delta=0.001`; checkpoint selection and patience therefore did not reset.

Loss definitions:

- Planner: cross-entropy on the next symbolic token, with padding excluded.
- Geometry: diffusion noise-prediction loss plus `0.1 × spacing loss +
  0.05 × turn loss + 0.1 × slider-shape loss`.
- The structural losses are normalized, noise-weighted comparisons with the
  same corrupted source example. They are not pixel error, star error, angle
  error in degrees, or a percentage of playable objects.

## Architecture and training setup

There are **113,585,507 parameters**: 93,202,273 in the planner and 20,383,234
in geometry. The planner is a hierarchical audio/bar encoder and autoregressive
symbolic decoder. Its width is 768, with 12 attention heads, four local layers,
three global layers, seven decoder layers, and one boundary-context layer.
Its configured limits are 512 bars and 2,048 decoder tokens.

Geometry is a six-layer, width-512 Transformer diffuser with eight attention
heads. It predicts object heads and up to 32 additional slider controls.
Heads use bounded playfield coordinates; slider controls are relative to their
head and may extend outside the canvas. Native slider families B, L, P, and C
are represented. The planner selects topology; geometry supplies coordinates.

Audio features are a 101-dimensional rhythm stream (96 mel bands plus rhythm
features) and a 128-dimensional pooled-mel stream. This run does **not** use a
pretrained MERT or AST music encoder. The legacy README metrics and legacy
architecture description do not describe this checkpoint.

The planner consumes six normalized profile controls: target stars, AR, CS,
combined OD/HP, slider multiplier, and tick rate. Geometry consumes planner
context plus event gaps, duration, and intended slider length.

Training used AdamW with learning rate `2e-4`, weight decay `0.05`, betas
`(0.9, 0.95)`, gradient clipping at 1.0, BF16, global batch 16, and seed 1337.
Model dropout is 0.1; training additionally drops profile and previous-bar
conditioning with probability 0.15 and symbolic geometry context with 0.1.
Geometry uses whole-pattern reflections. The planner is frozen during geometry
training; the stages are not jointly optimized end-to-end.

Four A4500s ran both stages in DDP. The run completed in 9:05:23, with a 48-hour
Slurm limit. Each training epoch sampled 2,400 bars, balanced over source
components. This does not mean every cached training bar appeared every epoch.
There were 150 global optimizer updates per epoch; checkpoint `global_step`
instead sums per-rank batch counts and should not be read as optimizer updates.

## Data and scope limits

The corpus contains 773 authentic charts from 188 source components:
150 train, 20 validation, and 18 test. It contains 66,903 / 8,555 / 6,818 bar
targets respectively. Audio/chart identities, split membership, and checksums
were verified. `production_ready: true` is an integrity and minimum-size guard,
not a musical-quality certification.

The upstream scan was limited to the first 300 raw tracks for storage capacity.
This is a small, not explicitly genre/difficulty-stratified corpus for a
113.6-million-parameter model. Neither 773 charts nor repeated bar sampling
provides 773 independent songs.

There were 88 rejected charts: 51 exceeded eight slider repeats, 24 exceeded
32 controls, three had unsupported P-slider topology, two exceeded the timing
residual representation, and eight had object heads outside the playfield.
This restricts the represented mapping styles.

Slider coverage is 68,858 P, 39,142 L, 30,384 B, and 105 C sliders. C sliders
are approximately 0.08% of this coverage, so competence on rare curve families
has very little supporting training evidence.

All current record timing variants use source-map oracle clocks. Geometry
targets remain bar-local. Generated previous-bar context, clock-estimation
errors, planner mistakes, and long-range geometry can therefore create a gap
between this test and raw-audio full-song generation. The test does not prove
that the network relies on audio rather than token history; an audio ablation
would be needed to quantify that.

## How to decide whether it is a good mapper

1. Generate complete maps for the held-out songs at several supported target
   difficulties and seeds. Compare oracle-clock generation with audio-only
   timing estimation, so clock failure is separated from mapping failure.
2. Report candidate failures and success rates before filtering as well as
   selected outputs. An accurate star rating among a few survivors can hide
   widespread generation failures.
3. Measure actual star error with the pinned calculator, valid object/slider
   rates, object density, silence/break handling, interval distributions,
   repeated patterns, spacing/turn distributions, and continuity across bars.
   The CLI's ±0.5-star tolerance and 10 ms minimum start gap are gates, not
   evidence that this model passes them reliably.
4. Use onset precision/recall at stated tolerances as one rhythm diagnostic;
   one human chart is not the only valid rhythm or spatial layout. Compare
   relative structure and distributions rather than demanding identical x/y.
5. Playtest blind, comparing model maps with simple baselines and human maps
   on the same audio and requested difficulty. Rate musical alignment,
   readability, flow, repetition, slider quality, and fun separately.
6. Report results per song and difficulty, including worst cases. Use a larger
   untouched song set for a dependable generalization claim.

The next useful experiment is a held-out full-song generation benchmark and
playtest, rather than another training run chosen solely by lower loss.
No such rollout, star-calibration measurement, or playtest was performed in
this assessment.

## Artifacts

- `v2-test-metrics-2026-10-14.json`: full held-out metrics and model configs.
- `v2-frequency-baseline-2026-10-14.json`: train-fitted baseline and test scores.
- `v2-training-curves-2026-10-14.png`: training/validation learning curves.
- `scripts/evaluate_v2_checkpoint.py`: reusable read-only evaluator.
- Checkpoint: `/common/users/asj102/osu_project/v2_runs/run_20261014_real/geometry/best.pt`.

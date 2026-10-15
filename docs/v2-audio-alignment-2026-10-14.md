# V2 musical alignment: investigation, decision, and implementation

## Decision

Keep the hierarchical event planner and geometry diffuser. Fix the planner's
within-bar audio bottleneck, add chart-timing supervision, and measure audio
dependence explicitly before investing in another larger model or full corpus
of separated stems. This is an additive, opt-in V2 configuration on the existing
`architecture-v2-hierarchical-beatgrid` branch. The trained baseline `adb0a55`
was clean and already pushed before these changes.

The user's suggestion—inspect different musical layers and see which chart
events align with them—is valuable. FL Studio's original project would expose
actual tracks/patterns/MIDI, but these cannot be recovered exactly from a mixed
recording. Its stem extraction estimates drums, bass, vocals and instruments;
Demucs offers a comparable separation workflow. Estimated stems may contain
bleed and artifacts. They are useful additional evidence, not ground-truth
labels saying which sound a mapper intended to follow. [FL Studio manual](https://www.image-line.com/fl-studio-learning-content/fl-studio-online-manual/html/playlist.htm),
[Demucs implementation](https://github.com/facebookresearch/demucs).

Do not impose rules such as “drum = circle” or “vocal = slider.” The chosen
musical layer can change within a song and across difficulty levels. Sliders
can express duration or rhythmic grouping, and not every audible attack must
be mapped. A chart is one interpretation of the audio.

## What the latest evaluation establishes

The supplied evaluation uses 773 authentic charts / 188 source groups, with
150/20/18 train/validation/test groups. Its 71.7% teacher-forced token accuracy
and 65.5% exact offset accuracy show learned structure, not proven musical
alignment during independent generation. Correct preceding tokens, previous-bar
history and human clocks are supplied in that measurement.

The subsequent playtest report is more concerning: none of 28 candidates passed
the requested ±0.5-star gate. Kaguya's delivered uncalibrated draft was 6.45 stars
for a 4.5-star request. Human-clock variants improved reference onset matching
on the held-out songs but still overshot difficulty. Clock estimation and event/
geometry calibration are therefore distinct issues. These are existing baseline
results, not outputs of the new code.

## A concrete architecture bottleneck

`TimestampedAudioEncoder` processes 100 Hz rhythm frames, but `encode_bars`
previously sampled only one learned vector at each bar midpoint. The decoder
cross-attended to those bar vectors. A `return_local_frames` capability existed,
but the planner did not use it. Local convolution and normalization give those
vectors context, so it would be wrong to say the rest of the audio was wholly
invisible. However, the decoder had no individually addressable within-bar
frame sequence to distinguish detailed attacks and sustains.

Adding four stem encoders while still collapsing each bar this way would leave
that bottleneck intact. Direct local audio memory is the smaller first change.

## Fresh paired audio-dependence experiment

Slurm job **507295** evaluated the unchanged trained geometry checkpoint's
planner on **80 validation bars from all 20 validation source groups**. Four
bars per group were sampled deterministically, seed 1731, for 2,676 scored tokens.
The same bar indices, reference current prefixes, profiles, clocks and masks
were used in each condition. No weights were updated.

| Condition | Token CE | Overall accuracy | Exact offset accuracy |
|---|---:|---:|---:|
| Aligned audio | 0.9390 | 70.03% | 60.63% |
| Audio shifted about 137 ms | 0.9539 | 69.62% | 58.54% |
| Audio shifted about 997 ms | 0.9469 | 69.51% | 57.14% |
| Zeroed audio features | 0.9486 | 69.51% | 59.58% |
| Previous-bar history removed | 1.0169 | 67.53% | 50.87% |

The small overall change with shifted/zero features, compared with removal of
previous-bar history, supports prioritizing audio grounding. It does not prove
that the model entirely ignores audio. Reference current prefixes remain a
strong cue, zeroed normalized features are out of distribution rather than
literal silence, and shifts wrap at the ends of valid streams. Each shift is
rounded to each stream's own frame rate. This small diagnostic is not a
statistical proof or a free-generation quality result.

Per-family and per-source measurements are retained in
`v2-audio-dependence-2026-10-14.json`. The new paired experiment used validation, not test. Previously generated
held-out playtest archives were inspected visually; they were not optimization
examples or a target for fitting the new parameters.

## Options weighed

| Option | Benefit | Cost/limitation | Decision |
|---|---|---|---|
| Direct local audio memory | Makes attacks and within-bar changes addressable by the decoder | Adds cross-attention memory; needs fine-tuning | Implement now |
| Auxiliary chart-alignment targets | Explicitly trains audio/profile representations to locate selected events and slider activity | One chart is one valid interpretation; weight needs ablation | Implement now |
| Stem decomposition | Separates some overlapping musical layers for inspection and potentially features | Separation errors, extra compute/storage, identical extraction needed in training and inference | Support optional exported-stem diagnostics now; defer mandatory training features |
| Pretrained transcription encoder | Transfers onset/note information learned from much larger datasets | Integration, model/cache versions and compute require a separate controlled experiment | Strong next feature experiment |
| Generic pretrained MERT features | Richer musical context than repeated pooled mel | Musical semantics alone do not guarantee precise event timing | Compare against transcription features, not automatically the first choice |
| Learned beat/downbeat tracker | Addresses tempo/phase estimation weakness | Requires proper clock/label alignment and an adapter, not arbitrary query shifts | Next separate timing experiment |
| More independent songs / balanced coverage | Improves diversity and generalization | Existing quota and preprocessing costs matter | Prioritize after the controlled architecture comparison |
| Larger planner | More capacity | Already ~113.6M total parameters on only 188 source groups; may worsen overfitting | Not selected |
| Force every onset / hard spacing rules | Easy to implement | Suppresses mapper choice; does not ensure difficulty or musicality | Not selected |

Beat-aligned sequence modeling provides precedent for explicitly aligning audio
and event sequences. [Yi et al., 2023](https://arxiv.org/abs/2311.13687).

A recent transcription-transfer paper reports better chart fitting and note-type
accuracy with a frozen AMT encoder than scratch encoders and MERT in its studied
setting. This motivates testing transcription features, but is not evidence that
they will improve this exact osu!standard pipeline without evaluation.
[An et al., 2026](https://arxiv.org/abs/2610.14955).

MERT is a broadly pretrained music representation rather than an explicit chart
alignment objective. Beat This! addresses beat/downbeat estimation and still
reports limitations on difficult material and continuity. Neither component
alone solves mapper choice or playability.
[MERT](https://arxiv.org/abs/2306.00107), [Beat This!](https://arxiv.org/abs/2407.21658).

## Implemented model path

The new preset is `configs/v2_audio_alignment_planner.json`:

```json
{"local_audio_tokens": 128}
```

1. Keep the whole-song bar encoder and original token decoder.
2. Retain the learned high-rate rhythm frames instead of discarding them.
3. Sample 128 positions across the target bar. Convert each beat-space query to
   milliseconds independently through the timing map; this handles tempo
   changes inside a bar rather than interpolating just two endpoints.
4. Fuse learned frame features with the raw rhythm features. The raw path keeps
   sharp attacks and relative energy available despite temporal normalization.
5. Add within-bar position and target-bar/profile context, then append these
   local tokens to the decoder memory. Out-of-audio queries are masked.
6. Use the same memory construction in teacher forcing, cached autoregressive
   generation, and the planner contexts passed to geometry.

The preset provides about 15.6 ms query spacing for a four-beat bar at 120 BPM.
This is a sampling resolution, not an accuracy claim. Slow bars have coarser
spacing, and 128 should be compared with other budgets on validation.

A four-channel auxiliary head predicts **circle onset, slider onset, spinner
onset, and slider-active interval** from audio plus bar/profile context. It does
not receive reference token prefixes. Targets come from existing chart tokens
and the stored oracle timing map. A slider begun in an earlier bar remains
active until its end. Onset labels have triangular tolerance of at least 25 ms,
widened to the grid step if needed to avoid missing positives on coarse grids.

The planner objective is:

```text
next-token cross entropy + alignment_loss_weight × chart-alignment BCE
```

The default auxiliary weight is 0.2; onset positive weights are fixed at five
and slider-active weight at one. These are experimental starting values. The
labels describe where the chart selected events, not all acoustic onsets and
not instrument identity. No alignment labels are passed into model inputs;
regression tests check this directly. The auxiliary head's predictions are
not hard gates that force object placement.

Existing records sometimes quantize an event just before the bar end onto the
endpoint. Label reconstruction tolerates only the codec's half-residual-step
rounding bound. Strict generation bar membership is unchanged.

## Compatibility and training

Legacy checkpoints omit `local_audio_tokens`, which defaults to zero. They keep
their previous parameter names and generation path. Existing authentic schema-3
records, features, source splits and vocabulary remain usable; this change does
not require redownloading/rebuilding the corpus.

New local-audio checkpoints require a matching query-grid configuration on
resume. The trainer rejects a silent grid-size change even when weight shapes
happen to match. `scripts/evaluate_v2_checkpoint.py` now supplies the required
queries when evaluating a new checkpoint.

Use **explicit initialization**, not resume, when moving the trained legacy
planner to the new configuration:

```text
--planner-config configs/v2_audio_alignment_planner.json
--initialize-planner /common/users/asj102/osu_project/v2_runs/run_20261014_real/planner/best.pt
--alignment-loss-weight 0.2
```

This validates feature/split/schema/vocabulary compatibility, copies the
existing planner weights, and leaves only the explicitly added modules newly
initialized. It does not import optimizer moments, epoch counters, RNG, or
claim that the geometry is already adapted to the changed planner context.

`configs/v2_audio_alignment_train.sbatch` prepares a fresh planner fine-tune
at learning rate 5e-5, followed by geometry training with the resulting planner.
It takes `SOURCE_DIR`, `DATA_DIR`, a new `RUN_DIR`, and `INITIAL_PLANNER`.
Both stages use the same local-audio config. It defaults to one GPU, BF16,
batch two, accumulation eight, 60 planner epochs and 100 geometry epochs with
patience ten. GPU count follows the Slurm allocation. These are configurable
run settings, not a promise of required runtime or optimal convergence.

The isolated source folder for this work is:

```text
/common/users/asj102/osu_project/codex_v2_audio_alignment_20261014
```

No full training job was submitted. Existing source deployments and trained
checkpoints were preserved. Job 450738 remains untouched.

## Diagnostic tools and plots

`audit_v2_audio_dependence.py` reproduces the paired intervention test; default
split is validation. `configs/v2_audio_audit.sbatch` runs it on one GPU.

`analyze_v2_audio_alignment.py` exports PNG/SVG figures, an event-level CSV and
JSON measurements for a shared-audio mapset:

```bash
python -m pip install -r requirements-audio-analysis.txt
python -m scripts.analyze_v2_audio_alignment \
  --osz path/to/mapset.osz --start 20 --seconds 15 --output-dir new-report
```

It shows waveform, spectrogram, frequency-band attack envelopes and one event
lane per chart. Slider intervals are horizontal segments. It reports measured
stars using the same pinned calculator as training/generation. It requires a
single shared audio asset when loading an archive; explicit `--audio`/`--maps`
pairings are user-supplied and not independently verified.

To inspect available FL Studio or Demucs exports:

```bash
python -m scripts.analyze_v2_audio_alignment \
  --audio song.wav --maps easy.osu insane.osu \
  --stems drums=drums.wav bass=bass.wav vocals=vocals.wav other=other.wav \
  --start 20 --seconds 15 --output-dir new-stem-report
```

Exports must preserve original time zero and duration. Length checks can catch
truncation but cannot prove phase alignment. The tool does not run a separator,
download its weights, infer original MIDI, or assign instrument labels to chart
events. Frequency-band views are not instrument separation. No stem-conditioned
training path is claimed in this implementation.

Nearest-attack proximity is a many-to-one acoustic diagnostic, not event F1 or
mapper/model intent. Actual musical layers must be assessed by listening and
controlled experiments. No source audio was added to Git.

The Star!! comparison illustrates why that distinction matters: median distance
to a mixture attack was 9 ms for the 9.25-star generated variant, 18 ms for the
4.66-star human chart, and 19 ms for the 5.17-star audio-only variant. A dense,
over-difficult chart can score well on this proximity measure. Matching nearby
spectral peaks alone is therefore an inadequate training or acceptance objective.

Local reports were generated for the original five-difficulty reference, the
trained Kaguya draft, and the same-audio Star!! human/AI playtest variants.
Readable copies and figures are under `codex-mds/osu/2026-10-14-audio-alignment`
on the Mac; original large artifacts are ignored under `data/audio-alignment-20261014`.

## Validation and next acceptance criteria

- Local full V2 suite: **106 tests and 28 subtests passed**. Tests cover variable
  tempo queries, carried sliders, rounding boundaries, label/input separation,
  audio perturbations, finite gradients, padding, legacy initialization, resume
  mismatch rejection, cached decoding and full-song sampling using local memory.
- After adding the exported-stem duration check, the focused local alignment
  suite passed **11 tests**. The combined alignment/training tests on iLab passed
  **14 tests**. Python compilation and Slurm shell syntax checks also passed.
- Slurm **507391**: three BF16 optimizer steps of the full-sized new planner on
  genuine **training** data. New projection weights changed, gradients stayed
  finite, and peak allocated GPU memory was ~2.49 GB for that one-example batch.
  This is not a throughput benchmark for arbitrary songs or a quality result.
  No weights were saved. The prior smoke job 507316 exposed the quantized bar
  boundary issue and failed before completing training; it was fixed and tested.
- Successful smoke losses were 0.7342, 0.7258, 0.6543. This tiny same-example
  sequence with dropout is only execution evidence, not generalization evidence.
- Baseline audio dependence is measured; improved musicality of the new model
  has **not** been established because full fine-tuning has not been run.

The next controlled comparison should use the same splits, budgets, profiles,
seeds and calculator: baseline; local audio with auxiliary weight zero; local
audio with weight 0.2. Evaluate per-family CE, source-balanced audio-intervention
deltas, and full-song rollouts with both human and estimated clocks. Report all
failed candidates, star error, silence behavior, density, reference onset F1 at
20/50 ms, and blind musical-alignment/playability judgments separately. Do not
select solely by combined loss or spectral-attack proximity. Keep further final-test evaluations out of iterative tuning; after repeated
playtest inspection, use fresh held-out songs for the final quality claim.

Only then decide whether separated stems or a frozen transcription encoder
justify corpus-wide extraction. Remaining difficulty overshoot, cross-bar
geometry, audio-only clock robustness and data diversity still need their own
experiments; this change does not claim to solve all of them.

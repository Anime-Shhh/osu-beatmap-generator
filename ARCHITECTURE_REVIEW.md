# Architecture audit and v2 direction

> Historical review: this document explains why the legacy and first
> beat-grid v2 scaffold were rejected. The implemented current architecture is
> the sparse-event system in
> [plans/tempo-posterior-hierarchical-event-diffusion.md](plans/tempo-posterior-hierarchical-event-diffusion.md),
> not the `src/v2/model.py` beat-grid prototype discussed below.

This review was made without training, GPU use, or touching the active iLab
run.  It is deliberately blunt because the next expensive experiment should be
based on trustworthy data and evaluation rather than a better-looking loss.

## Verdict

Do not scale the current full-song latent-flow experiment yet.  The default
project path is still a short-window AST + autoregressive Transformer;
`--stage flow` is an experimental, deterministic latent flow-matching path,
not a completed end-to-end diffusion system.  Neither reported legacy metric
(roughly 396 ms timing MAE and 0.108 hit F1) is evidence of a playable map.

The dominant problems are split leakage, teacher-forced evaluation, malformed
audio timing alignment, a lossy target representation, and unconstrained map
serialization.  More parameters will not repair these foundations.

## P0 findings in the current code

| Problem | Why it invalidates results | Evidence |
|---|---|---|
| Song leakage | Samples are windowed before a shard-level split. One song, its overlapping windows, and alternate difficulties can cross train/validation/test. Metadata does not retain an audio hash, map ID, or mapset ID. | `src/preprocess.py`, `src/splits.py`, `src/dataset.py` |
| “Validation” is teacher forced | Both training validation and standalone eval feed ground-truth prefixes, then argmax the next token. They do not measure autoregressive rollout, syntax validity, window stitching, or map playability. Empty predictions are excluded from timing averages. | `src/train.py`, `src/eval.py`, `src/metrics.py` |
| The multi-GPU launcher is unsafe | `configs/train_multi_gpu.sh` starts `torchrun`, but `src/train.py` has no process-group setup, DDP wrapping, distributed sampler, rank-local GPU selection, reduction, or rank-zero checkpoint policy. | `configs/train_multi_gpu.sh`, `src/train.py` |
| AST is not temporally decoded correctly | AST emits flattened frequency × time patch tokens plus special tokens. The code runs a 1-D temporal convolution/interpolation over that flattened stream without dropping specials, restoring the grid, or pooling frequencies. Its 200-ish ms patch stride cannot provide onset precision after 2× interpolation. | `src/model.py` |
| Pretrained loading can silently become random loading | A broad exception falls back to a randomly initialized AST and then freezes almost all of it. An offline Hugging Face failure can therefore look like successful training with a mostly frozen random encoder. | `src/model.py` |
| Full-song flow ignores padding | Collate creates signal masks, but the AE, AST conditioner, losses, and metrics do not consume them. Padded tails alter attention and losses. | `src/dataset.py`, `src/latent_ae.py`, `src/flow_model.py`, `src/train.py` |
| Augmentation corrupts labels | Streaming time stretch changes waveform duration but tokenization still reads unchanged `.osu` timestamps and BPM. Cached training does not apply it at all. | `src/dataset.py` |
| Existing inference does not match training windows | Targets are the middle three seconds of a six-second window; inference neither restricts objects to that interval nor uses the same tail offset. Five-ms deduplication can remove legitimate stacks. Scratch-WAV Slurm inference writes an `.osu` that references the temporary WAV filename. | `src/inference.py`, `configs/inference_*.sh` |

Other serious correctness gaps:

- The advertised teacher-forcing schedule is calculated and logged but never
  used; the cached path also bypasses the claimed curriculum and augmentation.
- `beam_size > 1` is not beam search.  Generation samples an arbitrary
  vocabulary token with no grammar/state mask.
- The token format clips rests above 7.94 beats, uses only the first BPM,
  loses SV/offset/time-signature changes, and cannot reproduce slider length,
  repeats, duration, hitsounds, or many control points.  Detokenization hard
  codes sliders to `length=100, slides=1`.
- Residual loss supervises every non-PAD token instead of only field tokens;
  zeros dominate and time/position units are unbalanced.
- The dense nine-channel signal invents linear cursor paths, polyline sliders,
  and circular spinners.  Its decoder re-snaps everything to one BPM.  Flow
  evaluation compares decoded generated signals to decoded lossy targets, not
  to the original map.
- The full-song AE has global quadratic attention over thousands of latent
  frames, while inference recomputes audio conditioning every ODE step (and
  twice with classifier-free guidance).  It is neither a cheap nor a robust
  full-song design.

## Does easy A100 training mean the model is too small?

No.  The legacy model is already roughly an AST-base encoder plus a sizeable
six-layer 768-dimensional decoder; most of AST is frozen and each training
example is only six seconds.  A 40 GB A100 should train that comfortably.

Easy optimization could instead mean: data leakage; heavily teacher-forced
tokens; duplicated nearby windows; a small cached subset; or multimodal mapper
labels being averaged into a weak objective.  Capacity should be changed only
after a song-disjoint, free-running validation exposes a real capacity limit.

## What comparable projects suggest

- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) is the best
  open architectural reference: a large Whisper-derived event model handles
  discrete structure, then a separate coordinate diffusion stage refines
  geometry, followed by deterministic repairs.
- [osu-diffusion](https://github.com/OliBomby/osu-diffusion) uses diffusion
  where it is appropriate—continuous coordinates after rhythm/type/spacing is
  supplied—not as a substitute for symbolic map structure.
- [osu!dreamer](https://github.com/jaswon/osu-dreamer) shows that raw-audio
  latent diffusion is feasible, but not that it is the best path to valid
  osu!standard maps.
- [cmyui/ai-osu-maps](https://github.com/cmyui/ai-osu-maps) is a useful modern
  baseline: frozen MERT features, a decoder-only event model, continuous
  difficulty, mapper conditioning, and constrained time postprocessing.
- Older [osumapper](https://github.com/kotritrona/osumapper) separated rhythm,
  momentum, sliders, and flow.  Its models are dated; its decomposition is
  still sound.
- [Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game
  Charts](https://arxiv.org/abs/2311.13687) found beat-aligned,
  tempo-normalized sequence modeling preferable to sparse framewise targets in
  osu!mania.  It is not a direct standard-mode result, but it is strong evidence
  for beat-aware targets.

## Recommended system

```text
audio
  ├─ high-rate onset / energy / meter features
  ├─ frozen music-SSL features and song-section embeddings
  └─ beat/downbeat/tempo-map estimator (with confidence)
       └─ beat-grid rhythm planner
            ├─ event occupancy, microtiming, type, duration, slider fields
            └─ grammar-constrained event decoder
                 └─ conditional geometry diffusion/refiner
                      ├─ circle positions
                      └─ slider anchors / paths
                           └─ parser + playability validator + reranker
```

Use a continuous beat coordinate and a 1/16 or 1/24 candidate grid plus a
small timing residual.  Do **not** hard-snap all outputs or reduce a tempo map
to one BPM.  Plan discrete events first: they are sparse, grammar constrained,
and need density calibration.  Apply diffusion only to positions/paths, where
there are many musically valid answers for one event plan.

Use local 8–16-second chunks with overlap for high-rate timing, plus a compact
bar/section-level encoder for song form.  A complete three-minute song at
hundreds of frames/sec must not be fed to unrestricted all-pairs attention.
Long-range context should be pooled to bars/sections, then injected back into
the local planner.

The new `src/v2/` package establishes this direction without breaking `main`:

- `timing.py` retains red timing points and inherited slider-velocity segments,
  maps milliseconds to continuous beats, and derives native slider duration;
- `events.py` keeps slider curves, repeats, length, spinner end times,
  hitsounds, and multi-lane simultaneous-event targets, and validates output;
- `data.py` provides deterministic source-audio-hash group splits;
- `model.py` contains a high-rate/local + section/global beat-grid planner and
  a separate coordinate denoiser;
- `losses.py` masks padding and only scores semantic fields where they exist.

It is an architecture foundation, not a trained claim.  A v2 preprocessing and
training command should be added only after the data schema and evaluation
contract below are agreed on.

## Audio representation: what to use

Keep log-mel.  It is a good local representation; the better upgrade is
multi-rate fusion, not a magical replacement feature.

| Stream | Suggested resolution | Purpose | Priority |
|---|---:|---|---|
| Log-mel + local energy/flux | 100 Hz (25 ms window, 10 ms hop) | attacks, density, timbre | required |
| Continuous onset activations | 100 Hz | likely attacks without forcing every hit to be an onset | required |
| Beat/downbeat likelihood, tempo phase, bar phase, confidence | frame + beat grid | meter and tempo changes | required |
| Frozen MERT-v1 95M layer mixture | 75 Hz, cached fp16 | melody, timbre, instrumentation, musical context | first ablation |
| Chroma/CQT | 25–50 Hz | harmony, melody, recurring patterns | second ablation |
| Section/novelty/energy embedding | 0.5–2 Hz | phrase density and chorus/verse variation | second ablation |
| Drum/vocal/bass/other summary features | 100 Hz | mapper choice of drums vs vocals/bass | third ablation |

[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) is a better first
music encoder than AudioSet-oriented AST: its official card specifies a 95M,
24 kHz, 75 Hz, 12-layer music SSL model.  Cache it frozen and learn a small
layer-mix/projection before considering fine tuning.  Its CC-BY-NC-4.0 license
fits a student research project but not an unexamined commercial path.

Use a dedicated beat/downbeat model such as [Beat
This!](https://arxiv.org/abs/2407.21658) as a soft feature source, not an
infallible rule.  It makes musical phase observable while retaining model
freedom to place off-beat objects.  [MusicFM](https://github.com/minzwon/musicfm)
is also informative: its results favor longer context and Conformer-like local
modeling, while showing data quality can matter more than scale.

### Speech envelope, auditory features, and lyrics

Do not make a broadband speech envelope the core representation.  It is
mostly amplitude rhythm, and much of it is already implicit in high-rate mel
features.  Bandwise envelopes/modulation features are cheap density/intensity
auxiliaries worth ablation; they do not contain lyrics, pitch, instrument
identity, or dense-stream detail.

Auditory research supports treating temporal modulation and beat hierarchy as
important cues, not copying a neuroscience model wholesale.  One large study
found different dominant slow modulation regimes for speech and music (about
5 Hz versus 2 Hz): [Ding et al.](https://pubmed.ncbi.nlm.nih.gov/28212857/).
That is a reason to retain multi-band envelope/energy features, not proof that
a cochleagram will beat log-mel.  Vocal separation plus vocal activity and
syllable/onset confidence may help vocal-focused maps; ASR/lyrics should remain
optional because ordinary `.osu` files do not label note-to-syllable alignment
and singing alignment is unreliable at hit-object timing resolution.

[Hybrid Transformer Demucs](https://arxiv.org/abs/2211.08553) is a reasonable
later offline stem extractor.  Cache low-dimensional drum/vocal/bass/onset
summaries rather than four full spectrograms.  Do not run it alongside an
uninterruptible training job.

## Dataset and split policy

The current source is much larger than the cached first 100,000 examples:
[project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps)
reports 213,068 beatmaps, 46,386 deduplicated audio tracks, and 2,526 audio
hours.  It carries `osu-terms` / research-only constraints; do not redistribute
copyrighted music or assume commercial rights.

Before any larger run:

1. Hash the encoded audio before windowing and assign **all** maps/difficulties
   for that audio to one split.  Persist audio hash, beatmap ID, mapset ID,
   mapper ID, source, status, year, and map settings in metadata.
2. Prefer ranked maps, with approved/loved as explicit quality tiers rather
   than a silent mixture.  Retain status so quality filtering is auditable.
3. Deduplicate across sources by hash first, then consider an acoustic
   fingerprint for re-encodes.  Do not simply concatenate datasets.
4. Use [Osu2MIR](https://github.com/ziyunliu4444/osu2mir) for a beat/downbeat
   auxiliary task or timing evaluation, not as a replacement standard-map
   corpus.
5. Draw random local chunks only after split assignment, and retain a small
   full-song section index for global context.

## Evaluation that corresponds to the goal

Report free-running, constrained decoding on held-out **audio groups**, not
teacher-forced next-token accuracy.  For each sample, include:

- parser-valid map rate and validator failures;
- onset/beat phase error distributions, event precision/recall/F1, density
  calibration, and object type/duration/slider-field accuracy;
- requested versus realized star/strain, spacing, velocity, and overlap rates;
- candidate diversity and repetition collapse;
- bootstrap confidence intervals by song/mapset;
- blinded mapper or player review, because one song has many valid maps.

Reference-map agreement is useful for reconstruction but not sufficient for
map quality.  [ChartGenEval](https://arxiv.org/abs/2607.12857) makes this
distinction explicitly.  Use the [osu! ranking
criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria) as a validator reference;
AI output should be framed as playable/co-creative, not rankable by itself.

## Compute plan

Rutgers' current [GPU scheduler documentation](https://resources.cs.rutgers.edu/docs/scheduler-for-gpu-jobs/)
lists A4000/A4500/A5000/A6000, RTX Pro 5000 Blackwell, and A100 PCIe 40 GB
types; public docs do not promise each type to every student.  Your A100 belief
is plausible, but query the live scheduler (`sinfo`, `scontrol show nodes`) or
ask iLab staff before planning around it.  Keep the user's stated Blackwell
access restriction as authoritative.

Use iLab for reproducible preprocessing and training.  [Colab's official
FAQ](https://research.google.com/colaboratory/faq.html) says GPU type,
availability, and runtime limits vary dynamically.  It is suitable for a
notebook demo, inference, or small ablation—not a dependable multi-stage run
over a large corpus.

A sensible first v2 budget is a frozen MERT-v1 feature cache, a 75–150M
parameter planner on a single 40 GB A100, and a smaller separate geometry
refiner.  Only scale the encoder or use multi-GPU after song-disjoint rollout
metrics show a bottleneck.  Implement DDP correctly before using the current
four-GPU launcher.

## Recommended experiment order

1. Repair/rebuild data with audio-group splits, source metadata, and a retained
   held-out-song test set.
2. Establish a grammar-constrained, rollout-evaluated log-mel-only beat-grid
   baseline.
3. Add onset/beat/downbeat/phase channels.
4. Add frozen MERT layer mixing and bar/section context.
5. Add geometry diffusion and deterministic validation/reranking.
6. Ablate compact stem features, then CQT/chroma.
7. Consider lyrics, cochleagrams, MERT-v2/MuQ, or full-song diffusion only if
   prior ablations demonstrate a specific missing signal.

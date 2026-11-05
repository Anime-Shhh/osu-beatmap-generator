# osu-dreamer v8: source audit and actual Kaguya inference

The earlier `ARCHITECTURE_REVIEW.md` cited osu-dreamer at a high level. The
audio/native flow work did not reproduce it or compare against its released
weights. This check closes that gap. The released model produces substantially
better Kaguya note coverage than our current pipeline, but not uniformly better
timing, object types, doubles or difficulty control.

## What was actually run

The official [notebook](https://colab.research.google.com/drive/1Th6v5OOrY5vcTWvIH3NKZsuj_RMnAEM5)
pins v8.0 and downloads `v8.pt` from the author's Hugging Face repository. We used
that same release, not an independently trained approximation:

- Source: [v8.0](https://github.com/jaswon/osu-dreamer/tree/v8.0), commit
  `0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc`.
- Weight URL: `https://huggingface.co/jaswon/osu-dreamer/resolve/main/v8.pt`.
- Weight SHA256: `010b3d80cdce13550dccd5f3f6677cd5dab7df5fd98fdd5c9d3e8e5ba6c149b6`.
- Exact original Kaguya audio SHA256:
  `6ee5686d629635b982d5ccd97e4bcac5a7fa01969ab13c37587ef180b90bbe17`.
- Slurm job **1045509**, completed. Run:
  `/common/home/asj102/osu_flow_runs/dreamer_v8_20261105`.
- 16 seeds, 1731–1746, eight denoising steps (official notebook default).
- Requested `(SR, AR, OD, CS, HP)` = `(4.5, 8.8, 6.5, 4.0, 6.5)`, from our
  existing training-only settings guide. No original chart properties or events
  are supplied to the model. Upstream predicts output difficulty settings;
  those predictions are retained, rather than silently forcing requested values.
- Select in-bounds object heads first, then closest measured stars. Reference
  events are loaded only after generation/selection and never choose the map.
- Upstream source unchanged. Existing Torch 2.10.0+cu128 was used rather than
  replacing the shared environment with upstream's pinned Torch 2.11; all
  model weights loaded strictly and complete inference succeeded. Missing
  `resonators==0.1.1` was installed in an isolated dependency directory.

The published model card is empty and gives no complete training manifest or
training-step count. Kaguya's inclusion in its training is unknown. This is a
practical development comparison, not a controlled architecture ablation or an
unseen-song generalization claim. No new model training was performed in this
experiment; the full released pretrained approach was exercised directly.

## Why this approach is promising

The following are source-confirmed design differences. Their individual causal
contributions have not been isolated experimentally.

1. **Joint rhythm and movement representation.** Nine dense signals represent
   onset, combo, first-slide extent, total sustain, three hitsound flags, and
   cursor X/Y. A chart autoencoder learns a common representation of all of
   them. Repeated positions and short movements can be represented directly,
   whereas our fixed geometry stage receives an already-decided event plan.
   Source: [encoding](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/data/beatmap/encode.py)
   and [cursor signal](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/data/beatmap/cursor.py).

2. **Audio remains available at the final decoding stage.** A multiscale audio
   encoder provides skip connections to the chart decoder. The decoder can
   recover detailed hits from audio while expanding compressed chart latents.
   Latent noise and contiguous masking during training encourage this use of
   audio. Frames are approximately 5.98ms (98 samples at 16,384Hz), with a
   72-band resonator spectrogram, rather than our 10ms feature frames.
   Source: [latent model](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/models/latent/model.py),
   [training](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/models/latent/train.py)
   and [audio](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/data/load_audio.py).

3. **Compression makes broad context practical.** Three stride-three stages
   compress 27 signal frames into each six-dimensional temporal latent. The
   generator uses eight Transformer blocks, width512, with 16 attention heads
   of size64. Inference processes the whole song in one call. Our rhythm stage
   uses overlapping 20.48s windows and geometry uses separate event windows.
   A 32-dimensional style code is shared across the generated song, sampled
   conditionally on the five difficulty attributes. Reconstruction swaps style
   codes between paired crop halves to encourage consistency.
   Source: [inference](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/models/inference/model.py)
   and [backbone](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/models/diffusion/backbone.py).

4. **Movement has direct supervision.** Training penalizes cursor position,
   velocity and acceleration errors, alongside hit signals and difficulty
   labels. That supplies a richer movement objective than evaluating only
   independent head/control-point coordinates. It does not guarantee stacks.

5. **The exporter contains empirical priors.** It fits curves to
   generated cursor traces and favors common simple slider shapes using empirical
   type frequencies. These heuristics contribute to the final appearance; it
   is not entirely an unconstrained neural output. Source:
   [slider decoder](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/data/beatmap/slider/decode_slider.py).

6. **Much greater capacity and pretrained weights.** The loaded model has
   **65,233,044 parameters**, about 11.3 times our combined phrase-rhythm and
   geometry models. The current data loader filters ranked standard maps from
   `project-riz/osu-beatmaps` and splits by audio group. Neither that source
   code nor the weight file proves the exact corpus size used for these weights;
   do not attribute the result to a guessed training-data count.

V8's generator uses a learned distance/direction field and iterative distance
marching; it is not identical to our time-conditioned flow-matching objective.
The representation, audio decoder, joint conditioning and pretrained capacity
are at least as important to test as the sampler name.

## Measured Kaguya results

All means below are across the same 16 seed identifiers. The original human
reference is Confession?, 4.4874 stars, 324 objects. F1 is onset matching, not
overall musical or ranked-map quality.

| Model | F1 ±50ms | F1 ±10ms | Mean objects | Reference short circle pairs recovered /36 |
| --- | ---: | ---: | ---: | ---: |
| Our original model | 70.60% | 47.22% | 249.81 | 5.88 |
| Our more-data control | 71.47% | 58.35% | 257.38 | 6.19 |
| Our more-data + phrase model | 70.73% | 58.54% | 254.06 | 6.19 |
| Published osu-dreamer v8 | **80.54%** | 55.35% | 267.50 | 4.63 |

V8 improves ±50ms onset F1 by **9.81 percentage points** over our latest model.
It is worse at the tighter ±10ms tolerance and does not improve mean recovery
of the reference's short circle pairs. This is evidence of better broad rhythm
selection/coverage on this example, not proof of superior instrument selection.

The selected v8 map (seed1733) measures **4.0323 stars** and has **317 objects**.
It matches **276/324 starts** within 50ms (85.2% recall; F1 86.1%), compared with
185 matches in our selected phrase map. Matched starts average 11.51ms absolute
error and +11.07ms signed error. No offset correction against the reference was
applied. The map has 245 circles and 72 sliders, versus 85 circles and 239 sliders in the
original: higher onset similarity does not reproduce its object-type style.

It recovers 10/36 reference short circle pairs and contains one close-position
pair within 8px, versus 13 in the original and zero in our selected map. Across
v8 seeds the count of close-position pairs ranges 0–8. The selected result alone
must not be used to claim consistently improved doubles.

Only 1/16 outputs falls within the requested 4.5±.5 star tolerance; mean stars
are 3.46. Two candidates have out-of-bounds object heads and were not selected.
The selected archive passes ZIP CRC checks and includes the exact original MP3.
It has not passed a complete ranking-criteria or human playability review.

## Important limitation in upstream export

[decode.py](https://github.com/jaswon/osu-dreamer/blob/v8.0/osu_dreamer/data/beatmap/decode.py)
still has `TODO: compute timing points`. Its base beat length is chosen from
slider velocities, then inherited timing points recover slider durations. That
is not a recovered musical tempo/offset. Convincing gameplay patterns therefore
do not imply a correctly timed, ranked-ready `.osu`. A future integration needs
an independently validated audio clock and native slider semantics; copying the
original's timing points would invalidate an MP3-only demonstration.

## Decision

Keep the pinned, pretrained v8 pipeline as a concrete baseline and starting
point for this direction, rather than adding more attention to our weak
separated onset model. Any adaptation should preserve its joint chart/cursor
representation, multiscale audio decoder and song-wide style, while explicitly
testing short patterns, object type/duration, actual stars and musical timing.
Before claiming causal gains from retraining, use known source-disjoint data,
exclude Kaguya from training, and compare against unchanged pretrained v8. A new
small model trained from scratch is not an equivalent test of this release.

Reproduction: `scripts/evaluate_dreamer_v8.py` and
`configs/dreamer_v8_inference.sbatch`. Source, dependencies, weights and results
are isolated under the run path above; historical runs are preserved. Evidence:
[full candidate metrics](evidence/osu-dreamer-v8-2026-11-05/comparison.json).
Local playtest:
`/Users/anime_shh/personalCS/projects/codex-mds/osu/2026-11-05-dreamer-v8/kaguya/Kaguya-osu-dreamer-v8.osz`.

# Kaguya upstream-model comparison

Verified November 9, 2026. Training and evaluation finished November 7; all seven jobs completed with exit code 0. Both cloned models were fine-tuned for 2,000 updates from published weights. These were bounded adaptation runs, not full pretraining. No jobs remain running from this comparison.

## What the results mean

Fine-tuning dreamer improved onset F1 at 25 ms by 1.88 percentage points (80.73% → 82.61%) and at 10 ms by 5.79 points. It did not solve short patterns: pair recovery fell from 27.08% to 26.39%. The selected fine-tuned map is 4.24 stars, outside the explicit ±0.2-star tolerance around the requested 4.6.

OliBomby/osu-diffusion generates geometry from supplied timestamps, object types and spacing. It has no audio input or direct star-rating control. Its 100% timing score with human rhythm is inherited from that input and cannot be compared as an audio-to-rhythm achievement. That version measures 4.58 stars. Using dreamer rhythm instead yields a fully generated chain, but the selected output measures 4.34 stars and retains geometry/export findings. Differences of fractions of a point in timing after the geometry stage reflect export rounding, not better musical understanding.

Our previous AE and VAE adaptations were re-evaluated using the same original audio, reference, four seeds and 4.6-star request. The new dreamer fine-tune has nearly the same 25 ms score as our AE (82.61% vs 82.40%), but lower short-pair recovery (26.39% vs 30.56%). Our VAE recovers more pairs here, but this one development song does not overturn its failed broader validation gate. None of these results establishes ranked-ready quality or justifies promoting a new model.

## Matched metrics

All scores below aggregate seeds 1731–1734 on Kaguya only. Higher F1/pair recovery is better; lower star error is better. Selected-star values describe one retained candidate, chosen solely by closeness to 4.6; other columns aggregate all four candidates.

| Model / conditioning | F1 10 ms | F1 25 ms | F1 50 ms | Short pairs | Mean star error | Selected stars |
|---|---:|---:|---:|---:|---:|---:|
| Published dreamer | 52.08% | 80.73% | 84.69% | 27.08% | 0.755 | 4.18 |
| Published osu-diffusion + human rhythm | 100.00% | 100.00% | 100.00% | 100.00% | 0.048 | 4.58 |
| Published dreamer → osu-diffusion | 52.55% | 81.12% | 84.61% | 27.08% | 0.787 | 4.25 |
| Fine-tuned dreamer | 57.87% | 82.61% | 85.06% | 26.39% | 0.698 | 4.24 |
| Fine-tuned osu-diffusion + human rhythm | 100.00% | 100.00% | 100.00% | 100.00% | 0.048 | 4.58 |
| Fine-tuned dreamer → osu-diffusion | 58.18% | 82.77% | 84.98% | 26.39% | 0.709 | 4.34 |
| Our previous AE adaptation | 58.37% | 82.40% | 85.62% | 30.56% | 0.653 | — |
| Our previous VAE adaptation | 57.66% | 81.65% | 85.15% | 33.33% | 0.641 | — |

## Timing misses, object selection and export

| Model / conditioning | Precision 25 ms | Recall 25 ms | Matched timing MAE | Type F1 25 ms | Slider duration median error | Strict invalid exports |
|---|---:|---:|---:|---:|---:|---:|
| Published dreamer | 82.86% | 78.70% | 8.88 ms | 48.22% | 97.91 ms | 0% |
| Published osu-diffusion + human rhythm | 100.00% | 100.00% | 0.37 ms | 100.00% | 0.69 ms | 100% |
| Published dreamer → osu-diffusion | 83.27% | 79.09% | 8.82 ms | 48.39% | 97.91 ms | 100% |
| Fine-tuned dreamer | 84.68% | 80.63% | 8.11 ms | 48.24% | 68.50 ms | 0% |
| Fine-tuned osu-diffusion + human rhythm | 100.00% | 100.00% | 0.37 ms | 100.00% | 0.69 ms | 100% |
| Fine-tuned dreamer → osu-diffusion | 84.85% | 80.79% | 8.02 ms | 48.33% | 68.50 ms | 100% |
| Our previous AE adaptation | 83.92% | 80.94% | 8.06 ms | 44.56% | 63.00 ms | 0% |
| Our previous VAE adaptation | 84.12% | 79.32% | 8.10 ms | 42.43% | 100.30 ms | 0% |

Timing MAE covers matched notes only: low error does not erase missed notes. Short pairs mean consecutive reference circles with a positive gap no greater than 0.51 beats, recovered as circles within 50 ms; this is not a full measure of stacked or repeated-position pattern quality. Type F1 additionally requires the correct circle/slider/spinner classification. The reference has 324 objects and 36 short-circle pairs.

Every geometry output triggered the strict validator. Findings include incomplete slider edge-sound/edge-set lists and rendered slider paths outside the playfield; some seeds also have out-of-bounds object coordinates, spinner-center violations or slider-occupancy overlaps. They remain in the denominator. These are raw upstream exports: no hidden snapping, offset correction, geometry repair or score filtering was applied. A strict validation failure is not proof that the osu! client will refuse to open the map. Conversely, dreamer passing this validator is not proof of ranked quality.

The human reference is “Confession?” and measures 4.4874 stars in pinned rosu-pp-py 4.0.2 (stable/no mods). The requested value remains 4.6. Client/difficulty-engine versions can produce different star ratings. Published training membership is unknown; Kaguya was excluded from our fine-tuning, but cannot be claimed unseen by upstream pretraining. Four seeds on one development song do not provide a source-level generalization test.

## Training and reproducibility

- Dreamer: 65,233,044 parameters; original codec and style sampler frozen, original RMS temporal latents and song-level style retained; distance/direction objective, 152-latent crops, EMA 0.99. 2,757 ranked charts from 633 training sources. No VAE and no KL term in this new upstream fine-tune.
- osu-diffusion: DiT-B, 170,370,054 checkpoint parameters, including 40,451,328 class-embedding parameters. Generic/null-style fine-tune with original 128-element geometry representation and diffusion objective; class embeddings frozen because upstream class IDs do not map to our corpus. 2,725 charts from 632 sources; 32 inputs were ineligible or failed conversion, retained in provenance records.
- Both: effective batch 32, microbatch 8, AdamW learning rate 3e-5, BF16, gradient clipping 1, fixed final update 2,000. Source-separated ranked training data only; no generated-map targets and no Kaguya training.
- Dreamer training job: 10m12s; geometry: 9m22s, run concurrently on separate RTX 4500 Ada GPUs on rlab1. Including both smoke tests and all inference/re-evaluation jobs: 2,040 allocated GPU-seconds, approximately 0.57 GPU-hours.
- Inference: original Kaguya audio, official eight-step dreamer sampler; 250-step cosine geometry sampler with null style/CFG 1 and band radius 128. Wrappers call upstream modules; this is not a claim of using untouched upstream CLIs.
- Checkpoint SHA-256, frozen-data binding, source revision, hyperparameters, conversion exclusions and histories are preserved in `training-evidence/`. Per-seed metrics are in each `comparison.json`. `packaging-audit.json` gives the final downloadable archive hashes; earlier `comparison.json` hashes describe archives before metadata-only relabeling.
- All six local archives passed ZIP integrity, final SHA-256 and original-audio SHA-256 checks. Packaging changed display metadata only and verified identical difficulty, timing and hit-object sections. No historical data or models were deleted for this comparison.

## Files and downloads

Selected fine-tuned maps are already in this folder:

- `finetuned/dreamer.osz`: independent audio-conditioned generation, 4.24 stars.
- `finetuned/diffusion-human-rhythm.osz`: osu-diffusion geometry with reference rhythm/types/spacing, 4.58 stars.
- `finetuned/diffusion-dreamer-rhythm.osz`: generated dreamer rhythm followed by osu-diffusion geometry, 4.34 stars.

The `pretrained/` folder contains the corresponding three published-weight baselines. These are experimental comparisons, not ranked-ready deliveries.

To fetch the fine-tuned versions again from your Mac terminal:

```bash
mkdir -p ~/Downloads/osu-kaguya-comparison
scp -F /Users/anime_shh/personalCS/projects/ilab-access/ssh-config rutgers-ilab:/common/home/asj102/osu_flow_runs/upstream_comparison_20261107/finetuned/dreamer.osz ~/Downloads/osu-kaguya-comparison/
scp -F /Users/anime_shh/personalCS/projects/ilab-access/ssh-config rutgers-ilab:/common/home/asj102/osu_flow_runs/upstream_comparison_20261107/finetuned/diffusion-human-rhythm.osz ~/Downloads/osu-kaguya-comparison/
scp -F /Users/anime_shh/personalCS/projects/ilab-access/ssh-config rutgers-ilab:/common/home/asj102/osu_flow_runs/upstream_comparison_20261107/finetuned/diffusion-dreamer-rhythm.osz ~/Downloads/osu-kaguya-comparison/
```

If the SSH session expires, run `/Users/anime_shh/personalCS/projects/ilab-access/ilab-connect` first. For published-weight baselines, replace `/finetuned/` with `/pretrained/` and use a separate local directory to avoid overwriting files.

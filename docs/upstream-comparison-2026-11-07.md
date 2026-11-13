# Upstream Kaguya comparison — 2026-11-07

Status: complete, verified November 9. Both 2,000-update training runs and all evaluations succeeded. See `upstream-comparison-results-2026-11-09.md` for matched results and delivered archives.

The user requested cloned osu-dreamer and OliBomby/osu-diffusion, training and Kaguya inference, a 4.6-star request where supported, matched metrics and downloadable OSZ archives.

## Fixed sources and interpretation

- osu-dreamer v8.0: `0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc`, published 65,233,044-parameter weights from the existing verified v8 baseline.
- OliBomby/osu-diffusion: `532dc5d3026d23bf1578e7b1ba755ee4471b8d78`; published checkpoint Google Drive ID `1oX8SPNnyswhaI8euWRkJ10tmncxaGsBG`. EMA extracted with restricted Torch deserialization allowing argparse.Namespace. The DiT-B state has 170,370,054 values including 40,451,328 beatmap-class embedding parameters.
- Geometry-only osu-diffusion receives existing object times, types and spacing. It has no audio input or star-rating conditioning. Evaluate human-rhythm and independently dreamer-generated-rhythm versions separately. Human-rhythm onset scores are inherited conditioning, not learned music alignment.
- Kaguya original archive/audio, nearest reference to 4.6 stars according to pinned rosu-pp-py 4.0.2. Four seeds: 1731–1734. Report actual reference stars and measured output stars separately from requested stars.
- Dreamer uses official eight-step distance marching. Geometry uses 250-step cosine diffusion, null class, CFG 1 and band radius 128. Adapter code invokes upstream models and exporters; this is not a claim of running the untouched CLI.

## Training

Bounded fine-tuning from published weights, not training from scratch. Maximum 2,000 updates or 5,400 seconds per arm, including feature preparation; effective batch 32, microbatch 8, BF16, learning rate 3e-5, clipping 1. Fixed final checkpoint; Kaguya is not used for checkpoint selection. No new model promotion is implied.

Dreamer freezes the original codec and style sampler. Full-song real-map temporal latents and style are prepared in RAM with the original RMS normalization; random 152-latent crops train the original distance/direction objective, weights 1 and 30, EMA .99, AdamW weight decay .01. No VAE or KL term is added.

Geometry uses ranked native maps, original 128-element sequence crops and coordinate-flip/time-offset augmentation, upstream diffusion training loss, EMA .9999, AdamW weight decay zero. Generic null-style conditioning avoids inventing correspondences between the published beatmap-class IDs and our corpus; the original embedding table is frozen. This is a limited generic-geometry adaptation.

Both use the existing source-separated ranked corpus. Kaguya is excluded; validation/test sources are excluded from training. Source/hash checks precede training. New latent/sequence caches are RAM-only. Home free-space gate remains 8 GiB. New training checkpoints use the recovered project allocation; historical artifacts are preserved.

## Runs

Remote root: `/common/home/asj102/osu_flow_runs/upstream_comparison_20261107`.
Training outputs: `/common/users/asj102/osu_project/upstream_comparison_20261107`.

- 1087294: pretrained Kaguya comparison. Initially requested A100, pending request changed to RTX 4500 Ada because all A100s were occupied.
- 1087307 / 1087308: two-step real-data Dreamer/geometry smoke tests.
- 1087334 / 1087335: Dreamer/geometry training, gated on successful respective smoke tests.
- 1087336: trained-model Kaguya inference, gated on both training jobs completing successfully.
- 1087337: retained adapted AE/VAE re-evaluation on the same four-seed 4.6-star Kaguya request; labels pretrained packages without changing gameplay.

The account has a one-node allocation limit. After initial inference on ilab3, subsequent jobs request rlab1 so multiple GPUs can run independent arms on one node.

Metrics retain failed exports and report onset P/R/F1 at 10/25/50ms, matched timing error, object/type counts, short-circle-pair recovery, slider duration errors, spatial patterns and measured stars. Strict native validity includes incomplete slider edge metadata and rendered slider bounds; a validation failure does not necessarily mean the osu! client refuses the file. Raw upstream geometry export findings are retained, not silently repaired.

At most one archive per variant is retained, selected by closest measured difficulty to 4.6, not by reference-matching score. All four seeds remain in aggregate metrics. ±0.2 stars is the comparison's explicit target tolerance; an outside result must not be described as meeting 4.6.

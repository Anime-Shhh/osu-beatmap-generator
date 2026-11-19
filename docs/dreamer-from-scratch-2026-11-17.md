# Dreamer reproduction from random initialization

The user clarified that the project objective is to train a model from scratch, not adapt pretrained weights. Previous Dreamer runs were pretrained adaptations and do not satisfy this objective. This run reuses the credited, pinned v8 architecture and upstream training objectives, but loads no published model weights or pretrained audio encoder. It is a from-scratch reproduction, not a claim of an original architecture.

Implementation: `src/dreamer_scratch/train.py` and `scripts/amarel/train_scratch.sbatch`. Every trainable component starts from its constructor's random initialization. Unit tests intercept `torch.load` during construction and verify reproducibility for a fixed seed, different initialization for another seed, and rejection of foreign checkpoints. Checkpoints bind to the source code, pinned upstream commit, configuration YAML hashes, feature/split manifest, seed, and stage. Generator/style checkpoints additionally bind to the newly trained codec hash. Existing published and fine-tuned models remain baselines only.

Data: verified native chart and v8 resonator-feature cache from the existing ranked corpus, with source-separated assignments, 24 frozen validation sources and locked test sources. Kaguya is excluded from training. Raw signal features are deterministic DSP outputs, not pretrained learned embeddings. All temporal/audio/style latent caches are newly produced by the scratch-trained codec. Train sampling is source-balanced. Validation does not optimize weights, test data are not loaded by this trainer, and padding is aligned to both half-window swapping and 27-frame compression.

Training stages:

1. Codec: jointly train audio encoder, chart encoder, temporal/style projections, chart decoder, and label predictor. Use upstream hit/cursor position/velocity/acceleration/label losses, style MMD regularization, cross-half style swapping, noise, and temporal masking. This is not a VAE and adds no KL term. Upstream learning rate 1e-3, effective batch 32, 2,052-frame crops, 2,000-update warmup. Up to 60,000 updates, with validation-score early stopping after ten unimproved epochs (minimum gain .001).
2. Encode: use the selected scratch codec to create separately bound temporal/audio/style representations for training and validation. Maximum new cache size 4 GiB; fail explicitly if exceeded. No old pretrained latent cache is accepted.
3. Temporal generator: random initialization, upstream distance/direction objective, learning rate 3e-4, effective batch 128, 152-latent crops, 1,000-update warmup, inverse-square-root decay after update 30,000, EMA .99. Maximum 60,000 updates. Select by validation loss as an initial training diagnostic; later full-song generation remains necessary for any quality claim.
4. Style generator: random initialization, upstream conditional distance/direction objective with label dropout .2, learning rate 3e-4, effective batch 512, EMA .99. Maximum 20,000 updates; select by validation energy distance.
5. Export: assemble only checkpoints from this scratch run. Verify each stage's weights changed and that the resulting artifact generates finite chart signals. This smoke verification is not musical-quality acceptance.

Training stages use BF16, clipping 1, preserved effective batches across supported GPU counts, and 48-hour Slurm allocations. An internal 47-hour timer saves resumable state; an incomplete stage fails rather than silently letting dependent stages use unfinished training. Keep one best and one last checkpoint per stage. At the November 17 resource check, two GPUs forecast an earlier start than four. Full training must depend on a successful real-data GPU smoke covering all stages and export. Smoke checkpoints are isolated and never initialize the full run.

The first import checks found missing TensorBoard and Matplotlib in the inference-focused Amarel environment. Installation temporarily upgraded NumPy; NumPy was restored to 1.26.4 with compatible plotting dependencies, and `pip check` passed. Torch was unchanged. Three initialization/provenance unit tests passed on Amarel. No claim of pretrained-level quality is made before evaluation.

## Submission interruption and November 18 continuation

The last confirmed GPU smoke submission is job 64002714, using immutable source `0125250` and run directory `/scratch/asj102/osu_project/runs/scratch-smoke-0125250`. Its latest observed state on November 17 was PENDING. The later attempted deployment of `dfc38f5` and dependent full training chain was **not executed**: automatic approval review encountered a usage-limit failure. Consequently there are no verified full scratch-training job IDs. Do not report the intended chain as submitted.

November 18 SSH attempts initially failed DNS lookup and then reached Amarel but failed authentication; the user was asked to reconnect. Local review added a fresh per-allocation timer budget on resume, explicit final optimizer-state checkpointing, rejection of nonfinite training losses, and explicit trusted full-state restoration after provenance checks. Syntax and diff checks pass locally; the two new regression tests and the updated end-to-end GPU smoke remain to be run on Amarel. Code changes require a new immutable snapshot and smoke directory because checkpoint bindings include the training program's hash.

Resume API reference: https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.trainer.trainer.Trainer . Full-state resume is necessary for our dataclass hyperparameters and optimizer/callback state; it does not authorize loading foreign checkpoints.

## November 18 verified submissions

SSH access was restored through the default `amarel` ControlMaster (the explicit `/tmp` socket was absent). Prior smoke 64002714 is confirmed COMPLETED in 10:24 with exit 0: two optimizer updates in each of codec/generator/style changed the initial weights, and the assembled artifact generated finite 2,052-frame signals. This validates plumbing, not map quality. Full training does not reuse those smoke weights.

Updated immutable source `31b2ec2` at `/scratch/asj102/osu_project/sources/scratch-31b2ec2` includes allocation-resume and distributed metric fixes. All six scratch unit tests passed on Amarel, and `pip check` found no broken requirements. The pipeline was actually submitted successfully this time, with job IDs persisted incrementally in `runs/dreamer-scratch-20261118/jobs.json`:

| Stage | Job | Dependency | GPUs | Limit |
|---|---:|---|---:|---|
| Updated real-data smoke | 64030937 | none | 2 | 20 minutes |
| Random-initialized codec | 64030938 | smoke success | 2 | 48 hours |
| New latent cache | 64030939 | codec success | 1 | 2 hours |
| Random-initialized temporal generator | 64030940 | cache success | 2 | 48 hours |
| Random-initialized style generator | 64030941 | cache success | 1 | 48 hours |
| Export and finite-inference check | 64030942 | both generators succeed | 1 | 20 minutes |
| Frozen validation + wrong audio | 64030943 | export success | 1 | 2 hours |

Full run root: `/scratch/asj102/osu_project/runs/dreamer-scratch-20261118`. Updated smoke root: `/scratch/asj102/osu_project/runs/scratch-smoke-31b2ec2`. Logs: `/scratch/asj102/osu_project/logs/scratch-JOBID.log`. At the first post-submission check, smoke was RUNNING on gpuk006 and all other stages were dependency-pending. Actual codec allocation was checked as 2 GPUs and `TimeLimit=2-00:00:00`. Non-submitting forecasts favored 2 GPUs (November 18 16:36) over 4 (23:43); no speculative duplicate full runs were submitted. Source-separated datasets and prior checkpoint baselines remain unchanged.

## November 19 completion audit

All six full-run stages completed successfully (exit 0), with final validation ending November 18 at 21:33:29 Eastern. Codec training ran 2:28:25 and early-stopped at 20,000 updates; best reconstruction score was at update 10,000. Generator ran 60,000 updates in 2:20:38, but best validation loss was at update 3,750; final training/validation losses were about 1.03/37.97, compared with best validation loss 11.69. This divergence is consistent with substantial overfitting and warrants investigation. Style completed 20,000 updates in 33:07; best validation energy distance was at update 2,772. Export used the retained best checkpoint from every stage, not the final weights. No jobs remain running from this pipeline.

The final artifact `/scratch/asj102/osu_project/runs/dreamer-scratch-20261118/scratch-inference.pt` was SHA-256 verified as `c4e6daa4b85b79bf03ce56f75611a51e79cd7a184d5bc7d9a80dd553a8c2afac`. Export provenance confirms `pretrained_weights=false`, and each learned stage changed from its recorded random initialization.

Independent full-song validation used 24 sources, four seeds each, with aligned and duration-matched wrong-song audio; locked test data were not used. Aligned onset F1 at 10/25/50 ms is 48.20%/60.15%/61.74%; typed onset F1 at 25 ms 35.61%; short-pair recovery 17.77%; mean absolute star error .345; median matched-slider duration error 78.58 ms. Wrong-song onset F1 at 25 ms is 15.27%, indicating substantial dependence on supplied audio (no paired significance test performed in this status check).

Critical limitation: **87/96 aligned outputs (90.625%) fail native validation**, all with at least one rendered body outside the playfield; five also report a degenerate slider without travel direction. Other errors include out-of-bounds event coordinates. Mean star error and onset metrics include invalid outputs and do not establish playable quality. This model is not promoted as a usable replacement for pretrained Dreamer. Do not compare its 24-source metrics directly with the earlier three-source Dreamer audit. No repair, clipping, re-training or new test evaluation was performed during this status check.

Compact verified evidence: `docs/evidence/dreamer-scratch-20261119/status-summary.json`. Full per-seed metrics remain under the run's `validation/metrics.json` and `validation/records.jsonl` on Amarel.

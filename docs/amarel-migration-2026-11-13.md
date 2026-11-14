# Amarel migration and generative-architecture direction

The user selected Amarel as the compute platform and a pivot toward a joint osu-dreamer-style generative model. Retain historical autoregressive code as evidence, but do not launch new autoregressive training.

Initial code push: `e3f4332095905bccf7725aff8364d21988e719ab`, branch `codex/audio-native-flow`, repository https://github.com/Anime-Shhh/osu-beatmap-generator. Author and committer are Animesh Jadhav, animeshjadhav05@gmail.com, with no co-author trailers.

Amarel clone: `/scratch/asj102/osu_project/repo`. The user authorized copying existing artifacts when simpler than retraining. Preserve iLab originals. Imported data maintains its original `common/...` directory hierarchy under `/scratch/asj102/osu_project/imported`.

`OSU_ARTIFACT_ROOT` resolves historical iLab paths without changing manifests, hashes, codec bindings or source splits. Missing copies fail rather than falling back to old paths. `OSU_STORAGE_ROOT` explicitly selects Amarel storage: its gate checks filesystem free space, not a personal quota. The original iLab home-quota check remains the default without this variable. Never represent filesystem free space as a quota measurement.

An isolated Python3.12 environment uses Torch2.10.0+cu128 and the previously exercised audio pipeline versions. FFmpeg shared libraries are supplied in a separate prefix. No other project environments are modified. Pinned upstream v8 remains commit0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc. Environment records are in the project logs directory.

Run GPU work through Slurm. Scripts in `scripts/amarel/` use partition `gpu`, account `general`, one Ada GPU, four CPUs and32GiB memory. Source `environment.sh` for path setup. Submit `smoke.sbatch` from the repository, directing output to the project logs directory. Only submit `train_dreamer.sbatch` after smoke succeeds, optionally with an `afterok` dependency. That script repeats the bounded2,000-update joint generator fine-tune from published weights; it does not resume an old optimizer, retrain an autoregressive model or imply a quality improvement. Outputs use job-specific directories and preserve earlier weights.

The first repeat is a migration/reproducibility check. More GPUs alone do not specify a new architecture or authorize indefinite training. Retain original validation/test assignments, Kaguya exclusion, and model-promotion criteria when designing the next experiment.

Cluster reference: https://user-guide-f4258a.gitlab.io/jobs/ and the live `scontrol show partition gpu`/account queries. Scratch stores active copies; the preserved iLab data and GitHub source remain separate recovery copies.

## Transfer and first repeat

Copied approximately 2.7 GB: the compact ranked feature cache, corresponding native maps, published v8 weights, retained AE/VAE codec and generator weights, the previous upstream fine-tune, and original Kaguya archive. All six transferred checkpoint SHA-256 hashes match their iLab sources. The remote ledger is `logs/checkpoint-transfer.json`. Data verification uses the unchanged manifest and verifies its referenced feature/native hashes before training.

The first GPU smoke job, 63885024, passed all 18 compatibility tests but stopped before training because `pytorch_lightning` was missing. The isolated environment was repaired; upstream inference imports and `pip check` then passed. Numerical pipeline versions remain pinned, including NumPy 1.26.4. See `amarel-environment-2026-11-13.txt` for the installed Python package versions; FFmpeg 7.1.1 is supplied separately.

Replacement smoke job: 63885266. Bounded training repeat: 63885268, with `afterok:63885266`. Kaguya evaluation: 63885269, with `afterok:63885268`. Training and inference use the immutable source worktree `sources/a03180c`. Their submission does not establish successful completion or a model-quality gain. Consult the job logs and completion records before using the outputs.

Smoke 63885266 completed successfully in 3 minutes 2 seconds on an NVIDIA L40S. All 18 tests passed, all manifest artifacts verified, and two real training updates completed with finite loss and saved checkpoints. Torch reported CUDA 12.8 and BF16 support. The full bounded repeat subsequently started on the same GPU model.

## Completed results, checked November 14

All three replacement jobs completed with exit code 0. Training 63885268 finished 2,000 updates in 10 minutes 18 seconds wall time; evaluation 63885269 finished in 34 seconds. Training used 2,757 charts from 633 training sources, with no conversion failures and Kaguya excluded. The ordinary upstream codec remained frozen; this repeat introduced no VAE or KL loss. Outputs are `runs/dreamer-63885268/{final,last}.pt` and `runs/kaguya-63885269/`.

Four fixed seeds, original audio, requested difficulty 4.6, eight sampling steps:

| Metric | Published v8 on Amarel | Amarel fine-tune |
|---|---:|---:|
| Onset F1, 10 ms | 52.00% | 57.65% |
| Onset F1, 25 ms | 80.81% | 82.48% |
| Onset F1, 50 ms | 84.69% | 85.01% |
| Onset-and-type F1, 25 ms | 48.22% | 48.51% |
| Short-circle-pair recovery | 27.08% | 26.39% |
| Median matched-slider duration error | 97.91 ms | 68.91 ms |
| Mean absolute requested-star error | 0.755 | 0.698 |
| Invalid exports in these four seeds | 0/4 | 0/4 |

At 25 ms, fine-tune precision is 84.59% and recall is 80.48%: roughly 19.5% of reference onsets are still missed at this tolerance. Onset improvement does not establish sound selection or human mapping quality. Type correspondence barely improved and short-circle-pair recovery declined. The selected seed 1732 maps measure 4.178 stars (published) and 4.241 stars (fine-tune); both fail the declared ±0.2-star target tolerance. The repeat's 1.68-point F1 gain on this one development song does not meet the +3-point multi-source promotion gate. No new model is promoted, and upstream pretraining membership remains unknown.

Full per-seed evidence is in `docs/evidence/amarel-20261114/`. Both selected `.osu` and `.osz` files are copied to the Mac's `osu-beatmap-generator/codex files/amarel-kaguya-20261114/`; archive hashes and original audio hashes were verified after transfer. Slurm reports no remaining jobs from this sequence. Small differences from the earlier iLab metrics mean this is a functional repeat, not a claim of bitwise reproducibility across runtimes/hardware.

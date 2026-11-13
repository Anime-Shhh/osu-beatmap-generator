# Amarel migration and generative-architecture direction

The user selected Amarel as the compute platform and a pivot toward a joint osu-dreamer-style generative model. Retain historical autoregressive code as evidence, but do not launch new autoregressive training.

Initial code push: `e3f4332095905bccf7725aff8364d21988e719ab`, branch `codex/audio-native-flow`, repository https://github.com/Anime-Shhh/osu-beatmap-generator. Author and committer are Animesh Jadhav, animeshjadhav05@gmail.com, with no co-author trailers.

Amarel clone: `/scratch/asj102/osu_project/repo`. The user authorized copying existing artifacts when simpler than retraining. Preserve iLab originals. Imported data maintains its original `common/...` directory hierarchy under `/scratch/asj102/osu_project/imported`.

`OSU_ARTIFACT_ROOT` resolves historical iLab paths without changing manifests, hashes, codec bindings or source splits. Missing copies fail rather than falling back to old paths. `OSU_STORAGE_ROOT` explicitly selects Amarel storage: its gate checks filesystem free space, not a personal quota. The original iLab home-quota check remains the default without this variable. Never represent filesystem free space as a quota measurement.

An isolated Python3.12 environment uses Torch2.10.0+cu128 and the previously exercised audio pipeline versions. FFmpeg shared libraries are supplied in a separate prefix. No other project environments are modified. Pinned upstream v8 remains commit0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc. Environment records are in the project logs directory.

Run GPU work through Slurm. Scripts in `scripts/amarel/` use partition `gpu`, account `general`, one Ada GPU, four CPUs and32GiB memory. Source `environment.sh` for path setup. Submit `smoke.sbatch` from the repository, directing output to the project logs directory. Only submit `train_dreamer.sbatch` after smoke succeeds, optionally with an `afterok` dependency. That script repeats the bounded2,000-update joint generator fine-tune from published weights; it does not resume an old optimizer, retrain an autoregressive model or imply a quality improvement. Outputs use job-specific directories and preserve earlier weights.

The first repeat is a migration/reproducibility check. More GPUs alone do not specify a new architecture or authorize indefinite training. Retain original validation/test assignments, Kaguya exclusion, and model-promotion criteria when designing the next experiment.

Cluster reference: https://user-guide-f4258a.gitlab.io/jobs/ and the live `scontrol show partition gpu`/account queries. Scratch stores active copies; the preserved iLab data and GitHub source remain separate recovery copies.

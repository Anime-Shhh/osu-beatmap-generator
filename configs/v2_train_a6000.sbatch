#!/usr/bin/env bash
# Tempo-Posterior Hierarchical Event Diffusion — explicit iLab A6000 launcher.
# This file is inert until the user submits it with sbatch and sets every
# required variable below. It never submits itself.
#SBATCH --job-name=osu-v2-planner
#SBATCH --output=osu-v2-%x-%j.out
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the osu-beatmap-generator checkout}"
: "${RECORDS_DIR:?Set RECORDS_DIR to v2 .pt song records}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the immutable source split JSON}"
: "${FEATURE_PIPELINE_KEY:?Set FEATURE_PIPELINE_KEY from FeatureCacheManifest.pipeline_key}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a durable checkpoint directory}"

# Refuse accidental allocation/runs if this template is invoked prematurely.
if [[ "${CONFIRM_TRAINING:-}" != "1" ]]; then
  echo "Set CONFIRM_TRAINING=1 only when you intend to start a v2 training run." >&2
  exit 64
fi

cd "${PROJECT_DIR}"

# Set VENV_ACTIVATE to a trusted environment activation script if needed, e.g.
# export VENV_ACTIVATE=/common/.../venv/bin/activate
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

V2_STAGE_VALUE="${V2_STAGE:-planner}"
TRAIN_ARGS=(
  --records-dir "${RECORDS_DIR}"
  --split-manifest "${SPLIT_MANIFEST}"
  --feature-pipeline-key "${FEATURE_PIPELINE_KEY}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --stage "${V2_STAGE_VALUE}"
  --epochs "${V2_EPOCHS:-1}"
  --batch-size "${V2_BATCH_SIZE:-1}"
  --gradient-accumulation "${V2_GRADIENT_ACCUMULATION:-8}"
  --num-workers "${V2_NUM_WORKERS:-4}"
  --seed "${V2_SEED:-1337}"
  --device cuda
  --mixed-precision bf16
  --confirm-training
)

# Geometry must consume a trained compatible planner; the Python entry point
# independently enforces this, while the template gives a clearer preflight.
if [[ "${V2_STAGE_VALUE}" == "geometry" ]]; then
  : "${V2_RESUME:?Set V2_RESUME to a compatible trained planner checkpoint for geometry stage}"
  TRAIN_ARGS+=(--resume "${V2_RESUME}")
elif [[ -n "${V2_RESUME:-}" ]]; then
  TRAIN_ARGS+=(--resume "${V2_RESUME}")
fi

python -m src.v2.train "${TRAIN_ARGS[@]}"

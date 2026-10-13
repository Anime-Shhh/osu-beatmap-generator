#!/usr/bin/env bash
#SBATCH --job-name=osu-v2-blackwell
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:5000_Blackwell:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/osu-v2-blackwell-%j.out
#SBATCH --error=logs/osu-v2-blackwell-%j.err

set -euo pipefail

# Default paths for iLab environment
export PROJECT_DIR="${PROJECT_DIR:-/common/users/asj102/osu_project}"
export VENV_PATH="${VENV_PATH:-/common/users/asj102/osu_project/env}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/common/users/asj102/osu_project/checkpoints/v2_blackwell_$(date +%Y%m%d_%H%M%S)}"
export CONFIRM_TRAINING=1

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${CHECKPOINT_DIR}"
cd "${PROJECT_DIR}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

echo "=== Slurm Job: ${SLURM_JOB_ID:-local} ==="
echo "=== Host: $(hostname) ==="
echo "=== GPU Allocation: ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

STAGE="${V2_STAGE:-planner}"
EPOCHS="${V2_EPOCHS:-5}"
BATCH_SIZE="${V2_BATCH_SIZE:-4}"
GRAD_ACCUM="${V2_GRAD_ACCUM:-4}"

echo "=== Starting V2 Training (Stage: ${STAGE}, Epochs: ${EPOCHS}) ==="

python -m src.v2.train \
    --stage "${STAGE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation "${GRAD_ACCUM}" \
    --num-workers 4 \
    --device cuda \
    --mixed-precision bf16 \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --confirm-training \
    "$@"

echo "=== Job Completed Successfully ==="

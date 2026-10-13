#!/usr/bin/env bash
#SBATCH --job-name=osu-v2-ddp
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab3
#SBATCH --gres=gpu:4500_ada:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/common/users/asj102/osu_project/v2_runs/run_20261010_v2/logs/osu-v2-ddp-%j.out
#SBATCH --error=/common/users/asj102/osu_project/v2_runs/run_20261010_v2/logs/osu-v2-ddp-%j.err

set -euo pipefail

export PROJECT_DIR="/common/users/asj102/osu_project"
export VENV_PATH="/common/users/asj102/osu_project/env"
export RUN_DIR="/common/users/asj102/osu_project/v2_runs/run_20261010_v2"
export DATA_DIR="${RUN_DIR}/data"
export CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
export SPLIT_MANIFEST="${DATA_DIR}/splits.json"
export FEATURE_PIPELINE_KEY="a5770810badaa88f92f98f198d2f49626a678227e16c7b168c44e6c093b66b14"
export PYTHONPATH="${PROJECT_DIR}"

mkdir -p "${RUN_DIR}/logs"
mkdir -p "${CHECKPOINT_DIR}"
cd "${PROJECT_DIR}"

source "${VENV_PATH}/bin/activate"

echo "=== Slurm Job: ${SLURM_JOB_ID:-local} ==="
echo "=== Host: $(hostname) ==="
echo "=== GPU Allocation: ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
echo "=== PyTorch Multi-GPU Inventory: ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "=== Launching torchrun DDP on ${NUM_GPUS} GPUs ==="

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    -m src.v2.train \
    --records-dir "${DATA_DIR}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --feature-pipeline-key "${FEATURE_PIPELINE_KEY}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --stage planner \
    --resume "${CHECKPOINT_DIR}/latest.pt" \
    --epochs 200 \
    --patience 15 \
    --min-delta 0.001 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --num-workers 4 \
    --mixed-precision bf16 \
    --learning-rate 2.0e-4 \
    --confirm-training

echo "=== Job Completed Successfully ==="

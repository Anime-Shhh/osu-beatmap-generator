#!/bin/bash
#SBATCH --job-name=osu-a100-train
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodelist=rlab2
#SBATCH --output=/common/users/asj102/osu_project/logs/osu-a100-%j.out
#SBATCH --error=/common/users/asj102/osu_project/logs/osu-a100-%j.err

set -euo pipefail

PROJECT_DIR=/common/users/asj102/osu_project
MODEL_DIR=/common/users/asj102/osu_project/models_a100_$(date +%Y%m%d_%H%M%S)
LOG_DIR=/common/users/asj102/osu_project/logs
VENV=/common/users/asj102/osu_project/env/bin/activate
COLD_DATA_DIR=/common/users/asj102/osu_project/data/shards

mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

if [[ -f "$VENV" ]]; then
    source "$VENV"
fi
cd "$PROJECT_DIR"

echo "=== Slurm A100 Training Job: ${SLURM_JOB_ID:-local} ==="
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "Data Directory: $COLD_DATA_DIR"
echo "Total Shards: $(ls "$COLD_DATA_DIR"/*.tar | wc -l)"
echo "=========================================="

python -m src.train \
    --data_dir "$COLD_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs 50 \
    --batch_size 32 \
    --accum_steps 2 \
    --lr 1e-4 \
    --num_workers 4 \
    --tf32 \
    --log_every 10

echo "=== Training Job Finished Successfully ==="

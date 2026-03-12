#!/bin/bash
#SBATCH --job-name=osu_train_ddp
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --output=/common/users/asj102/osu_project/logs/slurm_%j.out
#SBATCH --error=/common/users/asj102/osu_project/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL

set -e

PROJECT_DIR=/common/home/asj102/personalCS/osu_beatmap_generator
DATA_SRC=/freespace/local/asj102/osu_cache/shards
SHM_DIR=/dev/shm/asj102/osu_train
MODEL_DIR=/common/users/asj102/osu_project/models
LOG_DIR=/common/users/asj102/osu_project/logs
VENV=/common/users/asj102/osu_project/env/bin/activate

trap "rm -rf $SHM_DIR" EXIT

SHM_AVAIL=$(df --output=avail /dev/shm | tail -1)
if [ -d "$DATA_SRC" ] && [ "$(ls -A $DATA_SRC 2>/dev/null)" ] && [ "$SHM_AVAIL" -gt 100000000 ]; then
    mkdir -p "$SHM_DIR"
    rsync -a "$DATA_SRC/" "$SHM_DIR/" || true
    TRAIN_DATA="$SHM_DIR"
else
    TRAIN_DATA="$DATA_SRC"
fi

source "$VENV"
cd "$PROJECT_DIR"

echo "=== DDP Multi-GPU Job ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
echo "========================="

torchrun --nproc_per_node=4 -m src.train \
    --data_dir "$TRAIN_DATA" \
    --model_dir "$MODEL_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs 100 \
    --accum_steps 2 \
    --lr 1e-4 \
    --num_workers 4 \
    --use_wandb \
    "$@"

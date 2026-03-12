#!/bin/bash
#SBATCH --job-name=osu_train
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/common/users/asj102/osu_project/logs/slurm_%j.out
#SBATCH --error=/common/users/asj102/osu_project/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL

set -e

PROJECT_DIR=/common/home/asj102/personalCS/osu_beatmap_generator
MODEL_DIR=/common/users/asj102/osu_project/models
LOG_DIR=/common/users/asj102/osu_project/logs
VENV=/common/users/asj102/osu_project/env/bin/activate

source "$VENV"
cd "$PROJECT_DIR"

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "================"

# --- NEW DATA LOADING LOGIC ---
# 1. Define paths
COLD_DATA_DIR=/common/users/asj102/osu_project/data/shards
HOT_DATA_DIR=/dev/shm/$USER/osu_shards

echo "=== Preparing Data ==="
# 2. Make a clean folder in the node's RAM
rm -rf "$HOT_DATA_DIR"
mkdir -p "$HOT_DATA_DIR"

# 3. Copy files from the network warehouse into RAM
echo "Copying shards to fast RAM disk (/dev/shm)..."
cp "$COLD_DATA_DIR"/*.tar "$HOT_DATA_DIR"/
echo "Copy complete! Found $(ls "$HOT_DATA_DIR"/*.tar | wc -l) shards in RAM."
echo "======================"

# Batch size auto-detected from GPU VRAM
python -m src.train \
    --data_dir "$HOT_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs 100 \
    --batch_size 16 \
    --accum_steps 1 \
    --lr 1e-4 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --persistent_workers \
    --pin_memory \
    --tf32 \
    --compile \
    --log_every 10 \
    "$@"

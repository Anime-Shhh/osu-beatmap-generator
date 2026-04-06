#!/bin/bash
#SBATCH --job-name=osu_preproc
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/common/users/asj102/osu_project/logs/preproc_%j.out
#SBATCH --error=/common/users/asj102/osu_project/logs/preproc_%j.err
#SBATCH --mail-type=END,FAIL

set -e

PROJECT_DIR=/common/home/asj102/personalCS/osu_beatmap_generator
# Cold storage (shared across nodes)
OUTPUT_DIR=/common/users/asj102/osu_project/data/shards
VENV=/common/users/asj102/osu_project/env/bin/activate

source "$VENV"
cd "$PROJECT_DIR"

echo "=== Preprocessing Job ==="
echo "Node: $(hostname)"
echo "Output: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
TARGET_TOTAL_SHARDS=500
EXAMPLES_PER_SHARD=200
EXPECTED_SIZE_NOTE="~15-20GB for full 500x200 cache (data dependent)"
echo "Target shards: $TARGET_TOTAL_SHARDS"
echo "Examples per shard: $EXAMPLES_PER_SHARD"
echo "Expected size note: $EXPECTED_SIZE_NOTE"
echo "========================="

python -u -m src.preprocess \
    --output_dir "$OUTPUT_DIR" \
    --max_shards "$TARGET_TOTAL_SHARDS" \
    --examples_per_shard "$EXAMPLES_PER_SHARD" \
    "$@"

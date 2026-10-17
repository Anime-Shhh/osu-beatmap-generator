#!/usr/bin/env bash
#SBATCH --job-name=osu-rhythm-flow
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${CACHE_DIR:?}" "${OUTPUT_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENV_DIR=/common/users/asj102/osu_project/env
GPU_COUNT=$("$ENV_DIR/bin/python" -c 'import torch; print(torch.cuda.device_count())')
"$ENV_DIR/bin/torchrun" --standalone --nproc_per_node="$GPU_COUNT" -m scripts.train_rhythm_flow \
 --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-8}"

#!/usr/bin/env bash
#SBATCH --job-name=osu-onset-flow
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${CACHE_DIR:?}" "${OUTPUT_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/common/users/asj102/osu_project/env/bin/python -u -m scripts.train_onset_flow \
  --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" --epochs "${EPOCHS:-60}" \
  --batch-size "${BATCH_SIZE:-16}" --max-steps "${MAX_STEPS:-0}" \
  --val-batches "${VAL_BATCHES:-0}" --workers 4

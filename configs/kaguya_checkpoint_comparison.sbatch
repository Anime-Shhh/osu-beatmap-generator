#!/usr/bin/env bash
#SBATCH --job-name=osu-kaguya-compare
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
exec 2>&1
: "${SOURCE_DIR:?}" "${CHECKPOINT:?}" "${CALIBRATION:?}" "${OUTPUT_DIR:?}" "${CACHE_MANIFEST:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/common/users/asj102/osu_project/env/bin/python -u -m scripts.run_kaguya_comparison \
  --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" --output "$OUTPUT_DIR" \
  --cache-manifest "$CACHE_MANIFEST"

#!/usr/bin/env bash
#SBATCH --job-name=osu-flow-data-expand
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${RUN_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/common/users/asj102/osu_project/env/bin/python -u -m scripts.extend_rhythm_flow_cache \
 --base-cache "$RUN_DIR/cache" --output-dir "${EXPANDED_CACHE:-$RUN_DIR/expanded-cache}" \
 --hf-revision b82117a5de13b7df5290482aed7cffc49cb76e6d \
 --start-track 300 --max-tracks 200 --budget-mb 768

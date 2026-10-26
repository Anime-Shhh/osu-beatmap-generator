#!/usr/bin/env bash
#SBATCH --job-name=osu-expand-native
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${OUTPUT_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OSU_BASE=/common/users/asj102/osu_project
"$OSU_BASE/env/bin/python" -m unittest tests.test_v2_timing tests.test_flow_generation -q
"$OSU_BASE/env/bin/python" -u -m scripts.expand_native_flow_corpus \
  --base-cache "$OSU_BASE/flow_runs/run_20261025/cache" \
  --base-native "$OSU_BASE/geometry_runs/run_20261024/native" \
  --output-dir "$OUTPUT_DIR" \
  --hf-revision b82117a5de13b7df5290482aed7cffc49cb76e6d \
  --start-track 500 --max-tracks 1000 --budget-mb 4096

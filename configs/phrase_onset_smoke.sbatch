#!/usr/bin/env bash
#SBATCH --job-name=osu-phrase-smoke
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${OUTPUT_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OSU_BASE=/common/users/asj102/osu_project
"$OSU_BASE/env/bin/python" -m unittest tests.test_onset_flow tests.test_flow_generation tests.test_v2_timing -q
"$OSU_BASE/env/bin/python" -u -m scripts.train_onset_flow \
  --cache-dir "$OSU_BASE/flow_runs/run_20261025/cache" \
  --initialize "$OSU_BASE/flow_runs/run_20261025/onset/best.pt" \
  --output-dir "$OUTPUT_DIR" --epochs 20 --max-steps 64 --val-batches 2 \
  --context-layers 2 --density-auxiliary --learning-rate .0001 --batch-size 16 --workers 4

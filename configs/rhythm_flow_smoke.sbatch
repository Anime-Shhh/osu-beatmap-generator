#!/usr/bin/env bash
#SBATCH --job-name=osu-rhythm-flow-smoke
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${RUN_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/common/users/asj102/osu_project/env/bin/python
"$PYTHON" -u -m scripts.build_rhythm_flow_cache \
 --records-dir /common/users/asj102/osu_project/v2_runs/run_20261014_real/data_300 \
 --output-dir "$RUN_DIR/cache" --budget-mb 1024
"$PYTHON" -u -m scripts.train_rhythm_flow --cache-dir "$RUN_DIR/cache" \
 --output-dir "$RUN_DIR/smoke" --max-steps 64 --val-batches 4 --batch-size 8
"$PYTHON" -u -m scripts.sample_rhythm_flow --checkpoint "$RUN_DIR/smoke/best.pt" \
 --audio '/common/users/asj102/osu_project/data/inference_input/Kaguya-sama_ Love Is War_ Opening Theme (Limited Time Only).mp3' \
 --output-dir "$RUN_DIR/smoke-signals" --steps 8

#!/usr/bin/env bash
#SBATCH --job-name=osu-v2-real-data
#SBATCH --partition=unlimited
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${SOURCE_DIR:?Set SOURCE_DIR to the revised source checkout}"
: "${DATA_DIR:?Set DATA_DIR to a new empty record directory}"
VENV_PATH="${VENV_PATH:-/common/users/asj102/osu_project/env}"
cd "$SOURCE_DIR"
export PYTHONPATH="$SOURCE_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"$VENV_PATH/bin/python" -m scripts.materialize_v2_data \
  --hf-revision b82117a5de13b7df5290482aed7cffc49cb76e6d \
  --max-tracks "${MAX_TRACKS:-1000}" --output-dir "$DATA_DIR"

#!/usr/bin/env bash
#SBATCH --job-name=dreamer-pilot
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:05:00
set -euo pipefail
exec 2>&1
: "${SOURCE:?}" "${COMMAND:?}" "${CONFIG:?}"
export PYTHONPATH="$SOURCE:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/source:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/deps:/common/users/asj102/osu_project/env-beat-tracker-20261025/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SOURCE"
/common/users/asj102/osu_project/env/bin/python -u -m src.dreamer_adapter "$COMMAND" --config "$CONFIG"

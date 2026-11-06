#!/usr/bin/env bash
#SBATCH --job-name=dreamer-prepare
#SBATCH --partition=unlimited
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail
exec 2>&1
: "${PILOT:?}" "${SOURCE:?}"
export PYTHONPATH="$SOURCE:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/source:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PILOT/hf-cache"
cd "$SOURCE"
/common/users/asj102/osu_project/env/bin/python -u -m src.dreamer_adapter prepare \
 --native-manifest /common/home/asj102/osu_flow_runs/run_20261104/recovered-corpus/native/manifest.json \
 --output "$PILOT/cache"

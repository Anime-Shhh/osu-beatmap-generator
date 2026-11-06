#!/usr/bin/env bash
#SBATCH --job-name=dreamer-gated
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=47:50:00
set -euo pipefail
exec 2>&1
: "${PILOT:?}" "${SOURCE:?}" "${BEAT_CHECKPOINT:?}"
export PYTHONPATH="$SOURCE:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/source:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/deps:/common/users/asj102/osu_project/env-beat-tracker-20261025/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PILOT/hf-cache"
cd "$SOURCE"
/common/users/asj102/osu_project/env/bin/python -u -m src.dreamer_adapter.pilot \
 --root "$PILOT" --upstream /common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt \
 --manifest "$PILOT/cache-compact/manifest.json" --beat-checkpoint "$BEAT_CHECKPOINT" \
 --beat-checkpoint-sha256 8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 ${PARALLEL_INPUTS:+--parallel-inputs}

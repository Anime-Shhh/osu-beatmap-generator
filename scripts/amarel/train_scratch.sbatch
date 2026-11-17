#!/bin/bash
#SBATCH --job-name=dreamer-scratch
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:2
#SBATCH --constraint=adalovelace|ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
stage="$1";run="$2";shift 2
common=(--upstream "$OSU_PROJECT_ROOT/upstream/osu-dreamer" --manifest "$OSU_ARTIFACT_ROOT/common/home/asj102/osu_flow_runs/pilot_20261105/cache-compact/manifest.json" --run "$run")
if [[ "$stage" == smoke ]]; then
 python -m unittest tests.test_dreamer_scratch -v
 for part in codec encode generator style export; do
  python -m src.dreamer_scratch.train "$part" "${common[@]}" --smoke
 done
else
 python -m src.dreamer_scratch.train "$stage" "${common[@]}" "$@"
fi

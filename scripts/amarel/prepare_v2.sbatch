#!/bin/bash
#SBATCH --job-name=osu-v2-data
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m scripts.amarel.prepare_v2 \
 --source "$OSU_ARTIFACT_ROOT/common/users/asj102/osu_project/v2_runs/run_20261014_real/data_300" \
 --output "$OSU_PROJECT_ROOT/datasets/v2-ranked-20261115"

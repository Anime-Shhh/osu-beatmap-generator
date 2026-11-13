#!/bin/bash
#SBATCH --job-name=osu-dreamer-repeat
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_PROJECT_ROOT/repo"
python -m scripts.upstream_comparison.train --root "$OSU_PROJECT_ROOT/upstream" --output "$OSU_PROJECT_ROOT/runs/dreamer-$SLURM_JOB_ID" --kind dreamer --steps 2000 --seconds 5400

#!/bin/bash
#SBATCH --job-name=osu-v2-compare
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
export PYTHONUNBUFFERED=1
python -m scripts.amarel.compare_v2 --v2-checkpoint "$1" \
 --dreamer-checkpoint "$OSU_PROJECT_ROOT/runs/dreamer-63885268/final.pt" \
 --output "$OSU_PROJECT_ROOT/runs/v2-comparison-$SLURM_JOB_ID"

#!/bin/bash
#SBATCH --job-name=osu-amarel-kaguya
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_PROJECT_ROOT/repo"
python -m scripts.amarel.kaguya --checkpoint "$1" --output "$OSU_PROJECT_ROOT/runs/kaguya-$SLURM_JOB_ID"

#!/bin/bash
#SBATCH --job-name=osu-dreamer-set
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
python -m scripts.amarel.song_set --output "$OSU_PROJECT_ROOT/runs/song-set-$SLURM_JOB_ID" "$@"

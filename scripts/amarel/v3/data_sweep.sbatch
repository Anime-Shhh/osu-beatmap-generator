#!/bin/bash
# Pre-flight sweep of every training chart's first/last/random crop through the real samplers and
# collate functions (src/v3/train/sweep.py). Exits non-zero on any failure, so GPU jobs submitted with
# --dependency=afterok:<this job> never start on data that would crash them hours in.
#SBATCH --job-name=v3-sweep
#SBATCH --partition=main
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-sweep-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
export OMP_NUM_THREADS=1
python -m src.v3.train.sweep --workers "$SLURM_CPUS_PER_TASK" \
  --out "/scratch/asj102/osu_project/logs/v3-sweep-$SLURM_JOB_ID.json"

#!/bin/bash
# Single-GPU Phase 0 step: `sbatch ... phase0_gpu_single.sbatch offset|dedup`.
# Both steps are idempotent: a completed step verifies its done-marker and exits.
#SBATCH --job-name=v3-p0gpu
#SBATCH --partition=gpu
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace|ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-%x-%j.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
step="${1:?offset|dedup}"
python -m src.v3.data.jobs "$step" --device cuda

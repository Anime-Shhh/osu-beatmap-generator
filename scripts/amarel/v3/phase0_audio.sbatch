#!/bin/bash
# Audio decode (CPU threads, ffmpeg) feeding GPU STFT/mel feature extraction.
# Rerunnable: each song's .json is written last and acts as its done marker.
#SBATCH --job-name=v3-audio
#SBATCH --partition=gpu
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace|ampere
#SBATCH --array=0-15
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-audio-%A_%a.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
python -m src.v3.data.jobs audio --shards 16 --device cuda

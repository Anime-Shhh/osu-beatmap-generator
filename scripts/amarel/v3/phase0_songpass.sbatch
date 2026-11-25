#!/bin/bash
# Per-song onset envelope + duplicate fingerprints from cached features (GPU).
# Same 16-way sharding as phase0_audio, so task k can start right after audio task k.
#SBATCH --job-name=v3-songpass
#SBATCH --partition=gpu
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace|ampere
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-songpass-%A_%a.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
python -m src.v3.data.jobs songpass --shards 16 --device cuda

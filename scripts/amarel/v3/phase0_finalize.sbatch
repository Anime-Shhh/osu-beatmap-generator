#!/bin/bash
# Splits after dedup, eligibility, vocabularies, eval sets, round-trip gates.
#SBATCH --job-name=v3-finalize
#SBATCH --partition=main
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-finalize-%j.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
python -m src.v3.data.jobs finalize

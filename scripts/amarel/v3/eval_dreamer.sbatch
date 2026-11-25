#!/bin/bash
# osu-dreamer baseline generation on the eval set (one map per chart; 16 candidates, shared reranker).
# Resumable per chart. Usage: sbatch eval_dreamer.sbatch dreamer-v8|dreamer-ft|dreamer-scratch
#SBATCH --job-name=v3-evaldreamer
#SBATCH --partition=gpu
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace|ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-evaldreamer-%x-%A_%a.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
python -m src.v3.eval.dreamer --system "${1:?system}" --shards 8 ${V3_EVAL_ARGS:-}

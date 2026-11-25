#!/bin/bash
# Eval-set generation for one system (one map per eval chart, the chart's own conditioning).
# Resumable per chart: a requeue/resubmit skips charts whose sidecar exists.
# Usage: sbatch eval_generate.sbatch SYSTEM RHYTHM_RUN PLACEMENT_RUN [TIMING_RUN]   (no TIMING_RUN = oracle grid)
#SBATCH --job-name=v3-evalgen
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
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-evalgen-%A_%a.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
SYSTEM="${1:?system}"; R="${2:?rhythm run}"; G="${3:?placement run}"; T="${4:-}"
args=(--system "$SYSTEM" --rhythm-run "$R" --placement-run "$G" --shards 8)
[[ -n "$T" ]] && args+=(--timing-run "$T")
python -m src.v3.eval.harness generate "${args[@]}" ${V3_EVAL_ARGS:-}

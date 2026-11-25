#!/bin/bash
# Score systems on the eval set (CPU: .osu parsing + rosu-pp), then rebuild the comparison table.
# "human-other" builds the human-vs-human ceiling system first.
# Usage: sbatch eval_score.sbatch SYSTEM [SYSTEM ...]
#SBATCH --job-name=v3-evalscore
#SBATCH --partition=main
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-evalscore-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
for system in "$@"; do
  [[ "$system" == human-other ]] && python -m src.v3.eval.harness ceiling ${V3_EVAL_ARGS:-}
  python -m src.v3.eval.harness score --system "$system" ${V3_EVAL_ARGS:-}
done
python -m src.v3.eval.harness table ${V3_EVAL_ARGS:-}

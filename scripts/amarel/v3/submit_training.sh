#!/bin/bash
# Chain: [finalize] -> training caches -> T, R, G (teacher 0.3) and G (human-only ablation).
# Usage: submit_training.sh [UPSTREAM_JOBID]   (e.g. finalize, or the DDP smoke: caches wait on afterok)
# Each training job sizes itself to its GPU-hour budget (throughput probe -> plan.json),
# checkpoints every 20 min, and requeues itself before the walltime. Re-running this
# script resubmits; completed caches/runs verify and exit, unfinished runs resume.
set -euo pipefail
fin="${1:-}"
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
snap="$("$repo/scripts/amarel/v3/snapshot.sh" train)"
cd "$snap"
RUNS=/scratch/$USER/osu_project/runs
EXCL="${V3_EXCLUDE:-gpuk[001-018],halk[0001-0159]}"   # k nodes see a stale, diverged /scratch as of 2026-11-24
dep=(); [[ -n "$fin" ]] && dep=(--dependency="afterok:$fin")
cache=$(sbatch --parsable "${dep[@]}" scripts/amarel/v3/phase0_cache.sbatch)
# Gates (CPU): the 2-rank bf16 DDP smoke and the full-data sweep must pass before any GPU job starts;
# a failed gate leaves the GPU jobs pending with DependencyNeverSatisfied. V3_GATES=0 skips them.
gates="$cache" sweep=- smoke=-
if [[ "${V3_GATES:-1}" != 0 ]]; then
  sweep=$(sbatch --parsable --dependency="afterok:$cache" scripts/amarel/v3/data_sweep.sbatch)
  smoke=$(sbatch --parsable "${dep[@]}" scripts/amarel/v3/smoke_ddp.sbatch)
  gates="$cache:$sweep:$smoke"
fi
g=(--gres=gpu:2 --cpus-per-task=24 --mem=180G --exclude="$EXCL" --dependency="afterok:$gates")
STAGES="${V3_STAGES:-T R G G0}"   # e.g. V3_STAGES="G G0" resubmits only the placement runs
ids=()
for st in $STAGES; do
  case "$st" in
    T)  id=$(sbatch --parsable "${g[@]}" --job-name=v3-T scripts/amarel/v3/train_stage.sbatch "$RUNS/v3-timing-base" timing \
             --size base --batch 16 --lr 3e-4 --warmup 2000 --budget-gpu-hours 20 --val-every 2000) ;;
    R)  id=$(sbatch --parsable "${g[@]}" --job-name=v3-R scripts/amarel/v3/train_stage.sbatch "$RUNS/v3-rhythm-base" rhythm \
             --size base --batch 8 --lr 3e-4 --warmup 2000 --budget-gpu-hours 40 --val-every 2000) ;;
    G)  id=$(sbatch --parsable "${g[@]}" --job-name=v3-G scripts/amarel/v3/train_stage.sbatch "$RUNS/v3-placement-base" placement \
             --size base --batch 8 --lr 3e-4 --warmup 2000 --budget-gpu-hours 35 --val-every 2000 --teacher-p 0.3) ;;
    G0) id=$(sbatch --parsable "${g[@]}" --job-name=v3-G0 scripts/amarel/v3/train_stage.sbatch "$RUNS/v3-placement-base-human" placement \
             --size base --batch 8 --lr 3e-4 --warmup 2000 --budget-gpu-hours 35 --val-every 2000 --teacher-p 0.0) ;;
    *)  echo "unknown stage $st" >&2; exit 2 ;;
  esac
  ids+=("$st=$id")
done
echo "snapshot=$snap cache=$cache sweep=$sweep smoke=$smoke ${ids[*]}"

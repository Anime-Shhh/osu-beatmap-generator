#!/bin/bash
# CPU smoke of the exact production launch path: torchrun + 2-process DDP (gloo backend).
# Covers what the single-process smoke cannot: torchrun argv pass-through, DDP gradient sync with
# find_unused_parameters=False, rank-0 validation/checkpoint barriers, STOP-file agreement across
# ranks, resume, and the all-reduced budget probe. Exits non-zero on any failure, so training
# submitted with --dependency=afterok:<this job> never starts on a broken launch path.
#SBATCH --job-name=v3-smokeddp
#SBATCH --partition=main
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-smokeddp-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
export V3_DATA_ROOT=/scratch/asj102/osu_project/datasets/v3-smoke V3_AUTO_REQUEUE=0 OMP_NUM_THREADS=4
RUNS=/scratch/asj102/osu_project/runs/v3-smokeddp-$SLURM_JOB_ID
echo "== rebuild smoke caches with current code"
for k in 0 1 2 3 4 5 6 7; do python -m src.v3.data.jobs cache --shard $k; done
launch=(torchrun --standalone --nproc_per_node=2 -m src.v3.train.run)
# bf16 autocast on, as in production: the first G launch died at step 0 on an autocast-only bug
# (no_grad self-conditioning poisoned the bf16 weight cache) that a --no-bf16 smoke could not see.
common=(--size tiny --batch 2 --workers 1 --val-every 10 --log-every 5 --val-charts 8 --warmup 5)
state() { python -c "import json,sys; s=json.load(open('$1/status.json')); print(s); sys.exit(0 if s['state']==sys.argv[1] else 1)" "$2"; }
for stage in rhythm placement timing; do
  echo "== $stage: 2-rank DDP to step 30, forced stop after step 10, resume"
  ( while ! grep -q '"step": 10,' "$RUNS/$stage/metrics.jsonl" 2>/dev/null; do sleep 2; done; touch "$RUNS/$stage/STOP" ) &
  "${launch[@]}" "$stage" --run-dir "$RUNS/$stage" --max-steps 30 "${common[@]}"
  state "$RUNS/$stage" incomplete
  "${launch[@]}" "$stage" --run-dir "$RUNS/$stage" --max-steps 30 "${common[@]}"
  state "$RUNS/$stage" complete
done
echo "== budget probe under DDP (max_steps derived from all-reduced throughput)"
"${launch[@]}" timing --run-dir "$RUNS/timing-budget" --max-steps 0 --budget-gpu-hours 0.002 \
  --size tiny --batch 2 --workers 1 --val-every 1000 --log-every 5 --val-charts 4 --warmup 30
cat "$RUNS/timing-budget/plan.json"; echo
state "$RUNS/timing-budget" complete
echo SMOKE_DDP_DONE

#!/bin/bash
# CPU smoke of the export round-trip and all three training stages on the smoke root,
# including a forced mid-run stop + resume. Tiny models; plumbing only, not quality.
#SBATCH --job-name=v3-smoketrain
#SBATCH --partition=main
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-smoketrain-%j.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
export V3_DATA_ROOT=/scratch/asj102/osu_project/datasets/v3-smoke V3_AUTO_REQUEUE=0
RUNS=/scratch/asj102/osu_project/runs/v3-smoke-$SLURM_JOB_ID
echo "== rebuild smoke caches with current code"
for k in 0 1 2 3 4 5 6 7; do python -m src.v3.data.jobs cache --shard $k; done
echo "== export round-trip"; python -m src.v3.eval.roundtrip --n 300
common=(--size tiny --batch 2 --workers 2 --val-every 10 --log-every 5 --val-charts 8 --no-bf16 --warmup 5)
for stage in rhythm placement timing; do
  echo "== $stage: run to step 20, forced stop at ~12, resume"
  ( sleep 1; while [[ ! -f $RUNS/$stage/metrics.jsonl ]] || ! grep -q '"step": 10,' $RUNS/$stage/metrics.jsonl 2>/dev/null; do sleep 2; done; touch $RUNS/$stage/STOP ) &
  python -m src.v3.train.run $stage --run-dir $RUNS/$stage --max-steps 20 "${common[@]}"
  python -c "import json;print('after stop:', json.load(open('$RUNS/$stage/status.json')))"
  python -m src.v3.train.run $stage --run-dir $RUNS/$stage --max-steps 20 "${common[@]}"
  python -c "import json;print('final:', json.load(open('$RUNS/$stage/status.json')))"
  cat $RUNS/$stage/best.json 2>/dev/null
done
echo SMOKE_TRAIN_DONE

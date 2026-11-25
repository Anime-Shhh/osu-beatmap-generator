#!/bin/bash
# Phase-4 evaluation once training finishes: v3 on oracle grids (scores R+G) and the full cascade
# (T grids), scored with the human ceiling into one table.
# Usage: submit_eval.sh [AFTER_JOBIDS]   (colon-separated, e.g. the T:R:G training job ids)
set -euo pipefail
after="${1:-}"
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
snap="$("$repo/scripts/amarel/v3/snapshot.sh" eval)"
cd "$snap"
RUNS=/scratch/$USER/osu_project/runs
T=$RUNS/v3-timing-base R=$RUNS/v3-rhythm-base G=$RUNS/v3-placement-base G0=$RUNS/v3-placement-base-human
dep=(); [[ -n "$after" ]] && dep=(--dependency="afterok:$after")
o=$(sbatch --parsable "${dep[@]}" scripts/amarel/v3/eval_generate.sbatch v3-oracle "$R" "$G")
h=$(sbatch --parsable "${dep[@]}" scripts/amarel/v3/eval_generate.sbatch v3-oracle-humanG "$R" "$G0")
f=$(sbatch --parsable "${dep[@]}" scripts/amarel/v3/eval_generate.sbatch v3-full "$R" "$G" "$T")
s=$(sbatch --parsable --dependency="afterok:$o:$h:$f" scripts/amarel/v3/eval_score.sbatch \
     v3-oracle v3-oracle-raw v3-oracle-humanG v3-oracle-humanG-raw v3-full v3-full-raw)
echo "snapshot=$snap oracle=$o oracle_humanG=$h full=$f score=$s"

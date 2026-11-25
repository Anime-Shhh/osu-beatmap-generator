#!/bin/bash
# Build a small smoke data root from whatever audio features already exist, so the
# training stages can be exercised before Phase 0 finishes. Never used for real runs.
#SBATCH --job-name=v3-smokeroot
#SBATCH --partition=main
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-smokeroot-%j.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
S=/scratch/asj102/osu_project/datasets/v3-smoke
REAL=/scratch/asj102/osu_project/datasets/v3-phase0
mkdir -p "$S"
[[ -e $S/audio ]] || ln -s "$REAL/audio" "$S/audio"
[[ -e $S/charts ]] || ln -s "$REAL/charts" "$S/charts"
export V3_DATA_ROOT=$S V3_ALLOW_PARTIAL=1
python -m src.v3.data.jobs songpass --shard 0 --shards 16 --device cpu
python -m src.v3.data.jobs offset --device cpu
python -m src.v3.data.jobs dedup --device cpu
python -m src.v3.data.jobs finalize
for k in 0 1 2 3 4 5 6 7; do python -m src.v3.data.jobs cache --shard $k; done
echo SMOKE_ROOT_DONE

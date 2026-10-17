#!/usr/bin/env bash
#SBATCH --job-name=osu-coverage-baseline
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
cd /common/users/asj102/osu_project/codex_v2_audio_alignment_20261015_precision
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/common/users/asj102/osu_project/env/bin/python -u /common/users/asj102/osu_project/codex_v2_coverage_20261017/scripts/replay_v2_coverage_baseline.py \
 --output-dir /common/users/asj102/osu_project/v2_runs/run_20261017_coverage/baseline-replay

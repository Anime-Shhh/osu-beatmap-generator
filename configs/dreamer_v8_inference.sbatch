#!/usr/bin/env bash
#SBATCH --job-name=osu-dreamer-v8
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
exec 2>&1
: "${RUN_DIR:?}"
export PYTHONPATH="$RUN_DIR/source:$RUN_DIR/deps:$RUN_DIR/eval_source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RUN_DIR/eval_source"
/common/users/asj102/osu_project/env/bin/python -u -m scripts.evaluate_dreamer_v8 \
  --checkpoint "$RUN_DIR/v8.pt" \
  --audio /common/users/asj102/osu_project/flow_runs/run_20261025/kaguya-original-input/audio.mp3 \
  --original-osz '/common/users/asj102/osu_project/1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size) [no video].osz' \
  --profile-guide /common/users/asj102/osu_project/flow_runs/run_20261025/profile-guide.json \
  --output "$RUN_DIR/kaguya" --candidates 16 --steps 8

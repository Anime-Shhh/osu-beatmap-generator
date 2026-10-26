#!/usr/bin/env bash
#SBATCH --job-name=osu-fullsong-val
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SOURCE_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OSU_BASE=/common/users/asj102/osu_project
OSU_RUN="$OSU_BASE/flow_runs/run_20261025"
"$OSU_BASE/env/bin/python" -u -m scripts.evaluate_full_song_rhythm \
  --cache-dir "$OSU_RUN/cache" --native-dir "$OSU_BASE/geometry_runs/run_20261024/native" \
  --audio-dir "$OSU_RUN/expanded-corpus/native/audio" --checkpoint "$OSU_RUN/onset/best.pt" \
  --output-dir "$OSU_RUN/original-full-song-validation" --threshold .5

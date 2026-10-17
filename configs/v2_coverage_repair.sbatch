#!/usr/bin/env bash
#SBATCH --job-name=osu-coverage-repair
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:40:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${RUN_DIR:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/common/users/asj102/osu_project/env/bin/python -u -m scripts.regenerate_v2_coverage \
 --checkpoint /common/users/asj102/osu_project/v2_runs/run_20261015_audio_alignment_precision/geometry/best.pt \
 --audio '/common/users/asj102/osu_project/data/inference_input/Kaguya-sama_ Love Is War_ Opening Theme (Limited Time Only).mp3' \
 --output-dir "$RUN_DIR/kaguya" --candidates 8 --seed 1601

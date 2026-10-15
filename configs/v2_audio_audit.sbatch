#!/usr/bin/env bash
#SBATCH --job-name=osu-v2-audio-audit
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${DATA_DIR:?}" "${CHECKPOINT:?}" "${OUTPUT:?}"
VENV_PATH="${VENV_PATH:-/common/users/asj102/osu_project/env}"
cd "$SOURCE_DIR"
export PYTHONPATH="$SOURCE_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$VENV_PATH/bin/python" -m scripts.audit_v2_audio_dependence --checkpoint "$CHECKPOINT" \
  --records-dir "$DATA_DIR" --output "$OUTPUT" --split val --bars-per-source 4

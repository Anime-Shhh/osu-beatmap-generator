#!/usr/bin/env bash
#SBATCH --job-name=osu-v2-real-train
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${SOURCE_DIR:?Set SOURCE_DIR to the revised source checkout}"
: "${DATA_DIR:?Set DATA_DIR to verified real v3 records}"
: "${RUN_DIR:?Set RUN_DIR to a new run directory}"
VENV_PATH="${VENV_PATH:-/common/users/asj102/osu_project/env}"
cd "$SOURCE_DIR"
export PYTHONPATH="$SOURCE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
NUM_GPUS=$("$VENV_PATH/bin/python" -c 'import torch; print(torch.cuda.device_count())')
if (( NUM_GPUS < 1 )); then
  echo 'No Slurm-allocated CUDA devices are visible' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Launching $NUM_GPUS training ranks; job=${SLURM_JOB_ID}; started=$(date -Is)"
FEATURE_PIPELINE_KEY=$("$VENV_PATH/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["feature_pipeline_key"])' "$DATA_DIR/materialization.json")
common=(--records-dir "$DATA_DIR" --split-manifest "$DATA_DIR/splits.json"
        --feature-pipeline-key "$FEATURE_PIPELINE_KEY" --batch-size "${BATCH_SIZE:-4}"
        --gradient-accumulation "${GRADIENT_ACCUMULATION:-4}" --bars-per-song 16 --num-workers "${NUM_WORKERS:-2}"
        --mixed-precision bf16 --patience 15 --confirm-training)
# This template starts a fresh two-stage run. Same-stage continuation should
# use src.v2.train --resume explicitly, preserving the previous run directory.
"$VENV_PATH/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
  --stage planner --epochs 200 --checkpoint-dir "$RUN_DIR/planner"
"$VENV_PATH/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
  --stage geometry --epochs 100 --resume "$RUN_DIR/planner/best.pt" \
  --checkpoint-dir "$RUN_DIR/geometry"

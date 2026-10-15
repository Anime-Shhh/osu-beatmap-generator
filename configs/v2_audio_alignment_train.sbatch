#!/usr/bin/env bash
#SBATCH --job-name=osu-v2-audio-align
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${DATA_DIR:?}" "${RUN_DIR:?}" "${INITIAL_PLANNER:?}"
VENV_PATH="${VENV_PATH:-/common/users/asj102/osu_project/env}"
cd "$SOURCE_DIR"
export PYTHONPATH="$SOURCE_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUM_GPUS=$("$VENV_PATH/bin/python" -c 'import torch; print(torch.cuda.device_count())')
if (( NUM_GPUS < 1 )); then echo 'No Slurm GPU allocated' >&2; exit 1; fi
FEATURE_PIPELINE_KEY=$("$VENV_PATH/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["feature_pipeline_key"])' "$DATA_DIR/materialization.json")
common=(--records-dir "$DATA_DIR" --split-manifest "$DATA_DIR/splits.json"
        --feature-pipeline-key "$FEATURE_PIPELINE_KEY" --planner-config configs/v2_audio_alignment_planner.json
        --batch-size "${BATCH_SIZE:-2}" --gradient-accumulation "${GRADIENT_ACCUMULATION:-8}"
        --bars-per-song 16 --num-workers "${NUM_WORKERS:-1}" --mixed-precision bf16 --patience 10
        --alignment-loss-weight "${ALIGNMENT_LOSS_WEIGHT:-0.2}" --confirm-training)
"$VENV_PATH/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
  --stage planner --initialize-planner "$INITIAL_PLANNER" --learning-rate 0.00005 \
  --epochs "${PLANNER_EPOCHS:-60}" --checkpoint-dir "$RUN_DIR/planner"
# Planner context changed: geometry must be adapted/trained with the new planner.
"$VENV_PATH/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
  --stage geometry --resume "$RUN_DIR/planner/best.pt" --learning-rate 0.0002 \
  --epochs "${GEOMETRY_EPOCHS:-100}" --checkpoint-dir "$RUN_DIR/geometry"

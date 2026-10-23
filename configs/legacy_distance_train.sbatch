#!/usr/bin/env bash
#SBATCH --job-name=osu-legacy-distance
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${SOURCE_DIR:?}" "${RUN_DIR:?}" "${MODEL_DIR:?}" "${DATA_DIR:?}" "${SPLIT_FILE:?}" "${INITIAL_CHECKPOINT:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENV_DIR=/common/users/asj102/osu_project/env
echo "source_commit=$(git rev-parse HEAD) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/python" -m src.train --stage legacy \
 --data_dir "$DATA_DIR" --split_file "$SPLIT_FILE" \
 --initialize_from "$INITIAL_CHECKPOINT" --model_dir "$MODEL_DIR" --log_dir "$RUN_DIR/tensorboard" \
 --distance_loss_weight "${DISTANCE_LOSS_WEIGHT:-0.1}" --seed 1731 \
 --epochs "${EPOCHS:-30}" --max_updates "${MAX_UPDATES:-10000}" \
 --batch_size "${BATCH_SIZE:-4}" --accum_steps "${ACCUM_STEPS:-4}" \
 --num_workers 0 --lr 0.00005 --early_stop_patience 0 --log_every 50 \
 --checkpoint_top_k 1 --compact_checkpoints --tf32

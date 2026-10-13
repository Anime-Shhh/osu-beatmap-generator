#!/usr/bin/env bash
#SBATCH --job-name=osu-kaguya-gen
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/common/users/asj102/osu_project/data/inference_output/kaguya_gen_%j.out
#SBATCH --error=/common/users/asj102/osu_project/data/inference_output/kaguya_gen_%j.err

set -euo pipefail

export PROJECT_DIR="/common/users/asj102/osu_project"
export VENV_PATH="/common/users/asj102/osu_project/env"
export PYTHONPATH="${PROJECT_DIR}"

mkdir -p /common/users/asj102/osu_project/data/inference_output/kaguya_v2_output
cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

echo "=== Running GPU Generation for Kaguya-sama 4.5* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m src.v2.generate \
    --checkpoint /common/users/asj102/osu_project/v2_runs/run_20261010_v2/checkpoints/best.pt \
    --audio "/common/users/asj102/osu_project/data/inference_input/Kaguya-sama_ Love Is War_ Opening Theme (Limited Time Only).mp3" \
    --target-star 4.5 \
    --title "Love Dramatic" \
    --artist "Masayuki Suzuki" \
    --version "Insane (4.5*)" \
    --output-dir /common/users/asj102/osu_project/data/inference_output/kaguya_v2_output \
    --candidate-count 1 \
    --geometry-steps 8 \
    --overwrite \
    --device cuda

echo "=== Generation Completed Successfully ==="

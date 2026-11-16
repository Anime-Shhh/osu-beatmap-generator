#!/bin/bash
#SBATCH --job-name=osu-v2-retrain
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:2
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
DATA_DIR="$OSU_PROJECT_ROOT/datasets/v2-ranked-20261115"
RUN_DIR="$OSU_PROJECT_ROOT/runs/v2-$SLURM_JOB_ID"
mkdir "$RUN_DIR"
python -m pip freeze > "$RUN_DIR/environment.txt"
python - "$RUN_DIR" <<'PYINFO'
import json,re,sys
from pathlib import Path
import torch
repo=Path.cwd()
gitdir=Path((repo/'.git').read_text().removeprefix('gitdir: ').strip())
revision=(gitdir/'HEAD').read_text().strip()
if not re.fullmatch('[0-9a-f]{40}',revision):
    raise ValueError('Training requires an immutable detached source worktree')
root=Path(sys.argv[1]);(root/'source.txt').write_text(revision+'\n')
(root/'hardware.json').write_text(json.dumps({'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch':torch.__version__,'cuda':torch.version.cuda},indent=2))
PYINFO
python -c 'from pathlib import Path; import sys; from scripts.amarel.compare_v2 import checked_songs; checked_songs(Path(sys.argv[1]))' "$RUN_DIR"
FEATURE_PIPELINE_KEY=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["feature_pipeline_key"])' "$DATA_DIR/materialization.json")
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
common=(--records-dir "$DATA_DIR" --split-manifest "$DATA_DIR/splits.json"
 --feature-pipeline-key "$FEATURE_PIPELINE_KEY" --planner-config configs/v2_audio_alignment_planner.json
 --batch-size 2 --gradient-accumulation 4 --bars-per-song 16 --num-workers 1
 --mixed-precision bf16 --patience 10 --checkpoint-interval 0 --seed 1731 --confirm-training)
if [[ "${V2_SMOKE:-0}" == 1 ]]; then
 common+=(--max-steps 2)
 epochs=1
else
 epochs=100
fi
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
 --stage planner --learning-rate 0.0002 --epochs "$epochs" --checkpoint-dir "$RUN_DIR/planner"
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
 --stage geometry --learning-rate 0.0002 --epochs "$epochs" --resume "$RUN_DIR/planner/best.pt" \
 --checkpoint-dir "$RUN_DIR/geometry"
printf '%s\n' 'Both training stages completed; generation quality not yet evaluated.' > "$RUN_DIR/complete.txt"

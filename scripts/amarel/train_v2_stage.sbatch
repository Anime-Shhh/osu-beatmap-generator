#!/bin/bash
#SBATCH --job-name=osu-v2-stage
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:2
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
DATA_DIR="$OSU_PROJECT_ROOT/datasets/v2-ranked-20261115"
STAGE="${1:?planner or geometry required}"
CHECKPOINT="${2:?parent checkpoint required}"
case "$STAGE" in planner|geometry) ;; *) echo 'Invalid stage' >&2; exit 1;; esac
RUN_DIR="$OSU_PROJECT_ROOT/runs/v2-stage-$SLURM_JOB_ID"
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
mkdir "$RUN_DIR/$STAGE"
if [[ "$STAGE" == planner ]]; then
 # Carry the historical best forward even if resumed epochs never improve it.
 cp "${3:?historical best planner checkpoint required}" "$RUN_DIR/planner/best.pt"
fi
python - "$CHECKPOINT" "$RUN_DIR" <<'PYPARENT'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
(Path(sys.argv[2])/'parent.json').write_text(json.dumps({'checkpoint':str(p),'sha256':h.hexdigest()},indent=2))
PYPARENT
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" -m src.v2.train "${common[@]}" \
 --stage "$STAGE" --learning-rate 0.0002 --epochs 100 --resume "$CHECKPOINT" \
 --checkpoint-dir "$RUN_DIR/$STAGE"
printf '%s\n' "$STAGE completed; generation quality not yet evaluated." > "$RUN_DIR/complete.txt"

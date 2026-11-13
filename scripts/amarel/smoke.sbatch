#!/bin/bash
#SBATCH --job-name=osu-amarel-smoke
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
python -m pytest tests/test_dreamer_adapter.py -q
python -m scripts.upstream_comparison.train --root "$OSU_PROJECT_ROOT/upstream" --output "$OSU_PROJECT_ROOT/runs/smoke-$SLURM_JOB_ID" --kind dreamer --steps 2 --seconds 900 --smoke
python - <<'PY'
import json,torch
print(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'bf16':torch.cuda.is_bf16_supported()}))
PY

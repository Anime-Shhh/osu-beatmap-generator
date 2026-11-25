# Sourced by every v3 Slurm script. Jobs always run from an immutable source
# snapshot (the directory the job was submitted from), never the mutable checkout.
set -euo pipefail
export OSU_REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from a snapshot directory}"
source "$OSU_REPO_ROOT/scripts/amarel/environment.sh"
cd "$OSU_REPO_ROOT"
export OSU_FFMPEG="$OSU_PROJECT_ROOT/ffmpeg/bin/ffmpeg"
export V3_DATA_ROOT="${V3_DATA_ROOT:-$OSU_PROJECT_ROOT/datasets/v3-phase0}"
# Soft open-file limit is 1,024 (hard 524,288); each memory-mapped cache field holds a descriptor and
# a placement loader worker maps ~1,088 of them. (The Python entry points also raise it.)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTHONUNBUFFERED=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "[v3] job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) repo=$OSU_REPO_ROOT data=$V3_DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

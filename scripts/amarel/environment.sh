# Source from a checked-out repository, before Python or Slurm work.
export OSU_PROJECT_ROOT="${OSU_PROJECT_ROOT:-/scratch/$USER/osu_project}"
export OSU_ARTIFACT_ROOT="$OSU_PROJECT_ROOT/imported"
export OSU_STORAGE_ROOT="$OSU_PROJECT_ROOT"
export PYTHONPATH="$OSU_PROJECT_ROOT/repo:$OSU_PROJECT_ROOT/upstream/osu-dreamer${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$OSU_PROJECT_ROOT/ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$OSU_PROJECT_ROOT/env/bin:$OSU_PROJECT_ROOT/ffmpeg/bin:$PATH"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="$OSU_PROJECT_ROOT/runtime-cache/matplotlib"
export NUMBA_CACHE_DIR="$OSU_PROJECT_ROOT/runtime-cache/numba"

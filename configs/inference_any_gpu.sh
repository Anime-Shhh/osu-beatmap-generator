#!/bin/bash
#SBATCH --job-name=osu_infer
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/common/users/asj102/osu_project/logs/infer_%j.out
#SBATCH --error=/common/users/asj102/osu_project/logs/infer_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Repo (for python -m src.inference). Some compute nodes may not mount /common/home/...
PROJECT_DIR=/common/home/asj102/personalCS/osu_beatmap_generator
# Always available on cluster jobs in practice (venv, ckpt, logs).
OSU_PROJECT=/common/users/asj102/osu_project
LOG_DIR="$OSU_PROJECT/logs"
VENV="$OSU_PROJECT/env/bin/activate"
CHECKPOINT="$OSU_PROJECT/models/best.pt"
DIFFICULTY=4
BPM=138

# Slurm evidence (infer_124143.err): INPUT under PROJECT_DIR was missing on compute.
# Keep audio + written .osu under OSU_PROJECT so nodes see them without /common/home.
INFER_INPUT_DIR="$OSU_PROJECT/data/inference_input"
INFER_OUTPUT_DIR="$OSU_PROJECT/data/inference_output"
SAMPLE_MP3="Kaguya-sama_ Love Is War_ Opening Theme (Limited Time Only).mp3"
REPO_SONG="$PROJECT_DIR/data/tests/songs/$SAMPLE_MP3"
STAGED_SONG="$INFER_INPUT_DIR/$SAMPLE_MP3"

# region agent log
_agent_log() {
  # Writable on compute; workspace path may be missing on some nodes.
  export _AG_IN="${INPUT_AUDIO:-}" _AG_OUT="${OUTPUT_OSU:-}" _AG_PD="$PROJECT_DIR" _AG_OSU="$OSU_PROJECT"
  python3 -c "
import json, os, time, pathlib
rec = {
    'sessionId': '307c43',
    'hypothesisId': os.environ.get('_AG_H', 'H1'),
    'location': 'configs/inference_any_gpu.sh',
    'message': os.environ.get('_AG_MSG', 'event'),
    'data': {
        'input': os.environ.get('_AG_IN', ''),
        'output': os.environ.get('_AG_OUT', ''),
        'project_dir_has_src': os.path.isdir(os.path.join(os.environ.get('_AG_PD', ''), 'src')),
        'hostname': os.uname().nodename,
    },
    'timestamp': int(time.time() * 1000),
}
for path in (
    pathlib.Path(os.environ['_AG_OSU']) / 'logs' / 'debug_307c43.ndjson',
    pathlib.Path(os.environ['_AG_PD']) / '.cursor' / 'debug-307c43.log',
):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception:
        pass
" 2>/dev/null || true
}
# endregion

[[ -f "$VENV" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }

mkdir -p "$INFER_INPUT_DIR" "$INFER_OUTPUT_DIR" "$LOG_DIR"

# Optional: sbatch --export=INPUT_AUDIO_OVERRIDE=/path/to/file.mp3
if [[ -n "${INPUT_AUDIO_OVERRIDE:-}" ]]; then
  INPUT_AUDIO="$INPUT_AUDIO_OVERRIDE"
elif [[ -f "$STAGED_SONG" ]]; then
  INPUT_AUDIO="$STAGED_SONG"
elif [[ -f "$REPO_SONG" ]]; then
  # Staging helps when repo is visible here but you want a stable path under OSU_PROJECT.
  cp -f "$REPO_SONG" "$STAGED_SONG"
  INPUT_AUDIO="$STAGED_SONG"
else
  echo "Missing input audio." >&2
  echo "  Put a copy at: $STAGED_SONG" >&2
  echo "  (or set INPUT_AUDIO_OVERRIDE when submitting)" >&2
  echo "  Login-node one-liner:" >&2
  echo "  mkdir -p $INFER_INPUT_DIR && cp '$REPO_SONG' '$INFER_INPUT_DIR/'" >&2
  export _AG_H=H1 _AG_MSG=missing_input
  INPUT_AUDIO="" OUTPUT_OSU=""
  _agent_log
  exit 1
fi

OUTPUT_OSU="${OUTPUT_OSU_OVERRIDE:-$INFER_OUTPUT_DIR/kaguya.osu}"

[[ -f "$INPUT_AUDIO" ]] || { echo "INPUT_AUDIO not a file: $INPUT_AUDIO" >&2; exit 1; }

export _AG_H=H1 _AG_MSG=paths_resolved
_agent_log

source "$VENV"

if [[ ! -d "$PROJECT_DIR/src" ]]; then
  echo "ERROR: Project code not visible at PROJECT_DIR=$PROJECT_DIR on this node." >&2
  echo "  Inference needs the repo (python -m src.inference). Mount /common/home or clone to a shared path." >&2
  export _AG_H=H3 _AG_MSG=project_dir_missing
  _agent_log
  exit 1
fi
cd "$PROJECT_DIR"

mkdir -p "$(dirname "$OUTPUT_OSU")"

echo "=== Inference Job ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Input: $INPUT_AUDIO"
echo "Output: $OUTPUT_OSU"
echo "Checkpoint: $CHECKPOINT"
echo "====================="

# TorchCodec has been flaky on this cluster for MP3 decoding, so prefer a WAV
# scratch copy when ffmpeg is available. This does not change model behavior.
RUNTIME_INPUT="$INPUT_AUDIO"
if [[ "$INPUT_AUDIO" == *.mp3 ]] && command -v ffmpeg >/dev/null 2>&1; then
  SCRATCH_DIR="${SLURM_TMPDIR:-/tmp/${USER}/osu_infer_${SLURM_JOB_ID:-0}}"
  mkdir -p "$SCRATCH_DIR"
  RUNTIME_INPUT="$SCRATCH_DIR/$(basename "${INPUT_AUDIO%.mp3}").wav"
  echo "Converting MP3 to WAV for stable decoding: $RUNTIME_INPUT"
  ffmpeg -y -i "$INPUT_AUDIO" "$RUNTIME_INPUT"
fi

python -m src.inference \
  --input "$RUNTIME_INPUT" \
  --output "$OUTPUT_OSU" \
  --checkpoint "$CHECKPOINT" \
  --difficulty "$DIFFICULTY" \
  --bpm "$BPM" \
  "$@"

export _AG_H=H2 _AG_MSG=inference_finished_ok
_agent_log

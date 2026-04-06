#!/bin/bash
#SBATCH --job-name=osu_infer
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/common/users/asj102/osu_project/logs/infer_%j.out
#SBATCH --error=/common/users/asj102/osu_project/logs/infer_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Repo is under /common/home/... — /common/users/.../personalCS/... does not exist.
PROJECT_DIR=/common/home/asj102/personalCS/osu_beatmap_generator
OSU_PROJECT=/common/users/asj102/osu_project
LOG_DIR="$OSU_PROJECT/logs"
VENV="$OSU_PROJECT/env/bin/activate"

INPUT_AUDIO="$PROJECT_DIR/data/tests/songs/Kaguya-sama_ Love Is War_ Opening Theme (Limited Time Only).mp3"
OUTPUT_OSU="$PROJECT_DIR/data/tests/kaguya.osu"
CHECKPOINT="$OSU_PROJECT/models/best.pt"
DIFFICULTY=4
BPM=138

[[ -f "$VENV" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -f "$INPUT_AUDIO" ]] || { echo "Missing input audio: $INPUT_AUDIO" >&2; exit 1; }

source "$VENV"
cd "$PROJECT_DIR"

mkdir -p "$LOG_DIR"
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

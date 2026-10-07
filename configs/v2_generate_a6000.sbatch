#!/usr/bin/env bash
# Explicit offline generation job. Submit manually only after a compatible v2
# checkpoint, feature bundle, and clock-hypothesis JSON have been prepared.
#SBATCH --job-name=osu-v2-generate
#SBATCH --output=osu-v2-%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the osu-beatmap-generator checkout}"
: "${CHECKPOINT:?Set CHECKPOINT to a trained v2 .pt checkpoint}"
: "${FEATURE_BUNDLE:?Set FEATURE_BUNDLE to a safe extracted v2 .pt payload}"
: "${CLOCK_HYPOTHESES:?Set CLOCK_HYPOTHESES to resolver JSON}"
: "${AUDIO_FILENAME:?Set AUDIO_FILENAME to the basename used in output .osu}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated draft artifacts}"

cd "${PROJECT_DIR}"
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

python -m src.v2.generate \
  --checkpoint "${CHECKPOINT}" \
  --feature-bundle "${FEATURE_BUNDLE}" \
  --clock-hypotheses "${CLOCK_HYPOTHESES}" \
  --audio-filename "${AUDIO_FILENAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-star "${TARGET_STAR:-4.5}" \
  --candidate-count "${CANDIDATE_COUNT:-4}" \
  --slider-multiplier "${SLIDER_MULTIPLIER:-1.4}" \
  --slider-tick-rate "${SLIDER_TICK_RATE:-1.0}" \
  --seed "${SEED:-0}" \
  --device cuda

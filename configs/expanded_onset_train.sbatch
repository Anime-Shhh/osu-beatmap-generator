#!/usr/bin/env bash
#SBATCH --job-name=osu-expanded-onset
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -euo pipefail
exec 2>&1
: "${SOURCE_DIR:?}" "${OUTPUT_DIR:?}" "${CONTEXT_LAYERS:?}"
cd "$SOURCE_DIR"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OSU_BASE=/common/users/asj102/osu_project
OSU_RUN="$OSU_BASE/flow_runs/run_20261025"
OSU_CACHE="${CACHE_DIR:-$OSU_RUN/expanded-corpus/audio-cache}"
test -s "$OSU_CACHE/manifest.json"
extra=()
if [ "$CONTEXT_LAYERS" -gt 0 ]; then extra+=(--density-auxiliary); fi
"$OSU_BASE/env/bin/python" -m unittest tests.test_onset_flow tests.test_flow_clock -q
"$OSU_BASE/env/bin/python" -u -m scripts.train_onset_flow \
  --cache-dir "$OSU_CACHE" --output-dir "$OUTPUT_DIR" \
  --initialize "$OSU_RUN/onset/best.pt" --epochs 20 --learning-rate .0001 \
  --context-layers "$CONTEXT_LAYERS" "${extra[@]}" --batch-size 16 --workers 4
"$OSU_BASE/env/bin/python" -u -m scripts.evaluate_onset_flow \
  --cache-dir "$OSU_CACHE" --checkpoint "$OUTPUT_DIR/best.pt" \
  --output-dir "$OUTPUT_DIR/validation" --validation-only
"$OSU_BASE/env/bin/python" -u -m scripts.evaluate_full_song_rhythm \
  --cache-dir "$OSU_RUN/cache" --native-dir "$OSU_BASE/geometry_runs/run_20261024/native" \
  --audio-dir "$OSU_RUN/expanded-corpus/native/audio" --checkpoint "$OUTPUT_DIR/best.pt" \
  --output-dir "$OUTPUT_DIR/full-song-validation" --threshold .5
if [ "$CONTEXT_LAYERS" -eq 0 ]; then
  "$OSU_BASE/env/bin/python" -u -m scripts.evaluate_onset_flow \
    --cache-dir "$OSU_CACHE" --checkpoint "$OSU_RUN/onset/best.pt" \
    --output-dir "$OUTPUT_DIR/original-checkpoint-validation" --validation-only --allow-parent-checkpoint
fi

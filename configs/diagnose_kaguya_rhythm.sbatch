#!/usr/bin/env bash
#SBATCH --job-name=osu-kaguya-diagnose
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
: "${SOURCE_DIR:?}"
cd "$SOURCE_DIR"
OSU_BASE=/common/users/asj102/osu_project
OSU_RUN="$OSU_BASE/flow_runs/run_20261025"
export PYTHONPATH="$PWD:$OSU_BASE/env-beat-tracker-20261025/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME="$OSU_RUN/beat-tracker-weights"
"$OSU_BASE/env/bin/python" -u -m src.beatmap_flow.generate \
  "$OSU_RUN/kaguya-original-input/audio.mp3" --stars 4.5 \
  --rhythm-checkpoint "$OSU_RUN/onset/best.pt" --geometry-checkpoint "$OSU_RUN/geometry/best.pt" \
  --profile-guide "$OSU_RUN/profile-guide.json" --calibration "$OSU_RUN/evaluation/metrics.json" \
  --candidates 1 --seed 1737 --star-tolerance .6 --steps 48 \
  --output-dir "$OSU_RUN/kaguya-original-inference-v2"
"$OSU_BASE/env/bin/python" -u -m scripts.compare_native_rhythm \
  --original-osz "$OSU_BASE/1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size) [no video].osz" \
  --generated-osu "$OSU_RUN/kaguya-original-inference-v2/generated.osu" \
  --generated-audio "$OSU_RUN/kaguya-original-inference-v2/audio.mp3" \
  --cache-manifest "$OSU_RUN/cache/manifest.json" \
  --output-dir "$OSU_RUN/kaguya-original-comparison-v2"
"$OSU_BASE/env/bin/python" -u -m scripts.diagnose_beat_tracker \
  --audio "$OSU_RUN/kaguya-original-input/audio.mp3" --output "$OSU_RUN/kaguya-beat-tracker.json"

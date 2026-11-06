#!/usr/bin/env bash
#SBATCH --job-name=dreamer-compact
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
exec 2>&1
: "${PILOT:?}" "${SOURCE:?}"
export PYTHONPATH="$SOURCE:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/source:/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SOURCE"
/common/users/asj102/osu_project/env/bin/python -u -m src.dreamer_adapter compact --source "$PILOT/cache" --output "$PILOT/cache-compact"
# Delete only the new, redundant uncompressed pilot cache after full equality verification.
/common/users/asj102/osu_project/env/bin/python - "$PILOT" <<'PY'
import json,shutil,sys
from pathlib import Path
from src.dreamer_adapter.contracts import digest,write_json
from src.dreamer_adapter.data import read_manifest
root=Path(sys.argv[1]).resolve();old=root/'cache';new=root/'cache-compact'
audit=json.loads((new/'compression-audit.json').read_text())
assert audit['verified_equal'] and Path(audit['source']).resolve()==old
assert digest(old/'manifest.json')==audit['source_sha256']
read_manifest(new/'manifest.json',verify=True)
shutil.copy2(old/'manifest.json',new/'uncompressed-source-manifest.json')
shutil.copy2(old/'progress.json',new/'acquisition-journal.json')
write_json(root/'cleanup-uncompressed-cache.json',audit)
shutil.rmtree(old)
PY

#!/bin/bash
# Copy the current v3 working tree into an immutable snapshot and print its path.
# Usage: scripts/amarel/v3/snapshot.sh <label>
set -euo pipefail
label="${1:?label}"
src="$(cd "$(dirname "$0")/../../.." && pwd)"
project="${OSU_PROJECT_ROOT:-/scratch/$USER/osu_project}"
digest="$(cd "$src" && find src/v3 src/v2 scripts/amarel tests/v3 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) -print0 \
          | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-10)"
dest="$project/sources/v3-$label-$digest"
if [[ ! -d "$dest" ]]; then
  mkdir -p "$dest.tmp"
  rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' "$src/" "$dest.tmp/"
  (cd "$src" && git rev-parse HEAD 2>/dev/null; git -C "$src" status --short 2>/dev/null) > "$dest.tmp/SNAPSHOT_GIT.txt" || true
  echo "$digest" > "$dest.tmp/SNAPSHOT_DIGEST"
  chmod -R a-w "$dest.tmp"
  mv "$dest.tmp" "$dest"
fi
echo "$dest"

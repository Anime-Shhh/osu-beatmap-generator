#!/bin/bash
# Submit songpass -> {offset, dedup} -> finalize from an immutable snapshot.
# Usage: submit_phase0_downstream.sh [AUDIO_JOBID] [CHARTS_JOBID]
#   With job ids: songpass task k waits for audio task k (aftercorr); offset/dedup
#   wait for charts. Without ids (parents already finished): no dependencies.
# Safe to rerun at any time: completed steps verify their done-markers and exit.
set -euo pipefail
audio="${1:-}"; charts="${2:-}"
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
snap="$("$repo/scripts/amarel/v3/snapshot.sh" phase0-downstream)"
cd "$snap"
dep_sp=(); [[ -n "$audio" ]] && dep_sp=(--dependency="aftercorr:$audio")
sp=$(sbatch --parsable "${dep_sp[@]}" scripts/amarel/v3/phase0_songpass.sbatch)
after="afterok:$sp"; [[ -n "$charts" ]] && after="$after:$charts"
off=$(sbatch --parsable --job-name=v3-offset --dependency="$after" scripts/amarel/v3/phase0_gpu_single.sbatch offset)
ded=$(sbatch --parsable --job-name=v3-dedup --dependency="$after" scripts/amarel/v3/phase0_gpu_single.sbatch dedup)
fin=$(sbatch --parsable --dependency="afterok:$off:$ded" scripts/amarel/v3/phase0_finalize.sbatch)
echo "snapshot=$snap songpass=$sp offset=$off dedup=$ded finalize=$fin"

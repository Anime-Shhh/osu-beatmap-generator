#!/bin/bash
# One-command, read-only progress summary for v3 jobs.  Usage: status.sh
data="${V3_DATA_ROOT:-/scratch/$USER/osu_project/datasets/v3-phase0}"
runs="/scratch/$USER/osu_project/runs"
echo "== queue"; squeue -u "$USER" -o "%.12i %.14j %.8T %.10M %.10l %R" 2>/dev/null | grep -E "JOBID|v3-" || echo "(no v3 jobs queued)"
echo "== latest attempt per v3 job (last 3 days; superseded attempts only counted)"
sacct -u "$USER" -S now-3days -X -n -P -o JobID,JobName,State,Elapsed,ExitCode,NodeList 2>/dev/null | grep "|v3-" | python3 -c '
import sys, collections
latest, earlier = {}, collections.Counter()
for line in sys.stdin:
    jid, name, state, elapsed, code, node = line.rstrip("\n").split("|")
    task = jid.split("_")[1] if "_" in jid else None
    key = name if task is None else name + "[" + task + "]"
    if key in latest:
        earlier[key] += 1 if latest[key][1] != "COMPLETED" else 0
    latest[key] = (jid, state, elapsed, code, node)
arrays = collections.defaultdict(collections.Counter)
for key, (jid, state, *_rest) in latest.items():
    if "[" in key:
        arrays[key.split("[")[0]][state.split()[0]] += 1
for key, (jid, state, elapsed, code, node) in sorted(latest.items(), key=lambda kv: kv[1][0]):
    if "[" in key:
        continue
    note = f"  ({earlier[key]} earlier failed attempt(s), superseded)" if earlier[key] else ""
    print(f"  {key:14s} {jid:>14s} {state:12s} {elapsed:>9s} {code:6s} {node}{note}")
for name, c in sorted(arrays.items()):
    print("  %-14s %14s " % (name, "array") + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
'
echo "== phase0 stage status ($data/status)"
if [[ -d "$data/status" ]]; then
  python3 - "$data" <<'EOF'
import json, sys, glob, os, collections
root = sys.argv[1]
by = collections.defaultdict(collections.Counter)
fails = []
for f in glob.glob(os.path.join(root, "status", "*.json")):
    name = os.path.basename(f)[:-5]
    stage = name.rsplit("-", 1)[0] if name[-3:].isdigit() else name
    r = json.load(open(f)); by[stage][r["state"]] += 1
    if r["state"] == "failed": fails.append((name, r.get("error", "")[:160]))
for stage, c in sorted(by.items()): print(f"  {stage:10s} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
for name, err in sorted(fails)[:10]: print(f"  FAILED {name}: {err}")
done = len(glob.glob(os.path.join(root, "audio", "??", "*.json")))
print(f"  audio songs with features: {done} / 34413")
EOF
fi
for s in "$runs"/v3-*/status.json "$runs"/v3-*/*/status.json; do
  [[ "$s" == *smoke* ]] && continue
  [[ -f "$s" ]] && python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('  ', sys.argv[1].split('/runs/')[1], r.get('state'), 'step', r.get('step'), r.get('last_checkpoint',''), r.get('val',''))" "$s"
done
echo "== eval ($data/evals/val300): charts done per system, scored?"
if [[ -d "$data/evals/val300" ]]; then
  python3 - "$data/evals/val300" <<'EOF'
import glob, json, os, sys
root = sys.argv[1]
for d in sorted(glob.glob(os.path.join(root, "*"))):
    name = os.path.basename(d)
    if name in ("scores", "status") or not os.path.isdir(d):
        continue
    sides = [f for f in glob.glob(os.path.join(d, "*.json")) if not f.endswith("binding.json")]
    fails = sum("error" in json.load(open(f)) for f in sides)
    score = os.path.join(root, "scores", name + ".json")
    tag = ""
    if os.path.exists(score):
        m = json.load(open(score))["macro"]
        tag = f"scored: F1@10 {m['f1_10']['mean']:.3f} F1@25 {m['f1_25']['mean']:.3f} valid {m['valid']['mean']:.3f}"
    print(f"  {name:24s} {len(sides):3d}/300 charts ({fails} failed)  {tag}")
print(f"  full table: {os.path.join(root, 'table.md')}")
EOF
fi
exit 0

#!/bin/bash
# Smoke run (queue step 2): the synthetic self-test from README section 1, on one GPU.
# Tests the code path only; nothing it prints is a result.
# It also reports two cluster facts needed for step 3: outbound internet from a compute node,
# and whether /data/user_data/$USER mounts.
#
# Usage (from the repo folder; logs/ must exist BEFORE sbatch, Slurm opens the log file first):
#   mkdir -p logs && sbatch slurm/smoke.sbatch
#
#SBATCH --job-name=p2a-smoke
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=.venv/bin/python
TMP="${SLURM_TMPDIR:-/tmp}/p2a_smoke_${SLURM_JOB_ID}"
OUT="runs/smoke-${SLURM_JOB_ID}"
trap 'rm -rf "$TMP"' EXIT

echo "== node: $(hostname)  job: $SLURM_JOB_ID  $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== cluster facts (not part of the smoke test, failures here do not fail the job)"
for url in https://www.kaggle.com https://zenodo.org https://github.com; do
  code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$url" 2>&1 || true)
  echo "internet $url -> $code"
done
stat -c 'user_data %n owner=%U' "/data/user_data/$USER" 2>&1 || true
df -h "/data/user_data/$USER" 2>&1 | tail -1 || true

echo "== unit tests"
$PY -m pytest tests/test_core.py -q

echo "== synthetic data"
$PY scripts/make_synthetic.py --root "$TMP/synth"

echo "== train 300 steps (M3 code path: gen + calibration)"
$PY scripts/train.py --data "$TMP/synth" --mode gen --use-cal --base 16 --bs 32 --steps 300 \
    --val-every 300 --val-n 64 --val-k 9 --val-steps 10 --workers 0 --out "$OUT"

echo "== outputs"
ls -la "$OUT"
echo "SMOKE_OK"

echo "== resume check: kill a run after its first checkpoint (as a preemption would), then continue it"
R="runs/smoke-${SLURM_JOB_ID}-resume"
ARGS="--data $TMP/synth --mode gen --use-cal --base 16 --bs 32 --steps 600 --val-every 100 --val-n 64 --val-k 9 --val-steps 10 --workers 0 --out $R --resume"
ckstep() { $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(ck['step'], 'opt' in ck)" "$1"; }

$PY scripts/train.py $ARGS > "$TMP/resume1.log" 2>&1 &
pid=$!
for _ in $(seq 600); do [ -e "$R/last.pt" ] && break; sleep 0.2; done
kill -TERM "$pid" 2>/dev/null || true
wait "$pid" || true
read -r s1 opt1 < <(ckstep "$R/last.pt")
echo "killed run: checkpoint at step $s1, optimizer state saved: $opt1"
if [ "$s1" -le 0 ] || [ "$s1" -ge 600 ] || [ "$opt1" != True ]; then echo "RESUME_FAIL: no mid-run checkpoint to resume from"; exit 1; fi

$PY scripts/train.py $ARGS > "$TMP/resume2.log" 2>&1 || { cat "$TMP/resume2.log"; echo "RESUME_FAIL: resumed run crashed"; exit 1; }
grep -v '"val"' "$TMP/resume2.log"
grep -q "^resumed from step $s1\$" "$TMP/resume2.log" || { echo "RESUME_FAIL: did not resume from step $s1"; exit 1; }
grep -q '^done' "$TMP/resume2.log" || { echo "RESUME_FAIL: resumed run did not finish"; exit 1; }
grep -q '^best so far: step' "$TMP/resume2.log" || { echo "RESUME_FAIL: resumed run did not pick up best.pt"; exit 1; }
read -r s2 opt2 < <(ckstep "$R/last.pt")
echo "resumed run: final checkpoint at step $s2, optimizer state saved: $opt2"
if [ "$s2" -ne 600 ] || [ "$opt2" != False ]; then echo "RESUME_FAIL: wrong final checkpoint"; exit 1; fi

# a resume with a different arm-defining setting must be refused
if $PY scripts/train.py $ARGS --p-drop-cal 0.3 > "$TMP/resume3.log" 2>&1; then echo "RESUME_FAIL: accepted different settings"; exit 1; fi
tail -1 "$TMP/resume3.log"
echo "RESUME_OK"

echo "== best.pt check: it holds the validation step with the lowest MAP CRPS in log.jsonl, and loads for evaluation"
$PY - "$R" <<'PYEOF'
import json, sys
import torch
sys.path.insert(0, ".")
from p2a.evalcore import load_checkpoint

run = sys.argv[1]
val = {}                                    # the later record wins for a step that was logged twice
for line in open(f"{run}/log.jsonl"):
    d = json.loads(line)
    if "val" in d:
        val[d["step"]] = d["val"]["map"]["crps"]
ck = torch.load(f"{run}/best.pt", map_location="cpu", weights_only=False)
sel = ck["select"]
print("validation MAP CRPS by step:", {k: round(v, 3) for k, v in val.items()})
print("best.pt:", sel, "| has optimizer state:", "opt" in ck)
assert ck["step"] == sel["step"] and "opt" not in ck
assert abs(val[sel["step"]] - sel["value"]) < 1e-9, "best.pt score is not the one logged for its step"
assert sel["value"] <= min(val.values()) + 1e-9, "best.pt is not the lowest validation score"
model, args = load_checkpoint(f"{run}/best.pt", "cpu")
print("loads through p2a.evalcore.load_checkpoint:", type(model).__name__)
PYEOF
echo "BEST_OK"

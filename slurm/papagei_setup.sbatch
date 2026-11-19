#!/bin/bash
# Queue step 6, setup part: fetch the external PaPaGei code and weights (neither is part of this repo)
# into ext/, then run the two PaPaGei unit tests and check that the pretrained weights load.
# Safe to run again: it skips what is already there.
#
# Usage (from the repo folder):  mkdir -p logs && sbatch slurm/papagei_setup.sbatch
# Afterwards:  PAPAGEI=$PWD/ext/papagei-foundation-model   weights: $PAPAGEI/weights/papagei_s.pt
#
#SBATCH --job-name=p2a-papagei
#SBATCH --partition=preempt
# No GPU is used here. The job still asks for one because CPU-only jobs wait far longer in this queue.
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=.venv/bin/python
PAPAGEI="$PWD/ext/papagei-foundation-model"
W="$PAPAGEI/weights/papagei_s.pt"
URL=https://zenodo.org/api/records/13983110/files/papagei_s.pt/content
MD5=a4cdb32392e2a7b25999128af92813b5      # papagei_s.pt, 23339216 bytes, from the Zenodo record on 2026-11-16

echo "== node: $(hostname)  job: $SLURM_JOB_ID  $(date -Is)"
mkdir -p ext
[ -d "$PAPAGEI/.git" ] || git clone --depth 1 https://github.com/Nokia-Bell-Labs/papagei-foundation-model "$PAPAGEI"
git -C "$PAPAGEI" log -1 --format='PaPaGei commit %H  %cd'
du -sh "$PAPAGEI"

mkdir -p "$PAPAGEI/weights"
if ! echo "$MD5  $W" | md5sum -c --status 2>/dev/null; then
  curl -fsSL -o "$W.part" "$URL"
  echo "$MD5  $W.part" | md5sum -c -
  mv "$W.part" "$W"
fi
ls -la "$W"

echo "== unit tests with the real PaPaGei architecture (random init)"
PAPAGEI_REPO="$PAPAGEI" $PY -m pytest tests/test_core.py -q -rs

echo "== pretrained weights: strict load into the wrapper, one forward pass"
PAPAGEI="$PAPAGEI" W="$W" $PY - <<'EOF'
import os, sys
import torch
sys.path.insert(0, ".")
from p2a import SEG_LEN
from p2a.encoders import PPGEncoder

repo, weights = os.environ["PAPAGEI"], os.environ["W"]
torch.manual_seed(0)
x = torch.randn(4, 1, SEG_LEN)
out = {}
for mode in ("frozen", "lora", "full"):
    enc = PPGEncoder(repo, weights, mode=mode).eval()
    vec, fmap = enc(x)
    out[mode] = vec
    assert torch.isfinite(vec).all() and torch.isfinite(fmap).all()
    print(f"{mode:7s} vec {tuple(vec.shape)}  map {tuple(fmap.shape)}  trainable {enc.n_trainable() / 1e6:.2f}M")
assert torch.allclose(out["frozen"], out["lora"], atol=1e-5) and torch.allclose(out["frozen"], out["full"], atol=1e-5)
scratch = PPGEncoder(repo, mode="scratch").eval()
assert not torch.allclose(out["frozen"], scratch(x)[0], atol=1e-3), "pretrained output equals random init"
print("frozen, lora and full give the same output at start; it differs from random init")
EOF
echo "PAPAGEI_OK"

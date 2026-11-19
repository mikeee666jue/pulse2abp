#!/bin/bash
# Second-round evaluations on 7200 test segments with 19 draws and the default 50 sampling steps.
#
# Why these settings: the 7200 seeded segments are a subset of the 20000 used by the first evaluations, so
# the clean b1 / m1 / m3 results need no rerun (take those rows from results/{b1,m1,m3}.npz), and every arm
# can be compared row by row and fed to scripts/reanalyze.py. A check on M3 (results/n7200_k19_s25) showed
# that 19 draws change nothing, but that 25 sampling steps instead of 50 make the draws about 12% narrower
# and lower 90% coverage from 0.86 to 0.82. So the step count stays at 50.
#
#   noisy pulse                      b1, m1, m3   --ppg noise:0.5
#   one network, each conditional    m3 with the calibration off (--cal none) and with the pulse off (--w-ppg 0)
#   pulse weight doubled             m3 with --w-ppg 2.0
#
# A guidance weight other than 1 makes the sampler run the model on a triple batch at every step, so the
# two --w-ppg runs cost about three times a plain one; they run last.
#
# One job, run one after another: about 5 to 6 hours on one L40S. Safe to run again: an evaluation whose
# .json exists is skipped.
#
# Usage (from the repo folder; logs/ must exist):
#   DATA=/data/user_data/$USER/p2a/vital sbatch slurm/evalgrid_n7200.sbatch
#   on an account with the `normal` QoS add:  -p general --qos=normal
#
#SBATCH --job-name=p2a-grid
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|A6000|6000Ada|L40|A100_40GB|A100_80GB|A100_80G|H100|H200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${DATA:?set DATA=/path/to/prepared/dataset when calling sbatch}"
E="--n 7200 --k 19 --steps 50"
R=results/n7200_k19_s50
mkdir -p "$R"
echo "== node: $(hostname)  job: $SLURM_JOB_ID  restarts so far: ${SLURM_RESTART_COUNT:-0}  $(date -Is)"
ls "$DATA" > /dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {    # run <output name> <arguments for evaluate.py ...>
  local out="$R/$1.json" t0=$SECONDS
  shift
  if [ -e "$out" ]; then echo "== $out exists, skipping"; return; fi
  echo "== evaluate.py $* $E --out $out"
  .venv/bin/python scripts/evaluate.py --data "$DATA" --workers 8 "$@" $E --out "$out"
  echo "   took $((SECONDS - t0)) s"
}

run b1_noise  --ckpt runs/b1/best.pt --ppg noise:0.5
run m3_nocal  --ckpt runs/m3/best.pt --cal none
run m1_noise  --ckpt runs/m1/best.pt --ppg noise:0.5
run m3_noise  --ckpt runs/m3/best.pt --cal fixed --ppg noise:0.5
run m3_w0     --ckpt runs/m3/best.pt --cal fixed --w-ppg 0
run m3_w2     --ckpt runs/m3/best.pt --cal fixed --w-ppg 2.0
echo "GRID_OK"

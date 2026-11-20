#!/bin/bash
# Stage 2 signal grid (E1): the 12 cells listed below for ONE run, one after another.
#
#   --target   --visible            --cal
#   abp        ppg | ecg | ppg,ecg   none and fixed     6 cells
#   ppg        abp | ecg | abp,ecg   none               3 cells
#   ecg        ppg | abp | ppg,abp   none               3 cells
#
# Outputs: results/general/<run>/<target>_from_<visible joined by +>[_cal].json (and .npz), with the
# defaults of scripts/evaluate_general.py (7200 test segments, 19 draws, 50 sampling steps, seed 0).
# Safe to run again: a cell whose .json exists is skipped, so the ABP-from-PPG cell of step 5 is not redone.
# About 30 to 45 minutes per cell for a generator, seconds per cell for a regression twin.
#
# Usage (from the repo folder; logs/ must exist):
#   export DATA=/data/user_data/$USER/p2a/vital3
#   RUN=g1_twin sbatch -p general --qos=normal --time=00:30:00 --job-name=grid-g1_twin slurm/evalgrid_general.sbatch
#   RUN=g1 PART=abp sbatch -p general --qos=normal --time=05:00:00 --job-name=grid-g1-abp slurm/evalgrid_general.sbatch
# PART is all (default), abp, ppg or ecg: the cells with that target. Splitting a generator's grid into
# three jobs lets them run side by side.
#
#SBATCH --job-name=p2a-grid2
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
: "${DATA:?set DATA=/path/to/dataset prepared with the ECG when calling sbatch}"
: "${RUN:?set RUN=<folder under runs/>, for example RUN=g1}"
PART="${PART:-all}"
case "$PART" in all|abp|ppg|ecg) ;; *) echo "PART must be all, abp, ppg or ecg"; exit 2 ;; esac
R="results/general/$RUN"
mkdir -p "$R"
echo "== node: $(hostname)  job: $SLURM_JOB_ID  restarts so far: ${SLURM_RESTART_COUNT:-0}  $(date -Is)"
echo "== run: $RUN  part: $PART"
ls "$DATA" > /dev/null
ls "runs/$RUN/best.pt" > /dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cell() {    # cell <target> <visible> [cal]
  local target=$1 visible=$2 cal=${3:-none} name out t0=$SECONDS
  name="${target}_from_${visible//,/+}"
  [ "$cal" = none ] || name="${name}_cal"
  out="$R/$name.json"
  if [ -e "$out" ]; then echo "== $out exists, skipping"; return; fi
  echo "== evaluate_general.py --target $target --visible $visible --cal $cal --out $out"
  .venv/bin/python scripts/evaluate_general.py --data "$DATA" --workers 8 --ckpt "runs/$RUN/best.pt" \
      --target "$target" --visible "$visible" --cal "$cal" --out "$out"
  echo "   took $((SECONDS - t0)) s"
}

if [ "$PART" = all ] || [ "$PART" = abp ]; then
  for v in ppg ecg ppg,ecg; do cell abp "$v" none; cell abp "$v" fixed; done
fi
if [ "$PART" = all ] || [ "$PART" = ppg ]; then
  for v in abp ecg abp,ecg; do cell ppg "$v"; done
fi
if [ "$PART" = all ] || [ "$PART" = ecg ]; then
  for v in ppg abp ppg,abp; do cell ecg "$v"; done
fi
echo "GRID_OK $RUN $PART"

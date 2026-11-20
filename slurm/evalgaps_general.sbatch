#!/bin/bash
# Stage 2 gap filling (E2) for ONE run or for the model-free fillers.
#
#   target   abp | ppg | ecg
#   gap      0.25, 1, 2, 4, 8 seconds hidden in the target (the rest of the target stays visible)
#   given    "alone"  = nothing else visible (--visible none)
#            "others" = the two other signals fully visible
#
# A model (RUN=<folder under runs/>) gets 2 cells per target and gap: 30 in all.
# RUN=base runs the two fillers that use no model, copy-the-neighbouring-beat and straight line: 30 in all.
# Gap positions depend only on the seed and the segment index, so every model and filler fills the same gaps.
#
# Outputs: results/general/<run>/<target>_gap<seconds>_<alone|others>.json  (and .npz)
#          results/general/base/<target>_gap<seconds>_<copy|linear>.json
# with the defaults of scripts/evaluate_general.py (7200 test segments, 19 draws, 50 sampling steps, seed 0).
# Safe to run again: a cell whose .json exists is skipped. About 30 minutes per cell for a generator,
# seconds for a regression twin, under a minute for a filler.
#
# Usage (from the repo folder; logs/ must exist):
#   export DATA=/data/user_data/$USER/p2a/vital3
#   RUN=g1 PART=abp sbatch -p general --qos=normal --time=08:00:00 --job-name=gaps-g1-abp slurm/evalgaps_general.sbatch
#   RUN=g1_twin     sbatch -p general --qos=normal --time=00:40:00 --job-name=gaps-g1_twin slurm/evalgaps_general.sbatch
#   RUN=base        sbatch -p general --qos=normal --time=01:00:00 --job-name=gaps-base    slurm/evalgaps_general.sbatch
# PART is all (default), abp, ppg or ecg: the cells with that target.
# To evaluate another checkpoint of a run than its best.pt, give CKPT and a folder name for the results:
#   RUN=g1_twin CKPT=runs/g1_twin/last.pt OUT=g1_twin_last sbatch ... slurm/evalgaps_general.sbatch
#
#SBATCH --job-name=p2a-gaps
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
: "${RUN:?set RUN=<folder under runs/>, or RUN=base for the model-free fillers}"
PART="${PART:-all}"
case "$PART" in all) TARGETS="abp ppg ecg" ;; abp|ppg|ecg) TARGETS="$PART" ;; *) echo "PART must be all, abp, ppg or ecg"; exit 2 ;; esac
CKPT="${CKPT:-runs/$RUN/best.pt}"
R="results/general/${OUT:-$RUN}"
mkdir -p "$R"
echo "== node: $(hostname)  job: $SLURM_JOB_ID  restarts so far: ${SLURM_RESTART_COUNT:-0}  $(date -Is)"
echo "== run: $RUN  part: $PART  checkpoint: $([ "$RUN" = base ] && echo none || echo "$CKPT")  results: $R"
ls "$DATA" > /dev/null
[ "$RUN" = base ] || ls "$CKPT" > /dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

others() { case "$1" in abp) echo ppg,ecg ;; ppg) echo abp,ecg ;; ecg) echo ppg,abp ;; esac; }

cell() {    # cell <target> <gap seconds> <tag> <arguments for evaluate_general.py ...>
  local target=$1 gap=$2 tag=$3 out t0=$SECONDS
  shift 3
  out="$R/${target}_gap${gap}_${tag}.json"
  if [ -e "$out" ]; then echo "== $out exists, skipping"; return; fi
  echo "== evaluate_general.py --target $target --gap $gap $* --out $out"
  .venv/bin/python scripts/evaluate_general.py --data "$DATA" --workers 8 --target "$target" --gap "$gap" "$@" --out "$out"
  echo "   took $((SECONDS - t0)) s"
}

for target in $TARGETS; do
  for gap in 0.25 1 2 4 8; do
    if [ "$RUN" = base ]; then
      cell "$target" "$gap" copy   --baseline copy
      cell "$target" "$gap" linear --baseline linear
    else
      cell "$target" "$gap" alone  --ckpt "$CKPT" --visible none
      cell "$target" "$gap" others --ckpt "$CKPT" --visible "$(others "$target")"
    fi
  done
done
echo "GAPS_OK $RUN $PART"

#!/bin/bash
# Queue step 3: download the two VitalDB subset files from Kaggle and convert them (README section 2).
# The dataset is public: on 2026-11-16 it downloaded from Babel with no Kaggle token in place.
# If a download is refused, put a token in ~/.kaggle/ (kaggle.json, or access_token) and run again.
# This takes two of the five .mat files: 13.8 GB + 1.7 GB of the 19.14 GB total.
#
# Jobs here get preempted, so every step can be repeated cheaply:
#   1. download   the archive goes to raw/part/ and moves to raw/ only when complete; the Kaggle tool
#                 continues a partial file. Kaggle wraps large files in a zip; the zip is what is kept.
#   2. unpack     onto the compute node's own disk with unzip (about 6 MB/s when unpacked onto the
#                 shared disk by the Kaggle tool, which is why it is not done that way)
#   3. convert    scripts/prep_pulsedb.py reads the unpacked file and writes the npy files to the shared disk
#
# Usage (from the repo folder):  mkdir -p logs && sbatch slurm/data.sbatch
# Result:  /data/user_data/$USER/p2a/vital/{train,val,test}/   <- this is DATA for training and evaluation
# READ THE SANITY REPORT in the log before training on it.
#
#SBATCH --job-name=p2a-data
#SBATCH --partition=preempt
# No GPU is used here. The job still asks for one because CPU-only jobs wait far longer in this queue.
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=.venv/bin/python
KAGGLE=.venv/bin/kaggle
SLUG=weinanwangrutgers/pulsedb-balanced-training-and-testing
BASE="/data/user_data/$USER/p2a"
RAW="$BASE/raw"
DATA="$BASE/vital"
LOCAL="/scratch/$USER/p2a_unpack_$SLURM_JOB_ID"
mkdir -p "$LOCAL" 2>/dev/null || { LOCAL="/tmp/$USER-p2a_unpack_$SLURM_JOB_ID"; mkdir -p "$LOCAL"; }
trap 'rm -rf "$LOCAL"' EXIT
# KEEP_RAW=0: keep nothing but the converted data. The zips then go to the node's own disk and vanish with
# the job, so a later run has to download again. Use it where shared-disk space matters more than that.
if [ "${KEEP_RAW:-1}" = 0 ]; then RAW="$LOCAL/raw"; fi

echo "== node: $(hostname)  job: $SLURM_JOB_ID  restarts so far: ${SLURM_RESTART_COUNT:-0}  $(date -Is)"
mkdir -p "$RAW/part" "$DATA"
df -h "$BASE" "$LOCAL" | tail -2

echo "== files in the Kaggle dataset"
$KAGGLE datasets files "$SLUG"
listed_size() { $KAGGLE datasets files "$SLUG" --csv | awk -F, -v f="$1" '$1 == f {print $2}'; }

for f in VitalDB_Train_Subset.mat VitalDB_CalFree_Test_Subset.mat; do
  if [ ! -e "$RAW/$f.zip" ] && [ ! -e "$RAW/$f" ]; then
    echo "== download $f  (already in part/: $(du -sh "$RAW/part" | cut -f1))"
    ( while sleep 60; do echo "   $(date +%T)  $(du -sh "$RAW/part" | cut -f1) in part/"; done ) &
    progress=$!
    $KAGGLE datasets download "$SLUG" -f "$f" -p "$RAW/part" -q || { kill $progress; echo "DOWNLOAD FAILED: $f"; exit 1; }
    kill $progress
    rm -f "$RAW/part/$f.zip.kaggle-partial"
    if [ -e "$RAW/part/$f.zip" ]; then
      rm -f "$RAW/part/$f"                      # half-unpacked copy left by an earlier killed run
      mv "$RAW/part/$f.zip" "$RAW/$f.zip"
    else
      mv "$RAW/part/$f" "$RAW/$f"
    fi
  fi
  ls -la "$RAW/$f"* | sed 's/^/   /'
done
df -h "$BASE" | tail -1

convert() {    # convert <file> <done marker> <arguments for prep_pulsedb.py ...>
  local f=$1 marker=$2 mat="$RAW/$1" want got t0=$SECONDS
  shift 2
  if [ -e "$marker" ]; then echo "already converted, skipping (delete the split folders to redo)"; return; fi
  if [ -e "$RAW/$f.zip" ]; then
    unzip -o -q "$RAW/$f.zip" -d "$LOCAL"
    mat="$LOCAL/$f"
    echo "   unpacked to $LOCAL in $((SECONDS - t0)) s"
  fi
  want=$(listed_size "$f"); got=$(stat -c %s "$mat")
  if [ "$want" != "$got" ]; then echo "SIZE MISMATCH for $f: listed $want bytes, got $got"; exit 1; fi
  $PY scripts/prep_pulsedb.py --mat "$mat" --root "$DATA" "$@"
  rm -f "$LOCAL/$f"
}

echo "== convert: training file -> train + val (5% of subjects held out)"
convert VitalDB_Train_Subset.mat "$DATA/val/meta.npz" --split train --val-frac 0.05 --jobs 8

echo "== convert: calibration-free test file -> test"
convert VitalDB_CalFree_Test_Subset.mat "$DATA/test/meta.npz" --split test --jobs 8

echo "== outputs"
du -sh "$DATA"/*
[ "${KEEP_RAW:-1}" = 0 ] || du -sh "$RAW"
df -h "$BASE" | tail -1
echo "DATA_OK $DATA"

#!/bin/bash
# Stage 2: the general model (--mode gen) and its regression twin (--mode reg). DATA must be a root
# prepared with the ECG (DATA_NAME=vital3 PREP_EXTRA=--ecg sbatch slurm/data.sbatch).
# Usage (from the repo folder; logs/ must exist BEFORE sbatch):
#   export DATA=/data/user_data/$USER/p2a/vital3
#   sbatch -p general --qos=normal --job-name=g1      slurm/train_general.sbatch --mode gen --out runs/g1
#   sbatch -p general --qos=normal --job-name=g1_twin slurm/train_general.sbatch --mode reg --out runs/g1_twin
# Everything after the script name is passed to scripts/train_general.py. Evaluate with
#   SCRIPT=evaluate_general.py sbatch slurm/eval.sbatch --ckpt runs/g1/best.pt --target abp --visible ppg --out results/general/g1/abp_from_ppg.json
#
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
# Any GPU type that the installed PyTorch (2.13.0+cu126, built for sm_50..sm_90) can run on. That leaves out
# RTX_PRO_6000 (a newer generation this build does not list); a job landing there would be expected to crash.
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|A6000|6000Ada|L40|A100_40GB|A100_80GB|A100_80G|H100|H200"
# Small requests on purpose: the GPU sets the pace (data workers sit idle), a full run takes about 6 h on an
# L40S, and a smaller, shorter job is charged less and fits scheduling gaps more easily.
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${DATA:?set DATA=/path/to/prepared/dataset when calling sbatch}"
echo "== node: $(hostname)  job: $SLURM_JOB_ID  restarts so far: ${SLURM_RESTART_COUNT:-0}  $(date -Is)"
ls "$DATA" > /dev/null      # /data/user_data is mounted on first access; fail here with a clear error if it is not there
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python scripts/train_general.py --data "$DATA" --workers 4 --resume "$@"

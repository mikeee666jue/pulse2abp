#!/bin/bash
# Evaluation has to run inside a job (the data is only visible on compute nodes).
# Usage (from the repo folder; logs/ must exist BEFORE sbatch). Everything after the script name is passed
# to scripts/evaluate.py, or to scripts/$SCRIPT if SCRIPT is set:
#   export DATA=/data/user_data/$USER/p2a/vital
#   sbatch --job-name=ev-b1 slurm/eval.sbatch --ckpt runs/b1/best.pt --out results/b1.json
#   SCRIPT=information.py sbatch --job-name=info-m3 slurm/eval.sbatch --ckpt runs/m3/best.pt --cal fixed --out results/m3_info.json
# B0 (reference forecasts) uses no GPU but is submitted the same way: CPU-only jobs wait far longer here.
#   sbatch --job-name=ev-b0 slurm/eval.sbatch --baselines-only --out results/b0.json
#
# An evaluation that is preempted starts again from the beginning when Slurm requeues it.
# On an account with the `normal` QoS add `-p general --qos=normal` (not preemptible). To run right after
# a training job:  sbatch --dependency=afterok:<training job id> --kill-on-invalid-dep=yes ...
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
# Any GPU type the installed PyTorch can run on (see slurm/train.sbatch for why RTX_PRO_6000 is left out).
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|A6000|6000Ada|L40|A100_40GB|A100_80GB|A100_80G|H100|H200"
# 8 CPUs because the checker runs on 8 processes. A generative evaluation at the defaults (20000 segments,
# 39 draws, 50 steps) is estimated at about 3 h on an L40S, from the time one validation takes.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${DATA:?set DATA=/path/to/prepared/dataset when calling sbatch}"
mkdir -p results
echo "== node: $(hostname)  job: $SLURM_JOB_ID  restarts so far: ${SLURM_RESTART_COUNT:-0}  $(date -Is)"
echo "== scripts/${SCRIPT:-evaluate.py} $*"
ls "$DATA" > /dev/null      # /data/user_data is mounted on first access
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU in this job"
.venv/bin/python "scripts/${SCRIPT:-evaluate.py}" --data "$DATA" --workers 8 "$@"

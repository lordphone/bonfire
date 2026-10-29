#!/bin/bash
# Week 1 spike on one PSC Bridges-2 H100.
# Submit from the PSC checkout: sbatch experiments/spike/spike.sbatch
# Slurm copies this file at submit time, but run.sh is read when the job starts,
# so run.sh can keep changing (git pull on PSC) while the job waits in the queue.
#SBATCH -J bonfire-spike
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 2:00:00
#SBATCH -o /ocean/projects/cis260267p/lwen2/bonfire/logs/spike-%j.out

set -euo pipefail

B=$PROJECT/bonfire
SIF=$B/images/vllm-0.32.0.sif
MODEL=$B/models/Qwen3-4B
REV=1cfa9a7208912126459214e8b04321603b3df60c

# Fail fast (and cheaply) if the image or weights are not ready.
[ -f "$SIF" ] || { echo "PREFLIGHT_FAILED: missing $SIF"; exit 1; }
[ "$(cat "$MODEL/REVISION" 2>/dev/null)" = "$REV" ] || { echo "PREFLIGHT_FAILED: model revision mismatch"; exit 1; }

exec bash "$B/repo/experiments/spike/run.sh"

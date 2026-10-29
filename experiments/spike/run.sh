#!/bin/bash
# Spike body, run inside the Slurm allocation by spike.sbatch.
# Current scope: smoke test. Copy weights to node-local disk, start vLLM, wait for
# readiness, send one completion, and record wall-clock timestamps for each phase.
# The full spike measurements (prefill, decode, kill tests) replace this before the job starts.

set -euo pipefail

B=$PROJECT/bonfire
SIF=$B/images/vllm-0.32.0.sif
OUT=$B/results/spike/${SLURM_JOB_ID:-manual}
PORT=8000
mkdir -p "$OUT"

ts() { date +%s.%N; }
mark() { echo "$1 $(ts)" | tee -a "$OUT/phases.txt"; }

mark JOB_START
{
  echo "node=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
} | tee "$OUT/host.txt"

mark COPY_STARTED
cp -r "$B/models/Qwen3-4B" "$LOCAL/Qwen3-4B"
mark COPY_FINISHED

mark WORKER_START_REQUESTED
apptainer exec --nv --bind "$LOCAL" "$SIF" \
  vllm serve "$LOCAL/Qwen3-4B" \
    --served-model-name qwen3-4b \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --seed 20261026 \
    --no-enable-prefix-caching \
    --port "$PORT" \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

until curl -sf "localhost:$PORT/health" > /dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited before ready"; tail -50 "$OUT/vllm.log"; exit 1; }
  sleep 0.5
done
mark WORKER_READY

curl -sf "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","prompt":"Hello","max_tokens":20,"temperature":0}' \
  | tee "$OUT/completion.json"
echo
mark FIRST_COMPLETION_FINISHED

mark JOB_END

#!/usr/bin/env bash
# Start the long-lived bench container for the GPU matrix (docs/infrastructure/gpu-matrix.md).
# The supervisor runs inside it, so vLLM is an owned child process of the locked image.
#
#   scripts/gpu_container.sh                      # start (replaces an existing one)
#   docker exec bonfire-bench python3 scripts/run_gpu_matrix.py pilot --out experiments/gpu/results/<id>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$REPO/infra/environment.lock.json"
IMAGE="vllm/vllm-openai@$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['vllm']['image_amd64_digest'])" "$LOCK")"
REVISION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model']['revision'])" "$LOCK")"
MODEL_DIR="${BONFIRE_MODEL_DIR:-$HOME/.cache/bonfire-models/Qwen3-4B}"  # local snapshot at $REVISION
CACHE="${BONFIRE_CACHE:-$HOME/.cache/bonfire-spike}"                    # JIT caches shared with the spike
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)"

[ -f "$MODEL_DIR/config.json" ] || { echo "no model snapshot in $MODEL_DIR" >&2; exit 1; }
mkdir -p "$CACHE/vllm" "$CACHE/triton" "$CACHE/nv"
docker rm -f bonfire-bench >/dev/null 2>&1 || true
# --init reaps vLLM's engine process when the supervisor kills its parent.
docker run -d --init --name bonfire-bench --runtime=nvidia --gpus all --ipc=host --network host \
  -v "$REPO":/work -w /work -v "$MODEL_DIR":"/models/snapshots/$REVISION":ro \
  -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/triton":/root/.triton -v "$CACHE/nv":/root/.nv \
  -e CUDA_CACHE_MAXSIZE=4294967296 -e BONFIRE_GPU_UUID="$GPU_UUID" \
  -v "$REPO/experiments/spike/compat":/opt/bonfire-compat:ro -e PYTHONPATH=/opt/bonfire-compat:/work \
  --entrypoint sleep "$IMAGE" infinity

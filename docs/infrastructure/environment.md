# Reproducible Serving Environment

`infra/environment.lock.json` is the source of truth for every measured run. A run is comparable only when its `configuration_id` resolves to this lock plus explicit experiment overrides.

## Canonical GPU environment

| Component | Pinned value |
| --- | --- |
| OS / architecture | Ubuntu 24.04, linux/amd64 |
| vLLM | 0.32.0, release commit `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Container | `vllm/vllm-openai:v0.32.0-cu129-ubuntu2404` pinned to the amd64 digest in the lock |
| Python | 3.12 |
| PyTorch | 2.13.0 |
| CUDA toolkit | 12.9.1 |
| NCCL | 2.30.7 |
| Model | `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Precision | BF16 |
| Measured context limit | 16,384 tokens |

The preferred accelerator is one NVIDIA L40S 48 GB or A100 40 GB. A 24 GB NVIDIA GPU is the fallback for the 4B model. The canonical path passes the host GPU directly into the pinned container. Compatibility mode may use a newer host driver that supports CUDA 12.9 containers, but it must not rebuild or modify the image; record the actual GPU UUID, model, driver, and container digest in telemetry.

Prefix caching is explicitly disabled for recovery measurements. Otherwise a cache hit can make re-prefill appear cheaper than reconstructing state from the committed prefix. Tensor parallelism is fixed at one for the baseline.

## Preflight gate

Before vLLM starts, the launcher must:

1. Resolve the container to the locked linux/amd64 digest and fail on a mismatch.
2. Resolve the model snapshot to the locked 40-character revision and fail on a mismatch.
3. Capture `nvidia-smi` GPU model, GPU UUID, driver version, available memory, and host ID.
4. Verify NVIDIA driver `>=575.51.03`, BF16 support, and sufficient free memory for the configured run.
5. Generate the `configuration_id` from the immutable lock plus explicit runtime overrides.
6. Persist the preflight record before launching a worker.

Unsupported hardware or a failed check ends the run as `PREFLIGHT_FAILED`; it must not silently switch precision, model revision, context length, or image.

## CPU-only path

CPU runs validate protocol state transitions, schemas, deterministic fault schedules, reconciliation, and `RECOVERY_ABSTAIN`. They do not produce GPU latency or throughput claims and must use a distinct `configuration_id`. Run the local contract suite with:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/validate_week1_contracts.py
```

## Provenance

- vLLM release and image: <https://github.com/vllm-project/vllm/releases/tag/v0.32.0>
- vLLM GPU installation constraints: <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- vLLM serve options: <https://docs.vllm.ai/en/v0.32.0/cli/serve/>
- Qwen3-4B model card: <https://huggingface.co/Qwen/Qwen3-4B>
- CUDA 12.9 release notes: <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html>

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
| Served context limit (`max_model_len`) | 32,768 tokens; the spike measured contexts up to about 16.9K |

## Hardware

Every reported GPU number comes from one local workstation. The team switched from PSC Bridges-2 H100 to this machine on 2026-11-03 (no queue, Docker available); PSC H100 / L40S remain fallbacks and their results are reported separately, never pooled.

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32 GB, compute capability 12.0 (Blackwell) |
| Driver | 580.178.04 |
| Host | 16 CPU cores, 91 GB RAM, local NVMe |
| Container runtime | Docker 29.9 + NVIDIA Container Toolkit 1.20.1-1 (`--runtime=nvidia --gpus all`) |
| Weights | `~/models/Qwen3-4B` on local NVMe, mounted read-only |

The pinned container runs unmodified; record the GPU UUID, model, driver, and container digest in telemetry. The card is a consumer part that also drives the desktop display, so:

- run measurements with nothing else using the GPU, and log the SM clock, temperature, and clock event (throttle) reasons with every `GPU_SAMPLE`;
- `--gpu-memory-utilization 0.85` leaves room for the display;
- warm replicas are out of the current single-GPU process-restart scope; if one is ever added as future work, it would share this one GPU with the primary and must be labelled as such.

### Required runtime settings for the pinned image

Both were found in the Week 1 spike (`experiments/spike/report.md`). `experiments/spike/spike.py` (`Server.start`) is the reference launch command.

- **torchcodec shim.** The image bundles `torchcodec` 0.17.0 built for CUDA 13, so `vllm serve` crashes at import time with an uncaught `OSError` on any host. Mount `experiments/spike/compat` read-only and set `PYTHONPATH` to it. The shim raises `ImportError`, so vLLM skips audio decoding, which this workload doesn't use.
- **Persistent driver JIT cache.** The image has no native sm_120 kernels, so the CUDA driver JIT-compiles PTX at every start (about 65 s) unless `/root/.nv` is a persistent volume. Set `CUDA_CACHE_MAXSIZE=4294967296` too. Persist the vLLM torch.compile cache (`/root/.cache/vllm`) and the Triton cache (`/root/.triton`) as well. The harness records on every `WORKER_START_REQUESTED` whether these caches were warm (for the first cold-start run, the driver cache state was inferred; see the report).

## Prefix caching

Automatic prefix caching (APC) is an experiment factor. The lock pins it off as the baseline; a run that enables it declares `enable_prefix_caching=true` as an explicit runtime override, so it gets a distinct `configuration_id`. APC on and off results are reported side by side, never pooled. Tensor parallelism is fixed at one.

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
python -m pytest -q
python scripts/validate_recovery_contracts.py
python scripts/validate_infrastructure_contracts.py
```

See the [README](../../README.md) for the full CPU experiment workflow.

## Provenance

- vLLM release and image: <https://github.com/vllm-project/vllm/releases/tag/v0.32.0>
- vLLM GPU installation constraints: <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- vLLM serve options: <https://docs.vllm.ai/en/v0.32.0/cli/serve/>
- Qwen3-4B model card: <https://huggingface.co/Qwen/Qwen3-4B>
- CUDA 12.9 release notes: <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html>

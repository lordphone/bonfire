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
| Measured context limit | 32,768 tokens |

## Hardware

Every reported GPU number comes from PSC Bridges-2.

| Role | Node type | Slurm request |
| --- | --- | --- |
| Primary | NVIDIA H100 SXM 80 GB (`h100-80`) | `--gpus=h100-80:1`; warm replica uses `:2` |
| Fallback | NVIDIA L40S 48 GB (`l40s-48`) | `--gpus=l40s-48:1` |

Use the `GPU-shared` partition (at most 4 GPUs and 48 hours per job). Always name the GPU type, or the job can land on a V100. V100 nodes are not used: they lack BF16, and the pinned vLLM release reportedly does not support sm_70. L40S results are reported separately and never mixed with H100 results.

PSC does not allow Docker. Convert the pinned image to a Singularity/Apptainer `.sif` by digest, and pass the GPU in with `--nv`; the container must not be rebuilt or modified. Copy model weights to node-local storage (`$LOCAL`) before measuring cold start; a cold start from the shared filesystem is measured separately. Telemetry records the actual GPU UUID, model, driver, container digest, node name, and Slurm job ID.

## Prefix caching

Automatic prefix caching (APC) is an experiment factor, not a fixed setting. The lock pins it off as the baseline; a run that enables it must declare `enable_prefix_caching=true` as an explicit runtime override, so it gets a distinct `configuration_id`. APC on and off results are reported side by side, never pooled. Tensor parallelism is fixed at one for the baseline.

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

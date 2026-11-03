# Spike results: `20261103-rtx5090`

2598 telemetry events in 48 files, all valid against `telemetry-event.schema.json` with gap-free sequences.

GPU NVIDIA GeForce RTX 5090 (driver 580.178.04), host `lordphone-ubuntu`. Values are median (σ, n). Times in ms unless noted.

## Cold start (APC off)

| Weights | Driver JIT cache | Startup (s) | Weights load (s) | torch.compile (s) | CUDA graphs (s) | Engine init (s) |
| --- | --- | --- | --- | --- | --- | --- |
| nvme | empty | 86.2 | 2.0 | 0.1 | 70.0 | 71.7 |
| nvme | warm | 22.2 (σ 0.1, n=4) | 2.0 (σ 0.0, n=4) | 0.1 (σ 0.0, n=4) | 5.0 (σ 0.0, n=4) | 7.8 (σ 0.0, n=4) |
| page_cache | warm | 20.5 (σ 0.1, n=5) | 0.4 (σ 0.0, n=5) | 0.1 (σ 0.0, n=5) | 5.0 (σ 0.0, n=5) | 7.7 (σ 0.0, n=5) |

Sub-phases are vLLM's self-reported log values (diagnostic); startup is `WORKER_START_REQUESTED` → `WORKER_READY`. The vLLM torch.compile cache was warm for every start.

## Prefill (time to first token, max_tokens=1)

| Tokens | APC off (ms) | APC on, first (ms) | APC on, repeat (ms) | APC-off throughput (tok/s) |
| --- | --- | --- | --- | --- |
| 2048 | 89.6 (σ 2.0, n=5) | 96.9 (σ 2.8, n=5) | 14.9 (σ 0.8, n=5) | 22,869 |
| 8192 | 452.3 (σ 4.7, n=5) | 457.4 (σ 2.9, n=5) | 26.0 (σ 0.3, n=5) | 18,111 |
| 16384 | 1094.0 (σ 9.0, n=5) | 1111.0 (σ 1.8, n=5) | 42.0 (σ 1.6, n=5) | 14,977 |

## Decode

| Mode | Context / thinking | Decode (tok/s) | Generated tokens | TTFT (ms) |
| --- | --- | --- | --- | --- |
| fixed 512 tokens | 2048 ctx | 161.3 (σ 0.0, n=5) | 512 | – |
| fixed 512 tokens | 16384 ctx | 128.7 (σ 0.1, n=5) | 512 | – |
| chat, agent prompts | thinking off | 161.5 (σ 0.2, n=5) | 660.0 (σ 245.7, n=5) | 15.1 (σ 0.6, n=5) |
| chat, agent prompts | thinking on | 160.7 (σ 0.4, n=5) | 1598.0 (σ 499.0, n=5) | 15.1 (σ 0.7, n=5) |

## Recovery after `kill -9` mid-generation (step 6 of 10)

| Fault domain | Detect | Teardown | Startup | Reconcile | Re-prefill + 1st token | Crash → next token | Prefix tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inference kill (cold restart) | 387.3 (σ 18.8, n=5) | 21.0 (σ 0.6, n=5) | 20418.7 (σ 46.8, n=5) | 19.3 (σ 7.5, n=5) | 595.6 (σ 7.7, n=5) | 21562.9 (σ 47.2, n=5) | 10112.0 (σ 0.0, n=5) |
| Runner kill APC off | 7.0 (σ 0.3, n=5) | 0.6 (σ 0.1, n=5) | 32.8 (σ 0.7, n=5) | 1.2 (σ 0.1, n=5) | 584.1 (σ 1.5, n=5) | 625.4 (σ 1.4, n=5) | 10112.0 (σ 0.0, n=5) |
| Runner kill APC on | 7.0 (σ 8.1, n=5) | 0.7 (σ 0.1, n=5) | 33.2 (σ 0.7, n=5) | 1.2 (σ 0.1, n=5) | 31.3 (σ 0.9, n=5) | 73.5 (σ 0.8, n=5) | 10112.0 (σ 0.0, n=5) |

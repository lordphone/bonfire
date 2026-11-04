# Week 1 spike report

Lordphone Wen · 2026-11-03 · one RTX 5090 (32 GB, driver 580.178.04), pinned vLLM 0.32.0 cu129 image, Qwen3-4B BF16, `max_model_len` 32768. Raw data: [`results/20261103-rtx5090/`](results/20261103-rtx5090/) (2,630 telemetry events, all schema-valid); tables with per-cell σ: [`summary.md`](results/20261103-rtx5090/summary.md). Every value is the median of 5 runs unless noted. The cold start from NVMe with a warm cache is n=4, and the empty-driver-JIT cold start is a single first-start observation (n=1). Relative σ is under 3% for cold start, decode, APC-off prefill, and inference and APC-off runner recovery; it is up to 6% for the cached APC repeats and 11% for runner recovery with APC on (two slow crash detections). The kill-inference cell was re-run on 2026-11-03 after removing a 0.5 s harness wait that had landed inside the recovery interval; `run.log` shows the original run.

## Results

| Measurement | Result |
| --- | --- |
| Cold start (`WORKER_START_REQUESTED` → `WORKER_READY`) | **20.5 s** with weights in page cache, **22.2 s** from NVMe. **86 s** when the CUDA driver's JIT cache is empty (first start on a host; n=1, diagnostic only; cache state inferred, not recorded, for this run). |
| Prefill, APC off (2K / 8K / 16K) | 90 ms / 452 ms / **1.09 s** (23K / 18K / 15K tok/s) |
| Prefill, APC on, same prefix again | 15 ms / 26 ms / **42 ms** (6.5×, 17.6×, 26× faster than the APC-on first request); vLLM reports all but the last partial 16-token block cached |
| Decode, single stream | **161 tok/s** at 2K context, 129 tok/s at 16K |
| Typical answer length on agent prompts | 660 tokens with thinking off, 1,598 with thinking on (σ ≈ 30%) |
| Recovery, inference process `kill -9` (fault → next token) | **21.7 s**: 0.39 s detection, 20.6 s worker start, 0.61 s re-prefill of 10K tokens |
| Recovery, runner `kill -9`, APC off (fault → next token) | **633 ms**: 7 ms detection, 33 ms runner restart, 1 ms log replay, 584 ms re-prefill |
| Recovery, runner `kill -9`, APC on (fault → next token) | **81 ms**: 7 ms detection, 33 ms runner restart, 1 ms log replay, 31 ms re-prefill (cache hit) |

Phases don't sum exactly to the total; the remaining few milliseconds between boundaries are shown as "Other" in the figure. For the inference kill there is no log replay (the runner survives), so its reconcile phase is only the boundary events.

![Recovery time by fault domain](results/20261103-rtx5090/recovery.png)

Where the 20.5 s cold start goes (vLLM log timestamps for one start, diagnostic): about 0.2 s container start, 3.5 s API-server imports, 5 s engine-process spawn and config, 2 s weight load, 7.7 s engine init (5 s of it CUDA graph capture), and about 2 s of API-server setup before the first healthy `/health`. With an empty driver JIT cache, CUDA graph capture takes 70 s instead of 5 s.

## What this means for the RQ2 scope

These rows are the decision table from `docs/work-distribution.md`. This report records observations only; the project scope stays as set in the plan (single-GPU process restart), and Shirley makes the final call. Neither warm replicas nor decode checkpoints are implemented or measured here.

| If the spike shows | Measured | Supported observation |
| --- | --- | --- |
| Cold start ≫ 16K prefill | Yes: 20.5 s vs 1.09 s (19×) | Worker startup dominates inference-process recovery. Warm-replica failover would target this term; it is future work. |
| Decode throughput ≪ prefill throughput | Yes: 161 vs 15–23K tok/s (about 100×) | A lost decode token costs about 100 prefill tokens, and a thinking answer (about 1,600 tokens) is about 10 s of decode. Decode checkpoints are future work. |
| APC makes a repeated prefix much faster | Yes: 26× at 16K | APC materially changes recovery cost, so every result must report it as a factor |
| Runner-only recovery is almost all reconcile | No. With APC off it's 92% re-prefill; with APC on the whole recovery is 81 ms and reconcile is 1 ms. | Runner recovery cost is re-prefill, and APC sets it (7.8×). The APC-off arm isolates pure re-prefill cost. |

For context, a full restart at step 6 of the scripted task would also redo about 4 s of decode for five completed steps, before counting tool calls. Resume replaces that with about 0.6 s of re-prefill.

## Issues found

1. **The pinned image doesn't start as shipped.** It contains `torchcodec` 0.17.0 built for CUDA 13 (`libnvrtc.so.13`) next to a cu129 torch. The library load raises `OSError`, which `vllm/multimodal/media/audio.py` doesn't catch, so `vllm serve` crashes at import time on any host, including PSC. Fix: [`compat/torchcodec/`](compat/torchcodec/__init__.py) is mounted on `PYTHONPATH` and raises `ImportError` instead. That sends vLLM down its existing "torchcodec unavailable" path. The image itself is unchanged, and audio decoding isn't used.
2. **Driver JIT dominates first starts on Blackwell.** The image's kernels have no native sm_120 code, so the CUDA driver JIT-compiles their PTX, which costs about 65 s per start unless the cache in `/root/.nv` persists. The harness mounts that cache and raises `CUDA_CACHE_MAXSIZE` to 4 GB. Two consequences:
   - Every reported restart time must state whether the driver cache is warm. The harness now records it on every `WORKER_START_REQUESTED`. The cold-start run here predates that field, so its empty-cache label is inferred: only its first start followed the new cache mount, and that start's CUDA graph capture took 70 s instead of 5 s.
   - A fresh host pays this once (86 s in the single first start observed).
3. **Detection of an inference kill takes 387 ms.** That's mostly the `docker kill` CLI round trip before the stream breaks. A supervisor that watches the process directly would see the crash sooner. This doesn't change any conclusion, since detection is under 2% of the 21.7 s.
4. **A harness wait was inside the first kill-inference measurement.** `Server.start` slept 0.5 s for its log follower after `WORKER_READY`, which put 0.5 s of harness time between the new worker being ready and reconciliation. The sleep now happens only where vLLM log phases are read (cold start), and the kill-inference cell was re-run.

## Limits

- This is one consumer GPU that also drives a display, so numbers are reported as a separate platform and never pooled with PSC numbers. SM clock (median 2,835–2,865 MHz at ≥ 90% utilization, except about 2,730 MHz during the short prefill runs) and temperature (peak 86 °C, in kill-runner with APC off) were logged every 200 ms. Throttle reasons were logged only in the kill-inference re-run: software power-cap events in about 10% of samples, no thermal or hardware slowdown. Earlier runs can't rule out throttling.
- Prompts are random token IDs, which give the same attention and MLP cost as text. Generated tokens are greedy with `ignore_eos`.
- Reconcile here is a replay of the local durable log. There's no tool service yet, so a real `query(op_id)` round trip will add to it.
- The agent loop is scripted: 8K prefix, 128 generated plus 256 tool tokens per step, crash after 64 tokens of step 6. Five runs per cell, against the plan's ≥ 20 for reported performance cells.
- Warm replicas and decode checkpoints are out of scope here and are not measured; they are possible future work. A warm replica on this machine would have to share the one GPU with the primary instance.
- The empty-driver-JIT cold start (86 s) was observed once, on the first start; it is diagnostic, not a median.

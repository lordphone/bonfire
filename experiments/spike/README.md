# Week 1 spike (local RTX 5090)

Measures what sets the size of RQ2: cold start, prefill with APC on/off, decode speed, and
recovery after `kill -9` of the inference process or the runner. The report is
[`report.md`](report.md); raw data is under `results/<run-id>/`.

## Run

Prerequisites: Docker with the NVIDIA Container Toolkit, the pinned image pulled by digest
(`infra/environment.lock.json`), the pinned Qwen3-4B snapshot in `~/models/Qwen3-4B`, and
[uv](https://docs.astral.sh/uv/) for the download and for `analyze.py` (it installs the
analyzer's pinned dependencies).

```bash
uvx --from huggingface_hub hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir ~/models/Qwen3-4B
```

```bash
python3 experiments/spike/spike.py all --reps 5 --out experiments/spike/results/$(date +%Y%m%d-%H%M)
```

```bash
uv run experiments/spike/analyze.py experiments/spike/results/<run-id>
```

`spike.py` is stdlib-only, so the runner restart time is not inflated by imports. Each
experiment can also run on its own (`coldstart`, `prefill`, `decode`, `kill-runner`,
`kill-inference`). Keep the GPU otherwise idle while it runs.

## What each experiment does

| Experiment | Server(s) | Measured |
| --- | --- | --- |
| `coldstart` | one start per rep, weights evicted from page cache (NVMe) and cached (RAM) | `WORKER_START_REQUESTED` → `WORKER_READY`; vLLM's own log values for weight load, torch.compile, CUDA graphs |
| `prefill` | APC off, then APC on | time to first token with `max_tokens=1` at 2K / 8K / 16K random-token prefixes; with APC on, the same prefix again |
| `decode` | APC off | 512-token greedy decode at 2K and 16K context; natural-length chat answers to agent prompts with thinking on and off |
| `kill-runner` | APC off, then APC on | 10-step scripted agent loop (8K prefix, 128 generated + 256 tool tokens per step); runner SIGKILLs itself after 64 tokens of step 6; supervisor restarts it, it replays the durable log and re-prefills |
| `kill-inference` | APC off | same loop; the vLLM container is SIGKILLed after 64 tokens of step 6; a fresh container starts and the runner re-prefills |

## Files

- `spike.py`: harness (supervisor, runner, vLLM container control, telemetry emitter).
- `analyze.py`: validates every event against `schemas/telemetry-event.schema.json`, then
  writes `summary.md`, `prefill.png`, and `recovery.png` into the run directory.
- `compat/torchcodec/`: runtime shim for a packaging bug in the pinned image (see the report).

## Telemetry notes

- One JSONL file per process (`supervisor.jsonl`, `runner-attempt-N.jsonl`, `runner.jsonl`),
  each with a gap-free `sequence`. `CLOCK_MONOTONIC` is host-wide, so files from one run merge
  on `monotonic_ns`; `clock_id` carries the boot ID.
- GPU samples (200 ms, with SM clock) go to a separate `gpu.jsonl`, so boundary files stay
  in monotonic order.
- `PREFILL_FINISHED` is the arrival of the first streamed token, so it includes one decode step.

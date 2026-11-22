# GPU matrix on the RTX 5090

How the main GPU matrix runs: the integrated pilot gate and the preregistered tiers in `docs/decisions.md`. The environment itself is in `environment.md`.

## Running it

The supervisor runs inside the locked vLLM container, so the inference worker is a child process of the pinned image and the fault is a real `SIGKILL` of its process tree.

```bash
scripts/gpu_container.sh
```

```bash
docker exec -e PYTHONDONTWRITEBYTECODE=1 bonfire-bench python3 scripts/run_gpu_matrix.py pilot --out experiments/gpu/results/<id>
```

```bash
docker exec -e PYTHONDONTWRITEBYTECODE=1 bonfire-bench python3 scripts/run_gpu_matrix.py tier 1 --out experiments/gpu/results/<id>
```

`gpu_container.sh` mounts the repo at `/work`, the model snapshot at `/models/snapshots/<revision>` (read-only), the driver JIT, Triton and torch.compile caches, and the torchcodec shim (see `environment.md`). It uses `--init`, so vLLM's engine process is reaped when its parent is killed. The snapshot defaults to `~/.cache/bonfire-models/Qwen3-4B`; set `BONFIRE_MODEL_DIR` to use another copy.

A run whose `result.json` exists is skipped, so an interrupted tier resumes where it stopped. Within a tier the outer loop is seeds and the inner loop is prefixes; both methods run back to back on each seed, and the first method alternates by seed.

## One run

`experiments/gpu/matrix.py` (`Run.execute`):

1. Start the C2 tool service and a fresh vLLM worker (APC off, `--gpu-memory-utilization 0.85`).
2. Size the instruction padding so the initial prefix is the target (2,048, 8,192 or 16,384 tokens, within 2 tokens), using the live tokenizer.
3. `TASK_STARTED`, then launch `outcomeserve.runner` with a `DURING_GENERATION` barrier at step `int(20 × progress) + 1` (none for a fault-free run).
4. At the barrier, after the first token of that step has streamed: `FAULT_APPLIED`, `SIGKILL` of the worker tree, release the runner, start a replacement worker.
   - **OutcomeServe:** the runner survives (same epoch), waits for the replacement and replays its durable prefix.
   - **Restart:** the runner exits 4. The supervisor advances the epoch, resets the isolated task (`TOOL_RESET_*`), and starts a new runner from step 1 while the replacement worker loads.
5. `TASK_COMPLETED`, run `verify_business`, then `VERIFIED_COMPLETE` (or `VERIFICATION_FAILED`). Write `result.json` (run-result 1.1.0).

Each run directory keeps the workload, context spec, both event streams, `resources.json` (CPU, RSS and GPU samples every 200 ms), the vLLM log, the runner stderr, the SQLite state and the verifier report.

## Generation is constrained to the request

On the 5090, unconstrained greedy generation from the pilot prompt returns the right JSON object but does not stop after it; with a loose JSON schema it emits the next step's call instead. Following plan section 9 (constrained output, not a model-capability benchmark), the runner constrains each step with vLLM structured outputs to a schema that admits exactly that step's call. Generated tokens are still real model tokens with exact token IDs. A call that still differs is a model error: the runner emits `MODEL_ERROR` and exits 5, separate from protocol failures (exit 1).

The instruction lists only the calls (`tool_name`, `args`). With the CPU workload's step and op IDs and scripted token IDs, a 20-step task alone was 3,646 tokens, which made the 2K cell impossible.

## Metrics

All events in a run share one `clock_id` (the supervisor passes it to the runner), and `perf_counter` on one host is comparable across processes.

| Metric | From → to | Role |
| --- | --- | --- |
| `fault_to_verified_seconds` | supervisor `FAULT_APPLIED` → `VERIFIED_COMPLETE` | primary (crash to correct completion, plan section 6) |
| `task_to_verified_seconds` | `TASK_STARTED` → `VERIFIED_COMPLETE` | whole task; H for fault-free runs |
| `fault_to_first_new_token_seconds` | `FAULT_APPLIED` → runner `FIRST_NEW_TOKEN` | secondary; replacement startup + re-prefill + one decode step |
| `fault_to_replacement_ready_seconds` | `FAULT_APPLIED` → second `INFERENCE_READY` | worker restart cost |

`FIRST_NEW_TOKEN` is observed on the client stream, so it includes one decode step; a prefill/decode split would need serving-side timing.

## Runner event contract

The runner writes measurement-event 1.1.0 records (`schemas/measurement-event.schema.json`) to `runner-events.jsonl`, with `run_id`, `configuration_id` and `clock_id` from the supervisor. Every event's `details` carries `epoch` and `step_id`. Later attempts append to the same file and continue its sequence.

| Event | When | Details |
| --- | --- | --- |
| `RUNNER_STARTED` | attempt starts | `method` |
| `RECONCILIATION_STARTED` / `_FINISHED` | an open `STEP_PREPARED` is reconciled under its op_id | `op_id`, `outcome` |
| `RECOVERY_ABSTAIN` / `RECOVERY_HALTED` | outcome UNKNOWN / task already abstained | `op_id`, `reason` |
| `INFERENCE_LOST` | stream or request failed with no finish_reason | `reason` |
| `INFERENCE_REPLACEMENT_OBSERVED` | replacement worker serves the model | |
| `PREFILL_STARTED` | first generation after a recovery is sent | `committed_prefix_tokens`, `prompt_tokens` |
| `FIRST_NEW_TOKEN` | first streamed token after a recovery | `observed_at` |
| `BOUNDARY_REACHED` / `_RELEASED` | fault barrier | `boundary` |
| `ACKNOWLEDGMENT_LOST` | tool ack lost, outcome queried | |
| `MODEL_ERROR` | generated call differs from the request (exit 5) | `reason` |
| `TASK_COMPLETED` | last step committed (exit 0) | `committed_prefix_tokens` |

`FIRST_NEW_TOKEN` plays the role of the telemetry schema's `NEXT_TOKEN_EMITTED`; there is no separate `PREFILL_FINISHED`, because the client cannot see where prefill ends.

The supervisor's `measurements.jsonl` adds `RUN_STARTED`, `INFERENCE_START_REQUESTED` / `INFERENCE_READY` (with `instance`), `TASK_STARTED`, `RUNNER_LAUNCHED`, `FAULT_APPLIED`, `INFERENCE_KILLED`, `TOOL_RESET_STARTED` / `_FINISHED` (restart only), `TASK_COMPLETED`, `VERIFICATION_STARTED` and `VERIFIED_COMPLETE` / `VERIFICATION_FAILED`.

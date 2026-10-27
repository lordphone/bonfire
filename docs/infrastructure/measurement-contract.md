# Measurement Contract

All timing boundaries are structured `TelemetryEvent` records validated by `schemas/telemetry-event.schema.json`. Console text is diagnostic only and must not be used to reconstruct benchmark intervals.

## Identity and clocks

Every event carries experiment, run, task, step, configuration, model, hardware, and worker identity. It also carries a gap-detecting `sequence`, `accepted_state_mutation`, and the worker's `owner_epoch`. Replacement attempts share a `worker.lineage_id` but receive a new `instance_id`, increasing `attempt`, and increasing `owner_epoch`. A run cannot aggregate across configuration, model revision, hardware identity, worker lineage, or supervisor clock.

`wall_time` is RFC 3339 UTC for correlation. `monotonic_ns` from one supervisor `clock_id` defines durations, including intervals that span worker replacement. Events are persisted in monotonic order; missing, duplicate, or inverted boundaries invalidate the metric.

## Metric boundaries

| Metric | Start event | End event | Notes |
| --- | --- | --- | --- |
| Worker startup | `WORKER_START_REQUESTED` | `WORKER_READY` | Includes process launch, model load, and readiness gate. |
| Crash detection | `FAULT_INJECTED` or `PROCESS_EXITED` | `CRASH_DETECTED` | Report injected and organic failures separately. |
| Reconciliation | `RECONCILIATION_STARTED` | `RECONCILIATION_FINISHED` | End details include `COMMITTED`, `NOT_COMMITTED`, or `UNKNOWN`. |
| Re-prefill | `PREFILL_STARTED` | `PREFILL_FINISHED` | Prefix caching disabled; record committed token count. |
| Crash-to-next-token | `CRASH_DETECTED` | `NEXT_TOKEN_EMITTED` | Undefined for `RECOVERY_ABSTAIN`; report halt latency separately. |
| Failure-free overhead | first task event | terminal task event | Compare matched seeds and configuration, differing only in durable logging. |

Re-prefill throughput is `committed_prefix_tokens / prefill_seconds`. Regenerated tokens count model tokens recomputed by a full-restart baseline; re-prefilled tokens count committed prefix tokens submitted to vLLM. Never combine them.

## Validity rules

A run or metric is invalid when:

- a required boundary event is absent, duplicated, or reversed;
- `configuration_id`, model revision, hardware identity, worker lineage, or `clock_id` changes;
- replacement attempt numbers do not increase or an old fenced attempt emits accepted state mutations;
- a telemetry write gap covers either boundary;
- prefix caching is enabled for a re-prefill run;
- an injected fault misses its stated precondition or fires more than once.

Invalid runs remain in raw data with a reason and are excluded from aggregates. They are never repaired by inferring timestamps from logs.

## GPU sampling and retention

Sample GPU utilization, memory used, power, and temperature every 200 ms during measured GPU runs. Each sample is a `GPU_SAMPLE` event carrying GPU UUID and the same identities as the run. Retain append-only JSONL events, injector decisions, preflight records, stdout/stderr, and tool-state verification results for every run. Aggregated tables must link back to `run_id`; raw records are retained through final-report acceptance and archived with the artifact release.

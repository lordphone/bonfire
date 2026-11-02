# Measurement contract

Verified completion is the endpoint. A normal runner exit or TASK_COMPLETED alone does not establish success.

## Timing and evidence

| Metric | Start | End |
| --- | --- | --- |
| Whole-task time | TASK_STARTED | VERIFIED_COMPLETE |
| Primary recovery time | FAULT_APPLIED | VERIFIED_COMPLETE |
| Restart reset | TOOL_RESET_STARTED | TOOL_RESET_FINISHED |
| Inference startup/restart | Inference start/restart request | INFERENCE_READY |
| Crash-to-next-token | Fault timestamp | First real streamed token arrival after recovery |

Supervisor measurements share perf_counter_ns, clock_id, configuration_id, and a monotonic sequence; see schemas/measurement-event.schema.json. Missing, duplicate, or inverted boundaries do not define a valid interval. Worker events describe steps and reconciliation; resource samples come from an independent monitor. Console text alone is not timing evidence.

schemas/telemetry-event.schema.json defines serving worker identity/epoch events and invariant checks for runtime integration. Complete runtime telemetry integration has not been validated; experiments do not claim to emit all serving events.

Reset belongs inside both restart completion windows. Startup, reconciliation, and prefill may overlap and must not be added mechanically. Without serving-side prefill evidence, that decomposition remains null; HTTP elapsed time is not prefill time. CPU-only next-token and GPU metrics remain null.

## Correctness and validity

Fault domains are runner, inference, and tool_response. A tool timeout does not establish KV loss, nor does runner-only failure imply cold inference reconstruction. APC is an explicit factor; enabling it does not itself invalidate a run.

Injection must satisfy its preconditions. Retain invalid reasons when preconditions or required evidence are missing. An error after valid injection is a valid failed run; do not remove it or assign successful completion latency. The two faults in DURING_RECOVERY are reported separately from single-fault performance comparisons.

## Resource sampling

An independent process samples owned tool-service, runner, and inference process trees across attempts. Creation identity prevents PID reuse errors. CPU measurements of an existing driver subtract a baseline; exited processes retain their last sample.

Cumulative CPU is user plus system time. Peak host RAM is the maximum sampled sum of RSS, which may double-count shared pages. Sampling may miss CPU immediately before abrupt exit. Retain actual timestamps, target interval, sample_count, unsampled_pids, and measurement_valid. The default CPU sampling interval is 0.05 seconds; exact scheduling is not guaranteed.

GPU measurements require an allocated GPU and NVML, recording UUID, memory, and sampled utilization. Allocation-seconds equal GPU count times the measured reservation window; they represent reservation cost, not active cycles. Retain monitoring failures rather than inventing zero values.

## Statistical endpoints

Delta T is restart minus resume time from fault to verified completion for equivalent tasks, faults, and environments. Positive values favor resume, but a crossover is not assumed. Whole-task costs define failure-free overhead and H/S break-even; report reset cost and next-token latency separately.

Pair by seed and bootstrap differences. Missing or failed pairs suppress the primary estimate. Selected formal conditions target 20 repetitions; p95 is descriptive. Failure probability q is an assumed sensitivity parameter, not a deployment crash rate inferred from task success.
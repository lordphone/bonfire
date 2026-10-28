# Worker Lifecycle Contract

This contract covers fail-stop crashes of the agent runner or inference process. The durable log and tool service remain available, tools execute sequentially, and one fenced runner lineage owns a task.

## States

| State | Meaning | Allowed next states |
| --- | --- | --- |
| `STOPPED` | No worker process owns the attempt. | `STARTING` |
| `STARTING` | Supervisor issued a new fenced attempt and launched its process. | `READY`, `UNHEALTHY`, `FAILED`, `TERMINATING` |
| `READY` | Process, health endpoint, and model identity all passed. | `UNHEALTHY`, `TERMINATING` |
| `UNHEALTHY` | Liveness or bounded health policy failed; no new task work may start. | `TERMINATING`, `FAILED` |
| `TERMINATING` | Supervisor revoked ownership and is stopping the process. | `STOPPED`, `FAILED` |
| `FAILED` | Attempt ended without a clean stop; diagnosis is durable. | `STARTING` with a new attempt ID |

Readiness requires all three conditions: the process is alive, a bounded health request succeeds, and the returned model ID and revision match the environment lock. An open TCP port alone is not readiness.

## Detection and timestamps

The supervisor declares a crash on either:

- an observed process exit; or
- three consecutive failed health requests at one-second intervals after a previously `READY` worker.

The supervisor records every signal using one stable `clock_id`. `PROCESS_EXITED` or the final `HEALTH_CHECK_FAILED` is the failure signal; `CRASH_DETECTED` is emitted when the policy decides replacement is required. Wall time supports cross-process correlation; the supervisor's monotonic time is authoritative for durations.

## Ownership and fencing

Each task has a durable `owner_epoch`. A worker attempt receives `(lineage_id, instance_id, attempt, owner_epoch)`. The durable log and tool service reject mutations from any epoch other than the current one. Before replacement, the supervisor atomically advances `owner_epoch`; the old process therefore cannot resume writes even if it was only partitioned or slow.

A replacement gets a new `instance_id` and incremented `attempt`, but keeps the same `lineage_id`. It must reconcile the durable step before generating or dispatching new work:

- `COMMITTED`: append or reuse the stored result, then resume.
- `NOT_COMMITTED`: execute safely under the same operation ID, then append the result.
- `UNKNOWN`: persist `RECOVERY_ABSTAIN` and stop before any re-execution.

## Restart policy

The supervisor permits at most three replacement attempts per run. Each attempt and reason is durable. It never silently retries a model request or tool operation: retries require a new worker attempt event, and tool resubmission uses the original stable operation ID. Exhaustion ends the run as `FAILED`; it does not convert `UNKNOWN` into a retry.

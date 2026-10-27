# Deterministic Fault-Injection Contract

`FaultInjectionRequest` is defined by `schemas/fault-injection.schema.json`. A request is scheduled before a run and is immutable once its `injection_id` has been accepted.

## Identity and determinism

The tuple `(experiment_id, task_id, step_id, seed, boundary, action, target_worker_id, configuration_id)` fully determines an injection. The controller stores a canonical hash under `injection_id`. Reusing an ID with a different payload is a configuration error; replaying the identical payload reports the existing decision rather than scheduling a second fault. `one_shot` is always `true`.

The target worker acknowledges that `expected_precondition` holds, emits `FAULT_INJECTED`, and applies the action exactly once. If the precondition does not hold at the scheduled boundary, the run is invalid rather than shifting the fault to a later step.

## Boundaries

- `DURING_GENERATION`: after at least one token has been produced but before the generated step is durably prepared.
- `PRE_DISPATCH`: after the tool call and stable operation ID are prepared, before the tool service receives it.
- `POST_TOOL_COMMIT_PRE_RESULT_LOG`: after the tool service atomically commits the effect and result, before the runner durably appends that result.
- `POST_CONTEXT_COMMIT`: after the tool result is part of the committed transcript, before the next generation step.
- `DURING_RECOVERY`: after replacement begins reconciliation, before a terminal outcome is committed.

## Actions

`TERMINATE_PROCESS` sends a fail-stop termination to the targeted runner or inference process. The injector records the signal and does not kill the durable log or tool service.

`DROP_ACKNOWLEDGMENT` is legal only at `POST_TOOL_COMMIT_PRE_RESULT_LOG`. The tool service first commits the effect and stored result under the stable operation ID. The injector then suppresses only the response path to the runner. It never rolls back, repeats, or edits durable tool state. Recovery must query or resubmit that same operation ID; if neither query nor deduplication can establish a unique outcome, it records `UNKNOWN -> RECOVERY_ABSTAIN`.

This contract does not simulate storage failure, network-wide partitions, GPU cache replication, or sandbox rollback.

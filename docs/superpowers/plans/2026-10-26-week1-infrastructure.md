# Week 1 Infrastructure Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a reproducible serving environment plus machine-checkable worker, fault-injection, and performance-measurement contracts for Lordphone Wen's Week 1 ownership.

**Architecture:** The canonical GPU environment is an immutable vLLM container and immutable Qwen model revision. Human-readable design documents define lifecycle and measurement semantics; JSON Schema contracts and examples make fault and telemetry events independently testable before runtime code exists.

**Tech Stack:** vLLM 0.32.0, Python 3.12, PyTorch 2.13.0, CUDA 12.9.1, Qwen3-4B, JSON Schema 2020-12, Python 3.12, pytest 9.1.1, jsonschema 4.26.0.

**Spec:** `proposal/Team_33_llmsys_proposal_v2.pdf`

## Global Constraints

- Preserve fixed-model crash-consistent serving scope; do not add RL, direct KV replication, sandbox rollback, or general exactly-once claims.
- Pin the linux/amd64 vLLM image by digest, not a mutable tag alone.
- Pin Qwen3-4B by Hugging Face commit revision.
- Disable vLLM prefix caching for re-prefill measurements so cache hits do not contaminate recovery cost.
- Every fault and telemetry event carries `schema_version`, experiment/task/step identity, monotonic and wall-clock timestamps, worker identity, and configuration identity.
- Acknowledgment loss drops only the response path; it must never undo or repeat the tool-side effect.
- Measurement boundaries are event-defined, not inferred later from console logs.

## Review Focus

- Unsupported GPU or driver: preflight fails before launching vLLM and records the observed environment.
- Reused injection identity with different payload: schema/validator rejects the configuration.
- Acknowledgment-loss injection: tool commit is preserved while only the acknowledgment is suppressed.
- Non-monotonic event ordering: validator rejects negative or inverted duration boundaries.
- Mixed configuration or worker identity in one run: validator rejects the trace instead of aggregating it.

---

### Task 1: Reproducible environment lock

**Files:**
- Create: `infra/environment.lock.json`
- Create: `docs/infrastructure/environment.md`
- Create: `pyproject.toml`

**Interfaces:**
- Produces: immutable image/model identifiers and host preflight requirements consumed by all later tasks.

- [x] Record exact vLLM image tag, multi-arch digest, amd64 digest, release commit, Python, PyTorch, CUDA, model revision, precision, and context length.
- [x] Document canonical A100/L40S requirements, direct-driver baseline, compatibility-mode exception, and CPU-only contract-test path.
- [x] Pin `pytest==9.1.1` and `jsonschema==4.26.0` for contract validation.
- [x] Validate JSON syntax with `python -m json.tool infra/environment.lock.json`.

### Task 2: Worker lifecycle contract

**Files:**
- Create: `docs/infrastructure/worker-lifecycle.md`

**Interfaces:**
- Consumes: canonical environment from Task 1.
- Produces: worker states, health rules, ownership rules, restart policy, and structured lifecycle events.

- [x] Define `STOPPED`, `STARTING`, `READY`, `UNHEALTHY`, `TERMINATING`, and `FAILED` states.
- [x] Define readiness as process alive plus successful health request plus model identity match.
- [x] Define crash detection using process exit and bounded health failures, with explicit timestamps.
- [x] Define single-runner ownership and replacement fencing requirements.
- [x] Define restart attempt identity and the no-silent-retry rule.

### Task 3: Fault-injection contract

**Files:**
- Create: `schemas/fault-injection.schema.json`
- Create: `examples/fault-injection.request.json`
- Create: `docs/infrastructure/fault-injection.md`

**Interfaces:**
- Produces: `FaultInjectionRequest` with deterministic identity and an enumerated crash boundary/action.

- [x] Define boundaries for generation, pre-dispatch, post-tool-commit/pre-result-log, post-context-commit, and recovery.
- [x] Define actions for process termination and acknowledgment loss.
- [x] Require experiment/task/step, seed, target worker, expected precondition, and one-shot semantics.
- [x] State that acknowledgment loss suppresses the response after the tool service commits and never alters durable tool state.
- [x] Validate the example against the schema.

### Task 4: Measurement contract

**Files:**
- Create: `schemas/telemetry-event.schema.json`
- Create: `examples/telemetry-event.json`
- Create: `docs/infrastructure/measurement-contract.md`

**Interfaces:**
- Produces: `TelemetryEvent` and exact event-pair definitions for startup, detection, reconciliation, re-prefill, and crash-to-next-token latency.

- [x] Define required identity, timestamp, worker, hardware, model, and configuration fields.
- [x] Define the event vocabulary and start/end event pairs for each metric.
- [x] Separate wall-clock timestamps for cross-process correlation from monotonic timestamps for durations.
- [x] Define invalid-run rules for missing events, inverted time, configuration drift, worker mismatch, and telemetry gaps.
- [x] Specify GPU sampling cadence and raw-data retention.
- [x] Validate the example against the schema.

### Task 5: Contract validator and tests

**Files:**
- Create: `scripts/validate_week1_contracts.py`
- Create: `tests/test_week1_contracts.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: the environment lock, both schemas, and both examples.
- Produces: a zero-exit validation command for CI and local review.

- [x] Write failing tests for JSON syntax, schema validation, immutable pins, required fault boundaries, timestamp ordering, and configuration identity.
- [x] Run `python -m pytest -q` and confirm the tests fail before the validator exists.
- [x] Implement the validator using `jsonschema.Draft202012Validator` and explicit semantic checks.
- [x] Run `python -m pytest -q` and confirm all tests pass.
- [x] Run `python scripts/validate_week1_contracts.py` and confirm it reports every validated artifact.
- [x] Run `git diff --check` before commit.

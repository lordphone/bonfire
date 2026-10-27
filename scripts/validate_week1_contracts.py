#!/usr/bin/env python3
"""Validate OutcomeServe Week 1 environment, fault, and telemetry contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised when a Week 1 contract is internally inconsistent."""


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def validate_environment_lock(lock: dict[str, Any]) -> None:
    validate_instance(load_json("schemas/environment-lock.schema.json"), lock)
    try:
        image = lock["vllm"]["image"]
        amd64_digest = lock["vllm"]["image_amd64_digest"]
        index_digest = lock["vllm"]["image_index_digest"]
        release_commit = lock["vllm"]["release_commit"]
        revision = lock["model"]["revision"]
        prefix_caching = lock["serving"]["enable_prefix_caching"]
    except KeyError as exc:
        raise ContractError(f"environment lock missing {exc.args[0]}") from exc

    if not SHA256.fullmatch(amd64_digest) or not SHA256.fullmatch(index_digest):
        raise ContractError("vLLM image digests must be immutable sha256 values")
    if not image.endswith(f"@{amd64_digest}"):
        raise ContractError("vLLM image must be pinned to the linux/amd64 digest")
    if not GIT_SHA.fullmatch(release_commit) or not GIT_SHA.fullmatch(revision):
        raise ContractError("release and model revisions must be 40-character Git commits")
    if prefix_caching is not False:
        raise ContractError("prefix caching must be disabled for re-prefill measurements")


def validate_instance(schema: dict[str, Any], instance: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ContractError(details)


def validate_fault_schedule(requests: list[dict[str, Any]]) -> None:
    schema = load_json("schemas/fault-injection.schema.json")
    payload_by_id: dict[str, str] = {}
    for request in requests:
        validate_instance(schema, request)
        injection_id = request["injection_id"]
        canonical_payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
        previous = payload_by_id.setdefault(injection_id, canonical_payload)
        if previous != canonical_payload:
            raise ContractError(f"injection_id {injection_id!r} was reused with a different payload")


def _validate_interruptible_pairs(events: list[dict[str, Any]], start_type: str, end_type: str) -> None:
    open_start = False
    for event in events:
        event_type = event["event_type"]
        if event_type == start_type:
            if open_start:
                raise ContractError(f"duplicate {start_type} before {end_type}")
            open_start = True
        elif event_type == end_type:
            if not open_start:
                raise ContractError(f"{end_type} has no preceding {start_type}")
            open_start = False
        elif event_type == "CRASH_DETECTED" and open_start:
            open_start = False
    if open_start:
        raise ContractError(f"{start_type} is missing {end_type}")


def _health_failure_signal(events: list[dict[str, Any]]) -> bool:
    failures = [event for event in events if event["event_type"] == "HEALTH_CHECK_FAILED"]
    if len(failures) < 3:
        return False
    last_three = failures[-3:]
    gaps = [
        later["monotonic_ns"] - earlier["monotonic_ns"]
        for earlier, later in zip(last_three, last_three[1:])
    ]
    return all(900_000_000 <= gap <= 1_100_000_000 for gap in gaps)


def _validate_crash_cycles(events: list[dict[str, Any]]) -> None:
    detection_indexes = [index for index, event in enumerate(events) if event["event_type"] == "CRASH_DETECTED"]
    signal_types = {"PROCESS_EXITED", "HEALTH_CHECK_FAILED"}
    has_any_signal = any(
        event["event_type"] in signal_types
        or (event["event_type"] == "FAULT_INJECTED" and event["details"].get("action") == "TERMINATE_PROCESS")
        for event in events
    )
    if has_any_signal and not detection_indexes:
        raise ContractError("failure signal is missing CRASH_DETECTED")

    previous_detection = -1
    for detection_index in detection_indexes:
        window = events[previous_detection + 1 : detection_index]
        direct_signal = any(
            event["event_type"] == "PROCESS_EXITED"
            or (event["event_type"] == "FAULT_INJECTED" and event["details"].get("action") == "TERMINATE_PROCESS")
            for event in window
        )
        if not direct_signal and not _health_failure_signal(window):
            raise ContractError("CRASH_DETECTED is missing a process exit, termination fault, or three timed health failures")
        previous_detection = detection_index

    if detection_indexes:
        terminal_indexes = [
            index
            for index, event in enumerate(events)
            if index > detection_indexes[-1] and event["event_type"] in {"NEXT_TOKEN_EMITTED", "RECOVERY_ABSTAIN"}
        ]
        if len(terminal_indexes) != 1:
            raise ContractError("final CRASH_DETECTED requires one NEXT_TOKEN_EMITTED or RECOVERY_ABSTAIN")


def _validate_reconciliation_outcomes(events: list[dict[str, Any]]) -> None:
    pending_unknown: str | None = None
    halted = False
    for event in events:
        event_type = event["event_type"]
        if halted and (event_type == "NEXT_TOKEN_EMITTED" or event["accepted_state_mutation"]):
            raise ContractError("RECOVERY_ABSTAIN must halt before generation or state mutation")

        if event_type == "RECONCILIATION_FINISHED":
            outcome = event["details"]["outcome"]
            if pending_unknown is not None:
                raise ContractError("UNKNOWN must be followed by matching RECOVERY_ABSTAIN")
            if outcome == "UNKNOWN":
                pending_unknown = event["details"]["operation_id"]
        elif event_type == "RECOVERY_ABSTAIN":
            operation_id = event["details"]["operation_id"]
            if pending_unknown is None or operation_id != pending_unknown:
                raise ContractError("RECOVERY_ABSTAIN requires a matching UNKNOWN reconciliation outcome")
            pending_unknown = None
            halted = True
        elif pending_unknown is not None and (event_type == "NEXT_TOKEN_EMITTED" or event["accepted_state_mutation"]):
            raise ContractError("UNKNOWN must be followed by RECOVERY_ABSTAIN before resume or mutation")

    if pending_unknown is not None:
        raise ContractError("UNKNOWN must be followed by matching RECOVERY_ABSTAIN")


def _validate_step_boundaries(events: list[dict[str, Any]]) -> None:
    _validate_interruptible_pairs(events, "RECONCILIATION_STARTED", "RECONCILIATION_FINISHED")
    _validate_interruptible_pairs(events, "PREFILL_STARTED", "PREFILL_FINISHED")
    _validate_crash_cycles(events)
    _validate_reconciliation_outcomes(events)


def validate_trace(events: list[dict[str, Any]]) -> None:
    if not events:
        raise ContractError("trace must contain at least one telemetry event")

    schema = load_json("schemas/telemetry-event.schema.json")
    for event in events:
        validate_instance(schema, event)

    for field in ("experiment_id", "run_id", "task_id", "configuration_id", "clock_id"):
        values = {event[field] for event in events}
        if len(values) != 1:
            raise ContractError(f"trace contains mixed {field} values")

    lineage_ids = {event["worker"]["lineage_id"] for event in events}
    if len(lineage_ids) != 1:
        raise ContractError("trace contains mixed worker lineage_id values")

    hardware = {json.dumps(event["hardware"], sort_keys=True) for event in events}
    model = {json.dumps(event["model"], sort_keys=True) for event in events}
    if len(hardware) != 1:
        raise ContractError("trace contains mixed hardware identity")
    if len(model) != 1:
        raise ContractError("trace contains mixed model identity")

    monotonic_values = [event["monotonic_ns"] for event in events]
    if monotonic_values != sorted(monotonic_values):
        raise ContractError("trace has non-monotonic monotonic_ns values")

    sequences = [event["sequence"] for event in events]
    if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
        raise ContractError("trace has a telemetry sequence gap or is out of order")

    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ContractError("trace contains duplicate event_id values")

    active_events: list[dict[str, Any]] = []
    previous = events[0]["worker"]
    active_events.append(events[0])
    for event in events[1:]:
        current = event["worker"]
        if current["owner_epoch"] < previous["owner_epoch"]:
            if event["accepted_state_mutation"]:
                raise ContractError("stale worker owner_epoch emitted an accepted state mutation")
            continue
        if current["instance_id"] == previous["instance_id"]:
            if current["attempt"] != previous["attempt"] or current["owner_epoch"] != previous["owner_epoch"]:
                raise ContractError("one worker instance changed attempt or owner_epoch")
        elif current["attempt"] <= previous["attempt"] or current["owner_epoch"] <= previous["owner_epoch"]:
            raise ContractError("replacement worker must increase attempt and owner_epoch")
        previous = current
        active_events.append(event)

    by_step: dict[str, list[dict[str, Any]]] = {}
    for event in active_events:
        by_step.setdefault(event["step_id"], []).append(event)
    for step_events in by_step.values():
        _validate_step_boundaries(step_events)

    by_instance: dict[str, list[dict[str, Any]]] = {}
    for event in active_events:
        by_instance.setdefault(event["worker"]["instance_id"], []).append(event)
    for instance_events in by_instance.values():
        _validate_interruptible_pairs(instance_events, "WORKER_START_REQUESTED", "WORKER_READY")


def main() -> int:
    lock = load_json("infra/environment.lock.json")
    validate_environment_lock(lock)
    print("validated environment lock")

    validate_instance(load_json("schemas/environment-lock.schema.json"), lock)
    print("validated environment lock schema")

    pairs = [
        ("schemas/fault-injection.schema.json", "examples/fault-injection.request.json", "fault injection example"),
        ("schemas/telemetry-event.schema.json", "examples/telemetry-event.json", "telemetry example"),
    ]
    for schema_path, example_path, label in pairs:
        validate_instance(load_json(schema_path), load_json(example_path))
        print(f"validated {label}")

    validate_fault_schedule([load_json("examples/fault-injection.request.json")])
    print("validated fault schedule semantics")
    validate_trace([load_json("examples/telemetry-event.json")])
    print("validated telemetry trace semantics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

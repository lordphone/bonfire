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


def _validate_pair(events: list[dict[str, Any]], start_type: str, end_type: str) -> None:
    starts = [index for index, event in enumerate(events) if event["event_type"] == start_type]
    ends = [index for index, event in enumerate(events) if event["event_type"] == end_type]
    if not starts and not ends:
        return
    if len(starts) != 1:
        raise ContractError(f"expected exactly one {start_type}, found {len(starts)}")
    if len(ends) != 1:
        raise ContractError(f"expected exactly one {end_type}, found {len(ends)}")
    if starts[0] >= ends[0]:
        raise ContractError(f"{start_type} must precede {end_type}")


def _validate_step_boundaries(events: list[dict[str, Any]]) -> None:
    _validate_pair(events, "RECONCILIATION_STARTED", "RECONCILIATION_FINISHED")
    _validate_pair(events, "PREFILL_STARTED", "PREFILL_FINISHED")

    failure_signals = [
        index
        for index, event in enumerate(events)
        if event["event_type"] == "PROCESS_EXITED"
        or (
            event["event_type"] == "FAULT_INJECTED"
            and event["details"].get("action") == "TERMINATE_PROCESS"
        )
    ]
    detections = [index for index, event in enumerate(events) if event["event_type"] == "CRASH_DETECTED"]
    if failure_signals or detections:
        if not failure_signals:
            raise ContractError("CRASH_DETECTED is missing FAULT_INJECTED or PROCESS_EXITED")
        if len(detections) != 1:
            raise ContractError(f"expected exactly one CRASH_DETECTED, found {len(detections)}")
        if min(failure_signals) >= detections[0]:
            raise ContractError("failure signal must precede CRASH_DETECTED")

        terminals = [
            index
            for index, event in enumerate(events)
            if event["event_type"] in {"NEXT_TOKEN_EMITTED", "RECOVERY_ABSTAIN"}
        ]
        if len(terminals) != 1:
            raise ContractError("CRASH_DETECTED requires one NEXT_TOKEN_EMITTED or RECOVERY_ABSTAIN")
        if detections[0] >= terminals[0]:
            raise ContractError("CRASH_DETECTED must precede its recovery terminal event")


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

    previous = events[0]["worker"]
    for event in events[1:]:
        current = event["worker"]
        if current["owner_epoch"] < previous["owner_epoch"]:
            raise ContractError("worker owner_epoch decreased after replacement")
        if current["instance_id"] == previous["instance_id"]:
            if current["attempt"] != previous["attempt"] or current["owner_epoch"] != previous["owner_epoch"]:
                raise ContractError("one worker instance changed attempt or owner_epoch")
        elif current["attempt"] <= previous["attempt"] or current["owner_epoch"] <= previous["owner_epoch"]:
            raise ContractError("replacement worker must increase attempt and owner_epoch")
        previous = current

    by_step: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_step.setdefault(event["step_id"], []).append(event)
    for step_events in by_step.values():
        _validate_step_boundaries(step_events)

    by_instance: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_instance.setdefault(event["worker"]["instance_id"], []).append(event)
    for instance_events in by_instance.values():
        _validate_pair(instance_events, "WORKER_START_REQUESTED", "WORKER_READY")


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

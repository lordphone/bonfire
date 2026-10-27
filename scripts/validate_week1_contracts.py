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

    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ContractError("trace contains duplicate event_id values")


def main() -> int:
    lock = load_json("infra/environment.lock.json")
    validate_environment_lock(lock)
    print("validated environment lock")

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

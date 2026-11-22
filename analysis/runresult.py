"""C6 run results for the main GPU matrix, assembled only from retained evidence.

Owner: Victoria · Consumers: all result owners (see docs/run-result.md).

A GPU run directory is produced on the benchmark machine. This module turns its
retained evidence into one schema-valid ``result.json`` without rerunning or
editing anything:

* measurement events (``schemas/measurement-event.schema.json``), possibly from
  several processes, which must all share one ``clock_id``;
* the business verifier output (``outcomeserve.verify.business.verify_business``);
* the resource monitor summary;
* the run identity (``identity.json``, see ``IDENTITY_FIELDS``).

Timing intervals follow docs/measurement.md. A missing, duplicate or inverted
boundary yields ``None`` and never a guessed value. Classification never turns
a failed run into an excluded one: an error after a valid injection stays a
valid failed run with an ``outcome`` that the analysis counts per method.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from outcomeserve.contracts import validate_schema

SCHEMA_VERSION = "1.2.0"

IDENTITY_FIELDS = (
    "run_id", "seed", "method", "tier", "no_fault", "fault_domain", "boundary",
    "prefix_target_tokens", "requested_progress", "actual_progress", "apc", "method_order",
    "configuration_id", "context_spec_id", "business_fixture_sha256", "hardware_id", "gpu_name",
    "driver_version", "runtime_versions", "environment_lock_sha256",
)
OPTIONAL_IDENTITY_FIELDS = ("actual_prefix_tokens_at_fault",)

# Exit codes of the recovering runner (C3). Anything else is a protocol or
# infrastructure failure, distinguished below by the verifier.
EXIT_COMPLETED, EXIT_ABSTAINED, EXIT_MODEL_ERROR = 0, 3, 5


def pair_key(identity: dict) -> str:
    """Seed first, then the cell; ``no_fault`` last so break-even can strip it.

    Tier and method order are not part of the cell: they must match (tier) or
    differ (order) within a pair, which the analysis checks separately.
    """
    progress = "none" if identity["no_fault"] else identity["requested_progress"]
    boundary = "none" if identity["no_fault"] else identity["boundary"]
    domain = "none" if identity["no_fault"] else identity["fault_domain"]
    # Pilot gate runs reuse matrix seeds and cells but are never matrix samples.
    backend = "gpu-pilot" if identity["tier"] == "pilot" else "gpu"
    return (f"s{identity['seed']}:{backend}:L{identity['prefix_target_tokens']}:p{progress}:{domain}:{boundary}"
            f":apc-{str(identity['apc']).lower()}:{bool(identity['no_fault'])}")


def read_events(paths: Iterable[str | Path]) -> list[dict]:
    events = []
    for path in paths:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not JSON") from exc
            validate_schema("measurement-event", event)
            events.append(event)
    return events


def _single(events: list[dict], name: str) -> int | None:
    stamps = [e["monotonic_ns"] for e in events if e["event_type"] == name]
    return stamps[0] if len(stamps) == 1 else None


def _interval(start: int | None, end: int | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return (end - start) / 1e9


def timings(events: list[dict]) -> dict[str, float | None]:
    """Intervals from docs/measurement.md, on one shared clock."""
    task = _single(events, "TASK_STARTED")
    fault = _single(events, "FAULT_APPLIED")
    verified = _single(events, "VERIFIED_COMPLETE")
    # The first new token after the fault is the recovery endpoint; tokens of
    # earlier steps are not. Take the earliest one after the fault.
    after = sorted(e["monotonic_ns"] for e in events
                   if e["event_type"] == "FIRST_NEW_TOKEN" and fault is not None and e["monotonic_ns"] >= fault
                   and (verified is None or e["monotonic_ns"] <= verified))
    return {
        "task_to_verified_seconds": _interval(task, verified),
        "fault_to_verified_seconds": _interval(fault, verified),
        "fault_to_first_new_token_seconds": _interval(fault, after[0] if after else None),
        "reset_seconds": _interval(_single(events, "TOOL_RESET_STARTED"), _single(events, "TOOL_RESET_FINISHED")),
        "inference_restart_seconds": _interval(_single(events, "INFERENCE_RESTART_REQUESTED"),
                                               _single(events, "INFERENCE_REPLACEMENT_READY")),
    }


def _check_identity(identity: dict) -> list[str]:
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    unknown = sorted(set(identity) - set(IDENTITY_FIELDS) - set(OPTIONAL_IDENTITY_FIELDS))
    problems = [f"identity missing {', '.join(missing)}"] if missing else []
    if unknown:
        problems.append(f"identity has unknown fields {', '.join(unknown)}")
    return problems


def assemble(identity: dict, events: list[dict], verification: dict | None, resources: dict | None, *,
             runner_exit_code: int | None, failure_reason: str | None = None) -> dict:
    """Build and validate one C6 result. Raises if the identity itself is unusable."""
    problems = _check_identity(identity)
    if problems:
        raise ValueError("; ".join(problems))
    for event in events:
        validate_schema("measurement-event", event)
    invalid: list[str] = []
    clocks = {e["clock_id"] for e in events}
    runs = {e["run_id"] for e in events}
    if len(clocks) > 1:
        invalid.append("measurement events use more than one clock_id")
    if runs and runs != {identity["run_id"]}:
        invalid.append("measurement events belong to another run_id")
    configurations = {e["configuration_id"] for e in events}
    if configurations and configurations != {identity["configuration_id"]}:
        invalid.append("measurement events belong to another configuration_id")
    names = [e["event_type"] for e in events]
    if names.count("TASK_STARTED") != 1:
        invalid.append("exactly one TASK_STARTED required")
    fault_count = names.count("FAULT_APPLIED")
    if identity["no_fault"] and fault_count:
        invalid.append("fault recorded in a no-fault run")
    if not identity["no_fault"] and fault_count != 1:
        # Precondition not reached (or reached twice): the fault cell is not
        # established, so this is invalid rather than a failed recovery.
        invalid.append(f"one-fault run recorded {fault_count} FAULT_APPLIED events")
    task_stamp = _single(events, "TASK_STARTED")
    fault_stamp = _single(events, "FAULT_APPLIED")
    if task_stamp is not None and fault_stamp is not None and fault_stamp < task_stamp:
        invalid.append("FAULT_APPLIED precedes TASK_STARTED")
    fault_applied = fault_count == 1 and not identity["no_fault"]
    measured = timings(events) if not invalid else dict.fromkeys(timings([]), None)

    metrics = (verification or {}).get("metrics", {})
    duplicates = metrics.get("duplicate_effects")
    verified_ok = bool(verification and verification.get("task_success")) and duplicates == 0
    resources = resources if isinstance(resources, dict) and "measurement_valid" in resources else {
        "measurement_valid": False, "error": "resource summary missing"}

    if invalid:
        outcome = "INVALID"
    elif runner_exit_code == EXIT_MODEL_ERROR:
        outcome = "MODEL_ERROR"
    elif verification is None:
        outcome = "INFRASTRUCTURE_FAILURE"
    elif (duplicates is None or duplicates > 0 or metrics.get("audit_complete") is False
          or metrics.get("tool_state_mismatches")):
        # A committed result that disagrees with the oracle is a violation even
        # after a halt; only the unfinished final inventory is expected then.
        outcome = "RECOVERY_VIOLATION"
    elif runner_exit_code == EXIT_ABSTAINED:
        # Query capability does not establish availability at the decision time.
        # Keep abstention distinct; a final snapshot cannot prove a false abstain.
        outcome = "ABSTAINED"
    elif metrics.get("inventory_mismatches"):
        outcome = "RECOVERY_VIOLATION"
    elif (runner_exit_code == EXIT_COMPLETED and verified_ok
          and measured["task_to_verified_seconds"] is not None
          and (identity["no_fault"] or measured["fault_to_verified_seconds"] is not None)):
        outcome = "COMPLETED"
    elif runner_exit_code == EXIT_COMPLETED and verified_ok:
        outcome = "INFRASTRUCTURE_FAILURE"
        failure_reason = failure_reason or "completion timing boundaries missing, duplicated or inverted"
    else:
        outcome = "INFRASTRUCTURE_FAILURE"
    passed = outcome == "COMPLETED"
    if passed and not resources.get("measurement_valid"):
        # A run without a complete measurement window is not a usable timing sample.
        outcome, passed, failure_reason = "INFRASTRUCTURE_FAILURE", False, "resource sampling incomplete"
    valid = outcome != "INVALID"
    if not passed and failure_reason is None:
        failure_reason = "; ".join(invalid) or {
            "MODEL_ERROR": "runner reported a model error",
            "RECOVERY_VIOLATION": "business verification found inconsistent or incomplete effect evidence",
            "ABSTAINED": "runner abstained; decision correctness is not established by a final snapshot",
            "INFRASTRUCTURE_FAILURE": "runner, verification or completion evidence incomplete",
        }.get(outcome)

    gpu = {key: identity[key] for key in ("tier", "prefix_target_tokens", "method_order", "gpu_name", "driver_version")}
    if "actual_prefix_tokens_at_fault" in identity:
        gpu["actual_prefix_tokens_at_fault"] = identity["actual_prefix_tokens_at_fault"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "backend": "vllm", "purpose": "performance", "steps": 20,
        **{key: identity[key] for key in IDENTITY_FIELDS if key not in gpu},
        "pair_key": pair_key(identity), "gpu_matrix": gpu,
        "valid": valid, "passed": passed, "outcome": outcome, "task_success": passed,
        "fault_applied": fault_applied, "fault_count": fault_count, "invalid_reasons": invalid,
        "failure_reason": None if passed else failure_reason, "runner_exit_code": runner_exit_code,
        "metrics": metrics, "resources": resources, **measured,
    }
    for key, value in measured.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{key} is not finite")
    validate_schema("run-result", result)
    return result


def assemble_directory(run_dir: str | Path, *, event_files: Iterable[str] = ("measurements.jsonl",)) -> dict:
    """Assemble from the retained files of one run directory (see docs/run-result.md)."""
    run_dir = Path(run_dir)
    load = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8")) if (run_dir / name).exists() else None
    identity = load("identity.json")
    if identity is None:
        raise FileNotFoundError(run_dir / "identity.json")
    outcome = load("runner-exit.json") or {}
    verification = load("verification.json")
    if isinstance(verification, dict) and "business" in verification:
        verification = verification["business"]
    files = [run_dir / name for name in event_files if (run_dir / name).exists()]
    return assemble(identity, read_events(files), verification, load("resources.json"),
                    runner_exit_code=outcome.get("exit_code"), failure_reason=outcome.get("failure_reason"))

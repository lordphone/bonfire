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

SCHEMA_VERSION = "1.3.0"

IDENTITY_FIELDS = (
    "run_id", "task_id", "seed", "method", "tier", "no_fault", "fault_domain", "boundary",
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


RAW_FIELDS = {"event_type", "monotonic_ns", "clock_id"}


def _from_raw(raw: dict, identity: dict, sequence: int) -> dict:
    """A raw runner event (event_type, monotonic_ns, clock_id, other details) as a
    measurement event of this run. The clock is kept, never rewritten."""
    for field in ("run_id", "configuration_id", "task_id"):
        if field in raw and raw[field] != identity[field]:
            raise ValueError(f"raw runner event belongs to another {field}")
    return {"schema_version": "1.1.0", "run_id": identity["run_id"], "configuration_id": identity["configuration_id"],
            "clock_id": raw["clock_id"], "sequence": sequence, "event_type": raw["event_type"],
            "monotonic_ns": raw["monotonic_ns"], "details": {k: v for k, v in raw.items() if k not in RAW_FIELDS}}


def read_events(paths: Iterable[str | Path], identity: dict | None = None) -> list[dict]:
    """Measurement events from each file. With ``identity``, lines in the raw runner
    format (no ``schema_version``) are converted; sequences stay local to each file."""
    events = []
    for path in paths:
        sequence = 0
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not JSON") from exc
            sequence += 1
            if identity is not None and isinstance(event, dict) and "schema_version" not in event \
                    and RAW_FIELDS <= set(event):
                event = _from_raw(event, identity, sequence)
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


def _protocol_problem(identity, protocol, exit_code):
    if identity["method"] != "outcomeserve":
        return None
    if not isinstance(protocol, dict):
        return "OutcomeServe protocol report missing"
    if protocol.get("valid") is not True:
        return "protocol evidence is unreadable or invalid"
    for key, expected in (("task_id", identity["task_id"]), ("context_spec_id", identity["context_spec_id"]),
                          ("token_source", "model")):
        if protocol.get(key) != expected:
            return f"protocol report {key} mismatch"
    if protocol.get("checkpoint") not in {"completed", "halted", "in_progress"}:
        return "protocol checkpoint missing or unsupported"
    required = {EXIT_COMPLETED: "completed", EXIT_ABSTAINED: "halted"}.get(exit_code)
    if required and protocol["checkpoint"] != required:
        return f"runner exit {exit_code} requires protocol checkpoint {required}"
    if type(protocol.get("consistency_pass")) is not bool or not isinstance(protocol.get("issues"), list):
        return "protocol consistency verdict missing"
    if protocol["consistency_pass"] and protocol["issues"]:
        return "protocol verdict contradicts its issues"
    if exit_code == EXIT_COMPLETED and protocol["consistency_pass"] and protocol.get("task_success") is not True:
        return "protocol does not establish completed context"
    # Optional extra evidence. The model-mode log check already proves the replayed
    # prompt extends the durable committed prefix (PROMPT_TOKENS), so it is not required.
    replay = protocol.get("replay")
    if replay is not None and (not isinstance(replay, dict) or type(replay.get("valid")) is not bool
                               or replay.get("task_id") != identity["task_id"]
                               or replay.get("context_spec_id") != identity["context_spec_id"]):
        return "replay evidence is malformed or belongs to another task/context"
    return None


def assemble(identity: dict, events: list[dict], verification: dict | None, resources: dict | None, *,
             runner_exit_code: int | None, failure_reason: str | None = None, protocol: dict | None = None,
             evidence_errors: Iterable[str] = ()) -> dict:
    """Build and validate one C6 result. Raises if the identity itself is unusable.

    ``verification`` is the business verifier report. Successful OutcomeServe
    results require an identity-matched completed model-mode protocol report."""
    problems = _check_identity(identity)
    if problems:
        raise ValueError("; ".join(problems))
    if protocol is not None and identity["method"] != "outcomeserve":
        raise ValueError("only OutcomeServe runs keep a durable log to verify")
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

    business_report = verification if isinstance(verification, dict) else {}
    metrics = business_report.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    duplicates = metrics.get("duplicate_effects")
    verified_ok = business_report.get("task_success") is True and duplicates == 0
    resources = resources if isinstance(resources, dict) and "measurement_valid" in resources else {
        "measurement_valid": False, "error": "resource summary missing"}
    evidence_errors = list(evidence_errors)
    protocol_problem = _protocol_problem(identity, protocol, runner_exit_code)

    if invalid:
        outcome = "INVALID"
    elif runner_exit_code == EXIT_MODEL_ERROR:
        outcome = "MODEL_ERROR"
    elif business_report.get("valid") is not True or business_report.get("task_id") != identity["task_id"]:
        outcome = "INFRASTRUCTURE_FAILURE"
        failure_reason = failure_reason or "business report missing, invalid or belongs to another task_id"
    elif evidence_errors:
        outcome = "INFRASTRUCTURE_FAILURE"
        failure_reason = "; ".join(evidence_errors)
    elif (duplicates is None or duplicates > 0 or metrics.get("audit_complete") is False
          or metrics.get("tool_state_mismatches")
          or (protocol_problem is None and protocol is not None and (
              protocol.get("consistency_pass") is False or (isinstance(protocol.get("replay"), dict)
                                                           and protocol["replay"].get("valid") is False)))):
        # A committed result that disagrees with the oracle is a violation even
        # after a halt; only the unfinished final inventory is expected then.
        outcome = "RECOVERY_VIOLATION"
    elif protocol_problem:
        outcome = "INFRASTRUCTURE_FAILURE"
        failure_reason = protocol_problem
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
            "RECOVERY_VIOLATION": "business or protocol verification found inconsistent or incomplete evidence",
            "ABSTAINED": "runner abstained; decision correctness is not established by a final snapshot",
            "INFRASTRUCTURE_FAILURE": "runner, verification or completion evidence incomplete",
        }.get(outcome)
    protocol_report = protocol if isinstance(protocol, dict) else {}

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
        "evidence_errors": evidence_errors,
        "protocol_issues": None if protocol is None else [i.get("code", "UNKNOWN") for i in (protocol_report.get("issues") or [])
                                                         if isinstance(i, dict)],
        "protocol_verification": None if protocol is None else {key: protocol_report.get(key) for key in
            ("valid", "task_id", "context_spec_id", "token_source", "checkpoint", "consistency_pass", "task_success")},
    }
    if result["protocol_verification"] is not None:
        replay = protocol_report.get("replay")
        result["protocol_verification"]["replay_valid"] = replay.get("valid") if isinstance(replay, dict) else None
    for key, value in measured.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{key} is not finite")
    validate_schema("run-result", result)
    return result


def assemble_directory(run_dir: str | Path, *, event_files: Iterable[str] = ("measurements.jsonl",),
                       verification_file: str = "verification.json") -> dict:
    """Assemble from the retained files of one run directory (see docs/run-result.md)."""
    run_dir = Path(run_dir)
    load = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8")) if (run_dir / name).exists() else None
    identity = load("identity.json")
    if identity is None:
        raise FileNotFoundError(run_dir / "identity.json")
    outcome = load("runner-exit.json") or {}
    verification, protocol = load(verification_file), None
    if isinstance(verification, dict) and "business" in verification:
        verification, protocol = verification["business"], verification.get("protocol")
    files, missing = [], []
    for name in event_files:
        path = run_dir / name
        if not path.is_file():
            missing.append(f"required event file missing: {name}")
        else:
            files.append(path)
    return assemble(identity, read_events(files, identity), verification, load("resources.json"),
                    runner_exit_code=outcome.get("exit_code"), failure_reason=outcome.get("failure_reason"),
                    protocol=protocol, evidence_errors=missing)

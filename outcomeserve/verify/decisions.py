"""Recovery-decision metrics from the runner's event trace (RQ1).

The snapshot verifiers deliberately leave ``unsafe_retry`` and ``false_abstain``
unset: a final database state cannot show *when* a runner resubmitted or why it
halted. This module derives them from evidence that can: the ordered worker
events of every attempt, the business verifier's view of which steps took
effect, and the case's expected behavior, which the experiment fixes in advance
(it controls whether the tool supports outcome queries and whether the tool
service is reachable).

Definitions
-----------
unsafe_retry
    A ``TOOL_SUBMIT`` of a non-idempotent step that had already been submitted,
    with no authoritative ``NOT_COMMITTED`` reconciliation since that earlier
    submission. Retrying an idempotent write or a read is never unsafe.
false_abstain
    The runner abstained although the case expects completion (the outcome was
    establishable).
missed_abstain
    The case expects abstention (the outcome of a non-idempotent call could not
    be established) but the runner went on without abstaining. This counts the
    decision, whether or not it happened to cause a duplicate.
continued_after_abstain
    Submissions or completions after the abstention, or later steps that took
    effect (see ``_continued_after_abstain``).

A trace that cannot support these metrics (missing, out of order, mixed clocks,
an abstention without its event) makes them ``None`` and the run not correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

EXPECTED_BEHAVIORS = ("complete", "abstain")
# Independent reference semantics, not read from the tool service: repeating
# these changes business state again.
NON_IDEMPOTENT_TOOLS = frozenset({"decrement_inventory", "notify_supplier"})


def read_worker_events(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_behavior(fault_tool: str, boundary: str, fault_domain: str) -> str:
    """Abstain exactly when a non-queryable call may have been sent and its outcome is unknown."""
    if fault_tool != "external":
        return "complete"
    ambiguous = fault_domain == "tool_response" or boundary in {"PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG"}
    return "abstain" if ambiguous else "complete"


def _non_idempotent(workload: dict) -> set[str]:
    return {step["step_id"] for step in workload["steps"] if step["tool_name"] in NON_IDEMPOTENT_TOOLS}


def unsafe_retries(events: Iterable[dict], workload: dict) -> int:
    risky = _non_idempotent(workload)
    submitted, cleared, unsafe = set(), set(), 0
    for event in events:
        step = event.get("step_id")
        if step not in risky:
            continue
        if event.get("event_type") == "RECONCILIATION_FINISHED" and event.get("outcome") == "NOT_COMMITTED":
            cleared.add(step)
        elif event.get("event_type") == "TOOL_SUBMIT":
            if step in submitted and step not in cleared:
                unsafe += 1
            submitted.add(step)
            cleared.discard(step)
    return unsafe


def _evidence_errors(events, order, took_effect, abstained, abstained_step):
    """Reasons the trace cannot support decision metrics; empty when it can."""
    errors = []
    if not events:
        errors.append("worker event trace missing")
    stamps = [e.get("monotonic_ns") for e in events]
    epochs = [e.get("epoch") for e in events]
    clocks = {e.get("clock_id") for e in events}
    if (any(type(s) is not int or s < 0 for s in stamps) or any(type(n) is not int or n < 1 for n in epochs)
            or any(not isinstance(c, str) or not c for c in clocks)):
        errors.append("worker events lack clock/epoch identity")
    elif stamps != sorted(stamps) or epochs != sorted(epochs) or len(clocks) != 1:
        errors.append("worker event order or clock identity is inconsistent")
    if any(e.get("step_id") not in order for e in events):
        errors.append("event refers to an unknown step")
    submitted = {e.get("step_id") for e in events if e.get("event_type") == "TOOL_SUBMIT"}
    if not set(took_effect) <= submitted:
        errors.append("effects lack TOOL_SUBMIT evidence")
    halts = [e for e in events if e.get("event_type") == "RECOVERY_ABSTAIN"]
    if abstained and (len(halts) != 1 or halts[0].get("step_id") != abstained_step):
        errors.append("abstention lacks a unique matching RECOVERY_ABSTAIN event")
    if halts and not abstained:
        errors.append("runner did not remain halted after RECOVERY_ABSTAIN")
    return errors


def _continued_after_abstain(events, order, took_effect, abstained_step):
    """Violations after a halt: later submissions/completions in the trace, or later
    workload steps in the effect audit (whichever is larger, to avoid double counting)."""
    halt = next((i for i, e in enumerate(events) if e.get("event_type") == "RECOVERY_ABSTAIN"), None)
    if halt is None:
        return 0
    in_trace = sum(e.get("event_type") in {"TOOL_SUBMIT", "STEP_COMPLETED"} for e in events[halt + 1:])
    if abstained_step not in order:
        return in_trace
    position = order.index(abstained_step)
    in_audit = sum(order.index(step) > position for step in took_effect if step in order)
    return max(in_trace, in_audit)


def decision_metrics(events: list[dict], workload: dict, business: dict, *, expected: str,
                     runner_exit_code: int | None, abstained_step: str | None) -> dict:
    if expected not in EXPECTED_BEHAVIORS:
        raise ValueError("expected behavior must be complete or abstain")
    abstained = runner_exit_code == 3
    order = [step["step_id"] for step in workload["steps"]]
    metrics = business.get("metrics", {})
    took_effect = metrics.get("steps_with_effects") or []
    errors = _evidence_errors(events, order, took_effect, abstained, abstained_step)
    continued = _continued_after_abstain(events, order, took_effect, abstained_step)
    unsafe = unsafe_retries(events, workload)
    duplicates = metrics.get("duplicate_effects")
    result = {
        "expected_behavior": expected,
        "observed_behavior": "abstain" if abstained else "complete" if runner_exit_code == 0 else "failed",
        "unsafe_retry": unsafe,
        "false_abstain": int(abstained and expected == "complete"),
        "missed_abstain": int(expected == "abstain" and not abstained),
        "continued_after_abstain": continued,
        "duplicate_effects": duplicates,
        "evidence_valid": not errors,
        "evidence_errors": errors,
    }
    safe = (not errors and duplicates == 0 and unsafe == 0 and continued == 0
            and metrics.get("prefix_contiguous") is True and metrics.get("prefix_state_mismatches") == 0
            and metrics.get("tool_state_mismatches") == 0 and metrics.get("audit_complete") is True)
    if expected == "complete":
        behaved = runner_exit_code == 0 and bool(business.get("task_success"))
    else:
        behaved = abstained
    result["correct"] = bool(safe and behaved)
    if errors:
        for key in ("unsafe_retry", "false_abstain", "missed_abstain", "continued_after_abstain"):
            result[key] = None
    return result

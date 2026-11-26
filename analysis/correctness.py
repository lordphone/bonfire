"""RQ1 correctness matrix: case list, aggregation and exit criteria (work distribution §6).

Owner: Victoria. Every number here is computed from retained ``result.json``
files; nothing is typed in by hand. Invalid runs stay in the tables.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

METHODS = ("outcomeserve", "restart", "naive_retry", "transcript_checkpoint")
FAULT_TOOLS = ("decrement", "idempotent", "external")
RUNNER_BOUNDARIES = ("DURING_GENERATION", "PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "POST_CONTEXT_COMMIT")
TOOL_LABELS = {"decrement": "decrement (non-idempotent, C2)", "idempotent": "set_reorder_level (idempotent, C2)",
               "external": "notify_supplier (non-idempotent, unqueryable)"}


def cases(steps=(5, 10, 20), seeds=(33, 34, 35), progress=0.5):
    """Every cell of the CPU matrix, in a fixed order."""
    if (not steps or len(set(steps)) != len(steps)
            or any(type(n) is not int or n not in {5, 10, 20} for n in steps)):
        raise ValueError("steps must be distinct values from 5, 10, 20")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if type(progress) not in (int, float) or not math.isfinite(progress) or not 0 <= progress < 1:
        raise ValueError("progress must be finite and in [0,1)")
    out = []
    for n in steps:
        for seed in seeds:
            for tool in FAULT_TOOLS:
                for boundary in RUNNER_BOUNDARIES:
                    for method in METHODS:
                        out.append(dict(steps=n, seed=seed, progress=progress, fault_tool=tool,
                                        fault_domain="runner", boundary=boundary, method=method))
                for method in METHODS:
                    out.append(dict(steps=n, seed=seed, progress=progress, fault_tool=tool,
                                    fault_domain="tool_response", boundary="POST_TOOL_COMMIT_PRE_RESULT_LOG",
                                    method=method))
                if tool != "external":  # an external call has no reconciliation query to crash during
                    out.append(dict(steps=n, seed=seed, progress=progress, fault_tool=tool, fault_domain="runner",
                                    boundary="DURING_RECOVERY", method="outcomeserve"))
    return out


def case_name(case):
    return (f"n{case['steps']}-s{case['seed']}-{case['fault_tool']}-{case['fault_domain']}-"
            f"{case['boundary']}-{case['method']}")


ROW_FIELDS = ("case", "steps", "seed", "fault_tool", "fault_domain", "boundary", "method", "valid",
              "expected_behavior", "observed_behavior", "correct", "duplicate_effects", "unsafe_retry",
              "false_abstain", "missed_abstain", "continued_after_abstain", "actual_progress", "reason")


def row(case, result):
    decisions = result.get("decisions") or {}
    reason = ("; ".join(result.get("invalid_reasons") or []) or result.get("failure_reason")
              or "; ".join(decisions.get("evidence_errors") or []) or "")
    return {"case": case_name(case), **{k: case[k] for k in ("steps", "seed", "fault_tool", "fault_domain",
                                                            "boundary", "method")},
            "valid": bool(result.get("valid")), "expected_behavior": decisions.get("expected_behavior"),
            "observed_behavior": decisions.get("observed_behavior"), "correct": bool(result.get("correct")),
            **{k: decisions.get(k) for k in ("duplicate_effects", "unsafe_retry", "false_abstain", "missed_abstain",
                                             "continued_after_abstain")},
            "actual_progress": result.get("actual_progress"), "reason": reason}


def _sum(rows, key):
    values = [r.get(key) for r in rows]
    return None if any(value is None for value in values) else sum(values)


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["fault_tool"], r["fault_domain"], r["boundary"], r["method"]), []).append(r)
    table = []
    timeline = RUNNER_BOUNDARIES + ("DURING_RECOVERY",)
    for (tool, domain, boundary, method), members in sorted(groups.items(), key=lambda item: (
            FAULT_TOOLS.index(item[0][0]), item[0][1] != "runner", timeline.index(item[0][2]),
            METHODS.index(item[0][3]))):
        valid = [r for r in members if r["valid"]]
        table.append({"fault_tool": tool, "fault_domain": domain, "boundary": boundary, "method": method,
                      "runs": len(members), "invalid": len(members) - len(valid),
                      "unknown_metric_runs": sum(any(r.get(key) is None for key in
                          ("duplicate_effects", "unsafe_retry", "false_abstain", "missed_abstain", "continued_after_abstain"))
                          for r in valid),
                      "expected": "/".join(sorted({r["expected_behavior"] for r in valid if r["expected_behavior"]})),
                      "correct": sum(r["correct"] for r in valid),
                      "runs_with_duplicates": sum((r["duplicate_effects"] or 0) > 0 for r in valid),
                      "duplicate_effects": _sum(valid, "duplicate_effects"), "unsafe_retry": _sum(valid, "unsafe_retry"),
                      "false_abstain": _sum(valid, "false_abstain"), "missed_abstain": _sum(valid, "missed_abstain"),
                      "continued_after_abstain": _sum(valid, "continued_after_abstain")})
    return table


def exit_criteria(rows, *, expected_cases=None):
    """Work distribution §6 exit criteria, evaluated for OutcomeServe on valid runs."""
    ours = [r for r in rows if r["method"] == "outcomeserve"]
    valid = [r for r in ours if r["valid"]]
    supported = [r for r in valid if r["fault_tool"] in {"decrement", "idempotent"}]
    recovery = [r for r in valid if r["boundary"] == "DURING_RECOVERY"]
    checks = {
        "no invalid OutcomeServe runs": (len(ours) - len(valid)) == 0,
        "zero duplicate effects for supported (C2) tools": _sum(supported, "duplicate_effects") == 0,
        "zero unsafe retry, and abstention whenever an outcome is UNKNOWN":
            _sum(valid, "unsafe_retry") == 0 and _sum(valid, "missed_abstain") == 0,
        "no continuation after RECOVERY_ABSTAIN": _sum(valid, "continued_after_abstain") == 0,
        "no false abstention when the outcome is establishable": _sum(valid, "false_abstain") == 0,
        "repeated crash during recovery preserves the invariants": bool(recovery) and all(r["correct"] for r in recovery),
        "every OutcomeServe run correct": bool(valid) and all(r["correct"] for r in valid),
    }
    if expected_cases is not None:
        names = [r["case"] for r in rows]
        checks["complete declared matrix without duplicate or missing cases"] = (
            len(names) == len(set(names)) and set(names) == {case_name(case) for case in expected_cases})
        checks["all declared fault injections valid, including baselines"] = bool(rows) and all(r["valid"] for r in rows)
    return {"checks": checks, "all_pass": all(checks.values()), "outcomeserve_runs": len(ours),
            "model_errors": "not applicable: CPU runs use scripted traces without model inference"}


def write_outputs(rows, output: Path, *, config: dict):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    table = summarize(rows)
    criteria = exit_criteria(rows, expected_cases=cases(tuple(config["steps"]), tuple(config["seeds"]), config["progress"]))
    (output / "summary.json").write_text(json.dumps({"config": config, "cells": table, "exit_criteria": criteria},
                                                    indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(table, criteria, config), encoding="utf-8")
    return table, criteria


def render_report(table, criteria, config):
    lines = ["# CPU correctness matrix (RQ1)", "",
             f"Generated from {sum(r['runs'] for r in table)} retained runs: steps {config['steps']}, "
             f"seeds {config['seeds']}, requested progress {config['progress']}. "
             "Every count comes from `runs.csv`; invalid runs are listed, never dropped.", "",
             "## Exit criteria (OutcomeServe)", ""]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in criteria["checks"].items()]
    lines += [f"- Model errors: {criteria['model_errors']}", ""]
    for tool in FAULT_TOOLS:
        rows = [r for r in table if r["fault_tool"] == tool]
        if not rows:
            continue
        lines += [f"## Fault step: {TOOL_LABELS[tool]}", "",
                  "| Domain | Boundary | Method | Expected | Correct / runs | Runs with duplicates | Duplicate effects "
                  "| Unsafe retries | Missed abstain | False abstain | Continued after abstain | Invalid |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            r = {key: "unknown" if value is None else value for key, value in r.items()}
            lines.append(f"| {r['fault_domain']} | {r['boundary']} | {r['method']} | {r['expected']} | "
                         f"{r['correct']} / {r['runs']} | {r['runs_with_duplicates']} | {r['duplicate_effects']} | "
                         f"{r['unsafe_retry']} | {r['missed_abstain']} | {r['false_abstain']} | "
                         f"{r['continued_after_abstain']} | {r['invalid']} |")
        lines.append("")
    lines += ["## Definitions", "",
              "- *Correct*: the run behaved as the case requires (complete and verified, or abstain when an "
              "unqueryable call's outcome is unknown) with no duplicate effect, no unsafe retry, no effect after an "
              "abstention, and state consistent with the executed steps.",
              "- *Unsafe retry*: a resubmission of a non-idempotent step after an earlier submission, without an "
              "authoritative NOT_COMMITTED in between (from worker events, not the final snapshot).",
              "- *Missed abstain*: the case requires abstention but the runner continued, whether or not a "
              "duplicate resulted.", ""]
    return "\n".join(lines)

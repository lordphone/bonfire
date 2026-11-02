"""Paired completion/resource differences and hypothetical task-failure curves."""

import math
import random
from statistics import mean, median


def _metric(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if type(value) in (int, float) and math.isfinite(value) else None


def paired_comparison(records, *, metric="fault_to_verified_seconds", bootstrap_samples=2000, seed=33):
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("positive bootstrap sample count required")
    pairs = {}
    for record in records:
        if record.get("method") not in {"restart", "outcomeserve"}:
            continue
        key = record["pair_key"]
        pair = pairs.setdefault(key, {})
        if record["method"] in pair:
            raise ValueError("duplicate method within a pair; repetition IDs must be unique")
        pair[record["method"]] = record
    differences, incomplete = [], []
    for key, pair in sorted(pairs.items()):
        if set(pair) == {"restart", "outcomeserve"}:
            fields = ("backend", "steps", "actual_progress", "fault_domain", "boundary", "purpose", "no_fault",
                      "context_spec_id", "business_fixture_sha256", "hardware_id", "runtime_versions", "environment_lock_sha256", "apc")
            mismatched = [field for field in fields if pair["restart"].get(field) != pair["outcomeserve"].get(field)]
            if mismatched:
                incomplete.append({"pair_key": key, "reason": "mismatched setting: " + ",".join(mismatched)})
                continue
        values = {}
        for method in ("restart", "outcomeserve"):
            record = pair.get(method)
            if record is None or not record.get("valid") or not record.get("passed"):
                incomplete.append({"pair_key": key, "method": method, "reason": "missing, invalid or failed task"})
                break
            if metric.startswith("resources.") and not record.get("resources", {}).get("measurement_valid"):
                incomplete.append({"pair_key": key, "method": method, "reason": "resource sampling incomplete"})
                break
            value = _metric(record, metric)
            if value is None:
                incomplete.append({"pair_key": key, "method": method, "reason": "metric unavailable"})
                break
            values[method] = value
        if len(values) == 2:
            differences.append(values["restart"] - values["outcomeserve"])
    estimate = None
    # Never headline a success-only latency advantage when the input has failures.
    if differences and not incomplete:
        interval = None
        if len(differences) >= 2:
            rng = random.Random(seed)
            samples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(bootstrap_samples))
            interval = [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]
        estimate = {"mean_restart_minus_outcomeserve": mean(differences),
                    "median_restart_minus_outcomeserve": median(differences), "bootstrap_ci95": interval}
    return {"metric": metric, "total_pairs": len(pairs), "complete_pairs": len(differences),
            "incomplete_pairs": incomplete, "estimate": estimate,
            "all_run_count": len(records), "failed_or_invalid_runs": sum(not r.get("valid") or not r.get("passed") for r in records),
            "note": "positive difference favors OutcomeServe; no CI for a single pair; p95 is not a primary claim"}


def break_even_curve(overhead, one_fault_saving, probabilities=(0, 0.01, 0.1, 0.5, 1)):
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (overhead, one_fault_saving)):
        raise ValueError("finite H and S required")
    if any(type(q) not in (int, float) or not 0 <= q <= 1 for q in probabilities):
        raise ValueError("q is a hypothetical task probability in [0,1]")
    threshold = overhead / (overhead + one_fault_saving) if overhead >= 0 and one_fault_saving > 0 else None
    return {"H": overhead, "S": one_fault_saving, "break_even_q": threshold,
            "assumptions": "at most one fault per task, same fault-position distribution; q is not an observed or hourly rate",
            "source_metric": "whole-task task_to_verified_seconds, not fault_to_verified_seconds",
            "curve": [{"q": q, "expected_saving": q * one_fault_saving - (1 - q) * overhead} for q in probabilities]}


def whole_task_break_even(records):
    """H/S both come from whole-task correctly verified completion times."""
    clean = paired_comparison([r for r in records if r.get("no_fault")], metric="task_to_verified_seconds")
    fault = paired_comparison([r for r in records if not r.get("no_fault")], metric="task_to_verified_seconds")
    if not clean["estimate"] or not fault["estimate"]:
        return {"curve": None, "reason": "complete successful no-fault and one-fault pairs required"}
    return break_even_curve(-clean["estimate"]["mean_restart_minus_outcomeserve"],
                            fault["estimate"]["mean_restart_minus_outcomeserve"])

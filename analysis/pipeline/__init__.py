"""Paired completion/resource differences and hypothetical task-failure curves."""

import math
import random
from statistics import mean, median

from outcomeserve.contracts import validate_schema


PAIR_FIELDS = ("backend", "steps", "actual_progress", "fault_domain", "boundary", "purpose", "no_fault",
               "context_spec_id", "business_fixture_sha256", "hardware_id", "runtime_versions",
               "environment_lock_sha256", "apc")
# C6 1.3.0 main-matrix runs also pair on the cell and the platform identity.
GPU_PAIR_FIELDS = ("seed", "requested_progress", "gpu_matrix.tier", "gpu_matrix.prefix_target_tokens",
                   "gpu_matrix.gpu_name", "gpu_matrix.driver_version")
_MISSING = object()


def _field(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _is_gpu_matrix(record):
    return record.get("schema_version") == "1.3.0" and record.get("backend") == "vllm"


def _metric(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if type(value) in (int, float) and math.isfinite(value) else None


def paired_comparison(records, *, metric="fault_to_verified_seconds", bootstrap_samples=2000, seed=33):
    """Retain every input pair; invalid, failed or missing members suppress estimates.

    Complete-pair counts are diagnostic when estimate is null, not a success-only
    alternative estimate. Invalid runs are never silently filtered or replaced.
    """
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
            fields = PAIR_FIELDS + (GPU_PAIR_FIELDS if _is_gpu_matrix(pair["restart"]) or _is_gpu_matrix(pair["outcomeserve"]) else ())
            missing = [field for field in fields if _field(pair["restart"], field) is _MISSING
                       or _field(pair["outcomeserve"], field) is _MISSING]
            if missing:
                incomplete.append({"pair_key": key, "reason": "missing setting: " + ",".join(missing)})
                continue
            mismatched = [field for field in fields if _field(pair["restart"], field) != _field(pair["outcomeserve"], field)]
            if mismatched:
                incomplete.append({"pair_key": key, "reason": "mismatched setting: " + ",".join(mismatched)})
                continue
            if _is_gpu_matrix(pair["restart"]):
                orders = {_field(pair[m], "gpu_matrix.method_order") for m in pair}
                if orders != {1, 2}:
                    incomplete.append({"pair_key": key, "reason": "methods of a pair must run in positions 1 and 2"})
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
            "invalid_runs": sum(not r.get("valid") for r in records),
            "valid_failed_runs": sum(bool(r.get("valid")) and not r.get("passed") for r in records),
            "note": "invalid, failed or missing pairs suppress the estimate; positive difference favors OutcomeServe; no CI for a single pair; p95 is not a primary claim"}


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


OUTCOMES = ("COMPLETED", "ABSTAINED", "MODEL_ERROR", "RECOVERY_VIOLATION", "INFRASTRUCTURE_FAILURE", "INVALID")


def outcome_of(record):
    """C6 outcome; older 1.1.0 records are classified from their flags without guessing more."""
    if not record.get("valid"):
        return "INVALID"
    if record.get("outcome") in OUTCOMES:
        return record["outcome"]
    if record.get("passed"):
        return "COMPLETED"
    if record.get("model_or_backend_error") or record.get("model_error"):
        return "MODEL_ERROR"
    duplicates = (record.get("metrics") or {}).get("duplicate_effects")
    if duplicates:
        return "RECOVERY_VIOLATION"
    return "INFRASTRUCTURE_FAILURE"


def outcome_counts(records):
    """Failure counts and reasons per method: never compare successful runs only."""
    table = {}
    for record in records:
        row = table.setdefault(record.get("method"), {**dict.fromkeys(OUTCOMES, 0), "reasons": {}})
        outcome = outcome_of(record)
        row[outcome] += 1
        if outcome != "COMPLETED":
            reason = "; ".join(record.get("invalid_reasons") or []) or record.get("failure_reason") or "unspecified"
            row["reasons"][reason] = row["reasons"].get(reason, 0) + 1
    return table


def descriptive_p95(records, metric):
    """Nearest-rank p95 of completed runs per method; descriptive only at n=20."""
    out = {}
    for method in sorted({r.get("method") for r in records}):
        values = sorted(v for r in records if r.get("method") == method and outcome_of(r) == "COMPLETED"
                        for v in [_metric(r, metric)] if v is not None)
        out[method] = None if not values else {"n": len(values), "p95": values[math.ceil(0.95 * len(values)) - 1]}
    return out


def _gpu_cell(record):
    gpu = record["gpu_matrix"]
    return gpu["prefix_target_tokens"], record["requested_progress"]


def gpu_matrix_analysis(records, *, h_precision_fraction=0.25, bootstrap_samples=2000, seed=33):
    """Main-matrix (C6 1.3.0, vLLM) analysis by prefix length L and progress f.

    * Delta T(L, f): restart minus OutcomeServe, fault to verified completion (primary).
    * Crash-to-first-new-token difference (secondary) and reset time, reported separately.
    * H(L) from no-fault pairs and S(L, f) from one-fault pairs, both whole-task
      task_to_verified_seconds; break-even q = H / (H + S) with q an assumed probability.
    * Tier 1H precision: extend 1H by 10 seeds when the full width of H's 95% CI
      exceeds ``h_precision_fraction`` of the smallest Tier 1 (50%) saving S.

    Cells are never pooled across platforms; a mixed GPU set is refused.
    """
    gpu_rows = [r for r in records if _is_gpu_matrix(r)]
    for record in gpu_rows:
        validate_schema("run-result", record)
    # Pilot runs gate the matrix; they are counted but never become matrix samples.
    pilot = [r for r in gpu_rows if r["gpu_matrix"]["tier"] == "pilot"]
    rows = [r for r in gpu_rows if r["gpu_matrix"]["tier"] != "pilot"]
    platforms = {(r["gpu_matrix"]["gpu_name"], r["gpu_matrix"]["driver_version"]) for r in rows}
    if len(platforms) > 1:
        raise ValueError(f"GPU results from different platforms are never pooled: {sorted(platforms)}")
    # H and S must describe the same deployment, not just the same prefix size.
    # Seed/fixture and prefix/progress vary intentionally across the matrix.
    configuration_fields = ("hardware_id", "context_spec_id", "runtime_versions", "environment_lock_sha256",
                            "apc", "fault_domain", "boundary")
    for field in configuration_fields:
        if rows and any(_field(row, field) != _field(rows[0], field) for row in rows[1:]):
            raise ValueError(f"GPU results with different {field} are never pooled; analyze configurations separately")
    compare = lambda subset, metric: paired_comparison(subset, metric=metric,
                                                      bootstrap_samples=bootstrap_samples, seed=seed)
    fault_cells, clean_cells = {}, {}
    for record in rows:
        if record["no_fault"]:
            clean_cells.setdefault(record["gpu_matrix"]["prefix_target_tokens"], []).append(record)
        else:
            fault_cells.setdefault(_gpu_cell(record), []).append(record)

    overhead = {}
    for prefix, subset in sorted(clean_cells.items()):
        comparison = compare(subset, "task_to_verified_seconds")
        estimate = comparison["estimate"]
        interval = estimate and estimate["bootstrap_ci95"]
        overhead[prefix] = {
            "H_seconds": None if not estimate else -estimate["mean_restart_minus_outcomeserve"],
            "H_ci95": None if not interval else [-interval[1], -interval[0]],
            "complete_pairs": comparison["complete_pairs"], "incomplete_pairs": comparison["incomplete_pairs"],
            "outcomes": outcome_counts(subset)}

    cells = []
    for (prefix, progress), subset in sorted(fault_cells.items()):
        primary = compare(subset, "fault_to_verified_seconds")
        whole = compare(subset, "task_to_verified_seconds")
        saving = whole["estimate"] and whole["estimate"]["mean_restart_minus_outcomeserve"]
        h = overhead.get(prefix, {}).get("H_seconds")
        break_even = None
        if h is not None and saving is not None:
            break_even = break_even_curve(h, saving)
        cells.append({
            "prefix_target_tokens": prefix, "requested_progress": progress,
            "tiers": sorted({r["gpu_matrix"]["tier"] for r in subset}),
            "delta_T_fault_to_verified": primary,
            "whole_task_saving_S": whole,
            "secondary_first_new_token": compare(subset, "fault_to_first_new_token_seconds"),
            "reset_seconds_restart": descriptive_p95([r for r in subset if r["method"] == "restart"], "reset_seconds"),
            "p95_fault_to_verified": descriptive_p95(subset, "fault_to_verified_seconds"),
            "resources": {metric: compare(subset, metric) for metric in (
                "resources.process_cpu_seconds", "resources.gpu_allocation_seconds",
                "resources.peak_summed_rss_bytes", "resources.peak_gpu_memory_bytes")},
            "outcomes": outcome_counts(subset),
            "break_even": break_even})

    tier1_cells = [c for c in cells if "1" in c["tiers"]]
    tier1 = [c["whole_task_saving_S"]["estimate"]["mean_restart_minus_outcomeserve"] for c in tier1_cells
             if c["whole_task_saving_S"]["estimate"]]
    widths = {prefix: value["H_ci95"][1] - value["H_ci95"][0] for prefix, value in overhead.items() if value["H_ci95"]}
    precision = None
    required_prefixes = {2048, 8192, 16384}
    if (len(tier1) == len(tier1_cells) and len(tier1) == 3 and min(tier1) > 0
            and {c["prefix_target_tokens"] for c in tier1_cells} == required_prefixes
            and set(widths) == required_prefixes):
        threshold = h_precision_fraction * min(tier1)
        precision = {"rule": f"extend Tier 1H by 10 seeds if full 95% CI width of H > {h_precision_fraction} x smallest Tier 1 S",
                     "smallest_tier1_S_seconds": min(tier1), "threshold_seconds": threshold,
                     "H_ci95_width_seconds": widths,
                     "extend_tier_1H": any(width > threshold for width in widths.values())}
    return {"platform": None if not platforms else dict(zip(("gpu_name", "driver_version"), next(iter(platforms)))),
            "run_count": len(rows), "overhead_H": overhead, "cells": cells, "tier_1H_precision": precision,
            "outcomes_all_runs": outcome_counts(rows), "pilot_runs_excluded": outcome_counts(pilot),
            "note": "positive Delta T and S favor OutcomeServe; unmeasured cells are absent, never filled with predictions"}

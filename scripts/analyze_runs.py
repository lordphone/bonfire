#!/usr/bin/env python3
"""Analyze result files with complete-pair gating and separate total/recovery time."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.pipeline import gpu_matrix_analysis, outcome_counts, paired_comparison, whole_task_break_even
from outcomeserve.contracts import validate_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--results", nargs="+", type=Path)
    inputs.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve previous analysis")
    paths = args.results if args.results else sorted(args.results_root.rglob("result.json"))
    if not paths:
        parser.error("no result.json files found; no empty success analysis")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for record in records:
        validate_schema("run-result", record)
    # Validate grouping before creating output files, including the H/S linkage.
    gpu = [r for r in records if r.get("schema_version") == "1.2.0" and r.get("backend") == "vllm"]
    matrix = gpu_matrix_analysis(gpu) if gpu else None
    # Different settings are never pooled: pair_key without the seed identifies a cell.
    cells = {}
    for record in records:
        setting = record["pair_key"].split(":", 1)[1]
        cells.setdefault(setting, []).append(record)
    result = {setting: {metric: paired_comparison(rows, metric=metric) for metric in (
        "task_to_verified_seconds", "fault_to_verified_seconds", "resources.process_cpu_seconds",
        "resources.gpu_allocation_seconds")} for setting, rows in cells.items()}
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    settings = {}
    for record in records:
        key = record["pair_key"].split(":", 1)[1].rsplit(":", 1)[0]
        settings.setdefault(key, []).append(record)
    break_even = {setting: whole_task_break_even(rows) for setting, rows in settings.items()}
    (args.output / "break-even.json").write_text(json.dumps(break_even, indent=2) + "\n", encoding="utf-8")
    with (args.output / "failure-probability-curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "hypothetical_q", "expected_whole_task_saving_seconds"])
        for setting, values in break_even.items():
            for point in values.get("curve") or []:
                writer.writerow([setting, point["q"], point["expected_saving"]])
    with (args.output / "heatmap-cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "metric", "total_pairs", "complete_pairs", "mean_difference", "ci95", "incomplete_count"])
        for setting, metrics in result.items():
            for metric, comparison in metrics.items():
                estimate = comparison["estimate"] or {}
                writer.writerow([setting, metric, comparison["total_pairs"], comparison["complete_pairs"],
                    estimate.get("mean_restart_minus_outcomeserve"), estimate.get("bootstrap_ci95"), len(comparison["incomplete_pairs"])])
    outcomes = {setting: outcome_counts(rows) for setting, rows in cells.items()}
    (args.output / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    if matrix is not None:
        (args.output / "gpu-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
        with (args.output / "gpu-heatmap.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["prefix_target_tokens", "requested_progress", "tiers", "complete_pairs", "total_pairs",
                             "delta_T_mean_seconds", "delta_T_ci95_low", "delta_T_ci95_high",
                             "S_whole_task_mean_seconds", "H_seconds", "break_even_q", "non_completed_runs"])
            for cell in matrix["cells"]:
                primary = cell["delta_T_fault_to_verified"]
                estimate = primary["estimate"] or {}
                interval = estimate.get("bootstrap_ci95") or [None, None]
                saving = cell["whole_task_saving_S"]["estimate"] or {}
                h = matrix["overhead_H"].get(cell["prefix_target_tokens"], {}).get("H_seconds")
                failures = sum(count for row in cell["outcomes"].values()
                               for key, count in row.items() if key not in {"COMPLETED", "reasons"})
                writer.writerow([cell["prefix_target_tokens"], cell["requested_progress"], "+".join(cell["tiers"]),
                                 primary["complete_pairs"], primary["total_pairs"],
                                 estimate.get("mean_restart_minus_outcomeserve"), interval[0], interval[1],
                                 saving.get("mean_restart_minus_outcomeserve"), h,
                                 (cell["break_even"] or {}).get("break_even_q"), failures])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

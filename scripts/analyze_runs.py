#!/usr/bin/env python3
"""Analyze result files with complete-pair gating and separate total/recovery time."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.pipeline import paired_comparison, whole_task_break_even


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
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

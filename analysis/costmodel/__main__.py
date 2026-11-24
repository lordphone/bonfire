"""Fit the cost model and write costmodel.json plus a readable costmodel.md.

    python -m analysis.costmodel --spike experiments/spike/results/20261103-rtx5090 \
        --matrix experiments/gpu/results/20261122-rtx5090/runs.csv --out run-data/costmodel

With --matrix (v2) the logging cost is fitted to the matrix's fault-free cells and
every cell is compared with its measured mean; without it (v1) the logging cost is
a constant measured on this host and the task shape is assumed.
"""

import argparse
import json
from pathlib import Path

from analysis.costmodel import (LoggingCost, TaskShape, calibrate_from_spike, fixed_costs_from_cell,
                                matrix_predictions, measure_log_overhead, validate_on_matrix, validate_on_spike)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="analysis.costmodel")
    parser.add_argument("--spike", required=True, help="spike results directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--matrix", help="runs.csv of a GPU matrix (enables v2)")
    parser.add_argument("--pilot-prefix-lengths", help="actual-prefix-lengths.json from a pilot run (v1 only)")
    parser.add_argument("--log-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spike = Path(args.spike)
    calibration = calibrate_from_spike(spike)
    spike_check = validate_on_spike(spike)
    recovery = json.loads((spike / "recovery.json").read_text())
    runner_off = next(rows for name, rows in recovery.items() if name.startswith("Runner kill") and "APC off" in name)
    inference = next(rows for name, rows in recovery.items() if name.startswith("Inference kill"))
    report = {"spike": str(spike), "spike_validation": spike_check}

    if args.matrix:
        matrix_check = validate_on_matrix(spike, Path(args.matrix))
        shape = TaskShape(**matrix_check["task_shape"])
        logging = LoggingCost(**matrix_check["logging_cost"])
        report.update(version="v2", matrix=str(args.matrix), matrix_validation=matrix_check)
        logging_note = (f"Logging cost fitted to the fault-free cells: {logging.base * 1000:.1f} ms per step "
                        f"+ {logging.per_token * 1e6:.1f} µs per committed-prefix token.")
    else:
        overhead = measure_log_overhead(repeats=args.log_repeats)
        shape = (TaskShape.from_pilot(json.loads(Path(args.pilot_prefix_lengths).read_text()))
                 if args.pilot_prefix_lengths else TaskShape())
        logging = overhead["per_step_s"]
        report.update(version="v1", log_overhead=overhead)
        logging_note = f"Logging overhead measured on this host: {logging * 1000:.1f} ms per step (n={overhead['n']})."

    domains = (("inference", False, fixed_costs_from_cell(inference)), ("runner", False, fixed_costs_from_cell(runner_off)),
               ("runner", True, fixed_costs_from_cell(runner_off)))
    report["task_shape"] = shape.__dict__
    report["predictions"] = {f"{d}, APC {'on' if apc else 'off'}": matrix_predictions(
        calibration, fixed, logging=logging, domain=d, apc=apc, shape=shape) for d, apc, fixed in domains}
    (out / "costmodel.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Cost model {report['version']}", "", f"Spike: `{spike}`." + (f" Matrix: `{args.matrix}`." if args.matrix else ""),
             "", "## Recovery prediction vs spike", "", "| Cell | Basis | Measured (s) | Predicted (s) | Error |",
             "| --- | --- | --- | --- | --- |"]
    lines += [f"| {c['cell']} | {c['basis']} | {c['measured_s']:.4f} | {c['predicted_s']:.4f} | {c['relative_error']:+.1%} |"
              for c in spike_check["cells"]]
    lines += ["", logging_note, ""]
    if args.matrix:
        m = report["matrix_validation"]
        lines += ["## Prediction vs RTX 5090 matrix", "",
                  "S = restart − OutcomeServe, fault → verified; mean of paired seeds. v1 lets the steps after the fault cancel; "
                  "v2 charges OutcomeServe its logging on them.", "",
                  "| Cell | Pairs | Measured S | v1 S | v2 S | v2 logging after fault | Measured / predicted fault → first token |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {c['cell']} | {c['pairs']} | {c['measured_S']:.2f} | {c['v1_S']:.2f} | {c['v2_S']:.2f} | "
                  f"{c['v2_logging_after_fault']:.2f} | {c['measured_first_token']:.2f} / {c['predicted_first_token']:.2f} |"
                  for c in m["S"]]
        lines += ["", f"Mean absolute error of S: v1 {m['mean_abs_error_s']['v1']:.2f} s, v2 {m['mean_abs_error_s']['v2']:.2f} s.", "",
                  "| Prefix | Measured H | Predicted H (fit without this prefix) |", "| --- | --- | --- |"]
        lines += [f"| {h['prefix']} | {h['measured_H']:.2f} | {h['predicted_H']:.2f} |" for h in m["H"]]
        lines.append("")
    for name, cells in report["predictions"].items():
        lines += [f"## Model prediction: {name}", "", "| Prefix | Progress | S (s) | H (s) | Break-even q |",
                  "| --- | --- | --- | --- | --- |"]
        lines += [f"| {c['initial_prefix']} | {c['progress']:.0%} | {c['S']:.3f} | {c['H']:.3f} | "
                  f"{'—' if c['break_even_q'] is None else format(c['break_even_q'], '.3f')} |" for c in cells]
        lines.append("")
    (out / "costmodel.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

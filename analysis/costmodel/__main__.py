"""Fit the cost model to a spike run and write costmodel.json plus a readable costmodel.md.

    python -m analysis.costmodel --spike experiments/spike/results/20261103-rtx5090 --out run-data/costmodel

Rerun with --pilot-prefix-lengths to replace the assumed task shape with a pilot's measured one,
and on the GPU host so the logging overhead H reflects that host's storage.
"""

import argparse
import json
from pathlib import Path

from analysis.costmodel import (TaskShape, calibrate_from_spike, fixed_costs_from_cell, matrix_predictions,
                                measure_log_overhead, validate_on_spike)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="analysis.costmodel")
    parser.add_argument("--spike", required=True, help="spike results directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pilot-prefix-lengths", help="actual-prefix-lengths.json from a pilot run")
    parser.add_argument("--log-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spike = Path(args.spike)
    calibration = calibrate_from_spike(spike)
    validation = validate_on_spike(spike)
    runner_off = next(rows for name, rows in json.loads((spike / "recovery.json").read_text()).items()
                      if name.startswith("Runner kill") and "APC off" in name)
    fixed = fixed_costs_from_cell(runner_off)
    overhead = measure_log_overhead(repeats=args.log_repeats)
    shape = (TaskShape.from_pilot(json.loads(Path(args.pilot_prefix_lengths).read_text()))
             if args.pilot_prefix_lengths else TaskShape())
    matrix = {f"{domain}, APC {'on' if apc else 'off'}": matrix_predictions(
                  calibration, fixed, log_per_step=overhead["per_step_s"], domain=domain, apc=apc, steps=shape.steps)
              for domain, apc in (("inference", False), ("runner", False), ("runner", True))}
    report = {"spike": str(spike), "task_shape": shape.__dict__, "task_shape_source": args.pilot_prefix_lengths or "assumed defaults",
              "validation": validation, "log_overhead": overhead, "matrix": matrix}
    (out / "costmodel.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = ["# Cost model v1", "", f"Spike: `{spike}`. Task shape: {report['task_shape_source']}.", "",
             "## Recovery prediction vs spike", "", "| Cell | Basis | Measured (s) | Predicted (s) | Error |",
             "| --- | --- | --- | --- | --- |"]
    lines += [f"| {c['cell']} | {c['basis']} | {c['measured_s']:.4f} | {c['predicted_s']:.4f} | {c['relative_error']:+.1%} |"
              for c in validation["cells"]]
    lines += ["", f"Logging overhead H: {overhead['per_step_s'] * 1000:.1f} ms per step (n={overhead['n']}, this host).", ""]
    for name, cells in matrix.items():
        lines += [f"## Matrix prediction: {name}", "", "| Prefix | Progress | S (s) | H (s) | Break-even q |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {c['initial_prefix']} | {c['progress']:.0%} | {c['S']:.3f} | {c['H']:.3f} | "
                  f"{'—' if c['break_even_q'] is None else format(c['break_even_q'], '.3f')} |" for c in cells]
        lines.append("")
    (out / "costmodel.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

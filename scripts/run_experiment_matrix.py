#!/usr/bin/env python3
"""Run paired CPU pilots and retain every run, including unsafe baseline failures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.cpu.harness import run, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[33, 34])
    parser.add_argument("--steps", type=int, choices=[5, 10, 20], default=5)
    parser.add_argument("--progress", type=float, default=0.2)
    parser.add_argument("--purpose", choices=["performance", "correctness"], default="performance")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct paired repetition identities")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    methods = ["outcomeserve", "restart"] if args.purpose == "performance" else ["outcomeserve", "restart", "naive_retry", "transcript_checkpoint"]
    for seed in args.seeds:
        for no_fault in ([True, False] if args.purpose == "performance" else [False]):
            # Alternate pair order across seeds to reduce consistent order bias.
            order = methods if seed % 2 else list(reversed(methods))
            for method in order:
                folder = args.output_dir / f"seed-{seed}-{method}-{'clean' if no_fault else 'fault'}"
                result = run(folder, seed=seed, steps=args.steps, progress=args.progress,
                             method=method, purpose=args.purpose, no_fault=no_fault)
                records.append({"evidence_dir": folder.name, **result})
    summary = {"schema_version": "1.1.0", "backend": "scripted_cpu", "runs": len(records),
               "valid_runs": sum(r["valid"] for r in records), "passed_runs": sum(r["passed"] for r in records),
               "note": "unsafe correctness baselines may fail; CPU costs are not GPU serving results", "results": records}
    save(args.output_dir / "summary.json", summary)
    lines = ["# CPU experiment results", "", "Backend: scripted CPU. No GPU benchmark claim.", "",
             f"Runs: {summary['runs']}; valid: {summary['valid_runs']}; passed: {summary['passed_runs']}.", "",
             "| Seed | Method | No fault | Valid | Passed | Duplicate effects | Fault-to-verified s |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in records:
        values = [row["seed"], row["method"], row["no_fault"], row["valid"], row["passed"],
                  row.get("metrics", {}).get("duplicate_effects"), row.get("fault_to_verified_seconds")]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines += ["", "Correctness baseline violations are retained, not excluded.",
              "Small smoke samples are not stable tail-latency evidence.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0 if all(r["valid"] and (r["passed"] or args.purpose == "correctness" and r["method"] != "outcomeserve") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

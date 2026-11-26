#!/usr/bin/env python3
"""Run the integrated GPU pilot or one preregistered matrix tier (docs/decisions.md).

Runs inside the locked container (docs/infrastructure/gpu-matrix.md). A run whose
result.json exists is skipped, so an interrupted tier resumes where it stopped.

    python3 scripts/run_gpu_matrix.py pilot --out experiments/gpu/results/<id>
    python3 scripts/run_gpu_matrix.py tier 1 --out experiments/gpu/results/<id>
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gpu.matrix import cell_id, run_one  # noqa: E402

PREFIXES = (2048, 8192, 16384)
# Tier: (progress or None for fault-free, paired seeds). Order and sizes are preregistered in docs/decisions.md.
TIERS = {"1": (0.5, 20), "1H": (None, 10), "2": (0.9, 20), "3": (0.1, 20)}
METHODS = ("restart", "outcomeserve")


def schedule(tier: str, seeds: int | None):
    progress, count = TIERS[tier]
    for seed in range(1, (seeds or count) + 1):  # outer loop seeds, inner loop prefixes
        for prefix in PREFIXES:
            # both methods back to back on the same fixture; alternate which goes first
            for method in (METHODS if seed % 2 else METHODS[::-1]):
                yield method, prefix, progress, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["pilot", "tier"])
    parser.add_argument("tier", nargs="?", choices=sorted(TIERS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, help="override the tier's paired-seed count (only to extend 1H)")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="move each failed run to <cell>/rejected/ (kept as evidence) and run it again")
    args = parser.parse_args()
    if args.mode == "pilot":
        # Cut-rule gate: fault-free and inference-kill runs at 8K / 50% with outcomeserve.runner.
        plan = [("outcomeserve", 8192, None, 1), ("outcomeserve", 8192, 0.5, 1), ("restart", 8192, 0.5, 1)]
        base = args.out / "pilot"
    else:
        if args.tier is None:
            parser.error("tier mode needs a tier")
        plan, base = list(schedule(args.tier, args.seeds)), args.out / f"tier-{args.tier}"
    failures = 0
    for index, (method, prefix, progress, seed) in enumerate(plan, 1):
        out = base / cell_id(prefix, progress) / f"s{seed:03d}-{method}"
        if (out / "result.json").exists():
            if not (args.rerun_failed and not json.loads((out / "result.json").read_text())["passed"]):
                continue
            rejected = out.parent / "rejected"
            rejected.mkdir(exist_ok=True)
            out.rename(rejected / f"{out.name}-{len(list(rejected.glob(out.name + '-*'))) + 1}")
        result = run_one(out, method=method, prefix=prefix, progress=progress, seed=seed)
        failures += not result["passed"]
        print(json.dumps({"n": f"{index}/{len(plan)}", "run": result["run_id"], "passed": result["passed"],
                          "fault_to_verified_s": result.get("fault_to_verified_seconds"),
                          "task_to_verified_s": result.get("task_to_verified_seconds"),
                          "reason": result.get("failure_reason")}), flush=True)
    if args.mode == "pilot":
        rows = {p.parent.name + "/" + p.parent.parent.name: json.loads(p.read_text()) for p in base.rglob("result.json")}
        free = next(r for r in rows.values() if r["no_fault"])
        fault = next(r for r in rows.values() if r["fault_applied"] and r["method"] == "outcomeserve")
        gate = {"all_passed": all(r["passed"] for r in rows.values()),
                "zero_duplicates": all(r.get("duplicate_effects") == 0 for r in rows.values()),
                "fault_prefix_equals_fault_free": free.get("final_prefix_sha256") == fault.get("final_prefix_sha256")}
        gate["gate_passed"] = all(gate.values())
        (base / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        print(json.dumps(gate))
        return 0 if gate["gate_passed"] else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

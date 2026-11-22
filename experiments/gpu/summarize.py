"""Summarize GPU matrix results into summary.md and runs.csv (stdlib only).

    python3 experiments/gpu/summarize.py experiments/gpu/results/<id>

Per cell: runs, invalid runs with reasons, and paired medians by method. Pairs
are (configuration_id, seed); S = restart − OutcomeServe fault → verified, and
H = OutcomeServe − restart task → verified for fault-free cells. Confidence
intervals and exclusions are Victoria's analysis; this is the raw-data view.
"""

import csv
import json
from pathlib import Path
import statistics as st
import sys

FIELDS = ["tier", "cell", "seed", "method", "valid", "passed", "fault_applied", "initial_prefix_tokens",
          "final_prefix_tokens", "duplicate_effects", "task_to_verified_seconds", "fault_to_verified_seconds",
          "fault_to_first_new_token_seconds", "fault_to_replacement_ready_seconds", "failure_reason"]


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("tier-*/*/*/result.json")):
        result = json.loads(path.read_text())
        rows.append({"tier": path.parts[-4].removeprefix("tier-"), **{k: result.get(k) for k in FIELDS[1:]}})
    return rows


def med(values):
    values = [v for v in values if v is not None]
    return f"{st.median(values):.2f}" if values else "–"


def main(root: Path):
    rows = load(root)
    with (root / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md = [f"# GPU matrix: `{root.name}`", "", f"{len(rows)} runs; per-run data in `runs.csv`. Times in seconds, medians.", "",
          "| Tier | Cell | Pairs | Invalid / failed | Restart fault→verified | OutcomeServe fault→verified | Paired S (restart − OS) | "
          "Restart task | OutcomeServe task | Paired task OS − restart (H if fault-free) | OS fault→first token |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    cells = sorted({(r["tier"], r["cell"]) for r in rows}, key=lambda c: (c[0], c[1]))
    for tier, cell in cells:
        group = [r for r in rows if r["tier"] == tier and r["cell"] == cell]
        ok = [r for r in group if r["passed"]]
        by = {(r["seed"], r["method"]): r for r in ok}
        seeds = sorted({s for s, _ in by if (s, "restart") in by and (s, "outcomeserve") in by})
        saving = [by[s, "restart"]["fault_to_verified_seconds"] - by[s, "outcomeserve"]["fault_to_verified_seconds"]
                  for s in seeds if by[s, "restart"]["fault_to_verified_seconds"] is not None]
        overhead = [by[s, "outcomeserve"]["task_to_verified_seconds"] - by[s, "restart"]["task_to_verified_seconds"]
                    for s in seeds]
        bad = len(group) - len(ok)
        method = lambda m, k: med([r[k] for r in ok if r["method"] == m])  # noqa: E731
        md.append(f"| {tier} | {cell} | {len(seeds)} | {bad} | {method('restart', 'fault_to_verified_seconds')} | "
                  f"{method('outcomeserve', 'fault_to_verified_seconds')} | {med(saving)} | "
                  f"{method('restart', 'task_to_verified_seconds')} | {method('outcomeserve', 'task_to_verified_seconds')} | "
                  f"{med(overhead)} | {method('outcomeserve', 'fault_to_first_new_token_seconds')} |")
    failed = [r for r in rows if not r["passed"]]
    if failed:
        md += ["", "## Invalid or failed runs", "", "| Tier | Cell | Seed | Method | Reason |", "| --- | --- | --- | --- | --- |"]
        md += [f"| {r['tier']} | {r['cell']} | {r['seed']} | {r['method']} | {r['failure_reason']} |" for r in failed]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

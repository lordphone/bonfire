"""Numbers behind every report figure and table, computed only from retained results.

Owner: Victoria. No plotting here (``scripts/make_figures.py`` draws), so the
values can be tested and compared across regenerations without matplotlib.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.correctness import METHODS, summarize
from analysis.pipeline import gpu_matrix_analysis
from outcomeserve.contracts import validate_schema

PREFIXES = (2048, 8192, 16384)
PROGRESS = (0.1, 0.5, 0.9)
METHOD_LABELS = {"outcomeserve": "OutcomeServe", "restart": "Full restart", "naive_retry": "Naive retry",
                 "transcript_checkpoint": "Transcript ckpt."}


# ---------------------------------------------------------------- RQ1 (CPU correctness)

def read_correctness_rows(matrix_dir: Path) -> list[dict]:
    """runs.csv written by scripts/run_correctness_matrix.py, with types restored."""
    rows = []
    with (Path(matrix_dir) / "runs.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ("valid", "correct"):
                row[key] = row[key] == "True"
            for key in ("duplicate_effects", "unsafe_retry", "false_abstain", "missed_abstain",
                        "continued_after_abstain"):
                row[key] = None if row[key] == "" else int(row[key])
            rows.append(row)
    return rows


def correctness_overview(rows: list[dict]) -> list[dict]:
    """One row per method. Unknown counts stay None (never summed as zero)."""
    cells = summarize(rows)
    out = []
    for method in METHODS:
        members = [c for c in cells if c["method"] == method]
        if not members:
            continue
        total = lambda key: None if any(c[key] is None for c in members) else sum(c[key] for c in members)
        mine = [r for r in rows if r["method"] == method and r["valid"]]
        abstain_cases = [r for r in mine if r["expected_behavior"] == "abstain"]
        out.append({"method": method, "runs": total("runs"), "invalid": total("invalid"), "correct": total("correct"),
                    "runs_with_duplicates": total("runs_with_duplicates"),
                    "duplicate_effects": total("duplicate_effects"), "unsafe_retry": total("unsafe_retry"),
                    "missed_abstain": total("missed_abstain"), "false_abstain": total("false_abstain"),
                    "continued_after_abstain": total("continued_after_abstain"),
                    "abstained_when_required": sum(r["observed_behavior"] == "abstain" for r in abstain_cases),
                    "abstain_cases": len(abstain_cases),
                    # Incorrect runs split for the figure: a duplicate, else a missed abstention, else other.
                    "incorrect_duplicate": sum((not r["correct"]) and (r["duplicate_effects"] or 0) > 0 for r in mine),
                    "incorrect_missed_abstain": sum((not r["correct"]) and not (r["duplicate_effects"] or 0)
                                                    and (r["missed_abstain"] or 0) > 0 for r in mine)})
        last = out[-1]
        last["incorrect_other"] = (len(mine) - last["correct"] - last["incorrect_duplicate"]
                                   - last["incorrect_missed_abstain"])
    return out


def _cell(value):
    return "unknown" if value is None else str(value)


def latex_correctness_table(overview: list[dict]) -> str:
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Method & Correct / runs & Dup.\ effects & Unsafe retries & Abstained / required & Invalid \\",
             r"\midrule"]
    for row in overview:
        lines.append(f"{METHOD_LABELS[row['method']]} & {row['correct']} / {row['runs']} & "
                     f"{_cell(row['duplicate_effects'])} & {_cell(row['unsafe_retry'])} & "
                     f"{row['abstained_when_required']} / {row['abstain_cases']} & {row['invalid']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- RQ2/RQ3 (GPU matrix)

def read_gpu_results(results_root: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(results_root).rglob("result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_schema("run-result", record)
        if record.get("schema_version") == "1.3.0" and record.get("backend") == "vllm":
            rows.append(record)
    return rows


def _estimate(comparison):
    estimate = comparison["estimate"]
    if not estimate:
        return None
    interval = estimate["bootstrap_ci95"]
    return {"mean": estimate["mean_restart_minus_outcomeserve"],
            "ci95": None if interval is None else list(interval), "pairs": comparison["complete_pairs"]}


def heatmap_grid(matrix: dict) -> list[dict]:
    """All nine (L, f) cells in fixed order. Status: measured, unusable (measured but
    no estimate) or not_measured. Unmeasured cells are never filled in."""
    cells = {(c["prefix_target_tokens"], c["requested_progress"]): c for c in matrix["cells"]}
    grid = []
    for prefix in PREFIXES:
        for progress in PROGRESS:
            cell = cells.get((prefix, progress))
            if cell is None:
                grid.append({"prefix": prefix, "progress": progress, "status": "not_measured"})
                continue
            primary = cell["delta_T_fault_to_verified"]
            estimate = _estimate(primary)
            grid.append({"prefix": prefix, "progress": progress, "status": "measured" if estimate else "unusable",
                         "delta_T": estimate, "total_pairs": primary["total_pairs"],
                         "incomplete_pairs": len(primary["incomplete_pairs"]),
                         "next_token": _estimate(cell["secondary_first_new_token"])})
    return grid


def break_even_lines(matrix: dict) -> list[dict]:
    """Expected saving q*S - (1-q)*H for each measured cell with both H(L) and S(L, f)."""
    lines = []
    for cell in matrix["cells"]:
        if cell["break_even"] is None:
            continue
        lines.append({"prefix": cell["prefix_target_tokens"], "progress": cell["requested_progress"],
                      "H": cell["break_even"]["H"], "S": cell["break_even"]["S"],
                      "break_even_q": cell["break_even"]["break_even_q"]})
    return lines


RESOURCE_METRICS = (("resources.process_cpu_seconds", "CPU s"), ("resources.gpu_allocation_seconds", "GPU alloc.\\ s"),
                    ("resources.peak_summed_rss_bytes", "Peak RSS MiB"),
                    ("resources.peak_gpu_memory_bytes", "Peak GPU MiB"))


def _format(estimate, scale=1.0):
    if estimate is None:
        return "n/a"
    mean = estimate["mean"] / scale
    if estimate["ci95"] is None:
        return f"{mean:.1f}"
    low, high = (value / scale for value in estimate["ci95"])
    return f"{mean:.1f} [{low:.1f}, {high:.1f}]"


def latex_resource_table(matrix: dict) -> str:
    """Restart minus OutcomeServe per measured cell; positive favors OutcomeServe."""
    lines = [r"\begin{tabular}{ll" + "r" * len(RESOURCE_METRICS) + "}", r"\toprule",
             "Prefix & Progress & " + " & ".join(label for _, label in RESOURCE_METRICS) + r" \\", r"\midrule"]
    for cell in matrix["cells"]:
        values = []
        for metric, label in RESOURCE_METRICS:
            scale = 2 ** 20 if "bytes" in metric else 1.0
            values.append(_format(_estimate(cell["resources"][metric]), scale))
        lines.append(f"{cell['prefix_target_tokens'] // 1024}K & {int(cell['requested_progress'] * 100)}\\% & "
                     + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def gpu_report_data(results_root: Path) -> dict:
    matrix = gpu_matrix_analysis(read_gpu_results(results_root))
    return {"platform": matrix["platform"], "run_count": matrix["run_count"], "heatmap": heatmap_grid(matrix),
            "break_even": break_even_lines(matrix), "overhead_H": {str(k): v["H_seconds"] for k, v in
                                                                matrix["overhead_H"].items()},
            "tier_1H_precision": matrix["tier_1H_precision"], "outcomes": matrix["outcomes_all_runs"],
            "resource_table": latex_resource_table(matrix)}

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.7", "jsonschema==4.26.0"]
# ///
"""Regenerate every RQ1-RQ3 figure and table from retained results.

    uv run scripts/make_figures.py --output report-figures \
        --correctness run-data/correctness-matrix --gpu-results run-data/gpu

Either input may be omitted. Writes PDF+PNG figures, LaTeX tables and
figures-data.json (every plotted number). Regenerating from the same results
gives an identical figures-data.json; compare it to check reproducibility.
The output directory must be new.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from analysis.report_data import (METHOD_LABELS, PREFIXES, PROGRESS, correctness_overview,  # noqa: E402
                                  gpu_report_data, latex_correctness_table, read_correctness_rows)

# Same palette as experiments/spike/analyze.py, so midterm and final figures match.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4af"
COLUMN = 3.4  # inches: one paper column
plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.hashsalt": "outcomeserve", "pdf.compression": 9})


def save(fig, output: Path, name: str):
    fig.tight_layout()
    # No timestamps in the files, so regenerated figures are byte-stable.
    fig.savefig(output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)


def correctness_figure(overview, output):
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    names = [METHOD_LABELS[row["method"]] for row in overview][::-1]
    parts = (("correct", "Correct", SERIES[2]), ("incorrect_duplicate", "Duplicate effect", SERIES[1]),
             ("incorrect_missed_abstain", "Missed abstention", SERIES[3]), ("incorrect_other", "Other", OTHER))
    left = [0] * len(overview)
    for key, label, color in parts:
        values = [row[key] for row in overview][::-1]
        if not any(values):
            continue  # an empty category gets no legend entry
        ax.barh(names, values, left=left, color=color, label=label, height=0.6)
        left = [a + b for a, b in zip(left, values)]
    for y, row in enumerate(overview[::-1]):
        ax.text(left[y] + 2, y, f"{row['correct']}/{row['runs']}", va="center", color=INK, fontsize=6)
    ax.set_xlabel("Runs (valid fault injections)")
    ax.set_xlim(0, max(left) * 1.15)
    ax.legend(ncol=4, fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1, columnspacing=1)
    save(fig, output, "rq1-correctness")


def heatmap_figure(grid, output):
    values = [c["delta_T"]["mean"] for c in grid if c["status"] == "measured"]
    bound = max([abs(v) for v in values] + [1e-9])
    cmap = LinearSegmentedColormap.from_list("resume", [SERIES[1], "#ffffff", SERIES[0]])
    norm = TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    for cell in grid:
        x, y = PROGRESS.index(cell["progress"]), PREFIXES.index(cell["prefix"])
        if cell["status"] == "measured":
            estimate = cell["delta_T"]
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=cmap(norm(estimate["mean"]))))
            interval = estimate["ci95"]
            text = f"{estimate['mean']:+.1f} s" + ("" if interval is None else f"\n[{interval[0]:+.1f}, {interval[1]:+.1f}]")
            text += f"\nn={estimate['pairs']}"
        else:
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, facecolor="#f4f3f0", edgecolor=GRID, hatch="///"))
            text = "not measured" if cell["status"] == "not_measured" else \
                f"no estimate\n{cell['incomplete_pairs']} incomplete pair(s)"
        ax.text(x, y, text, ha="center", va="center", fontsize=6, color=INK)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(-.5, 2.5)
    ax.set_xticks(range(3), [f"{int(p * 100)}%" for p in PROGRESS])
    ax.set_yticks(range(3), [f"{p // 1024}K" for p in PREFIXES])
    ax.set_xlabel("Completed fraction at fault")
    ax.set_ylabel("Prefix length")
    ax.set_title("ΔT = restart − OutcomeServe, fault → verified completion (+ favors resume)", fontsize=6, color=INK2)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    save(fig, output, "rq2-delta-t-heatmap")


def break_even_figure(lines, output):
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    qs = [10 ** (-4 + 4 * i / 200) for i in range(201)]
    for index, line in enumerate(sorted(lines, key=lambda l: (l["prefix"], l["progress"]))):
        color = SERIES[index % len(SERIES)]
        ax.plot(qs, [q * line["S"] - (1 - q) * line["H"] for q in qs], color=color, linewidth=1,
                label=f"{line['prefix'] // 1024}K, {int(line['progress'] * 100)}%")
        if line["break_even_q"] is not None and line["break_even_q"] > 0:
            ax.axvline(line["break_even_q"], color=color, linewidth=0.5, linestyle=":")
    ax.axhline(0, color=INK2, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Assumed task-failure probability q (not a measured rate)")
    ax.set_ylabel("Expected saving per task (s)")
    ax.grid(color=GRID, linewidth=0.4)
    if lines:
        ax.legend(fontsize=5.5, frameon=False, ncol=3)
    save(fig, output, "rq3-break-even")


def finite(value):
    """JSON-safe numbers: a non-finite value is a bug upstream, never silently written."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value in figure data")
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, help="output directory of run_correctness_matrix.py")
    parser.add_argument("--gpu-results", type=Path, help="directory containing GPU result.json files")
    args = parser.parse_args(argv)
    if not args.correctness and not args.gpu_results:
        parser.error("give --correctness and/or --gpu-results")
    if args.output.exists():
        parser.error("output exists; figures are regenerated into a new directory")
    data = {}
    if args.correctness:
        data["rq1"] = {"overview": correctness_overview(read_correctness_rows(args.correctness))}
    if args.gpu_results:
        data["gpu"] = gpu_report_data(args.gpu_results)
        if not data["gpu"]["run_count"]:
            parser.error("no main-matrix GPU results found")
    args.output.mkdir(parents=True)
    if "rq1" in data:
        correctness_figure(data["rq1"]["overview"], args.output)
        (args.output / "rq1-correctness-table.tex").write_text(latex_correctness_table(data["rq1"]["overview"]),
                                                               encoding="utf-8")
    if "gpu" in data:
        heatmap_figure(data["gpu"]["heatmap"], args.output)
        break_even_figure(data["gpu"]["break_even"], args.output)
        (args.output / "rq2-resource-table.tex").write_text(data["gpu"].pop("resource_table"), encoding="utf-8")
    (args.output / "figures-data.json").write_text(json.dumps(finite(data), indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": sorted(p.name for p in args.output.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

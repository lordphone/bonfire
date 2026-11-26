"""Report figures and tables come only from retained results and never fill gaps."""

import csv
from itertools import count
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.correctness import ROW_FIELDS
from analysis.pipeline import gpu_matrix_analysis
from analysis.report_data import (correctness_overview, gpu_report_data, heatmap_grid, latex_correctness_table,
                                  read_correctness_rows)
from test_run_results import completed

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = count()


def matrix_row(method, *, correct=True, duplicates=0, missed=0, expected="complete", observed="complete",
               unsafe=0, valid=True):
    return {"case": f"{method}-{next(CASE_IDS)}", "steps": 5, "seed": 33, "fault_tool": "decrement",
            "fault_domain": "runner", "boundary": "PRE_DISPATCH", "method": method, "valid": valid,
            "expected_behavior": expected, "observed_behavior": observed, "correct": correct,
            "duplicate_effects": duplicates, "unsafe_retry": unsafe, "false_abstain": 0, "missed_abstain": missed,
            "continued_after_abstain": 0, "actual_progress": 0.4, "reason": ""}


def write_runs(tmp_path, rows):
    with (tmp_path / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_correctness_overview_splits_incorrect_runs_and_counts_required_abstentions(tmp_path):
    rows = [matrix_row("outcomeserve"), matrix_row("outcomeserve", expected="abstain", observed="abstain"),
            matrix_row("naive_retry", correct=False, duplicates=1, unsafe=1),
            matrix_row("naive_retry", correct=False, missed=1, expected="abstain"),
            matrix_row("naive_retry", valid=False, correct=False)]
    overview = {row["method"]: row for row in correctness_overview(read_correctness_rows(write_runs(tmp_path, rows)))}
    ours, naive = overview["outcomeserve"], overview["naive_retry"]
    assert (ours["correct"], ours["runs"], ours["abstained_when_required"], ours["abstain_cases"]) == (2, 2, 1, 1)
    assert (naive["incorrect_duplicate"], naive["incorrect_missed_abstain"], naive["incorrect_other"]) == (1, 1, 0)
    assert naive["invalid"] == 1 and naive["runs"] == 3  # invalid runs are counted, not dropped
    table = latex_correctness_table([ours, naive])
    assert "OutcomeServe & 2 / 2 & 0 & 0 & 1 / 1 & 0" in table and "Naive retry & 0 / 3" in table


def test_unknown_metrics_are_shown_as_unknown_not_zero(tmp_path):
    rows = [matrix_row("outcomeserve"), {**matrix_row("outcomeserve"), "unsafe_retry": None, "correct": False}]
    overview = correctness_overview(read_correctness_rows(write_runs(tmp_path, rows)))
    assert overview[0]["unsafe_retry"] is None and "unknown" in latex_correctness_table(overview)


def test_heatmap_marks_unmeasured_and_unusable_cells_instead_of_filling_them():
    rows = []
    for seed in range(33, 36):
        rows += [completed("restart", seed, prefix=2048, total=60), completed("outcomeserve", seed, prefix=2048, total=50)]
        rows += [completed("restart", seed, prefix=8192, total=60)]  # 8K pairs are missing a member
    grid = {(c["prefix"], c["progress"]): c for c in heatmap_grid(gpu_matrix_analysis(rows))}
    assert len(grid) == 9
    assert grid[(2048, 0.5)]["status"] == "measured" and grid[(2048, 0.5)]["delta_T"]["mean"] == pytest.approx(5)
    assert grid[(8192, 0.5)]["status"] == "unusable" and grid[(8192, 0.5)]["incomplete_pairs"] == 3
    assert grid[(16384, 0.9)] == {"prefix": 16384, "progress": 0.9, "status": "not_measured"}


def test_gpu_report_data_reads_only_main_matrix_results(tmp_path):
    for index, record in enumerate([completed("restart", 33, total=60), completed("outcomeserve", 33, total=50),
                                    completed("restart", 33, no_fault=True, total=30),
                                    completed("outcomeserve", 33, no_fault=True, total=31)]):
        (tmp_path / f"r{index}").mkdir()
        (tmp_path / f"r{index}" / "result.json").write_text(json.dumps(record))
    data = gpu_report_data(tmp_path)
    assert data["run_count"] == 4 and data["overhead_H"]["8192"] == pytest.approx(1)
    assert data["break_even"][0]["break_even_q"] == pytest.approx(1 / 11)
    assert "8K & 50\\%" in data["resource_table"]


def test_make_figures_regenerates_identically(tmp_path):
    pytest.importorskip("matplotlib")
    runs = write_runs(tmp_path, [matrix_row("outcomeserve"), matrix_row("restart", correct=False, duplicates=2)])
    outputs = []
    for name in ("a", "b"):
        done = subprocess.run([sys.executable, str(ROOT / "scripts/make_figures.py"), "--output", str(tmp_path / name),
                               "--correctness", str(runs)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        outputs.append(tmp_path / name)
    for file in ("figures-data.json", "rq1-correctness.pdf", "rq1-correctness-table.tex"):
        assert (outputs[0] / file).read_bytes() == (outputs[1] / file).read_bytes(), file
    again = subprocess.run([sys.executable, str(ROOT / "scripts/make_figures.py"), "--output", str(outputs[0]),
                            "--correctness", str(runs)], capture_output=True, text=True)
    assert again.returncode == 2 and "output exists" in again.stderr

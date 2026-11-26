import json
from statistics import median

import pytest

import csv

from analysis.costmodel import (MAX_MODEL_LEN, FixedCosts, LoggingCost, TaskShape, calibrate_from_spike,
                                fit_logging_cost, matrix_predictions, measure_log_overhead, one_fault_saving,
                                recovery_seconds, validate_on_matrix, validate_on_spike)
from analysis.costmodel.__main__ import main as costmodel_main
from analysis.pipeline import break_even_curve
from experiments.common import ROOT

SPIKE = ROOT / "experiments" / "spike" / "results" / "20261103-rtx5090"
FIXED = FixedCosts(detect=0.007, teardown=0.0006, runner_start=0.033, reconcile=0.0012)


@pytest.fixture(scope="module")
def cal():
    return calibrate_from_spike(SPIKE)


def test_calibration_reproduces_the_standalone_spike_medians(cal):
    rows = json.loads((SPIKE / "prefill-apc-off" / "summary.json").read_text())
    for tokens in (2048, 8192, 16384):
        measured = median(r["prefill_s"] for r in rows if r["tokens"] == tokens and r["pass"] == "first")
        assert cal.prefill(tokens) == pytest.approx(measured, rel=0.06)
    assert cal.prefill(16384, cached=True) < cal.prefill(2048)  # a full-prefix hit is cheaper than any miss
    assert cal.decode_rate(2048) > cal.decode_rate(16384)
    with pytest.raises(ValueError):
        cal.prefill(MAX_MODEL_LEN + 1)


def test_model_predicts_every_spike_recovery_cell_including_the_held_out_one():
    report = validate_on_spike(SPIKE)
    errors = {cell["cell"]: cell["relative_error"] for cell in report["cells"]}
    assert set(errors) == {"inference kill, APC off", "runner kill, APC off", "runner kill, APC on"}
    assert all(abs(error) < 0.05 for error in errors.values()), errors
    held_out = next(c for c in report["cells"] if c["cell"] == "runner kill, APC on")
    assert "held out" in held_out["basis"]


def test_inference_restart_never_counts_a_cache_hit(cal):
    lost = recovery_seconds(cal, FIXED, domain="inference", prefix_tokens=10_000, apc=True)
    survived = recovery_seconds(cal, FIXED, domain="runner", prefix_tokens=10_000, apc=True)
    assert not lost["cache_hit"] and survived["cache_hit"]
    assert lost["components"]["replay_prefill"] == pytest.approx(cal.prefill(10_000))
    assert lost["components"]["startup"] == cal.worker_startup


def test_saving_grows_with_progress_and_prefix(cal):
    def saving(prefix, progress):
        task = TaskShape(initial_prefix=prefix)
        return one_fault_saving(cal, FIXED, task, progress=progress, domain="inference", apc=False)["S"]

    assert saving(8192, 0.1) < saving(8192, 0.5) < saving(8192, 0.9)
    assert saving(2048, 0.5) < saving(8192, 0.5) < saving(16384, 0.5)
    # Faulting in the first step leaves nothing to redo, so resuming cannot win.
    assert saving(8192, 0.0) < 0


def test_matrix_break_even_matches_the_pipeline_definition(cal):
    cells = matrix_predictions(cal, FIXED, logging=0.02)
    assert len(cells) == 9
    for cell in cells:
        expected = break_even_curve(cell["H"], cell["S"])["break_even_q"]
        assert cell["break_even_q"] == pytest.approx(expected)
        assert cell["H"] == pytest.approx(20 * 0.02)


def test_task_shape_round_trips_through_pilot_prefix_lengths():
    shape = TaskShape(steps=5, initial_prefix=3000, step_prompt=37, generated=28, tool_result=19)
    rows, committed = [], shape.initial_prefix
    for _ in range(shape.steps):
        prompt = committed + shape.step_prompt
        committed = prompt + shape.generated + shape.tool_result
        rows.append({"prompt_tokens": prompt, "generated_tokens": shape.generated, "committed_prefix_tokens": committed})
    assert TaskShape.from_pilot(rows) == shape


def test_log_overhead_measures_real_prepare_and_commit():
    overhead = measure_log_overhead(steps=5, repeats=1)
    assert overhead["n"] == 5 and 0 < overhead["per_step_s"] < 1


def test_cli_writes_json_and_markdown_reports(tmp_path):
    assert costmodel_main(["--spike", str(SPIKE), "--out", str(tmp_path), "--log-repeats", "1"]) == 0
    report = json.loads((tmp_path / "costmodel.json").read_text())
    assert report["version"] == "v1"
    assert len(report["spike_validation"]["cells"]) == 3 and len(report["predictions"]) == 3
    assert "held out" in (tmp_path / "costmodel.md").read_text()


def test_logging_cost_fit_recovers_its_parameters():
    truth, shape = LoggingCost(base=0.02, per_token=2e-5), TaskShape()
    h = {p: truth.task_seconds(TaskShape(initial_prefix=p)) for p in (2048, 8192, 16384)}
    fitted = fit_logging_cost(h, shape)
    assert fitted.base == pytest.approx(truth.base, rel=1e-6)
    assert fitted.per_token == pytest.approx(truth.per_token, rel=1e-6)


def test_post_fault_logging_lowers_saving_most_when_the_fault_is_early(cal):
    task, cost = TaskShape(initial_prefix=8192), LoggingCost(base=0.02, per_token=2e-5)
    def delta(progress):
        v1 = one_fault_saving(cal, FIXED, task, progress=progress, domain="inference", apc=False)
        v2 = one_fault_saving(cal, FIXED, task, progress=progress, domain="inference", apc=False, logging=cost)
        assert v2["outcomeserve_logging_after_fault_s"] == pytest.approx(cost.steps(task, v2["completed_steps"], 20))
        return v1["S"] - v2["S"]
    assert delta(0.1) > delta(0.5) > delta(0.9) > 0


def synthetic_matrix(path, cal, cost, shape):
    """A runs.csv whose means are exactly what v2 predicts (initial/final prefixes match the shape)."""
    recovery = json.loads((SPIKE / "recovery.json").read_text())
    from analysis.costmodel import fixed_costs_from_cell
    fixed = fixed_costs_from_cell(next(rows for name, rows in recovery.items() if name.startswith("Inference kill")))
    rows = []
    for prefix in (2048, 8192, 16384):
        task = TaskShape(initial_prefix=prefix, step_prompt=shape.step_prompt, generated=shape.generated,
                         tool_result=shape.tool_result)
        final = task.prefix_after(20)
        for progress in (None, 0.1, 0.5, 0.9):
            cell = f"p{prefix // 1024}k-" + ("nofault" if progress is None else f"f{round(progress * 100)}")
            if progress is None:
                gap = cost.task_seconds(task)
            else:
                gap = one_fault_saving(cal, fixed, task, progress=progress, domain="inference", apc=False,
                                       logging=cost)["S"]
            for seed in (1, 2):
                base = 30.0 + seed
                os_time, restart_time = (base + gap, base) if progress is None else (base, base + gap)
                for method, value in (("outcomeserve", os_time), ("restart", restart_time)):
                    rows.append({"tier": "x", "cell": cell, "seed": str(seed), "method": method, "valid": "True",
                                 "passed": "True", "initial_prefix_tokens": str(prefix), "final_prefix_tokens": str(final),
                                 "task_to_verified_seconds": str(value), "fault_to_verified_seconds": str(value),
                                 "fault_to_first_new_token_seconds": "20.0"})
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_matrix_validation_scores_v2_against_measured_cells(cal, tmp_path):
    shape = TaskShape(step_prompt=30, generated=30, tool_result=19)
    synthetic_matrix(tmp_path / "runs.csv", cal, LoggingCost(base=0.02, per_token=2e-5), shape)
    report = validate_on_matrix(SPIKE, tmp_path / "runs.csv")

    assert report["task_shape"]["step_prompt"] + report["task_shape"]["generated"] + report["task_shape"]["tool_result"] == 79
    assert report["mean_abs_error_s"]["v2"] == pytest.approx(0, abs=1e-6)
    assert report["mean_abs_error_s"]["v1"] > 0.5  # v1 ignores logging after the fault
    assert all(h["predicted_H"] == pytest.approx(h["measured_H"], rel=1e-6) for h in report["H"])
    assert {c["cell"] for c in report["S"]} == {f"p{p}k-f{q}" for p in (2, 8, 16) for q in (10, 50, 90)}


def test_cli_v2_reports_matrix_comparison(cal, tmp_path):
    synthetic_matrix(tmp_path / "runs.csv", cal, LoggingCost(base=0.02, per_token=2e-5), TaskShape(step_prompt=30))
    assert costmodel_main(["--spike", str(SPIKE), "--matrix", str(tmp_path / "runs.csv"), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "costmodel.json").read_text())
    assert report["version"] == "v2" and len(report["matrix_validation"]["S"]) == 9
    assert "Prediction vs RTX 5090 matrix" in (tmp_path / "out" / "costmodel.md").read_text()

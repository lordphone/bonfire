import json
from statistics import median

import pytest

from analysis.costmodel import (MAX_MODEL_LEN, FixedCosts, TaskShape, calibrate_from_spike, matrix_predictions,
                                measure_log_overhead, one_fault_saving, recovery_seconds, validate_on_spike)
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
    cells = matrix_predictions(cal, FIXED, log_per_step=0.02)
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
    assert len(report["validation"]["cells"]) == 3 and len(report["matrix"]) == 3
    assert "held out" in (tmp_path / "costmodel.md").read_text()

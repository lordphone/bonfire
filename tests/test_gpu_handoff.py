"""SYNTHETIC ONLY: exercise the returned-artifact interface without vLLM or a GPU.

Token IDs, timings and resource values are invented test fixtures. This does
not run the CPU correctness matrix or measure recovery performance.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.runresult import assemble, assemble_directory, read_events
from outcomeserve.context import context_spec_id, scripted_context
from outcomeserve.contracts import validate_schema
from outcomeserve.storage import StateStore
from outcomeserve.tools import ToolService
from outcomeserve.verify import committed_tokens, verify_replay
from workloads import make_inventory_workload
from test_model_token_verification import model_run
from test_report_data import matrix_row, write_runs
from test_run_results import GOOD_RESOURCES, events, identity, protocol_report, verification
from test_correctness_cases import request

ROOT = Path(__file__).resolve().parents[1]


def test_replay_evidence_is_optional_for_inference_loss_recovery():
    ident = identity()
    protocol = {k: v for k, v in protocol_report(ident).items() if k != "replay"}
    result = assemble(ident, events(ident, fault=30, verified=60), verification(), GOOD_RESOURCES,
                      runner_exit_code=0, protocol=protocol)
    assert result["outcome"] == "COMPLETED" and result["protocol_verification"]["replay_valid"] is None


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def cli(script, *args, code=0):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                            capture_output=True, text=True)
    assert result.returncode == code, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("change", [
    None, {"valid": False}, {"task_id": "another-task"}, {"context_spec_id": "another-context"},
    {"token_source": "scripted"}, {"checkpoint": "in_progress"}, {"task_success": False},
    {"issues": [{"code": "LOG_TRANSITION"}]},
    {"replay": {"valid": True, "task_id": "other-task", "context_spec_id": "ctx-qwen3-4b"}},
])
def test_unverified_or_foreign_protocol_never_establishes_completion(change):
    ident = identity()
    protocol = None if change is None else {**protocol_report(ident), **change}
    result = assemble(ident, events(ident, fault=30, verified=60), verification(), GOOD_RESOURCES,
                      runner_exit_code=0, protocol=protocol)
    assert result["valid"] and not result["passed"]
    assert result["outcome"] == "INFRASTRUCTURE_FAILURE" and result["failure_reason"]


def test_missing_named_event_file_is_retained_as_failed_evidence(tmp_path):
    ident = identity()
    save(tmp_path / "identity.json", ident)
    save(tmp_path / "verification.json", {"business": verification(), "protocol": protocol_report(ident)})
    save(tmp_path / "resources.json", GOOD_RESOURCES)
    save(tmp_path / "runner-exit.json", {"exit_code": 0})
    (tmp_path / "measurements.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events(ident, fault=30, verified=60)))
    result = assemble_directory(tmp_path, event_files=["measurements.jsonl", "missing-runner.jsonl"])
    assert result["valid"] and result["outcome"] == "INFRASTRUCTURE_FAILURE"
    assert result["evidence_errors"] == ["required event file missing: missing-runner.jsonl"]
    cli("build_run_result.py", tmp_path, "--events", "measurements.jsonl", "missing-runner.jsonl")
    assert json.loads((tmp_path / "result.json").read_text())["passed"] is False


def test_raw_conversion_does_not_relabel_a_foreign_run(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"event_type": "FIRST_NEW_TOKEN", "clock_id": "clock-a", "monotonic_ns": 1,
                               "run_id": "someone-else"}) + "\n")
    with pytest.raises(ValueError, match="another run_id"):
        read_events([path], identity())


@pytest.mark.parametrize("method", ["restart", "outcomeserve"])
def test_business_verification_cannot_be_taken_from_another_task(method):
    ident = identity(method)
    foreign = {**verification(), "task_id": "someone-else"}
    result = assemble(ident, events(ident, fault=30, verified=60), foreign, GOOD_RESOURCES,
                      runner_exit_code=0, protocol=protocol_report(ident))
    assert result["outcome"] == "INFRASTRUCTURE_FAILURE" and "task_id" in result["failure_reason"]


def test_failed_replay_cannot_hide_behind_a_passing_protocol_flag():
    ident = identity()
    protocol = {**protocol_report(ident), "replay": {"valid": False, "task_id": ident["task_id"],
                                                    "context_spec_id": ident["context_spec_id"]}}
    result = assemble(ident, events(ident, fault=30, verified=60), verification(), GOOD_RESOURCES,
                      runner_exit_code=0, protocol=protocol)
    assert result["outcome"] == "RECOVERY_VIOLATION" and not result["passed"]


def test_old_gpu_results_remain_readable_but_are_not_formal_matrix_samples():
    from analysis.pipeline import gpu_matrix_analysis
    from test_run_results import completed
    current = completed("outcomeserve", 33)
    assert current["schema_version"] == "1.3.0"
    legacy = {key: value for key, value in current.items() if key not in {"task_id", "protocol_verification"}}
    legacy["schema_version"] = "1.2.0"
    validate_schema("run-result", legacy)
    assert gpu_matrix_analysis([legacy])["run_count"] == 0


def test_pilot_checks_historical_replay_without_requiring_identical_future_tokens(tmp_path):
    clean = model_run(tmp_path / "clean.sqlite")
    fault = model_run(tmp_path / "fault.sqlite", snapshot=tmp_path / "before-fault.sqlite", variant=99)
    prefix = committed_tokens(tmp_path / "before-fault.sqlite", fault["task_id"])
    replay = {"task_id": fault["task_id"], "context_spec_id": context_spec_id(scripted_context(fault["model_revision"])),
              "prefix_token_ids": prefix, "prompt_token_ids": prefix + [102]}
    save(tmp_path / "workload.json", fault)
    save(tmp_path / "recovery-request.json", replay)
    assert committed_tokens(tmp_path / "clean.sqlite", clean["task_id"]) != committed_tokens(tmp_path / "fault.sqlite", fault["task_id"])
    args = ("--db", tmp_path / "fault.sqlite", "--workload", tmp_path / "workload.json", "--token-source", "model",
            "--with-business", "--replay-snapshot", tmp_path / "before-fault.sqlite",
            "--replay-request", tmp_path / "recovery-request.json")
    cli("verify_run.py", *args, "--output", tmp_path / "verification.json")
    assert json.loads((tmp_path / "verification.json").read_text())["protocol"]["replay"]["valid"]
    cli("verify_run.py", *args, "--output", tmp_path / "verification.json", code=2)
    replay["prefix_token_ids"] = prefix + [999]
    assert not verify_replay(tmp_path / "before-fault.sqlite", fault, replay)["valid"]
    save(tmp_path / "recovery-request.json", replay)
    bad = cli("verify_run.py", *args, code=1)
    assert "PREFIX_REPLAY" in bad.stdout


def synthetic_run(folder, *, method, seed=33, prefix=2048, no_fault=False, pilot=False, bad=False):
    """Produce artificial returned files, then invoke the real verification/builder CLIs."""
    folder.mkdir(parents=True)
    (folder / "SYNTHETIC_ONLY.txt").write_text("No GPU/model was used. All timings/resources are made up.\n")
    task_id = "synthetic-task"
    database = folder / "runtime.sqlite"
    if method == "outcomeserve":
        workload = model_run(database, task_id=task_id, task_steps=20,
                             snapshot=None if no_fault else folder / "pre-fault.sqlite",
                             tamper="drop_generation" if bad else None)
    else:
        workload = make_inventory_workload(task_id, steps=20)
        store = StateStore(database)
        store.create_task(task_id, workload["initial_inventory"])
        store.bind_context(task_id, scripted_context(workload["model_revision"]))
        for step in workload["steps"]:
            ToolService(store).submit(**request(task_id, step))
    spec = scripted_context(workload["model_revision"])
    save(folder / "workload.json", workload)
    save(folder / "context.json", spec)
    ident = identity(method, seed, prefix, no_fault=no_fault, task_id=task_id,
                     context_spec_id=context_spec_id(spec), hardware_id="SYNTHETIC-NO-GPU",
                     gpu_name="SYNTHETIC-NOT-HARDWARE", driver_version="synthetic")
    if pilot:
        ident["tier"] = "pilot"
    save(folder / "identity.json", ident)
    total = (30 if method == "restart" else 31) if no_fault else (70 if method == "restart" else 60)
    supervisor = events(ident, fault=None if no_fault else 30, verified=total, failed=bad,
                        reset=(31, 32) if method == "restart" and not no_fault else None)
    if not no_fault:
        for name, stamp in (("INFERENCE_RESTART_REQUESTED", 31), ("INFERENCE_REPLACEMENT_READY", 39)):
            supervisor.append({**supervisor[0], "event_type": name, "monotonic_ns": stamp * 10**9,
                               "sequence": len(supervisor) + 1})
    (folder / "measurements.jsonl").write_text("".join(json.dumps(e) + "\n" for e in supervisor))
    raw = [{"event_type": "TASK_COMPLETED", "monotonic_ns": (total - 1) * 10**9,
            "clock_id": "clock-a", "epoch": 1, "step_id": "none"}]
    if not no_fault:
        raw.insert(0, {**raw[0], "event_type": "FIRST_NEW_TOKEN", "monotonic_ns": 40 * 10**9})
    (folder / "runner-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in raw))
    save(folder / "resources.json", GOOD_RESOURCES)
    save(folder / "runner-exit.json", {"exit_code": 0})
    args = ["--db", database, "--workload", folder / "workload.json", "--output", folder / "verification-rechecked.json"]
    if method == "outcomeserve":
        args += ["--token-source", "model", "--with-business"]
        if not no_fault:
            historical = committed_tokens(folder / "pre-fault.sqlite", task_id)
            save(folder / "recovery-request.json", {"task_id": task_id, "context_spec_id": ident["context_spec_id"],
                                                    "prefix_token_ids": historical, "prompt_token_ids": historical + [110]})
            args += ["--replay-snapshot", folder / "pre-fault.sqlite", "--replay-request", folder / "recovery-request.json"]
    else:
        args += ["--business-only"]
    cli("verify_run.py", *args, code=1 if bad else 0)
    cli("build_run_result.py", folder, "--verification-file", "verification-rechecked.json",
        "--events", "measurements.jsonl", "runner-events.jsonl")
    return json.loads((folder / "result.json").read_text())


def test_synthetic_returned_artifacts_to_all_tables_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    dataset = tmp_path / "SYNTHETIC_ONLY"
    for seed in (33, 34):
        for method in ("restart", "outcomeserve"):
            for no_fault in (False, True):
                synthetic_run(dataset / f"s{seed}-{method}-clean{no_fault}", method=method, seed=seed, no_fault=no_fault)
            synthetic_run(dataset / f"s{seed}-{method}-8k", method=method, seed=seed, prefix=8192,
                          bad=seed == 34 and method == "outcomeserve")
    for no_fault in (False, True):
        synthetic_run(dataset / f"pilot-clean{no_fault}", method="outcomeserve", pilot=True, no_fault=no_fault)
    analysis = tmp_path / "analysis"
    cli("analyze_runs.py", "--results-root", dataset, "--output", analysis)
    matrix = json.loads((analysis / "gpu-matrix.json").read_text())
    assert matrix["run_count"] == 12
    assert matrix["pilot_runs_excluded"]["outcomeserve"]["COMPLETED"] == 2
    cells = {cell["prefix_target_tokens"]: cell for cell in matrix["cells"]}
    assert cells[2048]["delta_T_fault_to_verified"]["estimate"]["mean_restart_minus_outcomeserve"] == 10
    assert cells[2048]["break_even"]["break_even_q"] == pytest.approx(1 / 11)
    assert cells[8192]["delta_T_fault_to_verified"]["estimate"] is None
    assert cells[8192]["outcomes"]["outcomeserve"]["RECOVERY_VIOLATION"] == 1
    with (analysis / "gpu-heatmap.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    # RQ1 rows are synthetic layout fixtures, not a run of the CPU matrix.
    rq1 = tmp_path / "synthetic-rq1"
    rq1.mkdir()
    write_runs(rq1, [matrix_row("outcomeserve"), matrix_row("restart", correct=False, duplicates=1)])
    first, second = tmp_path / "figures-a", tmp_path / "figures-b"
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset.rglob("*") if path.is_file()}
    for output in (first, second):
        cli("make_figures.py", "--gpu-results", dataset, "--correctness", rq1, "--output", output)
    expected = {"figures-data.json", "rq1-correctness-table.tex", "rq2-resource-table.tex"}
    expected |= {f"{name}.{ext}" for name in ("rq1-correctness", "rq2-delta-t-heatmap", "rq3-break-even") for ext in ("pdf", "png")}
    assert {p.name for p in first.iterdir()} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    cli("make_figures.py", "--gpu-results", dataset, "--output", first, code=2)
    assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset.rglob("*") if path.is_file()}

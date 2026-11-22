"""RQ1 correctness cases: idempotent writes, an unqueryable external call, decision metrics."""

import json
import sqlite3

import pytest

from experiments.cpu.harness import resolve_progress, run
from outcomeserve.contracts import ContractError, request_fingerprint
from outcomeserve.storage import StateStore
from outcomeserve.tools import ExternalSupplierAPI, ToolService, tool_capabilities
from outcomeserve.verify import verify_workload
from outcomeserve.verify.business import verify_business
from outcomeserve.verify.decisions import decision_metrics, expected_behavior, unsafe_retries
from workloads import make_inventory_workload, validate_workload, with_external_step


def request(task, step, *, epoch=1):
    return {"task_id": task, "step_id": step["step_id"], "op_id": step["op_id"], "tool_name": step["tool_name"],
            "args": step["args"], "request_fingerprint": request_fingerprint(step["tool_name"], step["args"]),
            "owner_epoch": epoch}


def test_default_workload_is_unchanged_and_mixed_adds_idempotent_writes():
    default = make_inventory_workload("t", seed=33, steps=10)
    assert {s["tool_name"] for s in default["steps"]} == {"read_inventory", "decrement_inventory"}
    mixed = make_inventory_workload("t", seed=33, steps=10, tools="mixed")
    assert [s["tool_name"] for s in mixed["steps"][:3]] == ["read_inventory", "decrement_inventory", "set_reorder_level"]
    external = with_external_step(mixed, "step-05")
    assert external["steps"][4]["tool_name"] == "notify_supplier" and external["steps"][4]["args"] == {"sku": "item-b"}
    with pytest.raises(ValueError):
        with_external_step(mixed, "step-03")  # only a decrement becomes the external call
    bad = json.loads(json.dumps(mixed))
    bad["steps"][2]["args"]["level"] = 2.0
    with pytest.raises(ValueError):
        validate_workload(bad)


def test_capabilities_mark_the_external_call_unqueryable():
    assert tool_capabilities("set_reorder_level")["idempotent_business_operation"] is True
    external = tool_capabilities("notify_supplier")
    assert not external["supports_outcome_query"] and not external["supports_deduplication"]
    with pytest.raises(ContractError):
        ToolService.capabilities("notify_supplier")  # not a C2 tool


def test_idempotent_write_is_deduplicated_and_external_call_is_not(tmp_path):
    workload = with_external_step(make_inventory_workload("t", seed=33, steps=5, tools="mixed"), "step-02")
    store = StateStore(tmp_path / "db.sqlite")
    store.create_task("t", workload["initial_inventory"])
    tools, external = ToolService(store), ExternalSupplierAPI(store)
    setter, notify = workload["steps"][2], workload["steps"][1]
    first = tools.submit(**request("t", setter))
    assert first["result"] == {"ok": True, "sku": setter["args"]["sku"], "reorder_level": setter["args"]["level"]}
    assert tools.submit(**request("t", setter))["commit_receipt"] == first["commit_receipt"]
    receipts = {external.submit(**request("t", notify))["commit_receipt"] for _ in range(2)}
    assert len(receipts) == 2  # every external submission is a new effect
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM external_effects").fetchone()[0] == 2
        assert connection.execute("SELECT level FROM reorder_levels").fetchone()[0] == setter["args"]["level"]


def test_business_verifier_counts_external_duplicates_and_checks_the_executed_prefix(tmp_path):
    workload = with_external_step(make_inventory_workload("t", seed=33, steps=5, tools="mixed"), "step-02")
    store = StateStore(tmp_path / "db.sqlite")
    store.create_task("t", workload["initial_inventory"])
    tools, external = ToolService(store), ExternalSupplierAPI(store)
    tools.submit(**request("t", workload["steps"][0]))
    for _ in range(2):
        external.submit(**request("t", workload["steps"][1]))
    report = verify_business(store.path, workload)
    metrics = report["metrics"]
    assert metrics["duplicate_effects"] == 1 and metrics["external_duplicate_effects"] == 1
    assert metrics["steps_with_effects"] == ["step-01", "step-02"]
    assert metrics["prefix_state_mismatches"] == 0  # halted after step 2: prefix state is consistent
    assert not report["task_success"] and "DUPLICATE_EFFECT" in report["issues"]


def test_verifiers_accept_databases_without_the_new_tables(tmp_path):
    workload = make_inventory_workload("t", seed=33, steps=5)
    store = StateStore(tmp_path / "db.sqlite")
    store.create_task("t", workload["initial_inventory"])
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE external_effects")
        connection.execute("DROP TABLE reorder_levels")
    assert verify_business(store.path, workload)["metrics"]["external_effects"] == 0
    assert verify_workload(store.path, workload, checkpoint="in_progress")["valid"]


@pytest.mark.parametrize("events, unsafe", [
    ([("TOOL_SUBMIT", "step-02")], 0),
    ([("TOOL_SUBMIT", "step-02"), ("TOOL_SUBMIT", "step-02")], 1),
    ([("TOOL_SUBMIT", "step-02"), ("RECONCILIATION_FINISHED", "step-02", "NOT_COMMITTED"), ("TOOL_SUBMIT", "step-02")], 0),
    ([("TOOL_SUBMIT", "step-02"), ("RECONCILIATION_FINISHED", "step-02", "UNKNOWN"), ("TOOL_SUBMIT", "step-02")], 1),
    ([("TOOL_SUBMIT", "step-03"), ("TOOL_SUBMIT", "step-03")], 0),  # idempotent write
    ([("TOOL_SUBMIT", "step-01"), ("TOOL_SUBMIT", "step-01")], 0),  # read
])
def test_unsafe_retry_is_a_blind_resubmission_of_a_non_idempotent_step(events, unsafe):
    workload = make_inventory_workload("t", seed=33, steps=5, tools="mixed")
    rows = [{"event_type": e[0], "step_id": e[1], **({"outcome": e[2]} if len(e) > 2 else {})} for e in events]
    assert unsafe_retries(rows, workload) == unsafe


def test_expected_behavior_abstains_only_where_an_unqueryable_outcome_is_unknown():
    assert expected_behavior("external", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "runner") == "abstain"
    assert expected_behavior("external", "PRE_DISPATCH", "runner") == "abstain"
    assert expected_behavior("external", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "tool_response") == "abstain"
    assert expected_behavior("external", "DURING_GENERATION", "runner") == "complete"
    assert expected_behavior("external", "POST_CONTEXT_COMMIT", "runner") == "complete"
    assert expected_behavior("decrement", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "runner") == "complete"


def business(effects, *, duplicates=0, success=False):
    return {"task_success": success, "metrics": {"duplicate_effects": duplicates, "steps_with_effects": effects,
            "prefix_state_mismatches": 0, "prefix_contiguous": True,
            "tool_state_mismatches": 0, "audit_complete": True}}


def trace(*items):
    return [{"event_type": kind, "step_id": step, "monotonic_ns": index, "clock_id": "clock", "epoch": 1}
            for index, (kind, step) in enumerate(items)]


def test_decision_metrics_classify_abstention_and_continuation():
    workload = make_inventory_workload("t", seed=33, steps=5, tools="mixed")
    evidence = trace(("TOOL_SUBMIT", "step-01"), ("TOOL_SUBMIT", "step-02"), ("RECOVERY_ABSTAIN", "step-02"))
    correct = decision_metrics(evidence, workload, business(["step-01", "step-02"]), expected="abstain",
                               runner_exit_code=3, abstained_step="step-02")
    assert correct["correct"] and correct["continued_after_abstain"] == 0
    continued_events = evidence + [{**evidence[-1], "event_type": "TOOL_SUBMIT", "step_id": "step-03", "monotonic_ns": 4}]
    continued = decision_metrics(continued_events, workload, business(["step-01", "step-02", "step-03"]), expected="abstain",
                                 runner_exit_code=3, abstained_step="step-02")
    assert not continued["correct"] and continued["continued_after_abstain"] == 1
    false = decision_metrics(evidence, workload, business(["step-01"]), expected="complete", runner_exit_code=3,
                             abstained_step="step-02")
    assert false["false_abstain"] == 1 and not false["correct"]
    missed = decision_metrics(evidence[:2], workload, business(["step-01", "step-02"], success=True), expected="abstain",
                              runner_exit_code=0, abstained_step=None)
    assert missed["missed_abstain"] == 1 and not missed["correct"]


def test_fault_tool_selects_the_step_kind_at_every_boundary():
    workload = make_inventory_workload("t", seed=33, steps=10, tools="mixed")
    for boundary in ("DURING_GENERATION", "PRE_DISPATCH", "POST_CONTEXT_COMMIT"):
        idempotent = resolve_progress(workload, 0.5, boundary=boundary, fault_tool="idempotent")
        assert workload["steps"][idempotent["fault_step"] - 1]["tool_name"] == "set_reorder_level"
    with pytest.raises(ValueError):
        run("unused", fault_tool="external", boundary="DURING_RECOVERY")
    with pytest.raises(ValueError):
        run("unused", fault_tool="external", purpose="performance", method="restart")


@pytest.mark.parametrize("method, fault_tool, boundary, domain, correct, behavior, duplicates", [
    ("outcomeserve", "external", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "runner", True, "abstain", 0),
    ("outcomeserve", "external", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "tool_response", True, "abstain", 0),
    ("outcomeserve", "external", "DURING_GENERATION", "runner", True, "complete", 0),
    ("naive_retry", "external", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "runner", False, "complete", 1),
    ("naive_retry", "idempotent", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "runner", True, "complete", 0),
    ("outcomeserve", "idempotent", "DURING_RECOVERY", "runner", True, "complete", 0),
])
def test_live_correctness_cases(tmp_path, method, fault_tool, boundary, domain, correct, behavior, duplicates):
    result = run(tmp_path / "run", method=method, seed=33, steps=10, progress=0.5, boundary=boundary,
                 fault_domain=domain, fault_tool=fault_tool)
    assert result["valid"] and result["correct"] is correct
    assert result["decisions"]["observed_behavior"] == behavior
    assert result["decisions"]["duplicate_effects"] == duplicates
    if behavior == "abstain":
        assert result["runner_exit_code"] == 3 and not result["passed"]


def test_matrix_covers_every_required_cell():
    from analysis.correctness import cases
    matrix = cases(steps=(5,), seeds=(33,))
    assert len(matrix) == 62  # 3 tools x (4 boundaries x 4 methods + 4 dropped acks) + 2 recovery crashes
    assert {c["boundary"] for c in matrix} == {"DURING_GENERATION", "PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG",
                                               "POST_CONTEXT_COMMIT", "DURING_RECOVERY"}
    assert not [c for c in matrix if c["boundary"] == "DURING_RECOVERY" and
                (c["method"] != "outcomeserve" or c["fault_tool"] == "external")]
    assert len(cases()) == 62 * 9


def matrix_row(**values):
    base = {"case": "c", "steps": 5, "seed": 33, "fault_tool": "decrement", "fault_domain": "runner",
            "boundary": "PRE_DISPATCH", "method": "outcomeserve", "valid": True, "expected_behavior": "complete",
            "observed_behavior": "complete", "correct": True, "duplicate_effects": 0, "unsafe_retry": 0,
            "false_abstain": 0, "missed_abstain": 0, "continued_after_abstain": 0, "actual_progress": 0.4, "reason": ""}
    return {**base, **values}


def test_exit_criteria_fail_on_any_outcomeserve_violation_and_ignore_baselines():
    from analysis.correctness import exit_criteria, summarize
    rows = [matrix_row(), matrix_row(boundary="DURING_RECOVERY"),
            matrix_row(method="naive_retry", correct=False, duplicate_effects=1, unsafe_retry=1)]
    assert exit_criteria(rows)["all_pass"]
    summary = {(r["boundary"], r["method"]): r for r in summarize(rows)}
    assert summary[("PRE_DISPATCH", "naive_retry")]["runs_with_duplicates"] == 1
    for bad in (dict(duplicate_effects=1, correct=False), dict(missed_abstain=1, correct=False),
                dict(valid=False, correct=False), dict(false_abstain=1, correct=False)):
        assert not exit_criteria(rows + [matrix_row(**bad)])["all_pass"], bad
    # A duplicate on the unqueryable call is not a "supported tool" duplicate, but it is still incorrect.
    external = exit_criteria(rows + [matrix_row(fault_tool="external", duplicate_effects=1, correct=False)])
    assert external["checks"]["zero duplicate effects for supported (C2) tools"] and not external["all_pass"]


def test_matrix_script_refuses_to_overwrite_output(tmp_path):
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "out").mkdir()
    done = subprocess.run([sys.executable, str(root / "scripts/run_correctness_matrix.py"), "--output-dir",
                           str(tmp_path / "out")], capture_output=True, text=True)
    assert done.returncode == 2 and "output exists" in done.stderr


@pytest.mark.parametrize("change", [
    {"fingerprint": "wrong"}, {"result_json": '{"ok":true,"sku":"item-a","notified":false}'},
    {"received_epoch": 999}, {"commit_receipt": ""}, {"tool_name": "read_inventory"},
])
def test_external_effect_corruption_cannot_pass_the_business_audit(tmp_path, change):
    workload = with_external_step(make_inventory_workload("t", seed=33, steps=5, tools="mixed"), "step-02")
    store = StateStore(tmp_path / "db.sqlite")
    store.create_task("t", workload["initial_inventory"])
    ToolService(store).submit(**request("t", workload["steps"][0]))
    ExternalSupplierAPI(store).submit(**request("t", workload["steps"][1]))
    with sqlite3.connect(store.path) as connection:
        for key, value in change.items():
            connection.execute(f"UPDATE external_effects SET {key}=?", (value,))
    result = verify_business(store.path, workload)
    assert result["metrics"]["audit_complete"] is False
    assert result["metrics"]["duplicate_effects"] is None
    assert "UNEXPECTED_EFFECT" in result["issues"]


def test_executed_steps_must_be_a_contiguous_prefix(tmp_path):
    workload = make_inventory_workload("t", steps=5, tools="mixed")
    store = StateStore(tmp_path / "db.sqlite")
    store.create_task("t", workload["initial_inventory"])
    ToolService(store).submit(**request("t", workload["steps"][2]))
    result = verify_business(store.path, workload)
    assert not result["metrics"]["prefix_contiguous"]
    assert "EXECUTED_ORDER" in result["issues"]


@pytest.mark.parametrize("step", ["step-01", "step-02", "step-03"])
def test_any_submission_after_abstention_is_detected_even_on_the_same_or_an_earlier_step(step):
    workload = make_inventory_workload("t", steps=5, tools="mixed")
    events = trace(("TOOL_SUBMIT", "step-01"), ("RECOVERY_ABSTAIN", "step-02"), ("TOOL_SUBMIT", step))
    result = decision_metrics(events, workload, business(["step-01"]), expected="abstain",
                              runner_exit_code=3, abstained_step="step-02")
    assert result["continued_after_abstain"] == 1 and not result["correct"]


@pytest.mark.parametrize("kind", ["missing", "clock", "order", "halt"])
def test_missing_or_corrupted_trace_never_establishes_a_safe_abstention(kind):
    workload = make_inventory_workload("t", steps=5, tools="mixed")
    events = trace(("TOOL_SUBMIT", "step-01"), ("RECOVERY_ABSTAIN", "step-02"))
    if kind == "missing":
        events = []
    elif kind == "clock":
        events[-1]["clock_id"] = "other"
    elif kind == "order":
        events[0]["monotonic_ns"] = 10
    else:
        events = events[:1]
    result = decision_metrics(events, workload, business(["step-01"]), expected="abstain",
                              runner_exit_code=3, abstained_step="step-02")
    assert not result["evidence_valid"] and not result["correct"]
    assert result["unsafe_retry"] is None and result["continued_after_abstain"] is None


def test_unknown_metrics_and_incomplete_matrix_cannot_pass_exit_criteria():
    from analysis.correctness import cases, exit_criteria, summarize
    rows = [matrix_row(), matrix_row(boundary="DURING_RECOVERY")]
    assert not exit_criteria(rows, expected_cases=cases((5,), (33,)))["all_pass"]
    rows[0]["unsafe_retry"] = None
    assert not exit_criteria(rows)["all_pass"]
    summary = summarize(rows)
    assert summary[0]["unsafe_retry"] is None and summary[0]["unknown_metric_runs"] == 1


def test_matrix_gate_requires_valid_baselines_and_no_duplicate_cases():
    from analysis.correctness import cases, case_name, exit_criteria
    plan = cases((5,), (33,))
    rows = [matrix_row(**{k: c[k] for k in ("steps", "seed", "fault_tool", "fault_domain", "boundary", "method")},
                       case=case_name(c)) for c in plan]
    assert exit_criteria(rows, expected_cases=plan)["all_pass"]
    baseline = next(r for r in rows if r["method"] == "restart")
    baseline["valid"] = False
    assert not exit_criteria(rows, expected_cases=plan)["all_pass"]
    baseline["valid"] = True
    assert not exit_criteria(rows + [rows[0]], expected_cases=plan)["all_pass"]


def test_resume_refuses_different_plan_source_and_result_identity(tmp_path, monkeypatch):
    from scripts.run_correctness_matrix import prepare_output, validate_case_result
    from analysis.correctness import cases
    monkeypatch.setattr("scripts.run_correctness_matrix.source_files", lambda: {"source.py": b"version 1"})
    output = tmp_path / "matrix"
    config = {"steps": [5], "seeds": [33], "progress": 0.5}
    plan = cases((5,), (33,))
    prepare_output(output, config, plan, resume=False)
    prepare_output(output, config, plan, resume=True)
    assert (output / "source-snapshot.zip").exists()
    with pytest.raises(ValueError, match="differs"):
        prepare_output(output, {**config, "progress": 0.1}, plan, resume=True)
    monkeypatch.setattr("scripts.run_correctness_matrix.source_files", lambda: {"source.py": b"version 2"})
    with pytest.raises(ValueError, match="differs"):
        prepare_output(output, config, plan, resume=True)
    case = plan[0]
    result = {"schema_version": "1.1.0", "backend": "scripted_cpu", "method": case["method"],
              "valid": False, "passed": False, "fault_applied": False, "invalid_reasons": ["test"],
              "resources": {"measurement_valid": False}, **case, "requested_progress": 0.1,
              "no_fault": False, "purpose": "correctness"}
    with pytest.raises(ValueError, match="requested_progress"):
        validate_case_result(case, result)


@pytest.mark.parametrize("kwargs", [
    {"seeds": (33, 33)}, {"seeds": (-1,)}, {"steps": (5, 5)}, {"progress": float("nan")}, {"seeds": ()},
])
def test_matrix_rejects_ambiguous_or_invalid_arguments(kwargs):
    from analysis.correctness import cases
    with pytest.raises(ValueError):
        cases(**kwargs)


def test_replacement_worker_keeps_durable_abstention_without_contacting_tools(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.cpu.worker import run as run_worker
    from outcomeserve.log import DurableLog
    result = run(tmp_path / "run", fault_tool="external", steps=5, progress=0.5)
    assert result["correct"] and result["runner_exit_code"] == 3
    workload_path = tmp_path / "run/workload.json"
    workload = json.loads(workload_path.read_text())
    store = StateStore(tmp_path / "run/runtime.sqlite")
    before = DurableLog(store).read(workload["task_id"])
    epoch = store.advance_epoch(workload["task_id"], 2)
    def forbidden(*args, **kwargs):
        pytest.fail("a durably halted task must not contact tools")
    monkeypatch.setattr("experiments.cpu.worker.ToolClient.query", forbidden)
    monkeypatch.setattr("experiments.cpu.worker.ToolClient.submit", forbidden)
    monkeypatch.setattr("experiments.cpu.worker.ToolClient.submit_external", forbidden)
    args = SimpleNamespace(workload=workload_path, db=store.path, tool_endpoint="http://unused",
                           events=tmp_path / "halt-events.jsonl", checkpoint=tmp_path / "unused.json",
                           method="outcomeserve", epoch=epoch, inject=False, clock_id="test-clock")
    assert run_worker(args) == 3
    assert DurableLog(store).read(workload["task_id"]) == before

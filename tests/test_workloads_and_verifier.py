"""Positive fixtures and deliberate SQL corruption for the independent oracle."""

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from outcomeserve.contracts import request_fingerprint
from outcomeserve.context import scripted_context
from outcomeserve.log import DurableLog
from outcomeserve.storage import StateStore
from outcomeserve.tools import ToolService
from outcomeserve.verify import verify_workload
from workloads import load_workload, make_inventory_workload, validate_workload
from workloads.exercise import exercise_workload


ROOT = Path(__file__).resolve().parents[1]


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def prepare(store, workload, index=0):
    step = workload["steps"][index]
    identity = {"task_id": workload["task_id"], "step_id": step["step_id"],
                "op_id": step["op_id"], "owner_epoch": 1}
    fingerprint = request_fingerprint(step["tool_name"], step["args"])
    spec = store.bind_context(workload["task_id"], scripted_context(workload["model_revision"]))
    record = {"schema_version": "1.1.0", "context_spec_id": spec, **identity, "record_id": "prepare",
              "model_revision": workload["model_revision"], "record_type": "STEP_PREPARED",
              "payload": {"tool_name": step["tool_name"], "args": step["args"],
                          "request_fingerprint": fingerprint, "generated_token_ids": [], "generation_prompt_token_ids": []}}
    assert DurableLog(store).append(workload["task_id"], "prepare", 0, 1, record)["status"] == "APPENDED"
    return {**identity, "tool_name": step["tool_name"], "args": step["args"],
            "request_fingerprint": fingerprint}


@pytest.fixture
def empty(tmp_path):
    workload = make_inventory_workload("fixture", seed=33)
    store = StateStore(tmp_path / "state.sqlite")
    store.create_task(workload["task_id"], workload["initial_inventory"])
    return store, workload


@pytest.fixture
def completed(empty):
    store, workload = empty
    exercise_workload(store, workload)
    return store, workload


@pytest.mark.parametrize("steps", [5, 10, 20])
def test_mixed_workload_completes_against_independent_oracle(tmp_path, steps):
    workload = make_inventory_workload(f"size-{steps}", seed=33, steps=steps)
    store = StateStore(tmp_path / "state.sqlite")
    store.create_task(workload["task_id"], workload["initial_inventory"])
    exercise_workload(store, workload)
    report = verify_workload(store.path, workload)
    assert report["valid"] and report["consistency_pass"] and report["task_success"]
    assert report["metrics"]["completed_steps"] == steps
    assert report["metrics"]["orphan_operations"] == 0
    assert report["metrics"]["inventory_mismatches"] == 0
    # Final snapshots cannot supply temporal retry rates or effect audit counts.
    assert report["metrics"]["unsafe_retry"] is None
    assert report["metrics"]["duplicate_effects"] is None


def test_generation_is_reproducible_and_namespaces_operation_ids():
    first = make_inventory_workload("task:one", seed=33)
    assert first == make_inventory_workload("task:one", seed=33)
    other = make_inventory_workload("task:two", seed=33)
    assert first["initial_inventory"] == other["initial_inventory"]
    assert first["steps"][0]["op_id"] != other["steps"][0]["op_id"]
    assert first != make_inventory_workload("task:one", seed=34)
    assert {step["tool_name"] for step in first["steps"]} == {"read_inventory", "decrement_inventory"}


def test_manifest_rejects_duplicate_operation_identity():
    workload = make_inventory_workload("fixture")
    workload["steps"][1]["op_id"] = workload["steps"][0]["op_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_workload(workload)


def test_manifest_rejects_float_quantity_and_duplicate_json_keys(tmp_path):
    workload = make_inventory_workload("fixture")
    workload["steps"][1]["args"]["quantity"] = 1.0
    with pytest.raises(ValueError, match="integer JSON"):
        validate_workload(workload)
    path = tmp_path / "duplicate.json"
    path.write_text('{"seed":33,"seed":34}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate workload key"):
        load_workload(path)


def test_empty_task_is_in_progress_not_completed(empty):
    store, workload = empty
    report = verify_workload(store.path, workload, checkpoint="in_progress")
    assert report["consistency_pass"] and not report["task_success"]
    assert "TASK_INCOMPLETE" in codes(verify_workload(store.path, workload))


@pytest.mark.parametrize("commit_tool", [False, True])
def test_pending_windows_are_allowed_only_at_nonterminal_checkpoint(empty, commit_tool):
    store, workload = empty
    req = prepare(store, workload)
    if commit_tool:
        ToolService(store).submit(**req)
    report = verify_workload(store.path, workload, checkpoint="in_progress")
    assert report["consistency_pass"] and not report["task_success"]
    assert report["metrics"]["pending_operations"] == int(commit_tool)
    terminal = verify_workload(store.path, workload)
    assert not terminal["consistency_pass"]
    assert terminal["metrics"]["orphan_operations"] == int(commit_tool)


def test_unprepared_tool_commit_is_detected_even_in_progress(empty):
    store, workload = empty
    step = workload["steps"][0]
    ToolService(store).submit(task_id=workload["task_id"], step_id=step["step_id"],
        op_id=step["op_id"], owner_epoch=1, tool_name=step["tool_name"], args=step["args"],
        request_fingerprint=request_fingerprint(step["tool_name"], step["args"]))
    report = verify_workload(store.path, workload, checkpoint="in_progress")
    assert "MISSING_PREPARE" in codes(report)


def test_unexplained_extra_inventory_decrement_is_detected(completed):
    store, workload = completed
    with store.transaction() as conn:
        conn.execute("UPDATE inventory SET quantity=quantity-1 WHERE task_id=? AND sku='item-a'", (workload["task_id"],))
    report = verify_workload(store.path, workload)
    assert report["valid"] and not report["consistency_pass"] and not report["task_success"]
    assert "INVENTORY_STATE" in codes(report)


def test_corrupt_operation_result_cannot_be_hidden_by_matching_log(completed):
    store, workload = completed
    first = workload["steps"][0]
    forged = {"ok": True, "sku": first["args"]["sku"], "remaining": 999}
    with store.transaction() as conn:
        conn.execute("UPDATE operations SET result_json=? WHERE task_id=? AND step_id=?",
                     (json.dumps(forged), workload["task_id"], first["step_id"]))
        row = conn.execute("SELECT * FROM log_records WHERE sequence=2").fetchone()
        record = json.loads(row["record_json"])
        record["payload"]["result"] = forged
        conn.execute("UPDATE log_records SET record_json=? WHERE sequence=2", (json.dumps(record),))
    report = verify_workload(store.path, workload)
    assert "TOOL_RESULT" in codes(report)
    assert not report["consistency_pass"]


@pytest.mark.parametrize("corruption,code", [
    ("pointer", "CONTEXT_POINTER"), ("halt", "HALT_STATE"),
    ("tokens", "CONTEXT_TOKENS"), ("receipt", "LOG_TOOL_MISMATCH"),
    ("sequence", "LOG_SEQUENCE"), ("epoch", "LOG_EPOCH"),
    ("fingerprint", "LOG_PREPARE"), ("result_ref", "LOG_RESULT_REFERENCE"),
    ("log_schema", "LOG_SCHEMA"),
])
def test_verifier_rejects_corrupt_persistent_evidence(completed, corruption, code):
    store, workload = completed
    with store.transaction() as conn:
        if corruption == "pointer":
            conn.execute("UPDATE tasks SET context_record_id='missing'")
        elif corruption == "halt":
            conn.execute("UPDATE tasks SET halted=1")
        else:
            sequence = {"tokens": 3, "receipt": 2, "sequence": 1, "epoch": 1,
                        "fingerprint": 1, "result_ref": 3, "log_schema": 1}[corruption]
            record = json.loads(conn.execute("SELECT record_json FROM log_records WHERE sequence=?", (sequence,)).fetchone()[0])
            if corruption == "tokens": record["payload"]["token_ids"] = [999]
            if corruption == "receipt": record["payload"]["commit_receipt"] = "fake"
            if corruption == "sequence": record["sequence"] = 99
            if corruption == "epoch": record["owner_epoch"] = 2
            if corruption == "fingerprint": record["payload"]["request_fingerprint"] = "sha256:" + "0" * 64
            if corruption == "result_ref": record["payload"]["result_record_id"] = "other"
            if corruption == "log_schema": del record["payload"]["tool_name"]
            conn.execute("UPDATE log_records SET record_json=? WHERE sequence=?", (json.dumps(record), sequence))
    report = verify_workload(store.path, workload)
    assert code in codes(report)
    assert not report["consistency_pass"] and not report["task_success"]


def test_halted_checkpoint_does_not_claim_task_success_or_decision_accuracy(empty):
    store, workload = empty
    prepare(store, workload)
    step = workload["steps"][0]
    halted = {"schema_version": "1.1.0",
              "context_spec_id": store.bind_context(workload["task_id"], scripted_context(workload["model_revision"])), "task_id": workload["task_id"], "owner_epoch": 1,
              "step_id": step["step_id"], "op_id": step["op_id"], "record_id": "halt",
              "model_revision": workload["model_revision"], "record_type": "RECOVERY_ABSTAIN",
              "payload": {"outcome": "UNKNOWN", "reason": "simulated ambiguity fixture"}}
    assert DurableLog(store).append(workload["task_id"], "halt", 1, 1, halted)["status"] == "APPENDED"
    report = verify_workload(store.path, workload, checkpoint="halted")
    assert report["consistency_pass"] and not report["task_success"]
    assert report["observed_state"] == "halted"
    assert report["metrics"]["false_abstain"] is None
    assert "TASK_INCOMPLETE" in codes(verify_workload(store.path, workload))


def test_read_only_verifier_does_not_call_runtime_validation(completed, monkeypatch):
    store, workload = completed
    import outcomeserve.contracts as runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("verifier called runtime validator")
    monkeypatch.setattr(runtime, "validate_log", forbidden)
    monkeypatch.setattr(runtime, "request_fingerprint", forbidden)
    connection = sqlite3.connect(store.path)
    before = list(connection.iterdump())
    assert verify_workload(store.path, workload)["consistency_pass"]
    assert list(connection.iterdump()) == before
    connection.close()


def test_in_progress_verifier_rejects_torn_result_context_commit(completed):
    store, workload = completed
    with store.transaction() as conn:
        # A result-only tail is corruption even when no terminal checkpoint is requested.
        conn.execute("DELETE FROM log_records WHERE sequence>=3")
        conn.execute("UPDATE tasks SET context_record_id=NULL")
    report = verify_workload(store.path, workload, checkpoint="in_progress")
    assert "LOG_ATOMIC_COMMIT" in codes(report)
    assert not report["consistency_pass"]


def test_missing_database_is_invalid_and_is_not_created(tmp_path):
    database = tmp_path / "missing.sqlite"
    report = verify_workload(database, make_inventory_workload("fixture"))
    assert not report["valid"] and report["consistency_pass"] is None
    assert not database.exists()


def test_cli_writes_reusable_evidence_and_refuses_output_reuse(tmp_path):
    output = tmp_path / "evidence"
    command = [sys.executable, str(ROOT / "scripts/run_cpu_experiment.py"),
               "--output-dir", str(output), "--no-fault"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "verification.json").read_text())["business"]["task_success"]
    verify = subprocess.run([sys.executable, str(ROOT / "scripts/verify_run.py"),
        "--db", str(output / "runtime.sqlite"), "--workload", str(output / "workload.json")],
        cwd=ROOT, text=True, capture_output=True, timeout=15)
    assert verify.returncode == 0, verify.stderr
    # A second fixture run must not erase prior business state or experiment data.
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
    assert repeated.returncode != 0


def test_cli_exits_nonzero_for_detected_failure(completed, tmp_path):
    store, workload = completed
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(workload), encoding="utf-8")
    with store.transaction() as conn:
        conn.execute("UPDATE tasks SET context_record_id=NULL")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_run.py"),
        "--db", str(store.path), "--workload", str(path)],
        cwd=ROOT, text=True, capture_output=True, timeout=15)
    assert result.returncode == 1
    assert not json.loads(result.stdout)["consistency_pass"]

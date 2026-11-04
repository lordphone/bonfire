"""Atomic commits, meaningful baselines, real process/HTTP boundaries."""

import copy
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from analysis.pipeline import paired_comparison, break_even_curve, whole_task_break_even
from experiments.cpu.harness import run, resolve_progress
from experiments.cpu.worker import run as run_worker
from outcomeserve.context import scripted_context
from outcomeserve.contracts import request_fingerprint, validate_schema
from outcomeserve.log import DurableLog
from outcomeserve.serving.client import CompletionBackend, InferenceError
from outcomeserve.serving.process import InferenceProcess
from outcomeserve.storage import StateStore
from outcomeserve.tools import ToolService
from outcomeserve.verify.business import verify_business
from workloads import make_inventory_workload
from workloads.exercise import exercise_workload
from workloads.reset import reset_isolated_task


ROOT = Path(__file__).resolve().parents[1]
REV = "1cfa9a7208912126459214e8b04321603b3df60c"


def prepare(tmp_path):
    store = StateStore(tmp_path / "atomic.sqlite")
    store.create_task("atomic", {"item": 10})
    spec = store.bind_context("atomic", scripted_context(REV))
    args = {"sku": "item", "quantity": 1}
    fp = request_fingerprint("decrement_inventory", args)
    identity = dict(task_id="atomic", step_id="step-1", op_id="atomic:step-1", owner_epoch=1)
    body = dict(schema_version="1.1.0", **identity, context_spec_id=spec, record_id="prepared", model_revision=REV,
                record_type="STEP_PREPARED", payload=dict(tool_name="decrement_inventory", args=args,
                request_fingerprint=fp, generated_token_ids=[11, 12], generation_prompt_token_ids=[9]))
    log = DurableLog(store)
    prepared = log.append("atomic", "prepared", 0, 1, body)["record"]
    response = ToolService(store).submit(**identity, tool_name="decrement_inventory", args=args, request_fingerprint=fp)
    return store, log, prepared, response


def test_atomic_commit_is_idempotent_and_preserves_context_identity(tmp_path):
    store, log, prepared, response = prepare(tmp_path)
    result = log.commit_step(prepared, response, [11, 12, 13], owner_epoch=1, expected_sequence=1, commit_id="commit")
    assert result["status"] == "COMMITTED" and len(result["records"]) == 2
    assert log.committed_prefix("atomic")["payload"]["token_ids"] == [11, 12, 13]
    assert log.commit_step(prepared, response, [11, 12, 13], owner_epoch=1, expected_sequence=99, commit_id="commit")["status"] == "ALREADY_COMMITTED"
    assert log.commit_step(prepared, response, [11, 12, 99], owner_epoch=1, expected_sequence=3, commit_id="commit")["status"] == "RECORD_ID_CONFLICT"
    changed_response = copy.deepcopy(response)
    changed_response["result"]["ok"] = 1
    assert log.commit_step(prepared, changed_response, [11, 12, 13], owner_epoch=1,
                           expected_sequence=3, commit_id="commit")["status"] == "RECORD_ID_CONFLICT"
    changed = scripted_context(REV)
    changed["enable_thinking"] = True
    with pytest.raises(ValueError, match="cannot change"):
        store.bind_context("atomic", changed)


@pytest.mark.parametrize("phase,expected_records", [("before_pointer", 1), ("after_commit", 3)])
def test_abrupt_exit_cannot_split_new_result_and_context_commit(tmp_path, phase, expected_records):
    store, log, prepared, response = prepare(tmp_path)
    child = '''
import json,os,sys
from contextlib import contextmanager
from outcomeserve.storage import StateStore
from outcomeserve.log import DurableLog
store=StateStore(sys.argv[1])
if sys.argv[2]=='before_pointer':
    original=store.connect
    def connect():
        c=original()
        c.set_trace_callback(lambda sql: os._exit(77) if sql.startswith('UPDATE tasks SET context_record_id=') else None)
        return c
    store.connect=connect
else:
    original=store.transaction
    @contextmanager
    def transaction():
        with original() as c: yield c
        os._exit(77)
    store.transaction=transaction
DurableLog(store).commit_step(json.loads(sys.argv[3]),json.loads(sys.argv[4]),[11,12,13],owner_epoch=1,expected_sequence=1,commit_id='commit')
'''
    result = subprocess.run([sys.executable, "-c", child, str(store.path), phase, json.dumps(prepared), json.dumps(response)], cwd=ROOT, capture_output=True, timeout=15)
    assert result.returncode == 77, result.stderr
    assert len(log.read("atomic")) == expected_records
    assert (log.committed_prefix("atomic") is not None) == (expected_records == 3)


def test_schema_requires_prepared_tokens(tmp_path):
    _, _, prepared, _ = prepare(tmp_path)
    del prepared["payload"]["generated_token_ids"]
    with pytest.raises(ValueError):
        validate_schema("durable-log", prepared)


def test_result_cannot_bypass_atomic_commit(tmp_path):
    store, log, prepared, response = prepare(tmp_path)
    body = {key: value for key, value in prepared.items() if key != "sequence"}
    body.update(record_id="unsafe-result", record_type="TOOL_RESULT_COMMITTED",
                payload={key: response[key] for key in ("request_fingerprint", "result", "commit_receipt")})
    result = log.append("atomic", "unsafe-result", 1, 1, body)
    assert result["status"] == "INVALID_TRANSITION"
    assert len(log.read("atomic")) == 1


def test_false_success_result_schema_is_rejected():
    with pytest.raises(ValueError):
        validate_schema("run-result", {"schema_version": "1.1.0", "backend": "scripted_cpu", "method": "outcomeserve",
            "valid": False, "passed": True, "fault_applied": True, "invalid_reasons": [],
            "resources": {"measurement_valid": True}, "task_success": True, "task_to_verified_seconds": 1.0})


def test_pair_setting_mismatch_prevents_a_performance_estimate():
    rows = [record(1, "restart", 2), record(1, "outcomeserve", 1)]
    rows[0]["context_spec_id"] = "different-model-context"
    result = paired_comparison(rows)
    assert result["estimate"] is None and "mismatched setting" in result["incomplete_pairs"][0]["reason"]


@pytest.mark.parametrize("field", [
    "backend", "steps", "actual_progress", "fault_domain", "boundary", "purpose", "no_fault",
    "context_spec_id", "business_fixture_sha256", "hardware_id", "runtime_versions", "environment_lock_sha256", "apc",
])
@pytest.mark.parametrize("missing_methods", [("restart",), ("outcomeserve",), ("restart", "outcomeserve")])
def test_missing_pair_identity_prevents_a_performance_estimate(field, missing_methods):
    rows = [record(1, "restart", 2), record(1, "outcomeserve", 1)]
    for row in rows:
        if row["method"] in missing_methods:
            del row[field]
    result = paired_comparison(rows)
    assert result["estimate"] is None
    assert result["complete_pairs"] == 0
    assert field in result["incomplete_pairs"][0]["reason"]


@pytest.mark.parametrize("append_status", [
    "APPENDED", "ALREADY_APPENDED", "STALE_EPOCH", "SEQUENCE_CONFLICT", "INVALID_TRANSITION", "RECORD_ID_CONFLICT",
])
def test_worker_requires_durable_abstention_before_unknown_exit(tmp_path, monkeypatch, append_status):
    workload = make_inventory_workload("unknown-outcome")
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload), encoding="utf-8")
    store = StateStore(tmp_path / "worker.sqlite")
    store.create_task(workload["task_id"], workload["initial_inventory"])
    original_append = DurableLog.append

    def injected_append(log, *args):
        if args[-1]["record_type"] != "RECOVERY_ABSTAIN":
            return original_append(log, *args)
        if append_status == "ALREADY_APPENDED":
            assert original_append(log, *args)["status"] == "APPENDED"
        elif append_status != "APPENDED":
            return {"status": append_status}
        result = original_append(log, *args)
        assert result["status"] == append_status
        return result

    monkeypatch.setattr(DurableLog, "append", injected_append)
    monkeypatch.setattr("experiments.cpu.worker.ToolClient.submit",
                        lambda *args, **kwargs: {"status": "UNKNOWN", "reason": "unqueryable"})
    args = SimpleNamespace(workload=workload_path, db=store.path, tool_endpoint="http://unused",
                           events=tmp_path / "events.jsonl", checkpoint=tmp_path / "checkpoint.json",
                           method="outcomeserve", epoch=1, inject=False, fault_step=1, boundary="PRE_DISPATCH",
                           clock_id="test-clock")
    durable = append_status in {"APPENDED", "ALREADY_APPENDED"}
    if durable:
        assert run_worker(args) == 3
    else:
        with pytest.raises(RuntimeError, match=append_status):
            run_worker(args)
    records = DurableLog(store).read(workload["task_id"])
    assert [r["record_type"] for r in records] == ["STEP_PREPARED"] + (["RECOVERY_ABSTAIN"] if durable else [])
    with store.transaction() as connection:
        assert bool(store.task(connection, workload["task_id"])["halted"]) is durable
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    assert not any(event["event_type"] == "STEP_COMPLETED" for event in events)


@pytest.mark.parametrize("domain", [None, "inference"])
def test_new_fault_contract_requires_valid_domain_boundary_combination(domain):
    request = json.loads((ROOT / "examples/fault-injection.request.json").read_text())
    request["schema_version"] = "1.1.0"
    request.pop("fault_domain", None)
    if domain:
        request["fault_domain"] = domain
    with pytest.raises(ValueError):
        validate_schema("fault-injection", request)


def test_effect_audit_and_stock_roll_back_if_result_storage_fails(tmp_path):
    store = StateStore(tmp_path / "rollback.sqlite")
    store.create_task("rollback", {"item": 10})
    with store.transaction() as connection:
        connection.execute("CREATE TRIGGER reject_result BEFORE INSERT ON operations BEGIN SELECT RAISE(ABORT,'injected result failure'); END")
    args = {"sku": "item", "quantity": 1}
    with pytest.raises(sqlite3.IntegrityError):
        ToolService(store).submit(task_id="rollback", step_id="step-1", op_id="op-1", owner_epoch=1,
            tool_name="decrement_inventory", args=args, request_fingerprint=request_fingerprint("decrement_inventory", args))
    connection = store.connect()
    try:
        assert connection.execute("SELECT quantity FROM inventory").fetchone()[0] == 10
        assert connection.execute("SELECT count(*) FROM tool_effects").fetchone()[0] == 0
    finally:
        connection.close()


def test_apc_is_an_explicit_factor_not_a_rejected_configuration():
    from scripts.validate_infrastructure_contracts import validate_environment_lock
    lock = json.loads((ROOT / "infra/environment.lock.json").read_text())
    assert lock["serving"]["max_model_len"] == 32768
    lock["serving"]["enable_prefix_caching"] = True
    validate_environment_lock(lock)


def test_progress_records_real_boundary_instead_of_silently_claiming_ninety_percent():
    workload = make_inventory_workload("progress", steps=20)
    assert resolve_progress(workload, 0.1, boundary="PRE_DISPATCH")["actual_progress"] == 0.1
    position = resolve_progress(workload, 0.9, boundary="PRE_DISPATCH")
    assert position["requested_progress"] == 0.9 and position["actual_progress"] == 0.95


@pytest.mark.parametrize("method,passed", [("outcomeserve", True), ("restart", False), ("naive_retry", False), ("transcript_checkpoint", False)])
def test_correctness_keeps_external_state_and_detects_unsafe_baselines(tmp_path, method, passed):
    result = run(tmp_path / method, method=method)
    assert result["valid"] and result["fault_applied"]
    assert result["passed"] is passed
    assert result["tool_service_survived"] and result["runner_replaced"]
    assert result["metrics"]["duplicate_effects"] == (0 if passed else 1)
    assert result["resources"]["measurement_valid"]
    assert result["resources"]["peak_summed_rss_bytes"] > 10_000_000
    if passed:
        assert result["task_to_verified_seconds"] >= result["fault_to_verified_seconds"] > 0


def test_tool_timeout_recovers_without_replacing_runner(tmp_path):
    result = run(tmp_path / "tool-timeout", fault_domain="tool_response")
    assert result["valid"] and result["passed"] and not result["runner_replaced"]
    assert result["tool_service_survived"] and result["metrics"]["duplicate_effects"] == 0


@pytest.mark.parametrize("boundary", ["DURING_GENERATION", "PRE_DISPATCH", "POST_CONTEXT_COMMIT", "DURING_RECOVERY"])
def test_expanded_cpu_boundaries_and_repeated_recovery_crash(tmp_path, boundary):
    result = run(tmp_path / boundary, boundary=boundary)
    assert result["valid"] and result["passed"], result
    assert result["fault_count"] == (2 if boundary == "DURING_RECOVERY" else 1)
    assert result["metrics"]["duplicate_effects"] == 0


def test_performance_restart_resets_only_isolated_simulator_and_measures_reset(tmp_path):
    output = tmp_path / "performance"
    result = run(output, method="restart", purpose="performance")
    assert result["valid"] and result["passed"]
    assert result["reset_seconds"] is not None and result["reset_seconds"] > 0
    assert (output / "before-reset.sqlite").exists()
    store = StateStore(tmp_path / "multi.sqlite")
    store.create_task("one", {"sku": 1})
    store.create_task("two", {"sku": 1})
    with pytest.raises(ValueError, match="isolated"):
        reset_isolated_task(store, "one", {"sku": 1}, expected_epoch=1)


def test_resource_monitor_failure_is_retained_instead_of_losing_run(tmp_path, monkeypatch):
    class BrokenMonitor:
        def __init__(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def stop(self, **kwargs): raise RuntimeError("injected monitor failure")
    monkeypatch.setattr("experiments.cpu.harness.ResourceMonitor", BrokenMonitor)
    output = tmp_path / "monitor-failed"
    result = run(output, purpose="performance", no_fault=True)
    assert result["task_success"] and not result["passed"]
    assert not result["resources"]["measurement_valid"]
    assert (output / "result.json").exists()


def test_missing_effect_audit_never_means_zero_duplicates(tmp_path):
    workload = make_inventory_workload("audit")
    store = StateStore(tmp_path / "audit.sqlite")
    store.create_task(workload["task_id"], workload["initial_inventory"])
    exercise_workload(store, workload)
    assert verify_business(store.path, workload)["consistency_pass"]
    with store.transaction() as connection:
        connection.execute("DELETE FROM tool_effects")
    report = verify_business(store.path, workload)
    assert not report["consistency_pass"]
    assert not report["metrics"]["audit_complete"]
    assert report["metrics"]["duplicate_effects"] is None


def spec():
    value = scripted_context(REV)
    value.update(model_id="Qwen/Qwen3-4B", tokenizer_id="Qwen/Qwen3-4B", backend="vllm")
    return value


def test_real_completion_contract_requires_returned_token_ids():
    backend = CompletionBackend("http://localhost", "Qwen/Qwen3-4B", spec())
    reply = {"choices": [{"text": '{"tool_name":"read_inventory","args":{"sku":"item"}}', "token_ids": [42, 43]}]}
    with patch("outcomeserve.serving.client.urlopen", return_value=BytesIO(json.dumps(reply).encode())) as transport:
        output = backend.generate([1, 2])
        payload = json.loads(transport.call_args.args[0].data)
    assert payload["prompt"] == [1, 2] and output["generated_token_ids"] == [42, 43]
    del reply["choices"][0]["token_ids"]
    with patch("outcomeserve.serving.client.urlopen", return_value=BytesIO(json.dumps(reply).encode())):
        with pytest.raises(InferenceError, match="no text re-tokenization"):
            backend.generate([1, 2])


def test_stream_fault_requires_an_unfinished_real_token_chunk():
    backend = CompletionBackend("http://localhost", "Qwen/Qwen3-4B", spec())
    calls = []
    event = b'data: {"choices":[{"token_ids":[42],"finish_reason":null}]}\n\n'
    with patch("outcomeserve.serving.client.urlopen", return_value=BytesIO(event)):
        backend.generate_until_fault([1], lambda tokens: calls.append(tokens))
    assert calls == [[42]]
    ended = b'data: {"choices":[{"token_ids":[42],"finish_reason":"stop"}]}\n\n'
    with patch("outcomeserve.serving.client.urlopen", return_value=BytesIO(ended)):
        with pytest.raises(InferenceError, match="precondition"):
            backend.generate_until_fault([1], lambda tokens: calls.append(tokens))


def test_streaming_completion_preserves_generated_ids_and_observes_first_arrival():
    backend = CompletionBackend("http://localhost", "Qwen/Qwen3-4B", spec())
    call = json.dumps({"tool_name": "read_inventory", "args": {"sku": "item"}})
    chunks = ["data: " + json.dumps({"choices": [{"text": call[:20], "token_ids": [1, 2], "finish_reason": None}]}),
              "data: " + json.dumps({"choices": [{"text": call[20:], "token_ids": [3], "finish_reason": "stop"}]})]
    arrivals = []
    with patch("outcomeserve.serving.client.urlopen", return_value=BytesIO(("\n\n".join(chunks) + "\n").encode())):
        result = backend.generate_stream([99], lambda tokens: arrivals.append(tokens))
    assert result["generated_token_ids"] == [1, 2, 3] and arrivals == [[1, 2]]


def test_inference_lifecycle_starts_and_terminates_owned_real_process():
    # Reserve without listening; no external pytest networking plugin required.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        process = InferenceProcess([sys.executable, "-c", "import time; time.sleep(10)"],
                                  f"http://127.0.0.1:{reserved.getsockname()[1]}", "model", timeout=2)
        try:
            with patch("outcomeserve.serving.process.urlopen", return_value=BytesIO(b'{"data":[{"id":"model"}]}')):
                pid = process.start()
            assert pid > 0 and process.process.poll() is None
        finally:
            process.stop()
        assert process.process.poll() is not None


def test_existing_endpoint_cannot_masquerade_as_a_cold_restart():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        process = InferenceProcess([sys.executable, "-c", "pass"], f"http://127.0.0.1:{port}", "model")
        with pytest.raises(RuntimeError, match="already in use"):
            process.start()
        assert process.process is None


def record(seed, method, value, *, no_fault=False, passed=True):
    return {"pair_key": f"s{seed}:setting:{no_fault}", "method": method, "valid": True, "passed": passed,
            "backend": "scripted_cpu", "steps": 5, "actual_progress": 0.4,
            "fault_domain": "runner", "boundary": "PRE_DISPATCH", "purpose": "performance",
            "context_spec_id": "test-context", "business_fixture_sha256": "test-fixture",
            "hardware_id": "test-host", "runtime_versions": {"python": "3.12"},
            "environment_lock_sha256": "test-environment", "apc": False,
            "no_fault": no_fault, "task_to_verified_seconds": value, "fault_to_verified_seconds": value / 2}


def test_paired_bootstrap_uses_differences_and_gates_failures():
    records = [record(1, "restart", 5), record(1, "outcomeserve", 3), record(2, "restart", 8), record(2, "outcomeserve", 4)]
    result = paired_comparison(records, metric="task_to_verified_seconds", bootstrap_samples=200)
    assert result["estimate"]["mean_restart_minus_outcomeserve"] == 3
    assert result["estimate"]["bootstrap_ci95"] == [2, 4]
    records[0]["passed"] = False
    assert paired_comparison(records)["estimate"] is None


def test_break_even_uses_total_task_cost_and_hypothetical_q():
    records = [record(1, "restart", 10, no_fault=True), record(1, "outcomeserve", 12, no_fault=True),
               record(1, "restart", 30), record(1, "outcomeserve", 22)]
    result = whole_task_break_even(records)
    assert result["H"] == 2 and result["S"] == 8 and result["break_even_q"] == 0.2
    assert break_even_curve(-1, 3)["break_even_q"] is None
    assert whole_task_break_even(records[:2])["curve"] is None


def test_duplicate_pairs_and_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison([record(1, "restart", 1), record(1, "restart", 2)])
    with pytest.raises(ValueError):
        break_even_curve(1, 2, [-0.1])


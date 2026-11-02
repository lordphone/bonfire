"""Database assertions and abrupt process exits for Victoria's CPU components."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier, Event

import pytest

from outcomeserve.contracts import ContractError, request_fingerprint, validate_exchange
from outcomeserve.context import scripted_context, context_spec_id
from outcomeserve.log import DurableLog
from outcomeserve.storage import OwnershipConflict, StateStore, TaskHalted
from outcomeserve.tools import ToolService


ROOT = Path(__file__).resolve().parents[1]
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def request(**overrides):
    args = {"sku": "item", "quantity": 1}
    result = dict(task_id="task", step_id="step-1", op_id="task:step-1",
                  tool_name="decrement_inventory", args=args, owner_epoch=1,
                  request_fingerprint=request_fingerprint("decrement_inventory", args))
    result.update(overrides)
    return result


def query(service, **overrides):
    req = request(**overrides)
    return service.query(**{k: v for k, v in req.items() if k not in {"args", "tool_name"}})


def body(kind="STEP_PREPARED", *, record_id="prepare", owner_epoch=1, payload=None):
    req = request()
    if payload is None:
        payload = {k: req[k] for k in ("tool_name", "args", "request_fingerprint")}
    if kind == "STEP_PREPARED":
        payload = {**payload, "generated_token_ids": [], "generation_prompt_token_ids": []}
    return dict(schema_version="1.1.0", context_spec_id=context_spec_id(scripted_context(REVISION)), task_id="task", step_id="step-1",
                op_id="task:step-1", owner_epoch=owner_epoch, record_id=record_id,
                model_revision=REVISION, record_type=kind, payload=payload)


def append(log, record, expected):
    return log.append(record["task_id"], record["record_id"], expected,
                      record["owner_epoch"], record)


def context_body(**overrides):
    return body("CONTEXT_COMMITTED", record_id="context", payload={
        "token_ids": [101, 102, 103], "result_record_id": "result",
    }, **overrides)


@pytest.fixture
def store(tmp_path):
    store = StateStore(tmp_path / "runtime.sqlite")
    store.create_task("task", {"item": 10})
    store.bind_context("task", scripted_context(REVISION))
    return store


def facts(store):
    connection = sqlite3.connect(store.path)
    try:
        return (
            connection.execute("SELECT quantity FROM inventory WHERE task_id='task' AND sku='item'").fetchone()[0],
            connection.execute("SELECT count(*) FROM operations WHERE task_id='task'").fetchone()[0],
        )
    finally:
        connection.close()


def test_duplicate_submit_is_durable_and_returns_original_result(store):
    service = ToolService(store)
    first = service.submit(**request())
    reopened = ToolService(StateStore(store.path))
    assert reopened.submit(**request()) == first
    assert query(reopened) == first
    validate_exchange({"schema_version": "1.0.0", "kind": "submit", **request()}, first)
    assert facts(store) == (9, 1)


def test_concurrent_submissions_execute_business_effect_once(store):
    gate = Barrier(8)
    def run(_):
        gate.wait(timeout=10)
        return ToolService(store).submit(**request())
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(run, range(8)))
    assert all(response == responses[0] for response in responses)
    assert facts(store) == (9, 1)


def test_changed_parameters_do_not_execute(store):
    service = ToolService(store)
    service.submit(**request())
    args = {"sku": "item", "quantity": 2}
    response = service.submit(**request(args=args, request_fingerprint=request_fingerprint("decrement_inventory", args)))
    assert response["status"] == "FINGERPRINT_MISMATCH"
    assert facts(store) == (9, 1)
    with pytest.raises(ContractError, match="fingerprint"):
        service.submit(**request(args=args))


def test_identity_collision_is_a_protocol_error(store):
    service = ToolService(store)
    service.submit(**request())
    with pytest.raises(ContractError, match="different step"):
        service.submit(**request(step_id="other-step"))
    with pytest.raises(ContractError, match="different op_id"):
        service.submit(**request(op_id="other-operation"))
    assert facts(store) == (9, 1)


def test_epoch_fences_both_tool_and_log_but_preserves_old_result(store):
    service, log = ToolService(store), DurableLog(store)
    assert append(log, body(), 0)["status"] == "APPENDED"
    response = service.submit(**request())
    assert store.advance_epoch("task", 1) == 2
    with pytest.raises(OwnershipConflict):
        store.advance_epoch("task", 1)
    assert service.submit(**request())["status"] == "STALE_EPOCH"
    assert query(service)["status"] == "STALE_EPOCH"
    assert append(log, body(), 0)["status"] == "STALE_EPOCH"
    recovered = query(service, owner_epoch=2)
    assert recovered["result"] == response["result"]
    assert recovered["commit_receipt"] == response["commit_receipt"]
    assert log.commit_step(log.read("task")[0], recovered, [101], owner_epoch=2,
                           expected_sequence=1, commit_id="recovered")["status"] == "COMMITTED"
    assert facts(store) == (9, 1)


def test_epoch_advance_serializes_with_inflight_effect(store):
    entered, release, advancing = Event(), Event(), Event()
    service = ToolService(store)
    original = service._execute_business
    def paused(connection, req):
        result = original(connection, req)
        entered.set()
        assert release.wait(10)
        return result
    service._execute_business = paused
    def advance():
        advancing.set()
        return store.advance_epoch("task", 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        effect = pool.submit(service.submit, **request())
        try:
            assert entered.wait(10)
            replacement = pool.submit(advance)
            assert advancing.wait(10)
        finally:
            release.set()
        assert effect.result(timeout=10)["status"] == "COMMITTED"
        assert replacement.result(timeout=10) == 2
    assert ToolService(store).submit(**request())["status"] == "STALE_EPOCH"
    assert facts(store) == (9, 1)


def test_inflight_query_timeout_is_unknown_not_absence(store):
    fast_store = StateStore(store.path, timeout=0.01)
    with store.transaction():
        assert query(ToolService(fast_store))["status"] == "UNKNOWN"
        assert ToolService(fast_store).submit(**request())["status"] == "UNKNOWN"
    assert query(ToolService(store))["status"] == "NOT_COMMITTED"
    assert facts(store) == (10, 0)


def test_business_rejection_is_committed_and_deduplicated(store):
    args = {"sku": "item", "quantity": 11}
    req = request(args=args, request_fingerprint=request_fingerprint("decrement_inventory", args))
    service = ToolService(store)
    response = service.submit(**req)
    assert response["status"] == "COMMITTED"
    assert response["result"]["error"] == "INSUFFICIENT_INVENTORY"
    assert service.submit(**req) == response
    assert facts(store) == (10, 1)


def test_post_commit_ack_loss_preserves_effect_and_receipt(store):
    def drop(response):
        assert response["status"] == "COMMITTED"
        assert facts(store) == (9, 1)
        raise TimeoutError("injected response loss")
    with pytest.raises(TimeoutError):
        ToolService(store, after_commit=drop).submit(**request())
    response = query(ToolService(store))
    assert response["status"] == "COMMITTED"
    assert ToolService(store, after_commit=drop).submit(**request()) == response
    assert facts(store) == (9, 1)


def test_context_commit_and_reopen_keep_prefix_consistent(store):
    log = DurableLog(store)
    assert append(log, body(), 0)["status"] == "APPENDED"
    response = ToolService(store).submit(**request())
    assert log.committed_prefix("task") is None
    assert log.commit_step(log.read("task")[0], response, [101, 102, 103], owner_epoch=1,
                           expected_sequence=1, commit_id="completed")["status"] == "COMMITTED"
    reopened = DurableLog(StateStore(store.path))
    assert reopened.committed_prefix("task") == reopened.read("task")[-1]
    assert reopened.committed_prefix("task")["payload"]["token_ids"] == [101, 102, 103]
    assert len(reopened.read("task", after_sequence=1)) == 2


def test_append_retry_conflict_and_invalid_transition_do_not_change_tail(store):
    log = DurableLog(store)
    prepared = body()
    assert append(log, prepared, 0)["status"] == "APPENDED"
    assert append(log, prepared, 99)["status"] == "ALREADY_APPENDED"
    changed = copy.deepcopy(prepared)
    changed["payload"]["args"]["quantity"] = 2
    assert append(log, changed, 1)["status"] == "RECORD_ID_CONFLICT"
    assert append(log, body(record_id="next"), 0)["status"] == "SEQUENCE_CONFLICT"
    assert append(log, context_body(), 1)["status"] == "INVALID_TRANSITION"
    assert len(log.read("task")) == 1
    assert log.committed_prefix("task") is None


def test_competing_log_appends_cannot_both_claim_same_tail(store):
    gate = Barrier(2)
    def run(record_id):
        gate.wait(timeout=10)
        return append(DurableLog(store), body(record_id=record_id), 0)["status"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(run, ["a", "b"]))
    assert sorted(statuses) == ["APPENDED", "SEQUENCE_CONFLICT"]
    assert len(DurableLog(store).read("task")) == 1


def test_abstain_persists_and_blocks_tools_across_epoch_replacement(store):
    log = DurableLog(store)
    append(log, body(), 0)
    halted = body("RECOVERY_ABSTAIN", record_id="halt", payload={"outcome": "UNKNOWN", "reason": "unqueryable external effect"})
    assert append(log, halted, 1)["status"] == "APPENDED"
    assert append(log, halted, 0)["status"] == "ALREADY_APPENDED"
    store.advance_epoch("task", 1)
    reopened = StateStore(store.path)
    with pytest.raises(TaskHalted):
        ToolService(reopened).submit(**request(owner_epoch=2))
    checkpoint = body(record_id="next-prepare", owner_epoch=2)
    assert append(DurableLog(reopened), checkpoint, 2)["status"] == "INVALID_TRANSITION"
    assert facts(store) == (10, 0)


@pytest.mark.parametrize("phase,expected", [("before_commit", (10, 0)), ("after_commit", (9, 1))])
def test_abrupt_process_exit_preserves_atomic_effect_and_result(store, phase, expected):
    child = '''
import json, os, sys
from outcomeserve.storage import StateStore
from outcomeserve.tools import ToolService
store = StateStore(sys.argv[1])
service = ToolService(store)
if sys.argv[2] == 'before_commit':
    execute = service._execute_business
    def crash(connection, request):
        execute(connection, request)
        os._exit(73)
    service._execute_business = crash
else:
    service.after_commit = lambda response: os._exit(73)
service.submit(**json.loads(sys.argv[3]))
'''
    process = subprocess.run(
        [sys.executable, "-c", child, str(store.path), phase, json.dumps(request())],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert process.returncode == 73, process.stderr
    assert facts(store) == expected
    service = ToolService(StateStore(store.path))
    assert query(service)["status"] == ("NOT_COMMITTED" if phase == "before_commit" else "COMMITTED")
    service.submit(**request())
    assert facts(store) == (9, 1)


def test_task_setup_cannot_reset_committed_external_state(store):
    ToolService(store).submit(**request())
    with pytest.raises(sqlite3.IntegrityError):
        store.create_task("task", {"item": 10})
    assert facts(store) == (9, 1)



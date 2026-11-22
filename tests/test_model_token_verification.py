"""Protocol verification of real-inference runs (model-generated token IDs) and their results."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.runresult import assemble, read_events
from outcomeserve.context import scripted_context
from outcomeserve.contracts import request_fingerprint
from outcomeserve.log import DurableLog
from outcomeserve.storage import StateStore
from outcomeserve.tools import ToolService
from outcomeserve.verify import committed_tokens, verify_workload
from test_run_results import GOOD_RESOURCES, events, identity, protocol_report, verification
from workloads import make_inventory_workload

ROOT = Path(__file__).resolve().parents[1]


def model_run(path, *, steps=None, tamper=None, task_id="gpu-task", task_steps=5, snapshot=None, variant=0):
    """Execute a workload as a real-inference runner would: tokens come from a 'model',
    each prompt extends the committed context, each context adds generation and result."""
    workload = make_inventory_workload(task_id, seed=33, steps=task_steps)
    store = StateStore(path)
    store.create_task(task_id, workload["initial_inventory"])
    context = store.bind_context(task_id, scripted_context(workload["model_revision"]))
    log, tools = DurableLog(store), ToolService(store)
    tokens = [7, 8, 9]  # tokenized instruction
    for index, step in enumerate(workload["steps"][:steps]):
        prompt = tokens + [100 + index]
        generated = [200 + index + (variant if index >= task_steps // 2 else 0), 201 + index]
        fingerprint = request_fingerprint(step["tool_name"], step["args"])
        body = {"schema_version": "1.1.0", "context_spec_id": context, "task_id": task_id,
                "step_id": step["step_id"], "op_id": step["op_id"], "owner_epoch": 1,
                "record_id": f"{step['step_id']}:prepare", "model_revision": workload["model_revision"],
                "record_type": "STEP_PREPARED", "payload": {
                    "tool_name": step["tool_name"], "args": step["args"], "request_fingerprint": fingerprint,
                    "generated_token_ids": generated, "generation_prompt_token_ids": prompt}}
        prepared = log.append(task_id, body["record_id"], len(log.read(task_id)), 1, body)["record"]
        response = tools.submit(task_id=task_id, step_id=step["step_id"], op_id=step["op_id"],
                                tool_name=step["tool_name"], args=step["args"], request_fingerprint=fingerprint,
                                owner_epoch=1)
        context_tokens = prompt + generated + [300 + index]
        if tamper == "drop_generation" and index == 2:
            context_tokens = prompt + [300 + index]
        result = log.commit_step(prepared, response, context_tokens, owner_epoch=1,
                                 expected_sequence=len(log.read(task_id)), commit_id=f"{step['step_id']}:commit")
        assert result["status"] == "COMMITTED", result
        tokens = context_tokens
        if snapshot is not None and index + 1 == task_steps // 2:
            import sqlite3
            source, target = store.connect(), sqlite3.connect(snapshot)
            source.backup(target)
            source.close()
            target.close()
        if tamper == "rewrite_history" and index == 1:
            tokens = [7, 8, 99] + tokens[3:]  # next prompt no longer extends the committed context
    return workload


def test_scripted_mode_rejects_model_tokens_and_model_mode_accepts_them(tmp_path):
    workload = model_run(tmp_path / "db.sqlite")
    scripted = verify_workload(tmp_path / "db.sqlite", workload)
    assert not scripted["consistency_pass"] and "CONTEXT_TOKENS" in {i["code"] for i in scripted["issues"]}
    model = verify_workload(tmp_path / "db.sqlite", workload, token_source="model")
    assert model["consistency_pass"] and model["task_success"], model["issues"]


@pytest.mark.parametrize("tamper, code", [("drop_generation", "CONTEXT_TOKENS"), ("rewrite_history", "PROMPT_TOKENS")])
def test_model_mode_requires_an_append_only_transcript(tmp_path, tamper, code):
    workload = model_run(tmp_path / "db.sqlite", tamper=tamper)
    report = verify_workload(tmp_path / "db.sqlite", workload, token_source="model")
    assert not report["consistency_pass"] and code in {i["code"] for i in report["issues"]}


def test_verify_run_compares_final_prefixes(tmp_path):
    workload = model_run(tmp_path / "a.sqlite")
    model_run(tmp_path / "b.sqlite")
    model_run(tmp_path / "c.sqlite", steps=4)
    (tmp_path / "w.json").write_text(json.dumps(workload))
    assert committed_tokens(tmp_path / "a.sqlite", "gpu-task") == committed_tokens(tmp_path / "b.sqlite", "gpu-task")
    command = [sys.executable, str(ROOT / "scripts/verify_run.py"), "--db", str(tmp_path / "a.sqlite"),
               "--workload", str(tmp_path / "w.json"), "--token-source", "model", "--same-prefix-as"]
    same = subprocess.run(command + [str(tmp_path / "b.sqlite")], capture_output=True, text=True)
    assert same.returncode == 0, same.stdout + same.stderr
    shorter = subprocess.run(command + [str(tmp_path / "c.sqlite")], capture_output=True, text=True)
    assert shorter.returncode == 1 and json.loads(shorter.stdout)["same_prefix"]["equal"] is False


def test_protocol_violation_is_a_recovery_violation_and_only_for_outcomeserve():
    ident = identity()
    rows = events(ident, fault=30, token=40, verified=60)
    bad = {**protocol_report(ident), "consistency_pass": False, "task_success": False, "issues": [{"code": "PROMPT_TOKENS"}]}
    result = assemble(ident, rows, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=bad)
    assert result["outcome"] == "RECOVERY_VIOLATION" and result["protocol_issues"] == ["PROMPT_TOKENS"]
    good = assemble(ident, rows, verification(), GOOD_RESOURCES, runner_exit_code=0,
                    protocol=protocol_report(ident))
    assert good["outcome"] == "COMPLETED" and good["protocol_issues"] == []
    with pytest.raises(ValueError, match="durable log"):
        assemble(identity("restart"), rows, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=bad)


def test_raw_runner_events_are_converted_with_the_run_identity(tmp_path):
    ident = identity()
    raw = tmp_path / "runner-events.jsonl"
    raw.write_text("\n".join(json.dumps(e) for e in (
        {"event_type": "FIRST_NEW_TOKEN", "monotonic_ns": 40_000_000_000, "clock_id": "clock-a", "epoch": 1,
         "step_id": "step-11", "observed_at": "client_stream"},
        {"event_type": "TASK_COMPLETED", "monotonic_ns": 55_000_000_000, "clock_id": "clock-a", "epoch": 1,
         "step_id": "none"})) + "\n")
    converted = read_events([raw], ident)
    assert converted[0]["run_id"] == ident["run_id"] and converted[0]["sequence"] == 1
    assert converted[0]["details"] == {"epoch": 1, "step_id": "step-11", "observed_at": "client_stream"}
    supervisor = [e for e in events(ident, fault=30, verified=60)]
    result = assemble(ident, supervisor + converted, verification(), GOOD_RESOURCES, runner_exit_code=0,
                      protocol=protocol_report(ident))
    assert result["outcome"] == "COMPLETED" and result["fault_to_first_new_token_seconds"] == 10
    with pytest.raises(Exception):
        read_events([raw])  # without an identity, raw lines are not measurement events

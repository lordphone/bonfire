"""Contract-level tests; transactional runtime acceptance remains separate."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


import outcomeserve.contracts as contracts

ROOT = Path(__file__).resolve().parents[1]


def log():
    return contracts.load_json("examples/durable-log.records.json")


def messages():
    return contracts.load_json("examples/tool-service.messages.json")


def test_contract_fixtures_and_cross_interface_identity():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_recovery_contracts.py")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_fingerprint_is_order_independent_and_unicode_preserving():
    args = {"name": "tea \U0001f375", "quantity": 1}
    actual = contracts.request_fingerprint("reserve", args)
    assert actual == contracts.request_fingerprint("reserve", {"quantity": 1, "name": args["name"]})
    assert actual != contracts.request_fingerprint("reserve", {**args, "quantity": 2})
    assert actual != contracts.request_fingerprint("release", args)


@pytest.mark.parametrize("args", [{"n": 1.0}, {"n": float("nan")}, {"n": 2**53}, {1: "key"}, {"nested": [0.5]}])
def test_unsupported_fingerprint_values_are_rejected(args):
    with pytest.raises(contracts.ContractError):
        contracts.request_fingerprint("tool", args)


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(contracts.ContractError, match="duplicate JSON key"):
        json.loads('{"n":1,"n":2}', object_pairs_hook=contracts._unique_keys)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r[1].update(sequence=7), "sequence"),
    (lambda r: r[1].update(record_id=r[0]["record_id"]), "record_id"),
    (lambda r: r[2].update(owner_epoch=0), "valid"),
    (lambda r: r[2].update(op_id="new-operation"), "identity"),
    (lambda r: r[2]["payload"].update(result_record_id="other-result"), "different result"),
    (lambda r: r[0]["payload"]["args"].update(quantity=9), "fingerprint"),
    (lambda r: r[1]["payload"].update(request_fingerprint="sha256:" + "0" * 64), "fingerprint"),
    (lambda r: r[2].update(model_revision="0" * 40), "model_revision"),
])
def test_log_rejects_corrupt_evidence(mutation, match):
    records = log()
    mutation(records)
    with pytest.raises(contracts.ContractError, match=match):
        contracts.validate_log(records)


def test_old_epoch_cannot_append_after_new_owner():
    records = log()
    records[1]["owner_epoch"] = 2
    with pytest.raises(contracts.ContractError, match="stale owner_epoch"):
        contracts.validate_log(records)


def test_context_cannot_skip_tool_result():
    records = log()
    del records[1]
    records[-1]["sequence"] = 2
    with pytest.raises(contracts.ContractError, match="staged tool result"):
        contracts.validate_log(records)


def test_duplicate_semantic_result_cannot_use_new_record_id():
    records = log()[:2]
    extra = copy.deepcopy(records[-1])
    extra.update(record_id="another-result", sequence=3)
    with pytest.raises(contracts.ContractError, match="unresolved prepared"):
        contracts.validate_log(records + [extra])


def test_abstain_remains_terminal_across_replacement():
    records = contracts.load_json("examples/durable-log.abstain.json")
    extra = log()[0]
    extra.update(record_id="after-abstain", sequence=3, owner_epoch=2)
    with pytest.raises(contracts.ContractError, match="terminal"):
        contracts.validate_log(records + [extra])


def test_known_result_cannot_be_converted_to_abstain():
    records = log()[:2]
    abstain = contracts.load_json("examples/durable-log.abstain.json")[-1]
    abstain.update(record_id="halt", sequence=3)
    with pytest.raises(contracts.ContractError, match="without known result"):
        contracts.validate_log(records + [abstain])


@pytest.mark.parametrize("field,value", [("op_id", "wrong"), ("task_id", "wrong"), ("owner_epoch", 2)])
def test_response_must_echo_request_identity(field, value):
    data = messages()
    response = data[3]
    response[field] = value
    with pytest.raises(contracts.ContractError, match="identity mismatch"):
        contracts.validate_exchange(data[1], response)


def test_submit_cannot_return_query_absence():
    data = messages()
    with pytest.raises(contracts.ContractError, match="submit cannot"):
        contracts.validate_exchange(data[1], data[4])


def test_unknown_response_cannot_claim_committed_result():
    response = messages()[5]
    response["result"] = {"remaining": 9}
    with pytest.raises(contracts.ContractError):
        contracts.validate_tool_message(response)


def test_stale_epoch_response_cannot_echo_current_epoch_as_stale():
    response = messages()[6]
    response["current_epoch"] = response["owner_epoch"]
    with pytest.raises(contracts.ContractError, match="different current epoch"):
        contracts.validate_tool_message(response)


def test_capabilities_cannot_claim_query_with_unsupported_consistency():
    message = messages()[0]
    message["query_consistency"] = "unsupported"
    with pytest.raises(contracts.ContractError, match="query capability"):
        contracts.validate_tool_message(message)


@pytest.mark.parametrize("schema_name", ["durable-log", "tool-service"])
def test_schemas_reject_nested_float_args_without_semantic_validator(schema_name):
    instance = log()[0] if schema_name == "durable-log" else messages()[1]
    body = instance["payload"] if schema_name == "durable-log" else instance
    body["args"] = {"nested": [{"quantity": 1.25}]}
    with pytest.raises(contracts.ContractError):
        contracts.validate_schema(schema_name, instance)


def test_next_step_cannot_start_before_context_commit():
    records = log()[:2]
    next_step = copy.deepcopy(records[0])
    next_step.update(sequence=3, record_id="next-prepare", step_id="step-02", op_id="task:step-02")
    with pytest.raises(contracts.ContractError, match="overlapping"):
        contracts.validate_log(records + [next_step])


def test_result_only_tail_is_not_a_valid_persisted_log():
    with pytest.raises(contracts.ContractError, match="atomically"):
        contracts.validate_log(log()[:2])



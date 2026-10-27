import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def load_validator_module():
    path = ROOT / "scripts" / "validate_week1_contracts.py"
    spec = importlib.util.spec_from_file_location("week1_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_lock_uses_immutable_runtime_and_model_pins():
    lock = load_json("infra/environment.lock.json")

    assert lock["vllm"]["version"] == "0.32.0"
    assert lock["vllm"]["image"].endswith("@sha256:f99aa709b3f873aa1e1aadf5c1db35970119e8af9b36fe15d2a667aae0ed005c")
    assert lock["vllm"]["release_commit"] == "5df63a7203cdb9328d824a27b682e77c7b9928be"
    assert lock["model"]["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert lock["serving"]["enable_prefix_caching"] is False


@pytest.mark.parametrize(
    "relative_schema,relative_example",
    [
        ("schemas/fault-injection.schema.json", "examples/fault-injection.request.json"),
        ("schemas/telemetry-event.schema.json", "examples/telemetry-event.json"),
    ],
)
def test_examples_validate_against_draft_2020_12_schemas(relative_schema, relative_example):
    schema = load_json(relative_schema)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(load_json(relative_example))


def test_fault_schema_covers_every_proposal_boundary():
    schema = load_json("schemas/fault-injection.schema.json")
    boundaries = set(schema["properties"]["boundary"]["enum"])

    assert boundaries == {
        "DURING_GENERATION",
        "PRE_DISPATCH",
        "POST_TOOL_COMMIT_PRE_RESULT_LOG",
        "POST_CONTEXT_COMMIT",
        "DURING_RECOVERY",
    }


def test_ack_loss_is_limited_to_post_commit_response_suppression():
    schema = load_json("schemas/fault-injection.schema.json")
    validator = Draft202012Validator(schema)
    request = load_json("examples/fault-injection.request.json")
    request["action"] = "DROP_ACKNOWLEDGMENT"
    request["boundary"] = "PRE_DISPATCH"

    assert list(validator.iter_errors(request)), "pre-dispatch acknowledgment loss must be rejected"


def test_fault_schedule_rejects_reused_identity_with_changed_payload():
    module = load_validator_module()
    request = load_json("examples/fault-injection.request.json")
    changed = copy.deepcopy(request)
    changed["seed"] += 1

    with pytest.raises(module.ContractError, match="injection_id"):
        module.validate_fault_schedule([request, changed])


def test_trace_validator_rejects_inverted_monotonic_boundaries():
    module = load_validator_module()
    event = load_json("examples/telemetry-event.json")
    start = copy.deepcopy(event)
    end = copy.deepcopy(event)
    start.update({"event_id": "evt-start", "event_type": "CRASH_DETECTED", "monotonic_ns": 200})
    end.update({"event_id": "evt-end", "event_type": "NEXT_TOKEN_EMITTED", "monotonic_ns": 100})

    with pytest.raises(module.ContractError, match="monotonic"):
        module.validate_trace([start, end])


def test_trace_validator_rejects_configuration_drift():
    module = load_validator_module()
    event = load_json("examples/telemetry-event.json")
    changed = copy.deepcopy(event)
    changed["event_id"] = "evt-other"
    changed["configuration_id"] = "cfg-different"

    with pytest.raises(module.ContractError, match="configuration_id"):
        module.validate_trace([event, changed])


def test_trace_validator_rejects_worker_lineage_drift():
    module = load_validator_module()
    event = load_json("examples/telemetry-event.json")
    replacement = copy.deepcopy(event)
    replacement["event_id"] = "evt-replacement"
    replacement["monotonic_ns"] += 1
    replacement["worker"] = {
        "lineage_id": "runner-b",
        "instance_id": "runner-b-attempt-1",
        "attempt": 1,
    }

    with pytest.raises(module.ContractError, match="lineage_id"):
        module.validate_trace([event, replacement])


def test_cli_validates_all_week1_contracts():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_week1_contracts.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "environment lock" in result.stdout
    assert "fault injection example" in result.stdout
    assert "telemetry example" in result.stdout

"""Failure-free fixture driver; not the project's agent runner/controller."""

from outcomeserve.contracts import request_fingerprint
from outcomeserve.context import scripted_context
from outcomeserve.log import DurableLog
from outcomeserve.tools import ToolService
from workloads import validate_workload


def exercise_workload(store, workload):
    """Execute into a previously initialized task. Never reset or auto-recover."""
    validate_workload(workload)
    log, tools = DurableLog(store), ToolService(store)
    context = store.bind_context(workload["task_id"], scripted_context(workload["model_revision"]))
    sequence = 0
    for step in workload["steps"]:
        identity = {"task_id": workload["task_id"], "step_id": step["step_id"],
                    "op_id": step["op_id"], "owner_epoch": 1}
        common = {"schema_version": "1.1.0", "context_spec_id": context, **identity, "model_revision": workload["model_revision"]}
        fingerprint = request_fingerprint(step["tool_name"], step["args"])

        def append(kind, suffix, payload):
            nonlocal sequence
            record_id = f"{step['step_id']}:{suffix}"
            body = {**common, "record_id": record_id, "record_type": kind, "payload": payload}
            result = log.append(workload["task_id"], record_id, sequence, 1, body)
            if result["status"] != "APPENDED":
                raise RuntimeError(f"fixture append failed: {result}")
            sequence += 1
            return record_id

        append("STEP_PREPARED", "prepare", {"tool_name": step["tool_name"], "args": step["args"],
                                               "request_fingerprint": fingerprint, "generated_token_ids": [],
                                               "generation_prompt_token_ids": []})
        response = tools.submit(**identity, tool_name=step["tool_name"], args=step["args"],
                                request_fingerprint=fingerprint)
        if response["status"] != "COMMITTED":
            raise RuntimeError(f"fixture submit failed: {response}")
        prepared = log.read(workload["task_id"])[-1]
        committed = log.commit_step(prepared, response, step["token_ids"], owner_epoch=1,
                                    expected_sequence=sequence, commit_id=f"{step['step_id']}:commit")
        if committed["status"] != "COMMITTED":
            raise RuntimeError(f"fixture commit failed: {committed}")
        sequence += 2

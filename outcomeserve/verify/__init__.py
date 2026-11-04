"""Read-only inventory oracle, independent of runtime state-transition code."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from jsonschema import Draft202012Validator

from workloads import validate_workload


ROOT = Path(__file__).resolve().parents[2]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _fingerprint(step):
    # Independent implementation of the published C2 encoding, not a runtime call.
    return "sha256:" + hashlib.sha256(_json({
        "tool_name": step["tool_name"], "args": step["args"]
    }).encode("utf-8")).hexdigest()


def _business_oracle(inventory, step):
    """Pure reference semantics; mutates only an in-memory reference state."""
    sku = step["args"]["sku"]
    if sku not in inventory:
        return {"ok": False, "error": "SKU_NOT_FOUND", "sku": sku}
    count = inventory[sku]
    if step["tool_name"] == "decrement_inventory":
        amount = step["args"]["quantity"]
        if amount > count:
            return {"ok": False, "error": "INSUFFICIENT_INVENTORY", "sku": sku, "remaining": count}
        inventory[sku] = count - amount
    return {"ok": True, "sku": sku, "remaining": inventory[sku]}


def _decode_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def _snapshot(database, task_id):
    # Do not instantiate StateStore, ToolService or DurableLog: opening verification
    # must not initialize tables, reset state, repair the log or call recovery.
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if task is None:
            raise ValueError("task not found")
        task = dict(task)
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_context'").fetchone():
            bound = connection.execute("SELECT context_spec_id FROM task_context WHERE task_id=?", (task_id,)).fetchone()
            task["context_spec_id"] = bound[0] if bound else None
        inventory = {row["sku"]: row["quantity"] for row in connection.execute(
            "SELECT * FROM inventory WHERE task_id=?", (task_id,)
        )}
        operations = [dict(row) for row in connection.execute(
            "SELECT * FROM operations WHERE task_id=? ORDER BY rowid", (task_id,)
        )]
        rows = [dict(row) for row in connection.execute(
            "SELECT * FROM log_records WHERE task_id=? ORDER BY sequence", (task_id,)
        )]
        for operation in operations:
            operation["result"] = _decode_json(operation["result_json"])
        for row in rows:
            row["record"] = _decode_json(row["record_json"])
        return task, inventory, operations, rows
    finally:
        connection.close()


def verify_workload(database, workload, *, checkpoint="completed"):
    """Check a consistent database snapshot against trusted scripted inputs.

    `in_progress` allows the expected prepared/committed-but-unlogged window.
    `completed` requires every step/context and no unreconciled tool operation.
    `halted` verifies a terminal halt, not that abstention was justified.
    """
    if checkpoint not in {"in_progress", "completed", "halted"}:
        raise ValueError("unsupported verification checkpoint")
    validate_workload(workload)
    report = {
        "schema_version": "1.0.0", "task_id": workload["task_id"],
        "checkpoint": checkpoint, "valid": True, "invalid_reasons": [],
        "consistency_pass": None, "task_success": None, "issues": [], "metrics": None,
    }
    try:
        task, inventory, operations, rows = _snapshot(database, workload["task_id"])
    except (sqlite3.Error, ValueError, TypeError) as exc:
        report.update(valid=False, invalid_reasons=[f"EVIDENCE_UNREADABLE: {exc}"])
        return report

    issues = report["issues"]
    def issue(code, detail, step_id=None):
        issues.append({"code": code, "step_id": step_id, "detail": detail})

    trace = workload["steps"]
    steps = {step["step_id"]: step for step in trace}
    ordered_ids = [step["step_id"] for step in trace]
    expected_all = dict(workload["initial_inventory"])
    expected_results = {step["step_id"]: _business_oracle(expected_all, step) for step in trace}
    expected_inventory = dict(workload["initial_inventory"])
    op_counts = Counter(op["step_id"] for op in operations)
    duplicates = sum(max(0, count - 1) for count in op_counts.values())
    if duplicates:
        issue("DUPLICATE_OPERATION", "more than one tool commit for a logical step")
    if [op["step_id"] for op in operations] != ordered_ids[:len(operations)]:
        issue("TOOL_ORDER", "committed operations are not a sequential workload prefix")

    by_step = {}
    for op in operations:
        step_id = op["step_id"]
        step = steps.get(step_id)
        by_step[step_id] = op
        if step is None:
            issue("UNEXPECTED_OPERATION", "operation does not belong to workload", step_id)
            continue
        if op["op_id"] != step["op_id"] or op["fingerprint"] != _fingerprint(step):
            issue("TOOL_IDENTITY", "operation ID/fingerprint differs from workload", step_id)
        if not op["commit_receipt"] or not 1 <= op["committed_epoch"] <= task["owner_epoch"]:
            issue("TOOL_EVIDENCE", "invalid receipt or commit epoch", step_id)
        if _json(op["result"]) != _json(expected_results[step_id]):
            issue("TOOL_RESULT", "stored result differs from independent business oracle", step_id)
        _business_oracle(expected_inventory, step)
    inventory_mismatches = sum(
        inventory.get(sku) != expected_inventory.get(sku)
        for sku in set(inventory) | set(expected_inventory)
    )
    if inventory_mismatches:
        issue("INVENTORY_STATE", "actual stock differs from committed-operation reference state")

    schema = json.loads((ROOT / "schemas/durable-log.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    prepared, staged, completed = {}, {}, []
    pending, last_context, halted, previous_epoch = None, None, False, 0
    seen_ids = set()
    for index, row in enumerate(rows, 1):
        record = row["record"]
        if list(validator.iter_errors(record)):
            issue("LOG_SCHEMA", f"malformed C1 record at SQL sequence {row['sequence']}")
            continue
        step_id, kind, payload = record["step_id"], record["record_type"], record["payload"]
        if row["sequence"] != index or record["sequence"] != index:
            issue("LOG_SEQUENCE", "SQL/body sequence is not gap-free", step_id)
        if record["record_id"] != row["record_id"] or record["record_id"] in seen_ids:
            issue("LOG_IDENTITY", "record ID mismatch or duplicate", step_id)
        seen_ids.add(record["record_id"])
        if record["task_id"] != workload["task_id"] or record["model_revision"] != workload["model_revision"]:
            issue("LOG_IDENTITY", "task/model identity differs from workload", step_id)
        if record.get("context_spec_id") != task.get("context_spec_id"):
            issue("CONTEXT_SPEC", "record context differs from immutable task binding", step_id)
        if not previous_epoch <= record["owner_epoch"] <= task["owner_epoch"]:
            issue("LOG_EPOCH", "accepted log has stale/future owner epoch", step_id)
        previous_epoch = record["owner_epoch"]
        if halted:
            issue("LOG_AFTER_HALT", "execution record follows terminal abstention", step_id)
        step = steps.get(step_id)
        if step is None or record["op_id"] != step["op_id"]:
            issue("LOG_IDENTITY", "step/op is not in workload", step_id)
            continue
        if kind == "STEP_PREPARED":
            next_id = ordered_ids[len(completed)] if len(completed) < len(trace) else None
            if pending is not None or step_id in prepared or step_id != next_id:
                issue("LOG_TRANSITION", "prepare overlaps or skips a step", step_id)
            if (payload["tool_name"] != step["tool_name"]
                    or _json(payload["args"]) != _json(step["args"])
                    or payload["request_fingerprint"] != _fingerprint(step)):
                issue("LOG_PREPARE", "prepared request differs from workload", step_id)
            prepared[step_id], pending = record, step_id
        elif kind == "TOOL_RESULT_COMMITTED":
            if pending != step_id or step_id in staged:
                issue("LOG_TRANSITION", "result requires a unique prepared step", step_id)
            op = by_step.get(step_id)
            if (op is None or op["op_id"] != record["op_id"]
                    or payload["request_fingerprint"] != op["fingerprint"]
                    or payload["commit_receipt"] != op["commit_receipt"]
                    or _json(payload["result"]) != _json(op["result"])):
                issue("LOG_TOOL_MISMATCH", "log result does not match tool-side commit", step_id)
            staged[step_id] = record
        elif kind == "CONTEXT_COMMITTED":
            result = staged.get(step_id)
            if pending != step_id or result is None or step_id in completed:
                issue("LOG_TRANSITION", "context lacks a unique staged result", step_id)
            elif payload["result_record_id"] != result["record_id"]:
                issue("LOG_RESULT_REFERENCE", "context references wrong result record", step_id)
            if payload["token_ids"] != step["token_ids"]:
                issue("CONTEXT_TOKENS", "scripted committed prefix differs from workload", step_id)
            completed.append(step_id)
            pending, last_context = None, record["record_id"]
        elif kind == "RECOVERY_ABSTAIN":
            if pending != step_id or step_id in staged:
                issue("LOG_TRANSITION", "abstain requires prepared unresolved step", step_id)
            halted = True

    if pending in staged:
        issue("LOG_ATOMIC_COMMIT", "persisted result lacks its atomic context commit", pending)
    if task["context_record_id"] != last_context:
        issue("CONTEXT_POINTER", "durable pointer does not identify latest committed context")
    if task["halted"] not in (0, 1) or bool(task["halted"]) != halted:
        issue("HALT_STATE", "durable halt flag disagrees with terminal log")
    for step_id in by_step:
        if step_id not in prepared:
            issue("MISSING_PREPARE", "tool commit has no durable intent", step_id)
    unreconciled = sum(op["step_id"] not in completed for op in operations)
    if checkpoint == "completed":
        if completed != ordered_ids or halted:
            issue("TASK_INCOMPLETE", "completion requires every workload step and no halt")
        if unreconciled:
            issue("ORPHAN_OPERATION", "tool commits remain outside completed transcript")
    elif checkpoint == "halted" and not halted:
        issue("TASK_NOT_HALTED", "expected terminal abstention is absent")

    report.update(
        consistency_pass=not issues,
        task_success=not issues and completed == ordered_ids and not halted
                     and inventory == expected_all and all(result["ok"] for result in expected_results.values()),
        observed_state="halted" if halted else "completed" if completed == ordered_ids else "in_progress",
        expected_final_inventory=expected_all, observed_inventory=inventory,
        metrics={
            "completed_steps": len(completed), "committed_operations": len(operations),
            "duplicate_operation_records": duplicates, "inventory_mismatches": inventory_mismatches,
            "tool_state_mismatches": sum(i["code"] in {"TOOL_RESULT", "LOG_TOOL_MISMATCH"} for i in issues),
            "pending_operations": unreconciled if checkpoint != "completed" else 0,
            "orphan_operations": unreconciled if checkpoint == "completed" else 0,
            # A final snapshot cannot establish temporal retry decisions or count
            # compensated duplicate effects. Do not invent zero-valued rates.
            "duplicate_effects": None, "unsafe_retry": None, "false_abstain": None,
        },
    )
    return report

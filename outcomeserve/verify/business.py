"""Common business oracle for all methods, including deliberately unsafe IDs."""

from collections import Counter
import json
from pathlib import Path
import sqlite3

from outcomeserve.verify import _business_oracle, _optional_rows, _fingerprint, _json, _decode_json
from workloads import validate_workload


def _decode(text):
    try:
        value = _decode_json(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _external_effect_valid(effect, steps, results, receipts, owner_epoch):
    """One external acknowledgment matches its workload step, result, receipt and epoch."""
    step = steps.get(effect["logical_step_id"])
    return (step is not None and step["tool_name"] == effect["tool_name"] == "notify_supplier"
            and effect["fingerprint"] == _fingerprint(step)
            and bool(effect["commit_receipt"]) and receipts[effect["commit_receipt"]] == 1
            and 1 <= effect["received_epoch"] <= owner_epoch
            and _json(_decode(effect["result_json"])) == _json(results[step["step_id"]]))


def verify_business(database, workload):
    validate_workload(workload)
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        task = workload["task_id"]
        owner_epoch = connection.execute("SELECT owner_epoch FROM tasks WHERE task_id=?", (task,)).fetchone()[0]
        inventory = dict(connection.execute("SELECT sku,quantity FROM inventory WHERE task_id=?", (task,)))
        operations = [dict(row) for row in connection.execute("""
            SELECT o.*,COALESCE(i.logical_step_id,o.step_id) AS logical_step_id
            FROM operations o LEFT JOIN operation_identity i ON o.task_id=i.task_id AND o.op_id=i.op_id
            WHERE o.task_id=? ORDER BY o.rowid
        """, (task,))]
        effects = [dict(row) for row in connection.execute("SELECT * FROM tool_effects WHERE task_id=? ORDER BY effect_id", (task,))]
        external = _optional_rows(connection, "external_effects", task, "effect_id")
        reorder = {row["sku"]: row["level"] for row in _optional_rows(connection, "reorder_levels", task)}
    finally:
        connection.close()
    expected, expected_reorder = dict(workload["initial_inventory"]), {}
    results = {step["step_id"]: _business_oracle(expected, step, expected_reorder) for step in workload["steps"]}
    steps = {step["step_id"]: step for step in workload["steps"]}
    order = [step["step_id"] for step in workload["steps"]]
    issues = []
    effects_by_step = Counter(effect["logical_step_id"] for effect in effects)
    external_by_step = Counter(effect["logical_step_id"] for effect in external)
    external_duplicates = sum(max(0, count - 1) for count in external_by_step.values())
    duplicates = sum(max(0, count - 1) for count in effects_by_step.values()) + external_duplicates
    external_receipts = Counter(effect["commit_receipt"] for effect in external)
    external_audit_complete = all(_external_effect_valid(effect, steps, results, external_receipts, owner_epoch)
                                  for effect in external)
    if not external_audit_complete:
        issues.append("UNEXPECTED_EFFECT")
    if duplicates:
        issues.append("DUPLICATE_EFFECT")
    state_mismatches = sum(inventory.get(key) != expected.get(key) for key in set(inventory) | set(expected))
    if state_mismatches:
        issues.append("INVENTORY_STATE")
    audit_state = dict(workload["initial_inventory"])
    for effect in effects:
        if effect["logical_step_id"] not in steps or effect["sku"] not in audit_state:
            issues.append("UNEXPECTED_EFFECT")
            continue
        step = steps[effect["logical_step_id"]]
        if step["tool_name"] != "decrement_inventory" or effect["sku"] != step["args"]["sku"] or effect["quantity_delta"] != -step["args"]["quantity"]:
            issues.append("EFFECT_ARGUMENT_MISMATCH")
        audit_state[effect["sku"]] += effect["quantity_delta"]
    if audit_state != inventory:
        issues.append("AUDIT_STATE_MISMATCH")
    receipts = Counter(effect["commit_receipt"] for effect in effects)
    audit_complete, result_mismatches = external_audit_complete, 0
    if any(code in issues for code in ("UNEXPECTED_EFFECT", "EFFECT_ARGUMENT_MISMATCH", "AUDIT_STATE_MISMATCH")):
        audit_complete = False
    operations_by_id = {op["op_id"]: op for op in operations}
    for effect in effects:
        op = operations_by_id.get(effect["op_id"])
        if op is None or op["commit_receipt"] != effect["commit_receipt"] or op["logical_step_id"] != effect["logical_step_id"] or op["committed_epoch"] != effect["committed_epoch"]:
            audit_complete = False
            issues.append("ORPHAN_EFFECT_AUDIT")
    for op in operations:
        step = steps.get(op["logical_step_id"])
        result = _decode(op["result_json"])
        if step is None:
            issues.append("UNEXPECTED_OPERATION")
            audit_complete = False
            continue
        if json.dumps(result, sort_keys=True) != json.dumps(results[op["logical_step_id"]], sort_keys=True):
            result_mismatches += 1
        if step["tool_name"] == "decrement_inventory" and result.get("ok") and receipts[op["commit_receipt"]] != 1:
            audit_complete = False
    if not audit_complete:
        issues.append("INCOMPLETE_EFFECT_AUDIT")
    if result_mismatches:
        issues.append("TOOL_RESULT")
    covered = {op["logical_step_id"] for op in operations} | set(external_by_step)
    if covered != set(steps):
        issues.append("TASK_INCOMPLETE")
    reorder_mismatches = sum(reorder.get(k) != expected_reorder.get(k) for k in set(reorder) | set(expected_reorder))
    if reorder_mismatches:
        issues.append("REORDER_STATE")
    # State after exactly the steps that took effect, in workload order. A halted
    # task must match this prefix reference even though it differs from the final one.
    executed = [step_id for step_id in order if step_id in covered]
    prefix_contiguous = executed == order[:len(executed)]
    if not prefix_contiguous:
        issues.append("EXECUTED_ORDER")
    prefix_inventory, prefix_reorder = dict(workload["initial_inventory"]), {}
    for step_id in executed:
        _business_oracle(prefix_inventory, steps[step_id], prefix_reorder)
    prefix_mismatches = (sum(inventory.get(k) != prefix_inventory.get(k) for k in set(inventory) | set(prefix_inventory))
                         + sum(reorder.get(k) != prefix_reorder.get(k) for k in set(reorder) | set(prefix_reorder)))
    return {"valid": True, "consistency_pass": not issues,
            "task_success": not issues and all(result["ok"] for result in results.values()),
            "issues": sorted(set(issues)), "observed_inventory": inventory, "expected_final_inventory": expected,
            "metrics": {"inventory_mismatches": state_mismatches, "tool_state_mismatches": result_mismatches,
                        "duplicate_effects": duplicates if audit_complete else None,
                        "audit_complete": audit_complete, "effect_count": len(effects),
                        "committed_operations": len(operations), "completed_logical_steps": len(covered),
                        "external_effects": len(external), "external_duplicate_effects": external_duplicates,
                        "reorder_mismatches": reorder_mismatches, "prefix_state_mismatches": prefix_mismatches,
                        "prefix_contiguous": prefix_contiguous,
                        "steps_with_effects": executed,
                        "unsafe_retry": None, "false_abstain": None}}

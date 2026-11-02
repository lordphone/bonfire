"""Common business oracle for all methods, including deliberately unsafe IDs."""

from collections import Counter
import json
from pathlib import Path
import sqlite3

from outcomeserve.verify import _business_oracle
from workloads import validate_workload


def verify_business(database, workload):
    validate_workload(workload)
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        task = workload["task_id"]
        inventory = dict(connection.execute("SELECT sku,quantity FROM inventory WHERE task_id=?", (task,)))
        operations = [dict(row) for row in connection.execute("""
            SELECT o.*,COALESCE(i.logical_step_id,o.step_id) AS logical_step_id
            FROM operations o LEFT JOIN operation_identity i ON o.task_id=i.task_id AND o.op_id=i.op_id
            WHERE o.task_id=? ORDER BY o.rowid
        """, (task,))]
        effects = [dict(row) for row in connection.execute("SELECT * FROM tool_effects WHERE task_id=? ORDER BY effect_id", (task,))]
    finally:
        connection.close()
    expected = dict(workload["initial_inventory"])
    results = {step["step_id"]: _business_oracle(expected, step) for step in workload["steps"]}
    steps = {step["step_id"]: step for step in workload["steps"]}
    issues = []
    effects_by_step = Counter(effect["logical_step_id"] for effect in effects)
    duplicates = sum(max(0, count - 1) for count in effects_by_step.values())
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
    audit_complete, result_mismatches = True, 0
    operations_by_id = {op["op_id"]: op for op in operations}
    for effect in effects:
        op = operations_by_id.get(effect["op_id"])
        if op is None or op["commit_receipt"] != effect["commit_receipt"] or op["logical_step_id"] != effect["logical_step_id"] or op["committed_epoch"] != effect["committed_epoch"]:
            audit_complete = False
            issues.append("ORPHAN_EFFECT_AUDIT")
    for op in operations:
        step = steps.get(op["logical_step_id"])
        result = json.loads(op["result_json"])
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
    covered = {op["logical_step_id"] for op in operations}
    if covered != set(steps):
        issues.append("TASK_INCOMPLETE")
    return {"valid": True, "consistency_pass": not issues,
            "task_success": not issues and all(result["ok"] for result in results.values()),
            "issues": sorted(set(issues)), "observed_inventory": inventory, "expected_final_inventory": expected,
            "metrics": {"inventory_mismatches": state_mismatches, "tool_state_mismatches": result_mismatches,
                        "duplicate_effects": duplicates if audit_complete else None,
                        "audit_complete": audit_complete, "effect_count": len(effects),
                        "committed_operations": len(operations), "completed_logical_steps": len(covered),
                        "unsafe_retry": None, "false_abstain": None}}

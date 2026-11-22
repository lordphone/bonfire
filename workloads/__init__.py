"""Deterministic scripted inputs for Victoria's CPU correctness checks."""

import json
from pathlib import Path
import random
from urllib.parse import quote

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def validate_workload(workload):
    schema = json.loads((ROOT / "schemas/inventory-workload.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(workload)
    if type(workload["seed"]) is not int or any(type(n) is not int for n in workload["initial_inventory"].values()):
        raise ValueError("seed and inventory must use integer JSON values")
    step_ids, op_ids = set(), set()
    for step in workload["steps"]:
        if any(type(token) is not int for token in step["token_ids"]):
            raise ValueError("token IDs must use integer JSON values")
        for key in ("quantity", "level"):
            if key in step["args"] and type(step["args"][key]) is not int:
                raise ValueError(f"{key} must use an integer JSON value")
        if step["step_id"] in step_ids or step["op_id"] in op_ids:
            raise ValueError("step_id and op_id must be unique within a workload")
        step_ids.add(step["step_id"])
        op_ids.add(step["op_id"])


def load_workload(path):
    """Load trusted experiment inputs without silently accepting duplicate keys."""
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate workload key: {key}")
            value[key] = item
        return value
    def reject_constant(value):
        raise ValueError(f"nonfinite workload value: {value}")
    workload = json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=pairs, parse_constant=reject_constant)
    validate_workload(workload)
    return workload


def make_inventory_workload(task_id: str, *, seed: int = 33, steps: int = 5, tools: str = "inventory"):
    """Generate 5/10/20 scripted steps; no model or learned policy.

    ``tools="inventory"`` (default, unchanged) mixes reads and decrements.
    ``tools="mixed"`` cycles read, decrement (non-idempotent) and
    set_reorder_level (idempotent write), as the proposal's order tasks require.
    """
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps not in {5, 10, 20}:
        raise ValueError("nonnegative integer seed and steps in {5,10,20} required")
    if tools not in {"inventory", "mixed"}:
        raise ValueError("tools must be inventory or mixed")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("nonempty globally unique task_id required")
    rng = random.Random(seed)
    trace, totals = [], {"item-a": 0, "item-b": 0}
    tokens = [1000]
    for index in range(1, steps + 1):
        sku = rng.choice(list(totals))
        tool = "read_inventory" if index % 3 == 1 else "decrement_inventory"
        if tools == "mixed" and index % 3 == 0:
            tool = "set_reorder_level"
        args = {"sku": sku}
        if tool == "set_reorder_level":
            args["level"] = rng.randint(2, 9)
        if tool == "decrement_inventory":
            args["quantity"] = rng.randint(1, 3)
            totals[sku] += args["quantity"]
        step_id = f"step-{index:02d}"
        tokens = tokens + [1000 + index, 2000 + index]
        trace.append({"step_id": step_id, "op_id": f"{quote(task_id, safe='')}:{step_id}",
                      "tool_name": tool, "args": args, "token_ids": tokens})
    workload = {"schema_version": "1.0.0", "task_id": task_id, "seed": seed,
                "trace_mode": "scripted", "model_revision": MODEL_REVISION,
                "initial_inventory": {sku: total + 5 for sku, total in totals.items()},
                "steps": trace}
    validate_workload(workload)
    return workload


def with_external_step(workload, step_id):
    """Copy of a workload whose decrement ``step_id`` becomes an unqueryable external call.

    The same position then forms a should-resume case (C2 decrement) and a
    should-abstain case (notify_supplier, no deduplication or outcome query).
    """
    changed = json.loads(json.dumps(workload))
    step = next((s for s in changed["steps"] if s["step_id"] == step_id), None)
    if step is None or step["tool_name"] != "decrement_inventory":
        raise ValueError("only a decrement step can be replaced by an external call")
    step["tool_name"], step["args"] = "notify_supplier", {"sku": step["args"]["sku"]}
    validate_workload(changed)
    return changed

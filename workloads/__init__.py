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
        if "quantity" in step["args"] and type(step["args"]["quantity"]) is not int:
            raise ValueError("quantity must use an integer JSON value")
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


def make_inventory_workload(task_id: str, *, seed: int = 33, steps: int = 5):
    """Generate 5/10/20 mixed read/write steps; no model or learned policy."""
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps not in {5, 10, 20}:
        raise ValueError("nonnegative integer seed and steps in {5,10,20} required")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("nonempty globally unique task_id required")
    rng = random.Random(seed)
    trace, totals = [], {"item-a": 0, "item-b": 0}
    tokens = [1000]
    for index in range(1, steps + 1):
        sku = rng.choice(list(totals))
        tool = "read_inventory" if index % 3 == 1 else "decrement_inventory"
        args = {"sku": sku}
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

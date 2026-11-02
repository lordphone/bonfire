#!/usr/bin/env python3
"""Validate C1/C2 fixtures and protocol evidence, not storage durability."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


class ContractError(ValueError):
    """An instance or a sequence contradicts C1/C2."""


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)


@lru_cache(maxsize=32)
def _validator(name: str):
    schema = load_json(f"schemas/{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(name: str, value: Any) -> None:
    errors = list(_validator(name).iter_errors(value))
    if errors:
        raise ContractError(errors[0].message)


def _check_args(value: Any) -> None:
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int and abs(value) <= 2**53 - 1:
        return
    if type(value) is list:
        for item in value:
            _check_args(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_args(item)
        return
    raise ContractError("args require JSON values with safe integers; floats are unsupported")


def request_fingerprint(tool_name: str, args: dict) -> str:
    if type(tool_name) is not str or not tool_name or type(args) is not dict:
        raise ContractError("tool_name and object args are required")
    _check_args(args)
    canonical = json.dumps(
        {"tool_name": tool_name, "args": args},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    try:
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except UnicodeEncodeError as exc:
        raise ContractError("args must be valid UTF-8") from exc


def validate_tool_message(message: dict) -> None:
    validate_schema("tool-service", message)
    if message["kind"] == "submit":
        expected = request_fingerprint(message["tool_name"], message["args"])
        if message["request_fingerprint"] != expected:
            raise ContractError("request fingerprint does not match tool and args")
    elif message["kind"] == "capabilities":
        query = message["supports_outcome_query"]
        if query != (message["query_consistency"] != "unsupported"):
            raise ContractError("query capability contradicts query consistency")
        if not message["supports_deduplication"] and message["deduplication_window_seconds"] is not None:
            raise ContractError("deduplication window requires deduplication support")
        if message["atomic_effect_and_result"] and not message["supports_deduplication"]:
            raise ContractError("C2 atomic tool requires operation-ID deduplication")
    elif message["kind"] == "response":
        if message["status"] == "STALE_EPOCH" and message["current_epoch"] == message["owner_epoch"]:
            raise ContractError("STALE_EPOCH requires a different current epoch")
        if message["status"] == "FINGERPRINT_MISMATCH" and message["stored_fingerprint"] == message["request_fingerprint"]:
            raise ContractError("FINGERPRINT_MISMATCH requires different fingerprints")


def validate_exchange(request: dict, response: dict) -> None:
    validate_tool_message(request)
    validate_tool_message(response)
    if request["kind"] not in {"submit", "query"} or response["kind"] != "response":
        raise ContractError("exchange requires request and response")
    for field in ("task_id", "step_id", "op_id", "request_fingerprint", "owner_epoch"):
        if request[field] != response[field]:
            raise ContractError(f"response identity mismatch: {field}")
    if request["kind"] == "submit" and response["status"] == "NOT_COMMITTED":
        raise ContractError("submit cannot return NOT_COMMITTED")


def validate_log(records: list[dict]) -> None:
    if not records:
        raise ContractError("log must not be empty")
    for record in records:
        validate_schema("durable-log", record)
    for field in ("task_id", "model_revision", "schema_version"):
        if len({record[field] for record in records}) != 1:
            raise ContractError(f"mixed {field} in task log")
    specs = {record.get("context_spec_id") for record in records}
    if len(specs) != 1:
        raise ContractError("mixed context specification in task log")
    if [record["sequence"] for record in records] != list(range(1, len(records) + 1)):
        raise ContractError("log sequence must start at 1 without gaps")
    if len({record["record_id"] for record in records}) != len(records):
        raise ContractError("duplicate record_id in persisted log")

    prepared = {}
    results = {}
    completed = set()
    operations = {}
    step_operations = {}
    pending = None
    halted = False
    epoch = 0
    for record in records:
        if halted:
            raise ContractError("RECOVERY_ABSTAIN is terminal")
        if record["owner_epoch"] < epoch:
            raise ContractError("stale owner_epoch in accepted log")
        epoch = record["owner_epoch"]
        step, op = record["step_id"], record["op_id"]
        if operations.setdefault(op, step) != step or step_operations.setdefault(step, op) != op:
            raise ContractError("step/op identity must remain stable and one-to-one")
        kind, payload = record["record_type"], record["payload"]
        if kind == "STEP_PREPARED":
            if pending is not None or step in prepared:
                raise ContractError("duplicate or overlapping prepared step")
            if payload["request_fingerprint"] != request_fingerprint(payload["tool_name"], payload["args"]):
                raise ContractError("prepared fingerprint mismatch")
            prepared[step] = payload
            pending = step
        elif kind == "TOOL_RESULT_COMMITTED":
            if pending != step or step in results:
                raise ContractError("result requires one unresolved prepared step")
            if payload["request_fingerprint"] != prepared[step]["request_fingerprint"]:
                raise ContractError("result fingerprint differs from prepare")
            results[step] = record
        elif kind == "CONTEXT_COMMITTED":
            if pending != step or step not in results:
                raise ContractError("context requires a staged tool result")
            if payload["result_record_id"] != results[step]["record_id"]:
                raise ContractError("context references a different result record")
            completed.add(step)
            pending = None
        elif kind == "RECOVERY_ABSTAIN":
            if pending != step or step in results:
                raise ContractError("abstain requires prepared step without known result")
            halted = True

    if pending in results:
        raise ContractError("result and context must be atomically committed")

#!/usr/bin/env python3
"""Validate C1/C2 fixtures using the shared runtime contract implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outcomeserve.contracts import (
    ContractError, load_json, validate_exchange, validate_log, validate_tool_message,
)


def main() -> int:
    for path in ("examples/durable-log.records.json", "examples/durable-log.abstain.json"):
        validate_log(load_json(path))
        print(f"validated C1 {path}")
    messages = load_json("examples/tool-service.messages.json")
    for message in messages:
        validate_tool_message(message)
    for response in messages[3:]:
        validate_exchange(messages[2], response)
        if response["status"] != "NOT_COMMITTED":
            validate_exchange(messages[1], response)
    prepared = load_json("examples/durable-log.records.json")[0]
    submit = messages[1]
    for field in ("task_id", "step_id", "op_id", "owner_epoch"):
        if prepared[field] != submit[field]:
            raise ContractError(f"C1/C2 fixture mismatch: {field}")
    for field in ("tool_name", "args", "request_fingerprint"):
        if prepared["payload"][field] != submit[field]:
            raise ContractError(f"C1/C2 fixture mismatch: {field}")
    print("validated C2 messages, response correlation, and C1/C2 identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

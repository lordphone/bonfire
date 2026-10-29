"""Stable operation identity: the same logical step always maps to the same op_id."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(tool: str, args: dict[str, Any]) -> str:
    return hashlib.sha256(f"{tool}\x00{canonical_args(args)}".encode()).hexdigest()


def op_id(task_id: str, step_id: int, tool: str, args: dict[str, Any]) -> str:
    material = f"{task_id}\x00{step_id}\x00{fingerprint(tool, args)}"
    return "op-" + hashlib.sha256(material.encode()).hexdigest()[:32]

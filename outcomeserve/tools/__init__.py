"""C2 local transactional inventory tool and post-commit injection hook."""

import copy
import json
import sqlite3
from uuid import uuid4

from outcomeserve.contracts import ContractError, validate_tool_message
from outcomeserve.storage import StateStore, TaskHalted


class ToolService:
    def __init__(self, store: StateStore, *, after_commit=None):
        self.store = store
        # Called outside the transaction, before returning an acknowledgment.
        # The fault injector may raise/exit here; it owns one-shot scheduling.
        self.after_commit = after_commit

    @staticmethod
    def capabilities(tool_name="decrement_inventory"):
        if tool_name not in {"decrement_inventory", "read_inventory"}:
            raise ContractError("unsupported tool")
        return {
            "schema_version": "1.0.0", "kind": "capabilities", "tool_name": tool_name,
            "idempotent_business_operation": tool_name == "read_inventory",
            "supports_deduplication": True, "supports_outcome_query": True,
            "query_consistency": "linearizable", "deduplication_window_seconds": None,
            "atomic_effect_and_result": True,
        }

    def submit(self, *, task_id, step_id, op_id, tool_name, args,
               request_fingerprint, owner_epoch):
        request = {
            "schema_version": "1.0.0", "kind": "submit", "task_id": task_id,
            "step_id": step_id, "op_id": op_id, "tool_name": tool_name,
            "args": copy.deepcopy(args), "request_fingerprint": request_fingerprint,
            "owner_epoch": owner_epoch,
        }
        validate_tool_message(request)
        self._validate_business_args(tool_name, request["args"])
        response, committed_now = self._perform(request)
        if committed_now and self.after_commit is not None:
            self.after_commit(copy.deepcopy(response))
        return response

    def query(self, *, task_id, step_id, op_id, request_fingerprint, owner_epoch):
        request = {
            "schema_version": "1.0.0", "kind": "query", "task_id": task_id,
            "step_id": step_id, "op_id": op_id,
            "request_fingerprint": request_fingerprint, "owner_epoch": owner_epoch,
        }
        validate_tool_message(request)
        return self._perform(request)[0]

    @staticmethod
    def _response(request, status, **extra):
        return {
            "schema_version": "1.0.0", "kind": "response", "status": status,
            **{key: request[key] for key in (
                "task_id", "step_id", "op_id", "request_fingerprint", "owner_epoch"
            )}, **extra,
        }

    def _perform(self, request):
        try:
            with self.store.transaction() as connection:
                task = self.store.task(connection, request["task_id"])
                if task["owner_epoch"] != request["owner_epoch"]:
                    return self._response(request, "STALE_EPOCH", current_epoch=task["owner_epoch"]), False
                operation = connection.execute(
                    "SELECT * FROM operations WHERE task_id=? AND op_id=?",
                    (request["task_id"], request["op_id"]),
                ).fetchone()
                if operation is not None:
                    if operation["step_id"] != request["step_id"]:
                        raise ContractError("op_id reused for a different step")
                    if operation["fingerprint"] != request["request_fingerprint"]:
                        return self._response(request, "FINGERPRINT_MISMATCH", stored_fingerprint=operation["fingerprint"]), False
                    if request["kind"] == "submit" and task["halted"]:
                        raise TaskHalted(request["task_id"])
                    return self._response(
                        request, "COMMITTED", result=json.loads(operation["result_json"]),
                        commit_receipt=operation["commit_receipt"],
                    ), False
                step_operation = connection.execute(
                    "SELECT op_id FROM operations WHERE task_id=? AND step_id=?",
                    (request["task_id"], request["step_id"]),
                ).fetchone()
                if step_operation is not None:
                    raise ContractError("step_id reused with a different op_id")
                if request["kind"] == "query":
                    return self._response(request, "NOT_COMMITTED", reason="authoritative absence under transaction lock"), False
                if task["halted"]:
                    raise TaskHalted(request["task_id"])
                result = self._execute_business(connection, request)
                receipt = "tx-" + uuid4().hex
                mapping = connection.execute(
                    "SELECT logical_step_id,execution_attempt FROM operation_identity WHERE task_id=? AND op_id=?",
                    (request["task_id"], request["op_id"]),
                ).fetchone()
                logical_step = mapping[0] if mapping else request["step_id"]
                attempt = mapping[1] if mapping else str(request["owner_epoch"])
                if request["tool_name"] == "decrement_inventory" and result.get("ok"):
                    connection.execute(
                        "INSERT INTO tool_effects(task_id,op_id,logical_step_id,execution_attempt,sku,quantity_delta,commit_receipt,committed_epoch) VALUES (?,?,?,?,?,?,?,?)",
                        (request["task_id"], request["op_id"], logical_step, attempt, request["args"]["sku"],
                         -request["args"]["quantity"], receipt, request["owner_epoch"]),
                    )
                connection.execute(
                    "INSERT INTO operations VALUES (?,?,?,?,?,?,?)",
                    (request["task_id"], request["op_id"], request["step_id"],
                     request["request_fingerprint"], json.dumps(result), receipt, request["owner_epoch"]),
                )
                response = self._response(request, "COMMITTED", result=result, commit_receipt=receipt)
            return response, True
        except sqlite3.OperationalError as exc:
            if (getattr(exc, "sqlite_errorcode", 0) & 255) not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            return self._response(request, "UNKNOWN", reason="transaction lock wait expired; outcome not established"), False

    @staticmethod
    def _validate_business_args(tool_name, args):
        ToolService.capabilities(tool_name)
        expected = {"sku", "quantity"} if tool_name == "decrement_inventory" else {"sku"}
        if set(args) != expected or not isinstance(args["sku"], str) or not args["sku"]:
            raise ContractError("invalid inventory arguments")
        if tool_name == "decrement_inventory" and (type(args["quantity"]) is not int or args["quantity"] < 1):
            raise ContractError("quantity must be a positive integer")

    @staticmethod
    def _execute_business(connection, request):
        args = request["args"]
        row = connection.execute(
            "SELECT quantity FROM inventory WHERE task_id=? AND sku=?",
            (request["task_id"], args["sku"]),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "SKU_NOT_FOUND", "sku": args["sku"]}
        remaining = row["quantity"]
        if request["tool_name"] == "decrement_inventory":
            if remaining < args["quantity"]:
                return {"ok": False, "error": "INSUFFICIENT_INVENTORY", "sku": args["sku"], "remaining": remaining}
            remaining -= args["quantity"]
            connection.execute(
                "UPDATE inventory SET quantity=? WHERE task_id=? AND sku=?",
                (remaining, request["task_id"], args["sku"]),
            )
        return {"ok": True, "sku": args["sku"], "remaining": remaining}

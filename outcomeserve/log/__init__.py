"""C1 append-only log, serialized with C2 commits and supervisor fencing."""

import copy
import json

from outcomeserve.contracts import ContractError, validate_log, validate_schema, validate_tool_message
from outcomeserve.storage import StateStore


def _same_json(left, right):
    # Python considers True == 1; evidence comparison must preserve JSON types.
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return canonical(left) == canonical(right)


class DurableLog:
    def __init__(self, store: StateStore):
        self.store = store

    @staticmethod
    def _records(connection, task_id):
        return [json.loads(row[0]) for row in connection.execute(
            "SELECT record_json FROM log_records WHERE task_id=? ORDER BY sequence", (task_id,)
        )]

    def read(self, task_id, after_sequence=0):
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be nonnegative")
        connection = self.store.connect()
        try:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT record_json FROM log_records WHERE task_id=? AND sequence>? ORDER BY sequence",
                (task_id, after_sequence),
            )]
        finally:
            connection.close()

    def committed_prefix(self, task_id):
        """A single snapshot joins the durable pointer to its full context record."""
        connection = self.store.connect()
        try:
            row = connection.execute("""
                SELECT l.record_json FROM tasks t LEFT JOIN log_records l
                ON t.task_id=l.task_id AND t.context_record_id=l.record_id WHERE t.task_id=?
            """, (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            return None if row[0] is None else json.loads(row[0])
        finally:
            connection.close()

    def append(self, task_id, record_id, expected_sequence, owner_epoch, record_body):
        body = copy.deepcopy(record_body)
        if type(expected_sequence) is not int or expected_sequence < 0:
            raise ValueError("expected_sequence must be nonnegative")
        for key, value in (("task_id", task_id), ("record_id", record_id), ("owner_epoch", owner_epoch)):
            if body.get(key) != value:
                raise ContractError(f"record_body disagrees with {key}")
        if "sequence" in body:
            raise ContractError("sequence is assigned by the store")
        validate_schema("durable-log", {**body, "sequence": 1})
        with self.store.transaction() as connection:
            task = self.store.task(connection, task_id)
            if task["owner_epoch"] != owner_epoch:
                return {"status": "STALE_EPOCH", "current_epoch": task["owner_epoch"]}
            bound = connection.execute("SELECT context_spec_id,specification_json FROM task_context WHERE task_id=?", (task_id,)).fetchone()
            if bound is None or bound[0] != body["context_spec_id"] or json.loads(bound[1])["model_revision"] != body["model_revision"]:
                return {"status": "INVALID_TRANSITION", "reason": "context spec is not bound to task"}
            records = self._records(connection, task_id)
            existing = next((r for r in records if r["record_id"] == record_id), None)
            if existing is not None:
                original = {key: value for key, value in existing.items() if key != "sequence"}
                if not _same_json(original, body):
                    return {"status": "RECORD_ID_CONFLICT"}
                return {"status": "ALREADY_APPENDED", "record": existing}
            if body["record_type"] in {"TOOL_RESULT_COMMITTED", "CONTEXT_COMMITTED"}:
                return {"status": "INVALID_TRANSITION", "reason": "result/context require atomic commit_step"}
            if len(records) != expected_sequence:
                return {"status": "SEQUENCE_CONFLICT", "current_sequence": len(records)}
            record = {**body, "sequence": len(records) + 1}
            try:
                validate_log(records + [record])
                if task["halted"]:
                    raise ContractError("task is halted")
            except ContractError as exc:
                return {"status": "INVALID_TRANSITION", "reason": str(exc)}
            connection.execute(
                "INSERT INTO log_records VALUES (?,?,?,?)",
                (task_id, record["sequence"], record_id, json.dumps(record, ensure_ascii=False, allow_nan=False)),
            )
            if record["record_type"] == "RECOVERY_ABSTAIN":
                connection.execute("UPDATE tasks SET halted=1 WHERE task_id=?", (task_id,))
            return {"status": "APPENDED", "record": record}

    def commit_step(self, prepared, tool_response, token_ids, *, owner_epoch,
                    expected_sequence, commit_id):
        """Atomically persist result + full context + prefix pointer.

        C2 remains a separate transaction. The caller supplies trusted C2 evidence;
        this method never reads the tool database or calls a remote tool while
        holding the log transaction. The independent verifier checks evidence.
        """
        validate_schema("durable-log", prepared)
        validate_tool_message(tool_response)
        if tool_response["status"] != "COMMITTED":
            raise ContractError("commit_step requires COMMITTED C2 evidence")
        if not isinstance(commit_id, str) or not commit_id or type(expected_sequence) is not int or expected_sequence < 0:
            raise ContractError("commit_id and nonnegative expected_sequence required")
        for key in ("task_id", "step_id", "op_id"):
            if prepared[key] != tool_response[key]:
                raise ContractError(f"C1/C2 commit identity mismatch: {key}")
        if tool_response["owner_epoch"] != owner_epoch or tool_response["request_fingerprint"] != prepared["payload"]["request_fingerprint"]:
            raise ContractError("commit evidence epoch/fingerprint mismatch")
        common = {key: prepared[key] for key in ("schema_version", "task_id", "step_id", "op_id", "model_revision")}
        common["context_spec_id"] = prepared["context_spec_id"]
        common["owner_epoch"] = owner_epoch
        result_body = {**common, "record_id": commit_id + ":result", "record_type": "TOOL_RESULT_COMMITTED",
                       "payload": {key: tool_response[key] for key in ("request_fingerprint", "result", "commit_receipt")}}
        context_body = {**common, "record_id": commit_id + ":context", "record_type": "CONTEXT_COMMITTED",
                        "payload": {"token_ids": list(token_ids), "result_record_id": result_body["record_id"]}}
        for body in (result_body, context_body):
            validate_schema("durable-log", {**body, "sequence": 1})
        with self.store.transaction() as connection:
            task = self.store.task(connection, prepared["task_id"])
            if task["owner_epoch"] != owner_epoch:
                return {"status": "STALE_EPOCH", "current_epoch": task["owner_epoch"]}
            records = self._records(connection, prepared["task_id"])
            existing = [r for r in records if r["record_id"] in {result_body["record_id"], context_body["record_id"]}]
            if existing:
                bodies = [{key: value for key, value in r.items() if key != "sequence"} for r in existing]
                if len(existing) == 2 and _same_json(bodies, [result_body, context_body]):
                    return {"status": "ALREADY_COMMITTED", "records": existing}
                return {"status": "RECORD_ID_CONFLICT"}
            if len(records) != expected_sequence:
                return {"status": "SEQUENCE_CONFLICT", "current_sequence": len(records)}
            if not any(_same_json(prepared, record) for record in records) or prepared["record_type"] != "STEP_PREPARED":
                return {"status": "INVALID_TRANSITION", "reason": "prepare is not durable evidence"}
            bound = connection.execute("SELECT context_spec_id,specification_json FROM task_context WHERE task_id=?", (prepared["task_id"],)).fetchone()
            if bound is None or bound[0] != prepared.get("context_spec_id") or json.loads(bound[1])["model_revision"] != prepared["model_revision"]:
                return {"status": "INVALID_TRANSITION", "reason": "context spec is not bound to task"}
            added = [{**result_body, "sequence": len(records) + 1}, {**context_body, "sequence": len(records) + 2}]
            try:
                validate_log(records + added)
                if task["halted"]:
                    raise ContractError("task is halted")
            except ContractError as exc:
                return {"status": "INVALID_TRANSITION", "reason": str(exc)}
            for record in added:
                connection.execute("INSERT INTO log_records VALUES (?,?,?,?)", (
                    record["task_id"], record["sequence"], record["record_id"],
                    json.dumps(record, ensure_ascii=False, allow_nan=False)))
            connection.execute("UPDATE tasks SET context_record_id=? WHERE task_id=?",
                               (context_body["record_id"], prepared["task_id"]))
            return {"status": "COMMITTED", "records": added}

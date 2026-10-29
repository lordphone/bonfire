"""SQLite stand-ins for C1 and C2 until Victoria's modules land.

Both stubs share one database file so state survives a SIGKILL of the
runner process. They implement the semantics in interfaces.py and nothing
more; replace them with outcomeserve.log / outcomeserve.tools once C1/C2
are frozen.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from .interfaces import (
    LogRecord,
    RecordType,
    StaleEpochError,
    ToolCapabilities,
    ToolOutcome,
    ToolResponse,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS epochs (task_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS log (seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tool_ops (op_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, tool TEXT NOT NULL, result TEXT);
CREATE TABLE IF NOT EXISTS effects (id INTEGER PRIMARY KEY AUTOINCREMENT, op_id TEXT NOT NULL, tool TEXT NOT NULL, args TEXT NOT NULL);
"""

# create_order: deduplicated by op_id and queryable (the safe case).
# lookup: read-only, so repeating it is harmless.
# notify_legacy: neither deduplicated nor queryable (forces RECOVERY_ABSTAIN).
TOOL_CAPABILITIES = {
    "create_order": ToolCapabilities(idempotent=True, queryable=True),
    "lookup": ToolCapabilities(idempotent=True, queryable=True),
    "notify_legacy": ToolCapabilities(idempotent=False, queryable=False),
}
READ_ONLY_TOOLS = {"lookup"}


class _Db:
    def __init__(self, path: str | Path):
        self.path = str(path)
        # executescript commits on its own, so it cannot run inside _tx.
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @staticmethod
    def _epoch(conn: sqlite3.Connection, task_id: str) -> int:
        row = conn.execute("SELECT epoch FROM epochs WHERE task_id = ?", (task_id,)).fetchone()
        return row[0] if row else 0


class SqliteLog(_Db):
    def append(self, record: LogRecord) -> None:
        with self._tx() as conn:
            current = self._epoch(conn, record.task_id)
            if record.owner_epoch != current:
                raise StaleEpochError(f"epoch {record.owner_epoch} != current {current}")
            payload = asdict(record)
            payload["type"] = record.type.value
            payload["token_ids"] = list(record.token_ids)
            conn.execute(
                "INSERT INTO log (task_id, payload) VALUES (?, ?)",
                (record.task_id, json.dumps(payload)),
            )

    def records(self, task_id: str) -> list[LogRecord]:
        with self._tx() as conn:
            rows = conn.execute("SELECT payload FROM log WHERE task_id = ? ORDER BY seq", (task_id,)).fetchall()
        out = []
        for (payload,) in rows:
            data = json.loads(payload)
            data["type"] = RecordType(data["type"])
            data["token_ids"] = tuple(data["token_ids"])
            out.append(LogRecord(**data))
        return out

    def current_epoch(self, task_id: str) -> int:
        with self._tx() as conn:
            return self._epoch(conn, task_id)

    def advance_epoch(self, task_id: str) -> int:
        with self._tx() as conn:
            new = self._epoch(conn, task_id) + 1
            conn.execute(
                "INSERT INTO epochs (task_id, epoch) VALUES (?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET epoch = excluded.epoch",
                (task_id, new),
            )
            return new


class SqliteToolService(_Db):
    def capabilities(self, tool: str) -> ToolCapabilities:
        return TOOL_CAPABILITIES[tool]

    def submit(
        self, op_id: str, fingerprint: str, tool: str, args: dict[str, Any], epoch: int, task_id: str
    ) -> ToolResponse:
        caps = self.capabilities(tool)
        with self._tx() as conn:
            if epoch != self._epoch(conn, task_id):
                return ToolResponse(ToolOutcome.STALE_EPOCH)
            if caps.idempotent:
                row = conn.execute("SELECT fingerprint, result FROM tool_ops WHERE op_id = ?", (op_id,)).fetchone()
                if row:
                    if row[0] != fingerprint:
                        return ToolResponse(ToolOutcome.FINGERPRINT_MISMATCH)
                    return ToolResponse(ToolOutcome.COMMITTED, json.loads(row[1]))
            result = self._apply(conn, op_id, tool, args)
            if caps.idempotent:
                conn.execute(
                    "INSERT INTO tool_ops (op_id, fingerprint, tool, result) VALUES (?, ?, ?, ?)",
                    (op_id, fingerprint, tool, json.dumps(result)),
                )
            return ToolResponse(ToolOutcome.COMMITTED, result)

    def query(self, op_id: str, tool: str) -> ToolResponse:
        if not self.capabilities(tool).queryable:
            return ToolResponse(ToolOutcome.UNKNOWN)
        with self._tx() as conn:
            row = conn.execute("SELECT result FROM tool_ops WHERE op_id = ?", (op_id,)).fetchone()
        if row is None:
            return ToolResponse(ToolOutcome.NOT_COMMITTED)
        return ToolResponse(ToolOutcome.COMMITTED, json.loads(row[0]))

    @staticmethod
    def _apply(conn: sqlite3.Connection, op_id: str, tool: str, args: dict[str, Any]) -> Any:
        if tool in READ_ONLY_TOOLS:
            return {"item": args.get("item"), "in_stock": True}
        cursor = conn.execute(
            "INSERT INTO effects (op_id, tool, args) VALUES (?, ?, ?)",
            (op_id, tool, json.dumps(args, sort_keys=True)),
        )
        return {"effect_id": cursor.lastrowid}

    def effect_counts(self) -> dict[str, int]:
        """Test helper: how many times each operation's side effect was applied."""
        with self._tx() as conn:
            rows = conn.execute("SELECT op_id, COUNT(*) FROM effects GROUP BY op_id").fetchall()
        return dict(rows)

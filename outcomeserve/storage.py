"""Shared SQLite ownership authority and transaction boundary for C1/C2."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3


class OwnershipConflict(ValueError):
    """Supervisor compare-and-swap did not match the durable epoch."""


class TaskHalted(ValueError):
    """An abstained task cannot perform more business mutations."""


class StateStore:
    """Local-disk CPU implementation; all writers must use this same database."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0):
        if str(path) == ":memory:" or timeout < 0:
            raise ValueError("a file-backed database and nonnegative timeout are required")
        self.path = Path(path).resolve()
        self.timeout = timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    owner_epoch INTEGER NOT NULL CHECK(owner_epoch > 0),
                    halted INTEGER NOT NULL DEFAULT 0,
                    context_record_id TEXT
                );
                CREATE TABLE IF NOT EXISTS inventory (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    sku TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK(quantity >= 0),
                    PRIMARY KEY(task_id, sku)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    op_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    commit_receipt TEXT NOT NULL,
                    committed_epoch INTEGER NOT NULL,
                    PRIMARY KEY(task_id, op_id),
                    UNIQUE(task_id, step_id)
                );
                CREATE TABLE IF NOT EXISTS log_records (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    sequence INTEGER NOT NULL,
                    record_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(task_id, sequence),
                    UNIQUE(task_id, record_id)
                );
                CREATE TABLE IF NOT EXISTS task_context (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
                    context_spec_id TEXT NOT NULL,
                    specification_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operation_identity (
                    task_id TEXT NOT NULL, op_id TEXT NOT NULL,
                    logical_step_id TEXT NOT NULL, execution_attempt TEXT NOT NULL,
                    PRIMARY KEY(task_id, op_id)
                );
                CREATE TABLE IF NOT EXISTS tool_effects (
                    effect_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL, op_id TEXT NOT NULL,
                    logical_step_id TEXT NOT NULL, execution_attempt TEXT NOT NULL,
                    sku TEXT NOT NULL, quantity_delta INTEGER NOT NULL,
                    commit_receipt TEXT NOT NULL, committed_epoch INTEGER NOT NULL
                );
            """)
        finally:
            connection.close()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def task(connection, task_id):
        task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        return task

    def create_task(self, task_id: str, inventory: dict[str, int]) -> None:
        """Administrative setup, exactly once; never reset external state on retry."""
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("nonempty task_id required")
        if not isinstance(inventory, dict) or any(
            not isinstance(sku, str) or not sku or type(count) is not int
            or not 0 <= count <= 2**53 - 1 for sku, count in inventory.items()
        ):
            raise ValueError("inventory requires SKU strings and nonnegative safe integers")
        with self.transaction() as connection:
            connection.execute("INSERT INTO tasks(task_id, owner_epoch) VALUES (?,1)", (task_id,))
            connection.executemany(
                "INSERT INTO inventory VALUES (?,?,?)",
                [(task_id, sku, count) for sku, count in inventory.items()],
            )

    def advance_epoch(self, task_id: str, expected_epoch: int) -> int:
        """Supervisor adapter: CAS serializes with both log and tool commits."""
        if type(expected_epoch) is not int or expected_epoch < 1:
            raise ValueError("positive expected_epoch required")
        with self.transaction() as connection:
            task = self.task(connection, task_id)
            if task["owner_epoch"] != expected_epoch:
                raise OwnershipConflict("owner_epoch compare-and-swap failed")
            connection.execute(
                "UPDATE tasks SET owner_epoch=owner_epoch+1 WHERE task_id=?", (task_id,)
            )
            return expected_epoch + 1

    def bind_context(self, task_id, specification):
        import json
        from outcomeserve.context import context_spec_id
        identity = context_spec_id(specification)
        with self.transaction() as connection:
            self.task(connection, task_id)
            existing = connection.execute("SELECT context_spec_id FROM task_context WHERE task_id=?", (task_id,)).fetchone()
            if existing is not None and existing[0] != identity:
                raise ValueError("task context specification cannot change on replay")
            connection.execute("INSERT OR IGNORE INTO task_context VALUES (?,?,?)",
                               (task_id, identity, json.dumps(specification, sort_keys=True)))
        return identity

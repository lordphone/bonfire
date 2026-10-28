"""Consumer-side view of C1 (durable log) and C2 (tool service).

These protocols state what the recovery controller needs from Victoria's
modules. They are a proposal for the C1/C2 drafts, not the final contracts:
names may change, but the semantics below are what reconciliation relies on.

C1 durable log
  - Records are appended per task in a total order and read back in that order.
  - An append carries the writer's owner_epoch; the log rejects it with
    StaleEpochError unless that epoch equals the task's current epoch.
  - advance_epoch is atomic and is called by the supervisor before a
    replacement worker starts.

C2 tool service
  - submit(op_id, ...) is keyed by a stable operation ID. A tool that
    supports deduplication returns the stored result for a repeated op_id
    instead of applying the effect again.
  - query(op_id) answers COMMITTED (with the stored result) or NOT_COMMITTED
    for queryable tools, and UNKNOWN for tools that cannot answer.
  - A submit whose epoch is not current returns STALE_EPOCH and has no effect.
  - Reusing an op_id with different arguments returns FINGERPRINT_MISMATCH.
  - Effect and stored result commit atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class RecordType(str, Enum):
    STEP_PREPARED = "STEP_PREPARED"
    TOOL_RESULT_COMMITTED = "TOOL_RESULT_COMMITTED"
    CONTEXT_COMMITTED = "CONTEXT_COMMITTED"
    RECOVERY_ABSTAIN = "RECOVERY_ABSTAIN"


@dataclass(frozen=True)
class LogRecord:
    type: RecordType
    task_id: str
    step_id: int
    owner_epoch: int
    op_id: str | None = None
    fingerprint: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: Any = None
    token_ids: tuple[int, ...] = ()
    reason: str | None = None


class StaleEpochError(RuntimeError):
    """A fenced-out writer tried to append."""


class DurableLog(Protocol):
    def append(self, record: LogRecord) -> None: ...

    def records(self, task_id: str) -> list[LogRecord]: ...

    def current_epoch(self, task_id: str) -> int: ...

    def advance_epoch(self, task_id: str) -> int: ...


class ToolOutcome(str, Enum):
    COMMITTED = "COMMITTED"
    NOT_COMMITTED = "NOT_COMMITTED"
    UNKNOWN = "UNKNOWN"
    STALE_EPOCH = "STALE_EPOCH"
    FINGERPRINT_MISMATCH = "FINGERPRINT_MISMATCH"


@dataclass(frozen=True)
class ToolResponse:
    outcome: ToolOutcome
    result: Any = None


@dataclass(frozen=True)
class ToolCapabilities:
    idempotent: bool
    queryable: bool


class AcknowledgmentLost(RuntimeError):
    """The tool committed but its response never reached the runner."""


class ToolService(Protocol):
    def capabilities(self, tool: str) -> ToolCapabilities: ...

    def submit(
        self, op_id: str, fingerprint: str, tool: str, args: dict[str, Any], epoch: int, task_id: str
    ) -> ToolResponse: ...

    def query(self, op_id: str, tool: str) -> ToolResponse: ...


@dataclass
class TaskState:
    """What the controller reconstructs from the log after a crash."""

    next_step: int = 0
    token_ids: list[int] = field(default_factory=list)
    abstained: bool = False
    recovered: bool = False  # the log already held records for this task
    reconciled_step: int | None = None  # the open step this recovery closed, if any

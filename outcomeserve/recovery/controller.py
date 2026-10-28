"""Recovery controller: rebuild task state from the durable log and close the open step.

Each step moves through prepare -> execute -> commit:

  STEP_PREPARED          op_id, tool call and generated tokens are durable
  TOOL_RESULT_COMMITTED  the tool's result is durable
  CONTEXT_COMMITTED      the step's tokens are part of the committed prefix

After a crash, at most the last step is open. Reconciliation resolves its tool
outcome under the original op_id and either commits it or, when the outcome
cannot be established, persists RECOVERY_ABSTAIN and stops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .interfaces import (
    DurableLog,
    LogRecord,
    RecordType,
    StaleEpochError,
    TaskState,
    ToolOutcome,
    ToolService,
)

Tokenize = Callable[[Any], list[int]]


@dataclass(frozen=True)
class Resolution:
    committed: bool
    result: Any = None
    reason: str | None = None


class RecoveryController:
    def __init__(self, log: DurableLog, tools: ToolService, task_id: str, epoch: int, tokenize: Tokenize):
        self.log = log
        self.tools = tools
        self.task_id = task_id
        self.epoch = epoch
        self.tokenize = tokenize

    def recover(self) -> TaskState:
        records = self.log.records(self.task_id)
        state = TaskState()
        open_step: dict[RecordType, LogRecord] = {}
        for record in records:
            if record.type is RecordType.RECOVERY_ABSTAIN:
                state.abstained = True
                state.next_step = record.step_id
                return state
            if record.type is RecordType.STEP_PREPARED:
                open_step = {RecordType.STEP_PREPARED: record}
            elif record.type is RecordType.TOOL_RESULT_COMMITTED:
                open_step[RecordType.TOOL_RESULT_COMMITTED] = record
            elif record.type is RecordType.CONTEXT_COMMITTED:
                state.token_ids.extend(record.token_ids)
                state.next_step = record.step_id + 1
                open_step = {}

        prepared = open_step.get(RecordType.STEP_PREPARED)
        if prepared is None:
            return state

        stored = open_step.get(RecordType.TOOL_RESULT_COMMITTED)
        if stored is not None:
            state.token_ids.extend(self.commit_context(prepared, stored.result))
            state.next_step = prepared.step_id + 1
            return state

        resolution = self.resolve(prepared)
        if not resolution.committed:
            self.abstain(prepared, resolution.reason or "unknown outcome")
            state.abstained = True
            state.next_step = prepared.step_id
            return state
        state.token_ids.extend(self.commit(prepared, resolution.result))
        state.next_step = prepared.step_id + 1
        return state

    def resolve(self, prepared: LogRecord) -> Resolution:
        """Establish the tool outcome for a prepared step without risking a second effect."""
        assert prepared.op_id and prepared.tool and prepared.fingerprint and prepared.args is not None
        caps = self.tools.capabilities(prepared.tool)

        if caps.queryable:
            answer = self.tools.query(prepared.op_id, prepared.tool)
            if answer.outcome is ToolOutcome.COMMITTED:
                return Resolution(True, answer.result)
            if answer.outcome is not ToolOutcome.NOT_COMMITTED:
                return Resolution(False, reason=f"query returned {answer.outcome.value}")
        elif not caps.idempotent:
            return Resolution(False, reason=f"{prepared.tool} is neither queryable nor idempotent")

        # Either the query proved NOT_COMMITTED, or the tool deduplicates by op_id.
        response = self.tools.submit(
            prepared.op_id, prepared.fingerprint, prepared.tool, prepared.args, self.epoch, self.task_id
        )
        if response.outcome is ToolOutcome.STALE_EPOCH:
            raise StaleEpochError(f"tool service fenced epoch {self.epoch}")
        if response.outcome is ToolOutcome.COMMITTED:
            return Resolution(True, response.result)
        return Resolution(False, reason=f"submit returned {response.outcome.value}")

    def commit(self, prepared: LogRecord, result: Any) -> list[int]:
        self.log.append(
            LogRecord(
                type=RecordType.TOOL_RESULT_COMMITTED,
                task_id=self.task_id,
                step_id=prepared.step_id,
                owner_epoch=self.epoch,
                op_id=prepared.op_id,
                result=result,
            )
        )
        return self.commit_context(prepared, result)

    def commit_context(self, prepared: LogRecord, result: Any) -> list[int]:
        tokens = list(prepared.token_ids) + self.tokenize(result)
        self.log.append(
            LogRecord(
                type=RecordType.CONTEXT_COMMITTED,
                task_id=self.task_id,
                step_id=prepared.step_id,
                owner_epoch=self.epoch,
                op_id=prepared.op_id,
                token_ids=tuple(tokens),
            )
        )
        return tokens

    def abstain(self, prepared: LogRecord, reason: str) -> None:
        self.log.append(
            LogRecord(
                type=RecordType.RECOVERY_ABSTAIN,
                task_id=self.task_id,
                step_id=prepared.step_id,
                owner_epoch=self.epoch,
                op_id=prepared.op_id,
                tool=prepared.tool,
                reason=reason,
            )
        )

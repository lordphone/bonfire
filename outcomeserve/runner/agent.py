"""Scripted agent runner for the CPU path.

The "model" is a fixed trace, so the committed token prefix of a crashed and
recovered run must equal the prefix of a failure-free run token for token.
The vLLM path (C3) replaces scripted_tokens with real generation later.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
from dataclasses import dataclass
from enum import Enum
from typing import Any

from outcomeserve.recovery.controller import RecoveryController
from outcomeserve.recovery.interfaces import (
    AcknowledgmentLost,
    DurableLog,
    LogRecord,
    RecordType,
    StaleEpochError,
    ToolOutcome,
    ToolService,
)
from outcomeserve.recovery.opid import fingerprint, op_id


class Boundary(str, Enum):
    DURING_GENERATION = "DURING_GENERATION"
    PRE_DISPATCH = "PRE_DISPATCH"
    POST_TOOL_COMMIT_PRE_RESULT_LOG = "POST_TOOL_COMMIT_PRE_RESULT_LOG"
    POST_CONTEXT_COMMIT = "POST_CONTEXT_COMMIT"


class Action(str, Enum):
    TERMINATE_PROCESS = "TERMINATE_PROCESS"
    DROP_ACKNOWLEDGMENT = "DROP_ACKNOWLEDGMENT"


@dataclass(frozen=True)
class FaultPlan:
    boundary: Boundary
    step: int
    action: Action = Action.TERMINATE_PROCESS

    def __post_init__(self) -> None:
        if self.action is Action.DROP_ACKNOWLEDGMENT and self.boundary is not Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG:
            raise ValueError("DROP_ACKNOWLEDGMENT is only legal at POST_TOOL_COMMIT_PRE_RESULT_LOG")


@dataclass(frozen=True)
class Step:
    thought: str
    tool: str
    args: dict[str, Any]


def scripted_trace(name: str = "orders", steps: int = 10) -> list[Step]:
    trace = []
    for i in range(steps):
        if i % 2 == 0:
            trace.append(Step(f"check stock for item {i}", "lookup", {"item": f"sku-{i}"}))
        else:
            trace.append(Step(f"place order {i}", "create_order", {"item": f"sku-{i - 1}", "qty": i}))
    if name == "legacy":
        trace[3] = Step("notify the warehouse", "notify_legacy", {"message": "ship sku-2"})
    elif name != "orders":
        raise ValueError(f"unknown trace {name!r}")
    return trace


def scripted_tokens(value: Any) -> list[int]:
    """Deterministic stand-in for tokenization: 4 token IDs per value."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    digest = hashlib.sha256(text.encode()).digest()
    return [int.from_bytes(digest[i : i + 2], "big") % 151_936 for i in range(0, 8, 2)]


@dataclass(frozen=True)
class RunOutcome:
    status: str  # COMPLETED or ABSTAINED
    token_ids: tuple[int, ...]
    next_step: int


class Runner:
    def __init__(
        self,
        log: DurableLog,
        tools: ToolService,
        task_id: str,
        trace: list[Step],
        fault: FaultPlan | None = None,
    ):
        self.log = log
        self.tools = tools
        self.task_id = task_id
        self.trace = trace
        self.fault = fault
        self.epoch = log.current_epoch(task_id)
        self.controller = RecoveryController(log, tools, task_id, self.epoch, scripted_tokens)

    def run(self) -> RunOutcome:
        state = self.controller.recover()
        tokens = list(state.token_ids)
        if state.abstained:
            return RunOutcome("ABSTAINED", tuple(tokens), state.next_step)

        for step_id in range(state.next_step, len(self.trace)):
            step = self.trace[step_id]
            self._maybe_fault(Boundary.DURING_GENERATION, step_id)
            prepared = LogRecord(
                type=RecordType.STEP_PREPARED,
                task_id=self.task_id,
                step_id=step_id,
                owner_epoch=self.epoch,
                op_id=op_id(self.task_id, step_id, step.tool, step.args),
                fingerprint=fingerprint(step.tool, step.args),
                tool=step.tool,
                args=step.args,
                token_ids=tuple(scripted_tokens(step.thought) + scripted_tokens([step.tool, step.args])),
            )
            self.log.append(prepared)
            self._maybe_fault(Boundary.PRE_DISPATCH, step_id)

            try:
                result = self._dispatch(prepared)
            except AcknowledgmentLost:
                resolution = self.controller.resolve(prepared)
                if not resolution.committed:
                    self.controller.abstain(prepared, resolution.reason or "acknowledgment lost")
                    return RunOutcome("ABSTAINED", tuple(tokens), step_id)
                result = resolution.result

            self._maybe_fault(Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG, step_id)
            tokens.extend(self.controller.commit(prepared, result))
            self._maybe_fault(Boundary.POST_CONTEXT_COMMIT, step_id)

        return RunOutcome("COMPLETED", tuple(tokens), len(self.trace))

    def _dispatch(self, prepared: LogRecord) -> Any:
        assert prepared.op_id and prepared.fingerprint and prepared.tool and prepared.args is not None
        response = self.tools.submit(
            prepared.op_id, prepared.fingerprint, prepared.tool, prepared.args, self.epoch, self.task_id
        )
        if response.outcome is ToolOutcome.STALE_EPOCH:
            raise StaleEpochError(f"tool service fenced epoch {self.epoch}")
        if response.outcome is not ToolOutcome.COMMITTED:
            raise RuntimeError(f"submit returned {response.outcome.value} for {prepared.op_id}")
        if self._fault_fires(Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG, prepared.step_id, Action.DROP_ACKNOWLEDGMENT):
            self.fault = None
            raise AcknowledgmentLost(prepared.op_id)
        return response.result

    def _fault_fires(self, boundary: Boundary, step_id: int, action: Action) -> bool:
        f = self.fault
        return f is not None and f.boundary is boundary and f.step == step_id and f.action is action

    def _maybe_fault(self, boundary: Boundary, step_id: int) -> None:
        if self._fault_fires(boundary, step_id, Action.TERMINATE_PROCESS):
            os.kill(os.getpid(), signal.SIGKILL)

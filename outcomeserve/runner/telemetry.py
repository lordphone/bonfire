"""C4 telemetry writer: append-only JSONL events valid against telemetry-event.schema.json.

Sequence numbers are allocated under a file lock from the event count, so a
replacement worker continues the run's sequence without gaps. Every event is
fsynced before emit returns; a FAULT_INJECTED event is therefore durable
before the process kills itself.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class WorkerIdentity:
    lineage_id: str
    instance_id: str
    attempt: int
    owner_epoch: int


@dataclass(frozen=True)
class RunContext:
    experiment_id: str
    run_id: str
    task_id: str
    configuration_id: str
    model: dict[str, str]
    hardware: dict[str, str | None]
    clock_id: str


def cpu_run_context(task_id: str, run_id: str, trace_definition: str) -> RunContext:
    """Identity for the CPU scripted path: no GPU, and the trace stands in for the model."""
    host = socket.gethostname()
    return RunContext(
        experiment_id="exp-cpu-scripted",
        run_id=run_id,
        task_id=task_id,
        configuration_id="cpu-scripted-sqlite-stub",
        model={
            "id": "outcomeserve/scripted-trace",
            "revision": hashlib.sha1(trace_definition.encode()).hexdigest(),
            "dtype": "none",
        },
        hardware={"host_id": host, "gpu_model": None, "gpu_uuid": None, "driver_version": None},
        clock_id=f"{host}-monotonic",
    )


def step_label(step_id: int) -> str:
    return f"step-{step_id:02d}"


class Telemetry(Protocol):
    def emit(
        self,
        event_type: str,
        step_id: int,
        worker: WorkerIdentity,
        details: dict[str, Any] | None = None,
        injection_id: str | None = None,
        accepted_state_mutation: bool = False,
    ) -> dict[str, Any] | None: ...


class NullTelemetry:
    def emit(self, *args: Any, **kwargs: Any) -> None:
        return None


class JsonlTelemetry:
    def __init__(self, path: str | Path, context: RunContext):
        self.path = Path(path)
        self.context = context

    def emit(
        self,
        event_type: str,
        step_id: int,
        worker: WorkerIdentity,
        details: dict[str, Any] | None = None,
        injection_id: str | None = None,
        accepted_state_mutation: bool = False,
    ) -> dict[str, Any]:
        with open(self.path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            sequence = sum(1 for line in handle if line.strip()) + 1
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_id": f"{self.context.run_id}-{sequence:06d}",
                "event_type": event_type,
                "experiment_id": self.context.experiment_id,
                "run_id": self.context.run_id,
                "task_id": self.context.task_id,
                "step_id": step_label(step_id),
                "injection_id": injection_id,
                "configuration_id": self.context.configuration_id,
                "sequence": sequence,
                "wall_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "monotonic_ns": time.monotonic_ns(),
                "clock_id": self.context.clock_id,
                "worker": {
                    "lineage_id": worker.lineage_id,
                    "instance_id": worker.instance_id,
                    "attempt": worker.attempt,
                    "owner_epoch": worker.owner_epoch,
                },
                "accepted_state_mutation": accepted_state_mutation,
                "hardware": dict(self.context.hardware),
                "model": dict(self.context.model),
                "details": details or {},
            }
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event


def read_events(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]

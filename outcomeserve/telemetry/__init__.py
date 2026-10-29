"""Telemetry events (C4) and GPU sampling.

Owner: Lordphone · Reviewer: Victoria
Contract: schemas/telemetry-event.schema.json, docs/infrastructure/measurement-contract.md

writer.py was started on the runner side so the CPU recovery path could emit
schema-valid events; Lordphone owns it from here.
"""

from .writer import (
    JsonlTelemetry,
    NullTelemetry,
    RunContext,
    Telemetry,
    WorkerIdentity,
    cpu_run_context,
    read_events,
    step_label,
)

__all__ = [
    "JsonlTelemetry",
    "NullTelemetry",
    "RunContext",
    "Telemetry",
    "WorkerIdentity",
    "cpu_run_context",
    "read_events",
    "step_label",
]

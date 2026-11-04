"""Same-host supervisor timing: correct completion and first token are separate."""

import json
from pathlib import Path
import socket
import time
from uuid import uuid4
from outcomeserve.contracts import validate_schema


class Measurements:
    def __init__(self, path, *, run_id, configuration_id):
        self.path = Path(path)
        self.run_id, self.configuration_id = run_id, configuration_id
        self.clock_id = socket.gethostname() + ":" + uuid4().hex
        self.events = []

    def emit(self, event_type, **details):
        event = {"schema_version": "1.1.0", "run_id": self.run_id,
                 "configuration_id": self.configuration_id, "clock_id": self.clock_id,
                 "sequence": len(self.events) + 1, "event_type": event_type,
                 "monotonic_ns": time.perf_counter_ns(), "details": details}
        event.update(clock_source="perf_counter", clock_resolution_seconds=time.get_clock_info("perf_counter").resolution)
        self.events.append(event)
        validate_schema("measurement-event", event)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        return event["monotonic_ns"]

    def duration(self, start, end):
        left = [e["monotonic_ns"] for e in self.events if e["event_type"] == start]
        right = [e["monotonic_ns"] for e in self.events if e["event_type"] == end]
        if len(left) != 1 or len(right) != 1 or right[0] < left[0]:
            return None
        return (right[0] - left[0]) / 1e9

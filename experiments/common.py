"""Shared experiment output and bounded process-readiness helpers."""
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def _ready(process, timeout):
    queue = Queue()
    def read():
        queue.put(process.stdout.readline())
    Thread(target=read, daemon=True).start()
    try:
        line = queue.get(timeout=timeout)
    except Empty as exc:
        raise TimeoutError("worker failed to reach scheduled boundary") from exc
    if not line:
        raise RuntimeError("worker exited before scheduled boundary")
    return json.loads(line)


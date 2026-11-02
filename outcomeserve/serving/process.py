"""Own and restart an actual inference process, independently of runner epoch."""

import json
import subprocess
import time
from urllib.request import urlopen
from urllib.parse import urlparse
import socket
from outcomeserve.processes import kill_owned


class InferenceProcess:
    def __init__(self, command, endpoint, model, *, timeout=300):
        if not command or isinstance(command, str):
            raise ValueError("explicit argument list required, never shell text")
        self.command, self.endpoint, self.model, self.timeout = list(command), endpoint.rstrip("/"), model, timeout
        self.process, self.instance = None, 0

    def start(self, *, stdout=None, stderr=None, on_launch=None):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("inference process is already running")
        address = urlparse(self.endpoint)
        try:
            probe = socket.create_connection((address.hostname, address.port or (443 if address.scheme == "https" else 80)), timeout=0.5)
        except OSError as exc:
            if isinstance(exc, PermissionError):
                raise
        else:
            probe.close()
            raise RuntimeError("inference endpoint already in use; cannot establish a fresh owned instance")
        self.process = subprocess.Popen(self.command, stdout=stdout, stderr=stderr)
        self.instance += 1
        if on_launch is not None:
            on_launch(self.process.pid)
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("inference process exited before readiness")
                try:
                    with urlopen(self.endpoint + "/v1/models", timeout=2) as response:
                        models = json.load(response)["data"]
                    if any(model["id"] == self.model for model in models):
                        return self.process.pid
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.2)
            raise TimeoutError("inference readiness deadline exceeded")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            kill_owned(self.process)
            self.process.wait(timeout=15)

"""C2 client; response loss never becomes a false NOT_COMMITTED answer."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AcknowledgmentLost(TimeoutError):
    pass


class ToolClient:
    def __init__(self, endpoint, timeout=35):
        self.endpoint, self.timeout = endpoint.rstrip("/"), timeout

    def post(self, path, value):
        request = Request(self.endpoint + path, data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 504:
                raise AcknowledgmentLost("tool response unavailable after possible commit") from exc
            raise ValueError(exc.read().decode()) from exc
        except (TimeoutError, URLError) as exc:
            raise AcknowledgmentLost("tool outcome not established") from exc

    def submit(self, request, *, pause=False, logical_step_id=None, attempt=None):
        envelope = {"request": request, "pause_after_commit": pause}
        if logical_step_id is not None:
            envelope.update(logical_step_id=logical_step_id, execution_attempt=attempt)
        return self.post("/submit", envelope)

    def query(self, request):
        try:
            return self.post("/query", {"request": request})
        except AcknowledgmentLost:
            return {"status": "UNKNOWN", "reason": "query transport unavailable"}

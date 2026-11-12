"""Recovering runner over a real token-ID completion endpoint.

On every start the runner trusts only durable state:

1. A task with RECOVERY_ABSTAIN stays halted.
2. A step with STEP_PREPARED but no CONTEXT_COMMITTED is reconciled through the
   tool service under its original op_id: COMMITTED commits the stored result,
   NOT_COMMITTED submits once, UNKNOWN persists RECOVERY_ABSTAIN.
3. The prompt prefix is rebuilt from the committed context record's token IDs
   and replayed to the inference worker (re-prefill); text is never re-tokenized
   to reconstruct history.

If the inference worker dies mid-step, the runner survives: it waits for the
supervisor's replacement, rebuilds the prefix from the durable log, and retries
the step. The supervisor owns process lifecycle, epochs and fault timing; the
runner only exposes barriers (ready file + blocking stdin line) at each boundary.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

from outcomeserve.contracts import request_fingerprint
from outcomeserve.log import DurableLog
from outcomeserve.serving.client import CompletionBackend, InferenceError, InferenceUnavailable
from outcomeserve.storage import StateStore
from outcomeserve.tools.client import AcknowledgmentLost, ToolClient

STEP_PROMPT = "\nExecute next request as JSON tool_name/args: "
RESULT_PROMPT = "\nTool result: "

EXIT_COMPLETED, EXIT_ABSTAINED, EXIT_RESTART_REQUIRED = 0, 3, 4

BOUNDARIES = ("DURING_GENERATION", "PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG",
              "POST_CONTEXT_COMMIT", "DURING_RECOVERY")


def build_instruction(template: str, workload: dict, padding_repeats: int = 0) -> str:
    """The pilot's instruction text; the only text tokenized before any generation."""
    return template.format(task=json.dumps(workload["steps"]), padding="context " * padding_repeats)


@dataclass
class RunnerConfig:
    workload: dict
    context_spec: dict
    instruction: str
    epoch: int
    method: str = "outcomeserve"  # or "restart"
    seed: int = 33
    pause_at: str | None = None  # boundary name
    pause_step: str | None = None  # step_id
    ready_file: Path | None = None
    inference_wait_seconds: float = 600.0
    max_tokens: int = 256


class RecoveringRunner:
    def __init__(self, config: RunnerConfig, *, store: StateStore, tools: ToolClient, model: CompletionBackend,
                 emit: Callable[..., None], wait_for_inference: Callable[[], None] | None = None):
        if config.method not in {"outcomeserve", "restart"}:
            raise ValueError("method must be outcomeserve or restart")
        if config.pause_at is not None and config.pause_at not in BOUNDARIES:
            raise ValueError(f"unknown boundary {config.pause_at}")
        self.config, self.store, self.tools, self.model, self.emit = config, store, tools, model, emit
        self.task = config.workload["task_id"]
        self.log = DurableLog(store)
        self.context_id = store.bind_context(self.task, config.context_spec)
        self.wait_for_inference = wait_for_inference or (lambda: wait_until_serving(
            model.endpoint, model.model, config.inference_wait_seconds))
        self._paused = False

    # ---------------------------------------------------------------- entry point

    def run(self) -> int:
        steps = self.config.workload["steps"]
        initial = self._with_inference(lambda: self.model.tokenize(self.config.instruction))
        if self.config.method == "outcomeserve":
            records = self.log.read(self.task)
            if any(r["record_type"] == "RECOVERY_ABSTAIN" for r in records):
                self.emit("RECOVERY_HALTED", "none")
                return EXIT_ABSTAINED
            if self._reconcile(records, initial) == EXIT_ABSTAINED:
                return EXIT_ABSTAINED
            records = self.log.read(self.task)
            done = {r["step_id"] for r in records if r["record_type"] == "CONTEXT_COMMITTED"}
            index = next((i for i, step in enumerate(steps) if step["step_id"] not in done), len(steps))
            prefix = self._durable_prefix(initial)
            resuming = bool(records)
        else:
            index, prefix, resuming = 0, list(initial), self.config.epoch > 1

        while index < len(steps):
            step = steps[index]
            try:
                prompt, answer = self._generate(step, prefix, measure_resume=resuming)
            except InferenceUnavailable as exc:
                self.emit("INFERENCE_LOST", step["step_id"], reason=str(exc))
                if self.config.method == "restart":
                    return EXIT_RESTART_REQUIRED
                self._await_replacement(step)
                replayed = self._durable_prefix(initial)
                if replayed != prefix:
                    raise RuntimeError("durable prefix differs from the runner's committed history")
                prefix, resuming = replayed, True
                continue
            resuming = False
            outcome = self._execute(step, prompt, answer)
            if outcome is None:
                return EXIT_ABSTAINED
            prefix = outcome
            self._barrier("POST_CONTEXT_COMMIT", step["step_id"])
            index += 1
        self.emit("TASK_COMPLETED", "none", committed_prefix_tokens=len(prefix))
        return EXIT_COMPLETED

    # ---------------------------------------------------------------- recovery

    def _reconcile(self, records: list[dict], initial: list[int]) -> int:
        committed = {r["step_id"] for r in records if r["record_type"] == "CONTEXT_COMMITTED"}
        pending = [r for r in records if r["record_type"] == "STEP_PREPARED" and r["step_id"] not in committed]
        if not pending:
            return EXIT_COMPLETED
        if len(pending) > 1 or pending[0]["sequence"] != len(records):
            raise RuntimeError("durable log has more than one open step or an open step that is not the tail")
        prepared = pending[0]
        step_id = prepared["step_id"]
        self.emit("RECONCILIATION_STARTED", step_id, op_id=prepared["op_id"])
        identity = {"task_id": self.task, "step_id": step_id, "op_id": prepared["op_id"],
                    "owner_epoch": self.config.epoch,
                    "request_fingerprint": prepared["payload"]["request_fingerprint"]}
        answer = self.tools.query(identity)
        outcome = answer["status"]
        self._barrier("DURING_RECOVERY", step_id)
        if outcome == "NOT_COMMITTED":
            request = {**identity, "tool_name": prepared["payload"]["tool_name"], "args": prepared["payload"]["args"]}
            answer = self._submit(request)
        if answer["status"] == "UNKNOWN":
            self.emit("RECONCILIATION_FINISHED", step_id, outcome="UNKNOWN", op_id=prepared["op_id"])
            self._abstain(prepared, answer.get("reason", "outcome could not be established"), len(records))
            return EXIT_ABSTAINED
        if answer["status"] != "COMMITTED":
            raise RuntimeError(f"reconciliation of {prepared['op_id']} returned {answer['status']}")
        tokens = (prepared["payload"]["generation_prompt_token_ids"] + prepared["payload"]["generated_token_ids"]
                  + self._result_tokens(answer))
        self._commit(prepared, answer, tokens, expected_sequence=len(records))
        self.emit("RECONCILIATION_FINISHED", step_id, outcome=outcome, op_id=prepared["op_id"])
        return EXIT_COMPLETED

    def _durable_prefix(self, initial: list[int]) -> list[int]:
        record = self.log.committed_prefix(self.task)
        return list(initial) if record is None else list(record["payload"]["token_ids"])

    def _await_replacement(self, step: dict) -> None:
        self.wait_for_inference()
        self.emit("INFERENCE_REPLACEMENT_OBSERVED", step["step_id"])

    # ---------------------------------------------------------------- one step

    def _generate(self, step: dict, prefix: list[int], *, measure_resume: bool):
        expected = {"tool_name": step["tool_name"], "args": step["args"]}
        prompt = prefix + self.model.tokenize(STEP_PROMPT + json.dumps(expected))

        def first_token(tokens: list[int]) -> None:
            if measure_resume:
                # Time to first token after recovery = re-prefill of the replayed
                # prefix plus one decode step; serving-side timing would split them.
                self.emit("FIRST_NEW_TOKEN", step["step_id"], observed_at="client_stream")
            self._barrier("DURING_GENERATION", step["step_id"], generated_tokens=len(tokens))

        if measure_resume:
            self.emit("PREFILL_STARTED", step["step_id"], committed_prefix_tokens=len(prefix),
                      prompt_tokens=len(prompt))
        answer = self.model.generate_stream(prompt, first_token, max_tokens=self.config.max_tokens,
                                            seed=self.config.seed)
        if json.dumps(answer["call"], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise InferenceError(f"generated call for {step['step_id']} does not match the business request")
        return prompt, answer

    def _execute(self, step: dict, prompt: list[int], answer: dict) -> list[int] | None:
        call = answer["call"]
        fingerprint = request_fingerprint(call["tool_name"], call["args"])
        identity = {"task_id": self.task, "step_id": step["step_id"], "op_id": step["op_id"],
                    "owner_epoch": self.config.epoch}
        request = {**identity, "tool_name": call["tool_name"], "args": call["args"], "request_fingerprint": fingerprint}
        pause_after_commit = self._targets("POST_TOOL_COMMIT_PRE_RESULT_LOG", step["step_id"])

        if self.config.method == "restart":
            response = self.tools.submit(request, pause=pause_after_commit, logical_step_id=step["step_id"],
                                         attempt=f"{self.config.epoch}-1")
            if response["status"] != "COMMITTED":
                raise RuntimeError(f"tool returned {response['status']}")
            return prompt + answer["generated_token_ids"] + self._result_tokens(response)

        records = self.log.read(self.task)
        body = {"schema_version": "1.1.0", **identity, "context_spec_id": self.context_id,
                "record_id": f"{step['step_id']}:prepare", "model_revision": self.config.context_spec["model_revision"],
                "record_type": "STEP_PREPARED", "payload": {
                    "tool_name": call["tool_name"], "args": call["args"], "request_fingerprint": fingerprint,
                    "generated_token_ids": answer["generated_token_ids"], "generation_prompt_token_ids": prompt}}
        prepared = self.log.append(self.task, body["record_id"], len(records), self.config.epoch, body)
        if prepared["status"] != "APPENDED":
            raise RuntimeError(f"prepare for {step['step_id']} not appended: {prepared}")
        prepared = prepared["record"]
        self._barrier("PRE_DISPATCH", step["step_id"])

        response = self._submit(request, pause=pause_after_commit)
        if response["status"] == "UNKNOWN":
            self._abstain(prepared, response.get("reason", "acknowledgment lost"), len(records) + 1)
            return None
        if response["status"] != "COMMITTED":
            raise RuntimeError(f"tool returned {response['status']} for {step['op_id']}")
        tokens = prompt + answer["generated_token_ids"] + self._result_tokens(response)
        self._commit(prepared, response, tokens, expected_sequence=len(records) + 1)
        return tokens

    # ---------------------------------------------------------------- durable writes

    def _submit(self, request: dict, *, pause: bool = False) -> dict:
        try:
            return self.tools.submit(request, pause=pause)
        except AcknowledgmentLost:
            query = {k: v for k, v in request.items() if k not in {"tool_name", "args"}}
            self.emit("ACKNOWLEDGMENT_LOST", request["step_id"])
            return self.tools.query(query)

    def _commit(self, prepared: dict, response: dict, tokens: list[int], *, expected_sequence: int) -> None:
        result = self.log.commit_step(prepared, response, tokens, owner_epoch=self.config.epoch,
                                      expected_sequence=expected_sequence, commit_id=f"{prepared['step_id']}:commit")
        if result["status"] not in {"COMMITTED", "ALREADY_COMMITTED"}:
            raise RuntimeError(f"commit for {prepared['step_id']} failed: {result}")

    def _abstain(self, prepared: dict, reason: str, expected_sequence: int) -> None:
        body = {key: value for key, value in prepared.items() if key != "sequence"}
        body.update(owner_epoch=self.config.epoch, record_id=f"{prepared['step_id']}:halt:{self.config.epoch}",
                    record_type="RECOVERY_ABSTAIN", payload={"outcome": "UNKNOWN", "reason": reason})
        answer = self.log.append(self.task, body["record_id"], expected_sequence, self.config.epoch, body)
        if answer["status"] not in {"APPENDED", "ALREADY_APPENDED"}:
            raise RuntimeError(f"RECOVERY_ABSTAIN was not persisted: {answer}")
        self.emit("RECOVERY_ABSTAIN", prepared["step_id"], op_id=prepared["op_id"], reason=reason)

    # ---------------------------------------------------------------- faults and transport

    def _targets(self, boundary: str, step_id: str) -> bool:
        return not self._paused and self.config.pause_at == boundary and self.config.pause_step == step_id

    def _barrier(self, boundary: str, step_id: str, **evidence: Any) -> None:
        """Block at a scheduled boundary until the supervisor kills us or releases the line."""
        if boundary == "POST_TOOL_COMMIT_PRE_RESULT_LOG" or not self._targets(boundary, step_id):
            return  # that boundary's barrier is held by the tool service
        self._paused = True
        if self.config.ready_file is None:
            raise RuntimeError("a pause boundary needs a ready file")
        self.config.ready_file.write_text(json.dumps(
            {"step_id": step_id, "boundary": boundary, "epoch": self.config.epoch, **evidence}))
        self.emit("BOUNDARY_REACHED", step_id, boundary=boundary)
        sys.stdin.readline()
        self.emit("BOUNDARY_RELEASED", step_id, boundary=boundary)

    def _result_tokens(self, response: dict) -> list[int]:
        # The tool has committed; losing inference here must not lose the commit.
        return self._with_inference(lambda: self.model.tokenize(RESULT_PROMPT + json.dumps(response["result"])))

    def _with_inference(self, call: Callable[[], Any]) -> Any:
        while True:
            try:
                return call()
            except InferenceUnavailable as exc:
                self.emit("INFERENCE_LOST", "none", reason=str(exc))
                self.wait_for_inference()


def wait_until_serving(endpoint: str, model: str, timeout: float) -> None:
    """Poll /v1/models until the served model ID is listed (the pilot's readiness check)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(endpoint.rstrip("/") + "/v1/models", timeout=2) as response:
                if model in {entry.get("id") for entry in json.load(response).get("data", [])}:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"no inference worker served {model} within {timeout}s")

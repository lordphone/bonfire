"""End-to-end recovery over a token-ID completion server with real process kills.

Each test plays the supervisor: it starts the C2 tool service, a fake vLLM
server (tests/fake_vllm.py) and runner attempts as separate processes, kills
one of them at a scheduled boundary, advances the owner epoch for runner
replacement, and checks the final state with the independent verifiers.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.common import ROOT, _ready, save
from outcomeserve.contracts import validate_log
from outcomeserve.log import DurableLog
from outcomeserve.serving.client import CompletionBackend, InferenceUnavailable
from outcomeserve.storage import StateStore
from outcomeserve.tools.client import ToolClient
from outcomeserve.verify.business import verify_business
from workloads import make_inventory_workload

TEMPLATE = ROOT / "examples" / "gpu-pilot-prompt.txt"
FAKE_VLLM = Path(__file__).resolve().parent / "fake_vllm.py"
REVISION = json.loads((ROOT / "infra" / "environment.lock.json").read_text())["model"]["revision"]
TARGET = "step-03"  # first decrement_inventory with a step before it


def context_spec():
    return {"model_id": "Qwen/Qwen3-4B", "model_revision": REVISION, "tokenizer_id": "Qwen/Qwen3-4B",
            "tokenizer_revision": REVISION, "chat_template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
            "enable_thinking": False, "backend": "vllm"}


class Stack:
    """Test supervisor: tool service, inference server, runner attempts."""

    def __init__(self, directory: Path, method: str = "outcomeserve"):
        self.dir, self.method = directory, method
        self.workload = make_inventory_workload("gpu-int", seed=33, steps=5)
        self.task = self.workload["task_id"]
        save(directory / "workload.json", self.workload)
        save(directory / "context.json", context_spec())
        self.db = directory / "runtime.sqlite"
        self.store = StateStore(self.db)
        self.store.create_task(self.task, self.workload["initial_inventory"])
        self.tool_barrier = directory / "tool-boundary.json"
        self.tool_server = subprocess.Popen(
            [sys.executable, "-m", "outcomeserve.tools.server", "--db", str(self.db), "--barrier-file", str(self.tool_barrier)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.tool_endpoint = f"http://127.0.0.1:{_ready(self.tool_server, 15)['port']}"
        self.inference, self.inference_port = None, 0
        self.start_inference()
        self.epoch, self.runners = 1, []

    def start_inference(self):
        self.inference = subprocess.Popen([sys.executable, str(FAKE_VLLM), "--port", str(self.inference_port)],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.inference_port = _ready(self.inference, 15)["port"]

    def kill_inference(self):
        self.inference.kill()
        self.inference.wait(timeout=10)

    def launch(self, *, pause_at=None, pause_step=TARGET, tool_endpoint=None):
        command = [sys.executable, "-m", "outcomeserve.runner", "--db", str(self.db),
                   "--workload", str(self.dir / "workload.json"),
                   "--tool-endpoint", tool_endpoint or self.tool_endpoint,
                   "--inference-endpoint", f"http://127.0.0.1:{self.inference_port}",
                   "--context-spec", str(self.dir / "context.json"), "--template", str(TEMPLATE),
                   "--events", str(self.events_path), "--clock-id", "test-clock",
                   "--epoch", str(self.epoch), "--method", self.method, "--inference-wait", "20"]
        if pause_at:
            command += ["--pause-at", pause_at, "--pause-step", pause_step,
                        "--ready-file", str(self.dir / f"runner-boundary-{self.epoch}.json")]
        runner = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                  stderr=(self.dir / f"runner-{self.epoch}.stderr.txt").open("w"), text=True)
        self.runners.append(runner)
        return runner

    @property
    def events_path(self):
        return self.dir / "runner-events.jsonl"

    def wait_for_barrier(self, runner, boundary):
        path = self.tool_barrier if boundary == "POST_TOOL_COMMIT_PRE_RESULT_LOG" else self.dir / f"runner-boundary-{self.epoch}.json"
        deadline = time.monotonic() + 20
        while True:
            try:
                return json.loads(path.read_text())
            except (FileNotFoundError, ValueError):
                pass
            assert runner.poll() is None, (self.dir / f"runner-{self.epoch}.stderr.txt").read_text()
            assert time.monotonic() < deadline, f"{boundary} was not reached"
            time.sleep(0.01)

    def crash_runner(self, runner, boundary):
        """Kill the runner at its barrier and fence it out, as the supervisor does."""
        self.wait_for_barrier(runner, boundary)
        runner.kill()
        runner.wait(timeout=10)
        if boundary == "POST_TOOL_COMMIT_PRE_RESULT_LOG":
            ToolClient(self.tool_endpoint).post("/release", {"drop": True})
        self.epoch = self.store.advance_epoch(self.task, self.epoch)

    def finish(self, runner):
        code = runner.wait(timeout=60)
        return code, (self.dir / f"runner-{self.epoch}.stderr.txt").read_text()

    def events(self, epoch=None):
        rows = [json.loads(line) for line in self.events_path.read_text().splitlines()]
        return [r for r in rows if epoch is None or r["epoch"] == epoch]

    def committed_prefix(self):
        record = DurableLog(self.store).committed_prefix(self.task)
        return record["payload"]["token_ids"]

    def close(self):
        for process in [*self.runners, self.inference, self.tool_server]:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout) if process is not None else ():
                if stream:
                    stream.close()


@pytest.fixture
def stack(tmp_path):
    created = []

    def make(method="outcomeserve", name="run"):
        directory = tmp_path / name
        directory.mkdir()
        created.append(Stack(directory, method))
        return created[-1]

    yield make
    for item in created:
        item.close()


@pytest.fixture
def reference_prefix(stack):
    clean = stack(name="reference")
    code, errors = clean.finish(clean.launch())
    assert code == 0, errors
    return clean.committed_prefix()


def assert_exactly_once(run):
    report = verify_business(run.db, run.workload)
    assert report["task_success"], report
    assert report["metrics"]["duplicate_effects"] == 0
    validate_log(DurableLog(run.store).read(run.task))


def test_failure_free_run_commits_every_step_through_the_real_services(stack):
    run = stack()
    code, errors = run.finish(run.launch())

    assert code == 0, errors
    assert_exactly_once(run)
    kinds = [e["event_type"] for e in run.events()]
    assert "RECONCILIATION_STARTED" not in kinds and "PREFILL_STARTED" not in kinds
    assert kinds[-1] == "TASK_COMPLETED"


@pytest.mark.parametrize("boundary, reconciled", [
    ("DURING_GENERATION", None),
    ("PRE_DISPATCH", "NOT_COMMITTED"),
    ("POST_TOOL_COMMIT_PRE_RESULT_LOG", "COMMITTED"),
    ("POST_CONTEXT_COMMIT", None),
])
def test_runner_crash_reconciles_and_replays_the_durable_prefix(stack, reference_prefix, boundary, reconciled):
    run = stack()
    run.crash_runner(run.launch(pause_at=boundary), boundary)
    code, errors = run.finish(run.launch())

    assert code == 0, errors
    assert run.committed_prefix() == reference_prefix
    assert_exactly_once(run)
    replacement = run.events(epoch=2)
    outcomes = [e["outcome"] for e in replacement if e["event_type"] == "RECONCILIATION_FINISHED"]
    assert outcomes == ([reconciled] if reconciled else [])
    prefill = next(e for e in replacement if e["event_type"] == "PREFILL_STARTED")
    assert prefill["committed_prefix_tokens"] > 0
    kinds = [e["event_type"] for e in replacement]
    assert kinds.index("PREFILL_STARTED") < kinds.index("FIRST_NEW_TOKEN")


def test_inference_loss_mid_generation_replays_the_durable_prefix_without_replacing_the_runner(stack, reference_prefix):
    run = stack()
    runner = run.launch(pause_at="DURING_GENERATION")
    run.wait_for_barrier(runner, "DURING_GENERATION")
    run.kill_inference()
    runner.stdin.write("\n")  # resume reading the now-dead stream
    runner.stdin.flush()
    time.sleep(0.3)  # the runner must wait for the replacement, not fail
    run.start_inference()
    code, errors = run.finish(runner)

    assert code == 0, errors
    assert run.epoch == 1  # inference loss does not fence the runner
    assert run.committed_prefix() == reference_prefix
    assert_exactly_once(run)
    kinds = [e["event_type"] for e in run.events()]
    lost = kinds.index("INFERENCE_LOST")
    assert kinds[lost:lost + 4] == ["INFERENCE_LOST", "INFERENCE_REPLACEMENT_OBSERVED", "PREFILL_STARTED", "FIRST_NEW_TOKEN"]


def test_crash_during_recovery_is_recovered_by_the_next_attempt(stack, reference_prefix):
    run = stack()
    run.crash_runner(run.launch(pause_at="POST_TOOL_COMMIT_PRE_RESULT_LOG"), "POST_TOOL_COMMIT_PRE_RESULT_LOG")
    run.crash_runner(run.launch(pause_at="DURING_RECOVERY"), "DURING_RECOVERY")
    code, errors = run.finish(run.launch())

    assert code == 0, errors
    assert run.epoch == 3
    assert run.committed_prefix() == reference_prefix
    assert_exactly_once(run)
    started = [e["epoch"] for e in run.events() if e["event_type"] == "RECONCILIATION_STARTED"]
    assert started == [2, 3]


def test_unreachable_tool_service_during_reconciliation_abstains_and_stays_halted(stack):
    run = stack()
    run.crash_runner(run.launch(pause_at="PRE_DISPATCH"), "PRE_DISPATCH")
    dead = "http://127.0.0.1:9"  # nothing listens: the query cannot establish an outcome
    code, _ = run.finish(run.launch(tool_endpoint=dead))

    assert code == 3
    records = DurableLog(run.store).read(run.task)
    assert records[-1]["record_type"] == "RECOVERY_ABSTAIN" and records[-1]["step_id"] == TARGET
    assert any(e["event_type"] == "RECOVERY_ABSTAIN" for e in run.events(epoch=2))

    run.epoch = run.store.advance_epoch(run.task, run.epoch)
    code, _ = run.finish(run.launch())
    assert code == 3
    assert DurableLog(run.store).read(run.task) == records  # halted: nothing retried or appended


def test_restart_baseline_asks_the_supervisor_to_reset_after_inference_loss(stack):
    run = stack(method="restart")
    runner = run.launch(pause_at="DURING_GENERATION")
    run.wait_for_barrier(runner, "DURING_GENERATION")
    run.kill_inference()
    runner.stdin.write("\n")
    runner.stdin.flush()
    code, _ = run.finish(runner)
    assert code == 4


def test_stream_cut_without_finish_reason_is_inference_loss_not_a_model_error(tmp_path):
    server = subprocess.Popen([sys.executable, str(FAKE_VLLM), "--chunk-delay", "0.05"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        port = _ready(server, 15)["port"]
        backend = CompletionBackend(f"http://127.0.0.1:{port}", "Qwen/Qwen3-4B", context_spec())
        prompt = backend.tokenize("\nExecute next request as JSON tool_name/args: " + json.dumps(
            {"tool_name": "read_inventory", "args": {"sku": "item-a"}}))
        with pytest.raises(InferenceUnavailable):
            backend.generate_stream(prompt, lambda tokens: (server.kill(), server.wait(timeout=10)))
    finally:
        if server.poll() is None:
            server.kill()
        server.wait(timeout=10)
        server.stdout.close()

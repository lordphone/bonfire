"""RTX 5090 supervisor for the main GPU matrix: one run = one task, end to end.

Runs inside the locked vLLM container (docs/infrastructure/gpu-matrix.md), so the
inference worker is an owned child process and its kill is a real SIGKILL of the
process tree. Per run the supervisor:

1. starts the C2 tool service and a fresh vLLM worker (APC off, single GPU);
2. sizes the instruction padding so the initial prefix is the target length;
3. launches ``outcomeserve.runner`` with a DURING_GENERATION barrier at the step
   after ``progress`` of the 20 steps (none for a fault-free run);
4. at the barrier: FAULT_APPLIED, SIGKILL of the worker, release the runner, and
   start a replacement worker. OutcomeServe's runner survives and replays its
   durable prefix; the restart baseline exits 4, and the supervisor advances the
   epoch, resets the isolated task and starts a new runner from step 1;
5. verifies the final business state and writes result.json (run-result 1.1.0).

Timestamps are perf_counter on one host; the runner gets this run's clock_id, so
FAULT_APPLIED → VERIFIED_COMPLETE (primary) and FAULT_APPLIED → FIRST_NEW_TOKEN
(secondary) are differences on one clock.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.common import ROOT, _ready, save
from outcomeserve.contracts import validate_log, validate_schema
from outcomeserve.log import DurableLog
from outcomeserve.measurement import Measurements
from outcomeserve.processes import kill_owned
from outcomeserve.runner.recovering import build_instruction
from outcomeserve.serving.client import CompletionBackend
from outcomeserve.serving.process import InferenceProcess
from outcomeserve.storage import StateStore
from outcomeserve.telemetry.resources import ResourceMonitor
from outcomeserve.verify.business import verify_business
from workloads import make_inventory_workload
from workloads.reset import reset_isolated_task

STEPS = 20
TEMPLATE = ROOT / "examples" / "gpu-pilot-prompt.txt"
LOCK = json.loads((ROOT / "infra" / "environment.lock.json").read_text())
REVISION = LOCK["model"]["revision"]
SNAPSHOT = Path("/models/snapshots") / REVISION  # mounted read-only by the container launcher
ENDPOINT = "http://127.0.0.1:8000"
RUNNER_EXIT = {0: "completed", 1: "protocol_failure", 3: "abstained", 4: "restart_required", 5: "model_error"}


def context_spec() -> dict:
    model = LOCK["model"]["id"]
    return {"model_id": model, "model_revision": REVISION, "tokenizer_id": model, "tokenizer_revision": REVISION,
            "chat_template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
            "enable_thinking": False, "backend": "vllm"}


def vllm_command() -> list[str]:
    serving = LOCK["serving"]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(SNAPSHOT),
            "--served-model-name", LOCK["model"]["id"], "--tokenizer", str(SNAPSHOT),
            "--max-model-len", str(serving["max_model_len"]), "--dtype", "bfloat16",
            "--no-enable-prefix-caching", "--tensor-parallel-size", "1", "--seed", str(serving["seed"]),
            "--gpu-memory-utilization", "0.85", "--port", "8000"]


def cell_id(prefix: int, progress: float | None) -> str:
    return f"p{prefix // 1024}k-" + ("nofault" if progress is None else f"f{round(progress * 100):02d}")


def configuration_id(prefix: int, progress: float | None) -> str:
    """Same for both methods and every seed of a cell, so runs pair on it."""
    cell = {"lock": LOCK, "context_spec": context_spec(), "steps": STEPS, "initial_prefix": prefix,
            "progress": progress, "apc": False, "fault_domain": "inference", "boundary": "DURING_GENERATION"}
    return "gpu-" + cell_id(prefix, progress) + "-" + hashlib.sha256(
        json.dumps(cell, sort_keys=True).encode()).hexdigest()[:12]


def size_padding(model: CompletionBackend, workload: dict, target: int) -> tuple[int, int]:
    """Padding repeats whose instruction tokenizes to the target length (±2 tokens)."""
    template = TEMPLATE.read_text(encoding="utf-8")
    repeats = max(0, target - len(model.tokenize(build_instruction(template, workload, 0))))
    for _ in range(8):
        tokens = len(model.tokenize(build_instruction(template, workload, repeats)))
        if abs(tokens - target) <= 2 or (repeats == 0 and tokens > target):
            return repeats, tokens
        repeats = max(0, repeats + target - tokens)
    raise RuntimeError(f"could not size the instruction to {target} tokens")


class Run:
    def __init__(self, out: Path, *, method: str, prefix: int, progress: float | None, seed: int):
        self.out, self.method, self.prefix, self.progress, self.seed = out, method, prefix, progress, seed
        self.cell = cell_id(prefix, progress)
        self.config_id = configuration_id(prefix, progress)
        self.run_id = f"{self.cell}-s{seed:03d}-{method}"
        # Same task for every method and fault setting at this prefix and seed, so final prefixes compare.
        self.workload = make_inventory_workload(f"gpu:p{prefix // 1024}k:s{seed}", seed=seed, steps=STEPS)
        self.task = self.workload["task_id"]
        self.fault_step = None if progress is None else self.workload["steps"][int(STEPS * progress)]["step_id"]
        self.processes: list[subprocess.Popen] = []

    def launch_runner(self, epoch: int, padding: int, *, pause: bool) -> subprocess.Popen:
        command = [sys.executable, "-m", "outcomeserve.runner", "--db", str(self.db),
                   "--workload", str(self.out / "workload.json"), "--tool-endpoint", self.tool_endpoint,
                   "--inference-endpoint", ENDPOINT, "--context-spec", str(self.out / "context.json"),
                   "--template", str(TEMPLATE), "--events", str(self.out / "runner-events.jsonl"),
                   "--run-id", self.run_id, "--configuration-id", self.config_id,
                   "--clock-id", self.measures.clock_id, "--epoch", str(epoch), "--method", self.method,
                   "--seed", str(self.seed), "--padding-repeats", str(padding), "--inference-wait", "600"]
        if pause:
            command += ["--pause-at", "DURING_GENERATION", "--pause-step", self.fault_step,
                        "--ready-file", str(self.out / "runner-boundary.json")]
        runner = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                  stderr=(self.out / f"runner-{epoch}.stderr.txt").open("w"), text=True)
        self.processes.append(runner)
        self.monitor.watch(runner.pid, f"runner-{epoch}")
        self.measures.emit("RUNNER_LAUNCHED", epoch=epoch, pid=runner.pid)
        return runner

    def wait_for_barrier(self, runner: subprocess.Popen) -> dict:
        path = self.out / "runner-boundary.json"
        deadline = time.monotonic() + 1800
        while True:
            try:
                return json.loads(path.read_text())
            except (FileNotFoundError, ValueError):
                pass
            if runner.poll() is not None:
                raise RuntimeError(f"runner exited {runner.returncode} before the fault boundary")
            if time.monotonic() > deadline:
                raise TimeoutError("fault boundary not reached")
            time.sleep(0.002)

    def execute(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=False)
        spec = context_spec()
        save(self.out / "workload.json", self.workload)
        save(self.out / "context.json", spec)
        self.db = self.out / "runtime.sqlite"
        store = StateStore(self.db)
        store.create_task(self.task, self.workload["initial_inventory"])
        self.measures = Measurements(self.out / "measurements.jsonl", run_id=self.run_id,
                                     configuration_id=self.config_id)
        report = {"schema_version": "1.1.0", "backend": "vllm", "method": self.method, "run_id": self.run_id,
                  "configuration_id": self.config_id, "cell": self.cell, "seed": self.seed, "valid": False,
                  "passed": False, "invalid_reasons": [], "fault_applied": False, "no_fault": self.progress is None,
                  "purpose": "performance", "apc": False, "steps": STEPS, "fault_domain": "inference",
                  "boundary": "DURING_GENERATION", "target_initial_prefix": self.prefix,
                  "requested_progress": self.progress, "fault_step": self.fault_step,
                  "pair_key": f"{self.config_id}:s{self.seed}", "gpu_uuid": os.environ.get("BONFIRE_GPU_UUID"),
                  "environment_lock_sha256": hashlib.sha256((ROOT / "infra/environment.lock.json").read_bytes()).hexdigest()}
        self.monitor = ResourceMonitor(self.out, gpu_indices=(0,), interval=0.2, start_ns=self.measures.emit("RUN_STARTED"))
        inference = InferenceProcess(vllm_command(), ENDPOINT, spec["model_id"], timeout=600)
        tool_server, verified = None, None
        try:
            tool_server = subprocess.Popen([sys.executable, "-m", "outcomeserve.tools.server", "--db", str(self.db),
                                            "--barrier-file", str(self.out / "tool-barrier.json")],
                                           cwd=ROOT, stdout=subprocess.PIPE, stderr=(self.out / "tool.stderr.txt").open("w"),
                                           text=True)
            self.processes.append(tool_server)
            self.monitor.watch(tool_server.pid, "tool_service")
            self.tool_endpoint = f"http://127.0.0.1:{_ready(tool_server, 30)['port']}"
            log = (self.out / "inference.log").open("w")
            self.measures.emit("INFERENCE_START_REQUESTED", instance=1)
            inference.start(stdout=log, stderr=subprocess.STDOUT, on_launch=lambda pid: self.monitor.watch(pid, "inference-1"))
            self.measures.emit("INFERENCE_READY", instance=1)
            model = CompletionBackend(ENDPOINT, spec["model_id"], spec)
            padding, initial = size_padding(model, self.workload, self.prefix)
            report.update(padding_repeats=padding, initial_prefix_tokens=initial)

            self.measures.emit("TASK_STARTED")
            runner = self.launch_runner(1, padding, pause=self.fault_step is not None)
            if self.fault_step is not None:
                evidence = self.wait_for_barrier(runner)
                self.measures.emit("FAULT_APPLIED", domain="inference", step_id=self.fault_step,
                                   generated_tokens_before_fault=evidence.get("generated_tokens"))
                inference.stop()  # SIGKILL of the owned worker tree
                self.measures.emit("INFERENCE_KILLED", instance=1)
                report["fault_applied"] = True
                runner.stdin.write("\n")  # release: the runner reads the dead stream
                runner.stdin.flush()
                if self.method == "restart":
                    code = runner.wait(timeout=120)
                    if code != 4:
                        raise RuntimeError(f"restart runner exited {code}, expected 4")
                    epoch = store.advance_epoch(self.task, 1)
                    self.measures.emit("TOOL_RESET_STARTED")
                    reset_isolated_task(store, self.task, self.workload["initial_inventory"], expected_epoch=epoch)
                    self.measures.emit("TOOL_RESET_FINISHED")
                    runner = self.launch_runner(epoch, padding, pause=False)
                self.measures.emit("INFERENCE_START_REQUESTED", instance=2)
                inference.start(stdout=log, stderr=subprocess.STDOUT,
                                on_launch=lambda pid: self.monitor.watch(pid, "inference-2"))
                self.measures.emit("INFERENCE_READY", instance=2)
            code = runner.wait(timeout=3600)
            report["runner_exit"] = RUNNER_EXIT.get(code, str(code))
            if code != 0:
                raise RuntimeError(f"runner exited {code} ({report['runner_exit']})")
            self.measures.emit("TASK_COMPLETED")
            self.measures.emit("VERIFICATION_STARTED")
            verification = verify_business(self.db, self.workload)
            prefix = self.committed_prefix(store)
            report.update(final_prefix_tokens=len(prefix) if prefix else None,
                          final_prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest() if prefix else None)
            verified = self.measures.emit("VERIFIED_COMPLETE" if verification["task_success"] else "VERIFICATION_FAILED")
            save(self.out / "verification.json", verification)
            report.update(valid=True, passed=verification["task_success"], task_success=verification["task_success"],
                          duplicate_effects=verification["metrics"]["duplicate_effects"])
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, never dropped
            report.update(failure_reason=f"{type(exc).__name__}: {exc}")
            report["invalid_reasons"].append(report["failure_reason"])
        finally:
            try:
                report["resources"] = self.monitor.stop(stop_ns=verified or time.perf_counter_ns())
            except (RuntimeError, OSError) as exc:
                report["resources"] = {"measurement_valid": False, "error": str(exc)}
            inference.stop()
            for process in self.processes:
                kill_owned(process)
                process.wait(timeout=30)
                for stream in (process.stdin, process.stdout):
                    if stream:
                        stream.close()
        report.update(self.timings())
        if report["passed"] and not report["resources"].get("measurement_valid"):
            report.update(passed=False, failure_reason="GPU resource sampling incomplete")
            report["invalid_reasons"].append(report["failure_reason"])
        validate_schema("run-result", report)
        save(self.out / "result.json", report)
        return report

    def committed_prefix(self, store: StateStore) -> list[int] | None:
        if self.method != "outcomeserve":
            return None  # the restart baseline keeps no durable log
        log = DurableLog(store)
        validate_log(log.read(self.task))
        return log.committed_prefix(self.task)["payload"]["token_ids"]

    def timings(self) -> dict:
        events = [json.loads(line) for line in (self.out / "measurements.jsonl").open()]
        runner = self.out / "runner-events.jsonl"
        events += [json.loads(line) for line in runner.open()] if runner.exists() else []
        for event in events:
            validate_schema("measurement-event", event)
        clocks = {event["clock_id"] for event in events}
        if len(clocks) != 1:
            raise RuntimeError(f"events span clocks {clocks}")

        def at(kind, after=0):
            return next((e["monotonic_ns"] for e in sorted(events, key=lambda e: e["monotonic_ns"])
                         if e["event_type"] == kind and e["monotonic_ns"] >= after), None)

        def seconds(a, b):
            return None if a is None or b is None else (b - a) / 1e9

        fault, start, done = at("FAULT_APPLIED"), at("TASK_STARTED"), at("VERIFIED_COMPLETE")
        return {"task_to_verified_seconds": seconds(start, done),
                "fault_to_verified_seconds": seconds(fault, done),
                "fault_to_first_new_token_seconds": seconds(fault, at("FIRST_NEW_TOKEN", fault or 0)) if fault else None,
                "fault_to_replacement_ready_seconds": seconds(fault, at("INFERENCE_READY", fault or 0)) if fault else None}


def run_one(out: Path, **kwargs) -> dict:
    return Run(out, **kwargs).execute()

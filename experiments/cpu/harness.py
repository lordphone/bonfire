"""Real CPU runner/tool-process experiments; GPU results are never simulated."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import socket

from experiments.common import ROOT, save, _ready
from outcomeserve.measurement import Measurements
from outcomeserve.log import DurableLog
from outcomeserve.context import context_spec_id, scripted_context
from outcomeserve.contracts import validate_schema
from outcomeserve.processes import kill_owned
from outcomeserve.storage import StateStore
from outcomeserve.telemetry.resources import ResourceMonitor
from outcomeserve.tools.client import ToolClient
from outcomeserve.verify import verify_workload
from outcomeserve.verify.business import verify_business
from workloads import make_inventory_workload
from workloads.reset import reset_isolated_task


METHODS = ("outcomeserve", "restart", "naive_retry", "transcript_checkpoint")


def resolve_progress(workload, progress, *, boundary):
    if not 0 <= progress < 1:
        raise ValueError("progress must be in [0,1)")
    completed = int(len(workload["steps"]) * progress)
    index = max(0, completed - 1) if boundary == "POST_CONTEXT_COMMIT" else completed
    # Tool windows require a write. Never silently claim the requested fraction.
    while (boundary in {"PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "DURING_RECOVERY"}
           and index < len(workload["steps"]) and workload["steps"][index]["tool_name"] != "decrement_inventory"):
        index += 1
    if index == len(workload["steps"]):
        raise ValueError("no state-changing step at/after requested progress")
    committed_contexts = index + 1 if boundary == "POST_CONTEXT_COMMIT" else index
    return {"requested_progress": progress, "completed_steps_at_fault": committed_contexts,
            "completed_steps_before_active": index,
            "actual_progress": committed_contexts / len(workload["steps"]), "fault_step": index + 1}


def run(output, *, method="outcomeserve", seed=33, steps=5, progress=0.2,
        boundary="POST_TOOL_COMMIT_PRE_RESULT_LOG", fault_domain="runner", purpose="correctness",
        no_fault=False, interval=0.05):
    if method not in METHODS or purpose not in {"correctness", "performance"}:
        raise ValueError("unsupported method/purpose")
    if fault_domain not in {"runner", "tool_response"}:
        raise ValueError("CPU experiment cannot simulate inference/GPU failure; use a real serving pilot")
    if boundary not in {"DURING_GENERATION", "PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "POST_CONTEXT_COMMIT", "DURING_RECOVERY"} or (fault_domain == "tool_response" and boundary != "POST_TOOL_COMMIT_PRE_RESULT_LOG"):
        raise ValueError("invalid fault domain/boundary")
    if boundary == "DURING_RECOVERY" and (method != "outcomeserve" or purpose != "correctness"):
        raise ValueError("repeated recovery crash is an OutcomeServe correctness test, not a one-fault performance cell")
    if purpose == "performance" and method not in {"outcomeserve", "restart"}:
        raise ValueError("performance main comparison is restart versus OutcomeServe")
    output = Path(output).resolve()
    workload = make_inventory_workload(f"cpu-{method}-s{seed}-n{steps}", seed=seed, steps=steps)
    position = resolve_progress(workload, progress, boundary=boundary)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "workload.json", workload)
    configuration = {"backend": "scripted_cpu", "purpose": purpose, "method": method,
                     "steps": steps, "fault_domain": fault_domain, "boundary": boundary,
                     "no_fault": no_fault, "sampling_interval": interval,
                     "context_spec_id": context_spec_id(scripted_context(workload["model_revision"])),
                     "environment_lock_sha256": hashlib.sha256((ROOT / 'infra/environment.lock.json').read_bytes()).hexdigest(), **position}
    config_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    target = workload["steps"][position["fault_step"] - 1]
    preconditions = {"DURING_GENERATION": "GENERATION_IN_PROGRESS", "PRE_DISPATCH": "OPERATION_PREPARED_NOT_DISPATCHED",
                    "POST_TOOL_COMMIT_PRE_RESULT_LOG": "TOOL_OUTCOME_DURABLY_COMMITTED",
                    "POST_CONTEXT_COMMIT": "CONTEXT_DURABLY_COMMITTED", "DURING_RECOVERY": "RECOVERY_RECONCILIATION_IN_PROGRESS"}
    fault_request = {"schema_version": "1.1.0", "injection_id": config_id[:16] + f":s{seed}",
        "experiment_id": "outcomeserve-cpu", "task_id": workload["task_id"], "step_id": target["step_id"],
        "seed": seed, "target_worker_id": f"{workload['task_id']}-attempt-{'2' if boundary == 'DURING_RECOVERY' else '1'}",
        "configuration_id": config_id, "boundary": boundary, "fault_domain": fault_domain,
        "action": "DROP_ACKNOWLEDGMENT" if fault_domain == "tool_response" else "TERMINATE_PROCESS",
        "expected_precondition": preconditions[boundary], "one_shot": True,
        "requested_progress": progress, "completed_steps_at_fault": position["completed_steps_at_fault"],
        "metadata": {"operation_id": target["op_id"] if method == "outcomeserve" else target["op_id"] + ":attempt-1-1"}}
    validate_schema("fault-injection", fault_request)
    if not no_fault:
        save(output / "fault-request.json", fault_request)
        if boundary == "DURING_RECOVERY":
            preparation = {**fault_request, "injection_id": fault_request["injection_id"] + ":prepare",
                           "boundary": "PRE_DISPATCH", "expected_precondition": preconditions["PRE_DISPATCH"],
                           "target_worker_id": workload["task_id"] + "-attempt-1"}
            validate_schema("fault-injection", preparation)
            save(output / "preparatory-fault-request.json", preparation)
    measures = Measurements(output / "measurements.jsonl", run_id=output.name, configuration_id=config_id)
    database = output / "runtime.sqlite"
    store = StateStore(database)
    store.create_task(workload["task_id"], workload["initial_inventory"])
    start = measures.emit("TASK_STARTED")
    monitor = ResourceMonitor(output, interval=interval, start_ns=start)
    server = runner = None
    verified_stop = None
    report = {"schema_version": "1.1.0", "configuration_id": config_id, **configuration,
              "context_spec_id": context_spec_id(scripted_context(workload["model_revision"])),
              "hardware_id": socket.gethostname(),
              "business_fixture_sha256": hashlib.sha256(json.dumps({"inventory": workload["initial_inventory"],
                  "requests": [{key: step[key] for key in ("tool_name", "args")} for step in workload["steps"]]}, sort_keys=True).encode()).hexdigest(),
              "seed": seed, "pair_key": f"s{seed}:n{steps}:{progress}:{boundary}:{fault_domain}:{purpose}:{no_fault}",
              "valid": False, "passed": False, "invalid_reasons": [], "fault_applied": False}
    try:
        with (output / "tool.stderr.txt").open("w") as errors:
            server = subprocess.Popen([sys.executable, "-m", "outcomeserve.tools.server", "--db", str(database),
                "--barrier-file", str(output / "tool-boundary.json")], cwd=ROOT, stdout=subprocess.PIPE, stderr=errors, text=True)
            monitor.watch(server.pid, "tool_service")
            ready = _ready(server, 15)
        endpoint = f"http://127.0.0.1:{ready['port']}"
        client = ToolClient(endpoint)
        command = [sys.executable, "-m", "experiments.cpu.worker", "--db", str(database),
            "--workload", str(output / "workload.json"), "--tool-endpoint", endpoint,
            "--events", str(output / "worker-events.jsonl"), "--checkpoint", str(output / "checkpoint.json"),
            "--clock-id", measures.clock_id,
            "--method", method, "--fault-step", str(position["fault_step"]), "--boundary", boundary]
        def launch(epoch, inject):
            stream = (output / f"attempt-{epoch}.stderr.txt").open("w")
            try:
                child = subprocess.Popen(command + ["--epoch", str(epoch), "--ready-file", str(output / f"runner-boundary-{epoch}.json")] + (["--inject"] if inject else []),
                    cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stream, text=True)
            finally:
                stream.close()
            monitor.watch(child.pid, "runner")
            return child
        measures.emit("WORKER_START_REQUESTED", attempt=1)
        runner = launch(1, not no_fault)
        if not no_fault:
            barrier = output / ("tool-boundary.json" if boundary == "POST_TOOL_COMMIT_PRE_RESULT_LOG" else "runner-boundary-1.json")
            deadline = time.monotonic() + 30
            while not barrier.exists():
                if runner.poll() is not None or server.poll() is not None:
                    raise RuntimeError("process exited before requested fault boundary")
                if time.monotonic() > deadline:
                    raise TimeoutError("requested fault boundary was not reached")
                time.sleep(0.01)
            # File may be visible before the writer closes it; retry only reading,
            # never shift the fault to a different application boundary.
            while True:
                try:
                    evidence = json.loads(barrier.read_text())
                    break
                except ValueError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.005)
            save(output / "precondition.json", evidence)
            target = workload["steps"][position["fault_step"] - 1]
            if evidence.get("logical_step_id", evidence.get("step_id")) != target["step_id"]:
                raise RuntimeError("fault step precondition mismatch")
            connection = store.connect()
            try:
                committed_count = connection.execute("SELECT count(*) FROM operations WHERE task_id=?", (workload["task_id"],)).fetchone()[0]
                expected_count = position["completed_steps_before_active"] + int(boundary in {"POST_TOOL_COMMIT_PRE_RESULT_LOG", "POST_CONTEXT_COMMIT"})
                if committed_count != expected_count:
                    raise RuntimeError("durable operation count does not match requested boundary")
                if method == "outcomeserve":
                    target_records = [r for r in DurableLog(store).read(workload["task_id"])
                                      if r["step_id"] == target["step_id"]]
                    expected_types = ([] if boundary == "DURING_GENERATION" else
                        ["STEP_PREPARED", "TOOL_RESULT_COMMITTED", "CONTEXT_COMMITTED"] if boundary == "POST_CONTEXT_COMMIT" else ["STEP_PREPARED"])
                    if [r["record_type"] for r in target_records] != expected_types:
                        raise RuntimeError("result/context already committed at requested tool boundary")
            finally:
                connection.close()
            measures.emit("PRECONDITION_CONFIRMED", committed_operations=committed_count)
            measures.emit("PREPARATORY_FAULT_APPLIED" if boundary == "DURING_RECOVERY" else "FAULT_APPLIED", domain=fault_domain, boundary=boundary)
            report["fault_applied"] = boundary != "DURING_RECOVERY"
            if boundary == "DURING_RECOVERY":
                report["preparatory_fault_applied"] = True
            report["fault_count"] = 1
            if fault_domain == "runner":
                kill_owned(runner)
                runner.communicate(timeout=10)
                measures.emit("PROCESS_EXITED", attempt=1)
                if boundary == "POST_TOOL_COMMIT_PRE_RESULT_LOG":
                    client.post("/release", {"drop": True})
                epoch = store.advance_epoch(workload["task_id"], 1)
                if purpose == "performance" and method == "restart":
                    connection = store.connect()
                    backup = sqlite3.connect(output / "before-reset.sqlite")
                    connection.backup(backup)
                    backup.close()
                    connection.close()
                    measures.emit("TOOL_RESET_STARTED")
                    reset_isolated_task(store, workload["task_id"], workload["initial_inventory"], expected_epoch=epoch)
                    (output / "checkpoint.json").unlink(missing_ok=True)
                    measures.emit("TOOL_RESET_FINISHED")
                measures.emit("WORKER_START_REQUESTED", attempt=epoch)
                runner = launch(epoch, boundary == "DURING_RECOVERY")
                if boundary == "DURING_RECOVERY":
                    recovery_barrier = output / f"runner-boundary-{epoch}.json"
                    deadline = time.monotonic() + 30
                    while not recovery_barrier.exists():
                        if runner.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("replacement failed to reach recovery fault boundary")
                        time.sleep(0.01)
                    while True:
                        try:
                            recovery_evidence = json.loads(recovery_barrier.read_text())
                            break
                        except ValueError:
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(0.005)
                    if recovery_evidence != {"step_id": target["step_id"], "boundary": "DURING_RECOVERY"}:
                        raise RuntimeError("recovery precondition handshake mismatch")
                    measures.emit("RECOVERY_FAULT_APPLIED")
                    measures.emit("FAULT_APPLIED", domain="runner", boundary="DURING_RECOVERY")
                    kill_owned(runner)
                    runner.communicate(timeout=10)
                    epoch = store.advance_epoch(workload["task_id"], epoch)
                    report["fault_count"] = 2
                    report["fault_applied"] = True
                    runner = launch(epoch, False)
            else:
                # Runner remains alive. Only response delivery fails; no inference
                # process exists or is restarted in this CPU configuration.
                client.post("/release", {"drop": True})
        stdout, _ = runner.communicate(timeout=40)
        (output / "runner.stdout.txt").write_text(stdout)
        measures.emit("TASK_COMPLETED", exit_code=runner.returncode)
        report.update(valid=True, runner_exit_code=runner.returncode, tool_service_survived=server.poll() is None,
                      runner_replaced=report["fault_applied"] and fault_domain == "runner")
        measures.emit("VERIFICATION_STARTED")
        business = verify_business(database, workload)
        protocol = verify_workload(database, workload) if method == "outcomeserve" else None
        passed = runner.returncode == 0 and business["task_success"] and (protocol is None or protocol["task_success"])
        verified_stop = measures.emit("VERIFIED_COMPLETE" if passed else "VERIFICATION_FAILED")
        save(output / "verification.json", {"business": business, "protocol": protocol})
        report.update(passed=passed, metrics=business["metrics"], task_success=passed,
                      task_to_verified_seconds=measures.duration("TASK_STARTED", "VERIFIED_COMPLETE"),
                      fault_to_verified_seconds=measures.duration("FAULT_APPLIED", "VERIFIED_COMPLETE"),
                      reset_seconds=measures.duration("TOOL_RESET_STARTED", "TOOL_RESET_FINISHED"),
                      crash_to_next_token_seconds=None)
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        report["passed"] = False
        if report["fault_applied"]:
            report.update(valid=True, failure_reason=str(exc))
        else:
            report["invalid_reasons"].append(str(exc))
    finally:
        # Stop measurement at verification/failure, not after teardown or queue wait.
        stop = verified_stop if verified_stop is not None else time.perf_counter_ns()
        try:
            try:
                report["resources"] = monitor.stop(stop_ns=stop)
            except (RuntimeError, OSError) as exc:
                report["resources"] = {"measurement_valid": False, "error": str(exc)}
        finally:
            for process in (runner, server):
                if process is not None:
                    if process.poll() is None:
                        kill_owned(process)
                        process.wait(timeout=10)
                    for stream in (process.stdin, process.stdout):
                        if stream is not None:
                            stream.close()
    if purpose == "performance" and not report.get("resources", {}).get("measurement_valid"):
        report.update(passed=False, failure_reason="resource sampling incomplete")
    validate_schema("run-result", report)
    save(output / "result.json", report)
    return report

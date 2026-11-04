"""Real single-GPU inference-loss pilot; requires locked assets and NVML.

No sleep-based generation, fabricated token IDs, or CPU fallback is provided.
Serving return_token_ids support is a pilot gate, not an assumed vLLM feature.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sqlite3
import shutil
import time
from importlib.metadata import version

from experiments.common import _ready
from experiments.common import ROOT, save
from outcomeserve.context import context_spec_id
from outcomeserve.contracts import request_fingerprint, validate_log, validate_schema
from outcomeserve.log import DurableLog
from outcomeserve.measurement import Measurements
from outcomeserve.processes import kill_owned
from outcomeserve.serving.client import CompletionBackend, InferenceError
from outcomeserve.serving.process import InferenceProcess
from outcomeserve.storage import StateStore
from outcomeserve.telemetry.resources import ResourceMonitor
from outcomeserve.tools.client import ToolClient
from outcomeserve.verify.business import verify_business
from workloads import make_inventory_workload
from workloads.reset import reset_isolated_task


def run_gpu_pilot(config, output):
    if config.get("exclusive_gpu") is not True:
        raise ValueError("pilot requires an explicitly exclusive allocated GPU")
    if config.get("method") not in {"restart", "outcomeserve"}:
        raise ValueError("pilot compares restart and OutcomeServe")
    if type(config.get("progress")) not in (int, float) or not 0 <= config["progress"] < 1:
        raise ValueError("progress must be in [0,1)")
    spec = config["context_spec"]
    identity = context_spec_id(spec)
    template = Path(config["template_path"]).read_text(encoding="utf-8")
    if hashlib.sha256(template.encode()).hexdigest() != spec["chat_template_sha256"]:
        raise ValueError("actual prompt template differs from pinned digest")
    snapshot = Path(config["model_snapshot_path"]).resolve()
    if not snapshot.is_dir() or snapshot.name != spec["model_revision"] or str(snapshot) not in config["command"]:
        raise ValueError("command must reference the local locked model snapshot")
    if spec["model_revision"] != spec["tokenizer_revision"] or spec["model_id"] != spec["tokenizer_id"]:
        raise ValueError("this pilot requires tokenizer from the same locked snapshot")
    if config.get("apc") not in (True, False) or type(config.get("apc")) is not bool:
        raise ValueError("explicit APC setting required")
    flag = "--enable-prefix-caching" if config["apc"] else "--no-enable-prefix-caching"
    if flag not in config["command"]:
        raise ValueError("inference command must explicitly match the APC setting")
    executable = shutil.which(config["command"][0]) or config["command"][0]
    if Path(executable).resolve() != Path(sys.executable).resolve():
        raise ValueError("run pilot inside the locked container and launch inference with the same Python environment")
    lock_bytes = (ROOT / "infra/environment.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    if spec["model_id"] != lock["model"]["id"] or spec["model_revision"] != lock["model"]["revision"]:
        raise ValueError("GPU context differs from the locked project model")
    runtime_versions = {name: version(name) for name in ("vllm", "torch")}
    if runtime_versions["vllm"].split("+")[0] != lock["vllm"]["version"] or runtime_versions["torch"].split("+")[0] != lock["runtime"]["pytorch"]:
        raise ValueError("pilot must run inside the locked vLLM/PyTorch environment")
    # Refuse to run an alleged GPU experiment without actual NVML hardware.
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(config["gpu_index"])
        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        gpu_uuid = gpu_uuid.decode() if isinstance(gpu_uuid, bytes) else gpu_uuid
        if config.get("gpu_uuid") != gpu_uuid:
            raise ValueError("allocated GPU UUID must explicitly match the sampled NVML device")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible and ("," in visible or not (visible == str(config["gpu_index"]) or gpu_uuid.startswith(visible))):
            raise ValueError("single CUDA_VISIBLE_DEVICES allocation does not match the sampled GPU")
        if not visible and config["gpu_index"] != 0:
            raise ValueError("set CUDA_VISIBLE_DEVICES to the allocated GPU before the pilot")
    finally:
        pynvml.nvmlShutdown()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "configuration.json", config)
    workload = make_inventory_workload("gpu-pilot", seed=config.get("seed", 33), steps=20)
    save(output / "business-workload.json", workload)
    database = output / "runtime.sqlite"
    store = StateStore(database)
    store.create_task(workload["task_id"], workload["initial_inventory"])
    store.bind_context(workload["task_id"], spec)
    log = DurableLog(store)
    model = CompletionBackend(config["endpoint"], spec["model_id"], spec)
    inference = InferenceProcess(config["command"], config["endpoint"], spec["model_id"], timeout=config.get("startup_timeout", 300))
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    measures = Measurements(output / "measurements.jsonl", run_id=output.name, configuration_id=config_id)
    start = measures.emit("TASK_STARTED")
    monitor = ResourceMonitor(output, gpu_indices=(config["gpu_index"],), interval=0.2, start_ns=start)
    monitor.watch(os.getpid(), "runner", existing=True)
    server, verified_stop = None, None
    report = {"schema_version": "1.1.0", "backend": "vllm", "method": config["method"],
              "seed": config.get("seed", 33), "valid": False, "passed": False,
              "invalid_reasons": [], "fault_applied": False, "context_spec_id": identity}
    report.update(configuration_id=config_id, purpose="performance", no_fault=bool(config.get("no_fault")),
                  runtime_versions=runtime_versions, environment_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(), apc=config["apc"],
                  cuda_visible_devices=visible,
                  steps=20, fault_domain="inference", boundary="DURING_GENERATION",
                  requested_progress=config["progress"], actual_progress=int(20 * config["progress"]) / 20,
                  pair_key=f"s{report['seed']}:gpu:{identity}:{config['progress']}:{config['apc']}:{config.get('padding_repeats',0)}:{bool(config.get('no_fault'))}")
    try:
        with (output / "tool.stderr.txt").open("w") as errors:
            server = subprocess.Popen([sys.executable, "-m", "outcomeserve.tools.server",
                "--db", str(database), "--barrier-file", str(output / "unused-barrier.json")],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=errors, text=True)
            monitor.watch(server.pid, "tool_service")
            ready = _ready(server, 15)
        tools = ToolClient(f"http://127.0.0.1:{ready['port']}")
        with (output / "inference.stdout.txt").open("w") as stdout, (output / "inference.stderr.txt").open("w") as stderr:
            measures.emit("INFERENCE_START_REQUESTED")
            inference.start(stdout=stdout, stderr=stderr, on_launch=lambda pid: monitor.watch(pid, "inference"))
            measures.emit("INFERENCE_READY")
            instruction = template.format(task=json.dumps(workload["steps"]), padding="context " * config.get("padding_repeats", 0))
            initial_prefix = model.tokenize(instruction)
            prefix = list(initial_prefix)
            actual_prefixes = []
            fault_after = int(20 * config["progress"])
            if not 0 <= fault_after < 20:
                raise ValueError("progress must be in [0,1)")
            index, fired, first_new_observed = 0, False, False
            while index < 20:
                step = workload["steps"][index]
                # Tokenize new instructions only; stored/generated history IDs
                # are never decoded/re-tokenized to rebuild the prefix.
                prompt = prefix + model.tokenize("\nExecute next request as JSON tool_name/args: " + json.dumps({"tool_name": step["tool_name"], "args": step["args"]}))
                if not config.get("no_fault") and not fired and index == fault_after:
                    def crash(tokens):
                        nonlocal fired
                        fired = True
                        measures.emit("FAULT_APPLIED", domain="inference", generated_tokens_before_fault=len(tokens), actual_prefix_tokens=len(prompt))
                        report["fault_applied"] = True
                        inference.stop()
                    model.generate_until_fault(prompt, crash, seed=report["seed"])
                    if not fired:
                        raise InferenceError("inference-loss precondition absent")
                    measures.emit("INFERENCE_RESTART_REQUESTED")
                    inference.start(stdout=stdout, stderr=stderr, on_launch=lambda pid: monitor.watch(pid, "inference"))
                    measures.emit("INFERENCE_REPLACEMENT_READY", instance=inference.instance, reusable_kv=False)
                    if config["method"] == "restart":
                        source, target = store.connect(), sqlite3.connect(output / "before-reset.sqlite")
                        source.backup(target)
                        source.close()
                        target.close()
                        measures.emit("TOOL_RESET_STARTED")
                        reset_isolated_task(store, workload["task_id"], workload["initial_inventory"], expected_epoch=1)
                        measures.emit("TOOL_RESET_FINISHED")
                        prefix, index = list(initial_prefix), 0
                    continue
                if fired and not first_new_observed:
                    def first_token(tokens):
                        nonlocal first_new_observed
                        first_new_observed = True
                        measures.emit("FIRST_NEW_TOKEN", observed_at="client_stream", generation_step=index)
                    answer = model.generate_stream(prompt, first_token, seed=report["seed"])
                else:
                    answer = model.generate(prompt, seed=report["seed"])
                actual_call = answer["call"]
                expected_call = {"tool_name": step["tool_name"], "args": step["args"]}
                if json.dumps(actual_call, sort_keys=True, allow_nan=False) != json.dumps(expected_call, sort_keys=True, allow_nan=False):
                    report["model_error"] = "generated call does not satisfy independent business request"
                    raise InferenceError(report["model_error"])
                generated_prefix = prompt + answer["generated_token_ids"]
                fp = request_fingerprint(actual_call["tool_name"], actual_call["args"])
                op = {"task_id": workload["task_id"], "step_id": step["step_id"], "op_id": step["op_id"], "owner_epoch": 1}
                body = {"schema_version": "1.1.0", **op, "context_spec_id": identity,
                        "record_id": f"{step['step_id']}:prepare", "model_revision": spec["model_revision"],
                        "record_type": "STEP_PREPARED", "payload": {"tool_name": actual_call["tool_name"], "args": actual_call["args"],
                            "request_fingerprint": fp, "generated_token_ids": answer["generated_token_ids"],
                            "generation_prompt_token_ids": prompt}}
                records = log.read(workload["task_id"])
                if config["method"] == "outcomeserve":
                    prepared = log.append(op["task_id"], body["record_id"], len(records), 1, body)
                    if prepared["status"] != "APPENDED":
                        raise RuntimeError(str(prepared))
                result = tools.submit({**op, "tool_name": actual_call["tool_name"], "args": actual_call["args"], "request_fingerprint": fp})
                prefix = generated_prefix + model.tokenize("\nTool result: " + json.dumps(result["result"]))
                if config["method"] == "outcomeserve":
                    committed = log.commit_step(prepared["record"], result, prefix, owner_epoch=1,
                                                expected_sequence=len(records) + 1, commit_id=f"{step['step_id']}:commit")
                    if committed["status"] != "COMMITTED":
                        raise RuntimeError(str(committed))
                actual_prefixes.append({"step_id": step["step_id"], "prompt_tokens": len(prompt),
                                        "generated_tokens": len(answer["generated_token_ids"]), "committed_prefix_tokens": len(prefix),
                                        "inference_instance": inference.instance})
                index += 1
            measures.emit("TASK_COMPLETED")
            measures.emit("VERIFICATION_STARTED")
            verification = verify_business(database, workload)
            if config["method"] == "outcomeserve":
                validate_log(log.read(workload["task_id"]))
                if log.committed_prefix(workload["task_id"])["payload"]["token_ids"] != prefix:
                    raise RuntimeError("stored replay prefix differs from actual model history")
            verified_stop = measures.emit("VERIFIED_COMPLETE" if verification["task_success"] else "VERIFICATION_FAILED")
            save(output / "verification.json", verification)
            save(output / "actual-prefix-lengths.json", actual_prefixes)
            report.update(valid=True, passed=verification["task_success"], task_success=verification["task_success"],
                          metrics=verification["metrics"],
                          fault_to_verified_seconds=measures.duration("FAULT_APPLIED", "VERIFIED_COMPLETE"),
                          task_to_verified_seconds=measures.duration("TASK_STARTED", "VERIFIED_COMPLETE"),
                          prefill_seconds=None, fault_to_first_new_token_seconds=measures.duration("FAULT_APPLIED", "FIRST_NEW_TOKEN"),
                          note="prefill/decode split requires serving-side timing; not inferred from generation wall time")
    except Exception as exc:
        report.update(valid=bool(report["fault_applied"]), failure_reason=str(exc))
        if isinstance(exc, InferenceError):
            report["model_or_backend_error"] = str(exc)
        if not report["fault_applied"]:
            report["invalid_reasons"].append(str(exc))
    finally:
        try:
            try:
                report["resources"] = monitor.stop(stop_ns=verified_stop or time.perf_counter_ns())
            except (RuntimeError, OSError) as exc:
                report["resources"] = {"measurement_valid": False, "error": str(exc)}
        finally:
            inference.stop()
            if server is not None:
                kill_owned(server)
                if server.stdout:
                    server.stdout.close()
    report["hardware_id"] = ",".join(device["uuid"] for device in report["resources"].get("gpu_identity", []))
    if not report["resources"].get("measurement_valid"):
        report.update(passed=False, failure_reason="GPU resource sampling incomplete")
    validate_schema("run-result", report)
    save(output / "result.json", report)
    return report

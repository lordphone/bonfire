"""Scripted CPU execution with a separately surviving tool-service process."""

import argparse
import json
from pathlib import Path
import sys
import time

from outcomeserve.context import scripted_context
from outcomeserve.contracts import request_fingerprint
from outcomeserve.log import DurableLog
from outcomeserve.storage import StateStore
from outcomeserve.tools.client import ToolClient, AcknowledgmentLost
from workloads import load_workload


def run(args):
    workload = load_workload(args.workload)
    store, task = StateStore(args.db), workload["task_id"]
    context_id = store.bind_context(task, scripted_context(workload["model_revision"]))
    log, tools = DurableLog(store), ToolClient(args.tool_endpoint)
    events = Path(args.events)
    checkpoint = Path(args.checkpoint)
    def emit(kind, step, **details):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event_type": kind, "monotonic_ns": time.perf_counter_ns(),
                "clock_id": args.clock_id, "epoch": args.epoch, "step_id": step["step_id"], **details}) + "\n")
    def pause(step, boundary):
        evidence = {"step_id": step["step_id"], "boundary": boundary}
        if boundary == "DURING_GENERATION":
            emit("SCRIPTED_TOKEN_PRODUCED", step, token_id=step["token_ids"][-1])
            evidence["scripted_tokens_produced"] = 1
        Path(args.ready_file).write_text(json.dumps(evidence))
        sys.stdin.readline()
        raise RuntimeError("runner should have been terminated at scheduled boundary")
    start = 0
    if args.method != "outcomeserve" and args.epoch > 1:
        if args.method == "naive_retry":
            start = args.fault_step - 1
        elif args.method == "transcript_checkpoint" and checkpoint.exists():
            start = json.loads(checkpoint.read_text())["next_index"]
    records = log.read(task) if args.method == "outcomeserve" else []
    for index, step in enumerate(workload["steps"]):
        if index < start:
            continue
        step_records = [r for r in records if r["step_id"] == step["step_id"]]
        if any(r["record_type"] == "CONTEXT_COMMITTED" for r in step_records):
            continue
        target = args.inject and index == args.fault_step - 1
        if target and args.boundary == "DURING_GENERATION":
            # CPU boundary only: this is a scripted generation phase, not GPU work.
            pause(step, "DURING_GENERATION")
        fingerprint = request_fingerprint(step["tool_name"], step["args"])
        identity = {"task_id": task, "step_id": step["step_id"], "op_id": step["op_id"], "owner_epoch": args.epoch}
        request = {**identity, "tool_name": step["tool_name"], "args": step["args"], "request_fingerprint": fingerprint}
        prepared = next((r for r in step_records if r["record_type"] == "STEP_PREPARED"), None)
        if args.method == "outcomeserve" and prepared is None:
            body = {"schema_version": "1.1.0", "context_spec_id": context_id, **identity,
                    "record_id": f"{step['step_id']}:prepare", "model_revision": workload["model_revision"],
                    "record_type": "STEP_PREPARED", "payload": {"tool_name": step["tool_name"], "args": step["args"],
                        "request_fingerprint": fingerprint, "generated_token_ids": [],
                        "generation_prompt_token_ids": workload["steps"][index - 1]["token_ids"] if index else [1000]}}
            answer = log.append(task, body["record_id"], len(records), args.epoch, body)
            if answer["status"] != "APPENDED":
                raise RuntimeError(str(answer))
            prepared = answer["record"]
            records.append(prepared)
        if target and (args.boundary == "PRE_DISPATCH" or args.boundary == "DURING_RECOVERY" and args.epoch == 1):
            pause(step, "PRE_DISPATCH")
        query = {key: value for key, value in request.items() if key not in {"tool_name", "args"}}
        response = None
        if args.method == "outcomeserve" and step_records:
            emit("RECONCILIATION_STARTED", step)
            response = tools.query(query)
            emit("RECONCILIATION_FINISHED", step, outcome=response["status"])
            if target and args.boundary == "DURING_RECOVERY":
                pause(step, "DURING_RECOVERY")
        call = 0
        if response is None or response["status"] == "NOT_COMMITTED":
            while True:
                call += 1
                try:
                    emit("TOOL_SUBMIT", step)
                    response = tools.submit(request, pause=target and call == 1 and args.boundary == "POST_TOOL_COMMIT_PRE_RESULT_LOG",
                        logical_step_id=step["step_id"] if args.method != "outcomeserve" else None,
                        attempt=f"{args.epoch}-{call}")
                    break
                except AcknowledgmentLost:
                    emit("ACKNOWLEDGMENT_LOST", step)
                    target = False
                    if args.method == "outcomeserve":
                        response = tools.query(query)
                        emit("RECONCILIATION_FINISHED", step, outcome=response["status"])
                        break
                    if call >= 2:
                        raise
        if response["status"] != "COMMITTED":
            if args.method == "outcomeserve" and response["status"] == "UNKNOWN":
                body = {key: value for key, value in prepared.items() if key != "sequence"}
                body.update(owner_epoch=args.epoch, record_id=f"{step['step_id']}:halt:{args.epoch}",
                            record_type="RECOVERY_ABSTAIN", payload={"outcome": "UNKNOWN", "reason": response.get("reason", "unknown")})
                log.append(task, body["record_id"], len(records), args.epoch, body)
                return 3
            raise RuntimeError(f"tool returned {response['status']}")
        if args.method == "outcomeserve":
            committed = log.commit_step(prepared, response, step["token_ids"], owner_epoch=args.epoch,
                                        expected_sequence=len(records), commit_id=f"{step['step_id']}:commit")
            if committed["status"] != "COMMITTED":
                raise RuntimeError(str(committed))
            records.extend(committed["records"])
        else:
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps({"next_index": index + 1}))
            temporary.replace(checkpoint)
        emit("STEP_COMPLETED", step)
        if target and args.boundary == "POST_CONTEXT_COMMIT":
            pause(step, "POST_CONTEXT_COMMIT")
    return 0


def main():
    parser = argparse.ArgumentParser()
    for name in ("db", "workload", "tool-endpoint", "events", "checkpoint", "ready-file", "clock-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--method", choices=["outcomeserve", "restart", "naive_retry", "transcript_checkpoint"], required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--fault-step", type=int, required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--inject", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

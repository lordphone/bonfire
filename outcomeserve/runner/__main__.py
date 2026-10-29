"""Run one scripted task: python -m outcomeserve.runner --db state.sqlite --task t1 [--crash-at ...]

Exit code 0 means COMPLETED, 3 means ABSTAINED. A TERMINATE_PROCESS fault
kills this process with SIGKILL, so the caller sees -9.
"""

from __future__ import annotations

import argparse
import json
import sys

from outcomeserve.recovery.stubs import SqliteLog, SqliteToolService
from outcomeserve.runner.agent import Action, Boundary, FaultPlan, Runner, scripted_trace
from outcomeserve.telemetry import JsonlTelemetry, WorkerIdentity, cpu_run_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="outcomeserve.runner")
    parser.add_argument("--db", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trace", default="orders", choices=["orders", "legacy"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--crash-at", choices=[b.value for b in Boundary])
    parser.add_argument("--crash-step", type=int)
    parser.add_argument("--action", default=Action.TERMINATE_PROCESS.value, choices=[a.value for a in Action])
    parser.add_argument("--telemetry", help="append C4 events to this JSONL file")
    parser.add_argument("--run-id")
    parser.add_argument("--lineage")
    parser.add_argument("--attempt", type=int, help="defaults to the task's owner_epoch")
    parser.add_argument("--recovery-step", type=int, help="step the previous attempt crashed in (set by the supervisor)")
    parser.add_argument(
        "--fence", action="store_true", help="advance owner_epoch before running, as the supervisor would"
    )
    args = parser.parse_args(argv)

    fault = None
    if args.crash_at is not None:
        if args.crash_step is None:
            parser.error("--crash-at requires --crash-step")
        fault = FaultPlan(Boundary(args.crash_at), args.crash_step, Action(args.action))

    log = SqliteLog(args.db)
    epoch = log.advance_epoch(args.task) if args.fence else log.current_epoch(args.task)
    if args.telemetry and epoch < 1:
        parser.error("telemetry needs owner_epoch >= 1: pass --fence or start under a supervisor")
    lineage = args.lineage or f"runner-{args.task}"
    attempt = args.attempt or max(epoch, 1)
    worker = WorkerIdentity(lineage, f"{lineage}-attempt-{attempt}", attempt, epoch)
    telemetry = None
    if args.telemetry:
        run_id = args.run_id or f"run-{args.task}"
        telemetry = JsonlTelemetry(args.telemetry, cpu_run_context(args.task, run_id, f"{args.trace}:{args.steps}"))

    outcome = Runner(
        log,
        SqliteToolService(args.db),
        args.task,
        scripted_trace(args.trace, args.steps),
        fault,
        telemetry=telemetry,
        worker=worker,
        recovery_step=args.recovery_step,
    ).run()
    print(json.dumps({"status": outcome.status, "next_step": outcome.next_step, "token_ids": outcome.token_ids}))
    return 0 if outcome.status == "COMPLETED" else 3


if __name__ == "__main__":
    sys.exit(main())

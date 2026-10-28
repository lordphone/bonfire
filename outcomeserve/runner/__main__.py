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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="outcomeserve.runner")
    parser.add_argument("--db", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trace", default="orders", choices=["orders", "legacy"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--crash-at", choices=[b.value for b in Boundary])
    parser.add_argument("--crash-step", type=int)
    parser.add_argument("--action", default=Action.TERMINATE_PROCESS.value, choices=[a.value for a in Action])
    args = parser.parse_args(argv)

    fault = None
    if args.crash_at is not None:
        if args.crash_step is None:
            parser.error("--crash-at requires --crash-step")
        fault = FaultPlan(Boundary(args.crash_at), args.crash_step, Action(args.action))

    outcome = Runner(
        SqliteLog(args.db), SqliteToolService(args.db), args.task, scripted_trace(args.trace, args.steps), fault
    ).run()
    print(json.dumps({"status": outcome.status, "next_step": outcome.next_step, "token_ids": outcome.token_ids}))
    return 0 if outcome.status == "COMPLETED" else 3


if __name__ == "__main__":
    sys.exit(main())

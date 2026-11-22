#!/usr/bin/env python3
"""Inspect an existing SQLite snapshot without repairing or executing it.

    python scripts/verify_run.py --db run/runtime.sqlite --workload run/workload.json
    python scripts/verify_run.py --db fault/runtime.sqlite --workload fault/workload.json \
        --token-source model --same-prefix-as clean/runtime.sqlite

``--token-source model`` checks real-inference logs. ``--replay-snapshot`` and
``--replay-request`` check historical replay against the faulted task's snapshot.
``--same-prefix-as`` is an optional deterministic-test diagnostic, not the real
GPU pilot gate: future generated tokens need not match a separate clean run.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from outcomeserve.verify import committed_tokens, verify_replay, verify_workload
from outcomeserve.verify.business import verify_business
from workloads import load_workload


def write_report(report, output):
    encoded = json.dumps(report, indent=2) + "\n"
    if output:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["completed", "in_progress", "halted"], default="completed")
    parser.add_argument("--token-source", choices=["scripted", "model"], default="scripted")
    parser.add_argument("--same-prefix-as", type=Path, metavar="OTHER_DB")
    parser.add_argument("--other-task", help="task_id in OTHER_DB (default: this workload's task_id)")
    parser.add_argument("--with-business", action="store_true", help="also independently verify business state")
    parser.add_argument("--business-only", action="store_true", help="restart baseline: verify business state without a durable log")
    parser.add_argument("--output", type=Path, help="write report to a new file; never overwrite retained evidence")
    parser.add_argument("--replay-snapshot", type=Path, help="SQLite backup of this task immediately before inference loss")
    parser.add_argument("--replay-request", type=Path, help="actual replacement request's prefix/prompt token IDs")
    args = parser.parse_args(argv)
    if bool(args.replay_snapshot) != bool(args.replay_request):
        parser.error("--replay-snapshot and --replay-request must be used together")
    if args.replay_snapshot and args.token_source != "model":
        parser.error("replay evidence requires --token-source model")
    if args.output and args.output.exists():
        parser.error("output exists; verification evidence is never overwritten")
    if args.business_only and (args.with_business or args.same_prefix_as or args.replay_snapshot):
        parser.error("--business-only cannot be combined with protocol/replay checks")
    workload = load_workload(args.workload)
    if args.business_only:
        business = verify_business(args.db, workload)
        write_report({"business": business, "protocol": None}, args.output)
        return 0 if business["task_success"] else 1
    report = verify_workload(args.db, workload, checkpoint=args.checkpoint, token_source=args.token_source)
    if args.same_prefix_as is not None:
        mine = committed_tokens(args.db, workload["task_id"])
        other = committed_tokens(args.same_prefix_as, args.other_task or workload["task_id"])
        report["same_prefix"] = {"other_db": str(args.same_prefix_as), "equal": mine is not None and mine == other,
                                 "tokens": None if mine is None else len(mine),
                                 "other_tokens": None if other is None else len(other)}
    if args.replay_snapshot:
        request = json.loads(args.replay_request.read_text(encoding="utf-8"))
        report["replay"] = verify_replay(args.replay_snapshot, workload, request)
        if report["replay"]["context_spec_id"] != report.get("context_spec_id"):
            report["replay"]["valid"] = False
            report["replay"]["issues"].append("snapshot and final context identities differ")
        if not report["replay"]["valid"]:
            report["consistency_pass"], report["task_success"] = False, False
            report["issues"].append({"code": "PREFIX_REPLAY", "step_id": None,
                                     "detail": "; ".join(report["replay"]["issues"])})
    business = verify_business(args.db, workload) if args.with_business else None
    output = {"business": business, "protocol": report} if args.with_business else report
    write_report(output, args.output)
    if not report["valid"]:
        return 2
    passed = (report["consistency_pass"] and report.get("same_prefix", {"equal": True})["equal"]
              and report.get("replay", {"valid": True})["valid"]
              and (business is None or business["task_success"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

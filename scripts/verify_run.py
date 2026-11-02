#!/usr/bin/env python3
"""Inspect an existing SQLite snapshot without repairing or executing it."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from outcomeserve.verify import verify_workload
from workloads import load_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["completed", "in_progress", "halted"], default="completed")
    args = parser.parse_args()
    workload = load_workload(args.workload)
    report = verify_workload(args.db, workload, checkpoint=args.checkpoint)
    print(json.dumps(report, indent=2))
    return 2 if not report["valid"] else 0 if report["consistency_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

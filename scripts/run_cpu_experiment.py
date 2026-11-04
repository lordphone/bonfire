#!/usr/bin/env python3
"""CPU protocol/cost experiment; no GPU measurements are simulated."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.cpu.harness import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["outcomeserve", "restart", "naive_retry", "transcript_checkpoint"], default="outcomeserve")
    parser.add_argument("--purpose", choices=["correctness", "performance"], default="correctness")
    parser.add_argument("--fault-domain", choices=["runner", "tool_response"], default="runner")
    parser.add_argument("--boundary", choices=["DURING_GENERATION", "PRE_DISPATCH", "POST_TOOL_COMMIT_PRE_RESULT_LOG", "POST_CONTEXT_COMMIT", "DURING_RECOVERY"], default="POST_TOOL_COMMIT_PRE_RESULT_LOG")
    parser.add_argument("--progress", type=float, default=0.2)
    parser.add_argument("--steps", type=int, choices=[5, 10, 20], default=5)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--no-fault", action="store_true")
    args = vars(parser.parse_args())
    args["output"] = args.pop("output_dir")
    result = run(**args)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] and result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

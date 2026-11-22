#!/usr/bin/env python3
"""Write result.json for one retained GPU run directory (C6, docs/run-result.md).

    python scripts/build_run_result.py run-data/gpu/tier1/s33-L8192-p0.5-outcomeserve \
        --events measurements.jsonl runner-measurements.jsonl

Reads identity.json, the listed measurement event files, verification.json,
resources.json and runner-exit.json from the directory. Refuses to overwrite an
existing result.json: raw run data is immutable.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.runresult import assemble_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--events", nargs="+", default=["measurements.jsonl"],
                        help="measurement event files inside run_dir, all on one clock_id")
    parser.add_argument("--output", type=Path, help="default: run_dir/result.json")
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "result.json"
    if output.exists():
        parser.error(f"{output} exists; raw results are never rewritten")
    result = assemble_directory(args.run_dir, event_files=args.events)
    # Exclusive creation also prevents a concurrent builder from overwriting it.
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error(f"{output} exists; raw results are never rewritten")
    print(json.dumps({"output": str(output), "outcome": result["outcome"], "valid": result["valid"],
                      "invalid_reasons": result["invalid_reasons"]}))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

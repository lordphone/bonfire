#!/usr/bin/env python3
"""Run a real allocated-GPU pilot from an explicit local configuration file."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gpu.pilot import run_gpu_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_gpu_pilot(json.loads(args.config.read_text(encoding="utf-8")), args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] and result["passed"] else 1)

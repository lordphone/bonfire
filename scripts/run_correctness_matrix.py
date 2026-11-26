#!/usr/bin/env python3
"""Run the full CPU correctness matrix (RQ1) and write runs.csv, summary.json and report.md.

    python scripts/run_correctness_matrix.py --output-dir run-data/correctness-matrix
    python scripts/run_correctness_matrix.py --output-dir run-data/correctness-matrix --resume

Cells: 5/10/20 steps x seeds x fault step tool (C2 decrement, C2 idempotent
write, unqueryable external call) x runner boundaries x four methods, plus a
dropped tool acknowledgment and, for OutcomeServe, a crash during recovery.
Each run is a fresh directory with real processes. ``--resume`` reuses runs that
already have result.json and stops on a half-written run instead of replacing it.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.correctness import case_name, cases, row, write_outputs
from experiments.cpu.harness import run
from outcomeserve.contracts import validate_schema


ROOT = Path(__file__).resolve().parents[1]


def source_files():
    """Exact uncommitted implementation and tests used by this experiment."""
    files = {}
    for directory in ("analysis", "experiments/cpu", "outcomeserve", "workloads", "schemas", "scripts", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    return files


def prepare_output(output, config, matrix, *, resume):
    """Freeze the plan before execution; never mix different settings or sources."""
    files = source_files()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    plan = {"schema_version": "1.0.0", "config": config, "cases": matrix, "source_sha256": hashes,
            "python": sys.version, "platform": platform.platform(), "executable": sys.executable}
    manifest = output / "matrix-plan.json"
    if output.exists():
        if not resume:
            raise ValueError("output exists; choose a new directory or pass --resume")
        if not manifest.exists():
            raise ValueError("cannot resume without matrix-plan.json; use a new output directory")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != plan:
            raise ValueError("resume plan, source code or environment differs; use a new output directory")
        if not (output / "source-snapshot.zip").exists():
            raise ValueError("cannot resume without source-snapshot.zip")
    else:
        output.mkdir(parents=True)
        with zipfile.ZipFile(output / "source-snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")
        try:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            head = "unavailable (the source snapshot is authoritative)"
        (output / "git-head.txt").write_text(head + "\n", encoding="utf-8")


def validate_case_result(case, result):
    validate_schema("run-result", result)
    required = {**case, "requested_progress": case["progress"], "purpose": "correctness",
                "backend": "scripted_cpu", "no_fault": False}
    required.pop("progress")
    for field, value in required.items():
        if result.get(field) != value:
            raise ValueError(f"retained result differs from requested case: {field}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", choices=[5, 10, 20], default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[33, 34, 35])
    parser.add_argument("--progress", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="reuse completed runs in an existing output directory")
    args = parser.parse_args(argv)
    config = {"steps": args.steps, "seeds": args.seeds, "progress": args.progress}
    try:
        matrix = cases(tuple(args.steps), tuple(args.seeds), args.progress)
        prepare_output(args.output_dir, config, matrix, resume=args.resume)
    except ValueError as exc:
        parser.error(str(exc))
    runs_dir = args.output_dir / "runs"
    rows, started = [], time.monotonic()
    for number, case in enumerate(matrix, 1):
        folder = runs_dir / case_name(case)
        if folder.exists():
            if not (folder / "result.json").exists():
                parser.error(f"{folder} has no result.json; inspect it and rerun into a new output directory")
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        else:
            folder.parent.mkdir(parents=True, exist_ok=True)
            result = run(folder, method=case["method"], seed=case["seed"], steps=case["steps"],
                         progress=case["progress"], boundary=case["boundary"], fault_domain=case["fault_domain"],
                         fault_tool=case["fault_tool"])
        validate_case_result(case, result)
        rows.append(row(case, result))
        mark = "ok" if rows[-1]["correct"] else "INVALID" if not rows[-1]["valid"] else "incorrect"
        print(f"[{number}/{len(matrix)}] {case_name(case)}: {mark}", flush=True)
    config = {**config, "runs": len(rows),
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    _, criteria = write_outputs(rows, args.output_dir, config=config)
    print(json.dumps(criteria, indent=2))
    # Baselines are expected to fail; the exit code reports OutcomeServe's criteria only.
    return 0 if criteria["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

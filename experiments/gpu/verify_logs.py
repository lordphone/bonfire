"""Offline C1 check of every committed GPU run with the independent verifier.

    python3 experiments/gpu/verify_logs.py experiments/gpu/results/<id>

The runtime schema-checks only records it has not persisted yet (see
docs/infrastructure/gpu-matrix.md). This reruns `verify_workload`, which
schema-checks every stored record, on each OutcomeServe run's runtime.sqlite
(pilot and rejected/ originals included) and writes log-verification.json next to
summary.md. Restart runs get the business check only: their reset deletes the
log, so they keep no durable prefix to verify.

It uses the model-mode check (`token_source="model"`, #15): committed tokens are
real model output, so instead of matching scripted token IDs, every prompt must
extend the durable committed prefix and every committed context must extend the
prompt and generation recorded at prepare time. Until #15 merges, put a checkout
of it first on PYTHONPATH:

    PYTHONPATH=<victoria checkout> python3 experiments/gpu/verify_logs.py experiments/gpu/results/<id>
"""

import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.append(str(Path(__file__).resolve().parents[2]))  # PYTHONPATH wins, see above
from outcomeserve.verify import verify_workload  # noqa: E402


def check(run: Path, scratch: Path) -> dict:
    database = scratch / "runtime.sqlite"
    with gzip.open(run / "runtime.sqlite.gz") as source, database.open("wb") as target:
        shutil.copyfileobj(source, target)
    report = verify_workload(database, json.loads((run / "workload.json").read_text()), token_source="model")
    database.unlink()
    return {"valid": report["valid"], "consistency_pass": report["consistency_pass"], "task_success": report["task_success"],
            "issues": sorted({issue["code"] for issue in report["issues"]}) + report["invalid_reasons"]}


def main(root: Path):
    found = [p for pattern in ("pilot/*/*/result.json", "tier-*/*/*/result.json", "tier-*/*/rejected/*/result.json")
             for p in sorted(root.glob(pattern))]
    runs = [p.parent for p in found if json.loads(p.read_text())["method"] == "outcomeserve"]
    with tempfile.TemporaryDirectory() as scratch:
        results = {str(run.relative_to(root)): check(run, Path(scratch)) for run in runs}
    failed = {name: r for name, r in results.items() if not (r["valid"] and r["consistency_pass"] and r["task_success"])}
    out = {"verifier": "outcomeserve.verify.verify_workload(token_source='model', checkpoint='completed')", "checked": len(results), "passed": len(results) - len(failed), "failed": failed, "runs": results}
    (root / "log-verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("checked", "passed", "failed")}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))

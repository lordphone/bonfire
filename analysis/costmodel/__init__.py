"""RQ3 cost model: predict recovery time and the one-fault saving S from component costs.

Owner: Shirley · Reviewers: Victoria / Lordphone

analysis.pipeline measures H and S from completed paired runs. This module
explains them: it calibrates per-component costs from standalone experiments
(cold start, prefill with APC off/hit, decode), predicts the recovery cells and
the planned 2K/8K/16K x 10/50/90% matrix, and reports prediction error so the
model's claims stay checkable against retained evidence.

v2 adds the durable-logging cost as a function of the committed prefix, and
charges OutcomeServe that cost on every step it still runs after the fault; v1
let the remaining steps cancel, which the 480-run RTX 5090 matrix showed was
wrong (OutcomeServe loses at 10% progress). validate_on_matrix checks v1 and v2
against that matrix.

Units are seconds and tokens. Scope matches the plan: one GPU, process restart,
at most one fault per task, no warm replica and no decode checkpoint.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections import defaultdict
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from analysis.pipeline import break_even_curve

MAX_MODEL_LEN = 32768


# ------------------------------------------------------------------ calibration


def _least_squares(rows: list[list[float]], targets: list[float]) -> list[float]:
    """Solve the normal equations for a small dense system (no numpy dependency)."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * t for r, t in zip(rows, targets)) for i in range(n)]
    for col in range(n):  # Gaussian elimination with partial pivoting
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot], b[col], b[pivot] = a[pivot], a[col], b[pivot], b[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
            b[row] -= factor * b[col]
    x = [0.0] * n
    for row in reversed(range(n)):
        x[row] = (b[row] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
    return x


@dataclass(frozen=True)
class Calibration:
    """Component costs, each with the experiment it came from."""

    prefill_miss: tuple[float, float]  # t = a*L + b*L^2  (APC off or cache miss)
    prefill_hit: tuple[float, float]  # t = c0 + c1*L    (whole prefix cached)
    decode_points: tuple[tuple[int, float], ...]  # (context tokens, tok/s), interpolated
    worker_startup: float  # inference process start to ready, warm caches
    source: dict

    def prefill(self, tokens: int, *, cached: bool = False) -> float:
        if not 0 <= tokens <= MAX_MODEL_LEN:
            raise ValueError(f"prefix of {tokens} tokens is outside max_model_len {MAX_MODEL_LEN}")
        if cached:
            c0, c1 = self.prefill_hit
            return c0 + c1 * tokens
        a, b = self.prefill_miss
        return a * tokens + b * tokens * tokens

    def decode_rate(self, context: int) -> float:
        points = sorted(self.decode_points)
        if context <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if context <= x1:
                return y0 + (y1 - y0) * (context - x0) / (x1 - x0)
        return points[-1][1]

    def decode(self, tokens: int, context: int) -> float:
        return tokens / self.decode_rate(context)


def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def calibrate_from_spike(results: Path) -> Calibration:
    """Fit component costs from the spike's standalone cells (not its recovery cells)."""
    results = Path(results)
    miss = [r for r in _rows(results / "prefill-apc-off" / "summary.json") if r["pass"] == "first"]
    hit = [r for r in _rows(results / "prefill-apc-on" / "summary.json") if r["pass"] == "repeat"]
    lengths = sorted({r["tokens"] for r in miss})
    miss_medians = [median(r["prefill_s"] for r in miss if r["tokens"] == L) for L in lengths]
    hit_medians = [median(r["prefill_s"] for r in hit if r["tokens"] == L) for L in lengths]
    a, b = _least_squares([[L, L * L] for L in lengths], miss_medians)
    c0, c1 = _least_squares([[1.0, L] for L in lengths], hit_medians)

    fixed = [r for r in _rows(results / "decode" / "summary.json") if r["kind"] == "fixed"]
    contexts = sorted({r["context_tokens"] for r in fixed})
    decode = tuple((c, median(r["decode_tok_s"] for r in fixed if r["context_tokens"] == c)) for c in contexts)

    starts = [r for r in _rows(results / "coldstart" / "summary.json")
              if r["weights"] == "page_cache" and r["compile_cache"] == "warm"]
    startup = median(r["startup_s"] for r in starts)
    return Calibration(prefill_miss=(a, b), prefill_hit=(c0, c1), decode_points=decode, worker_startup=startup,
                       source={"prefill_miss": f"prefill-apc-off first pass, medians at {lengths}",
                               "prefill_hit": f"prefill-apc-on repeat pass, medians at {lengths}",
                               "decode": f"decode fixed-512 medians at contexts {contexts}",
                               "worker_startup": f"coldstart page_cache/warm median, n={len(starts)}"})


# ------------------------------------------------------------------ recovery prediction


@dataclass(frozen=True)
class FixedCosts:
    """Recovery costs the model does not predict: supervisor policy and process start.

    detect: fault -> crash detected; teardown: detected -> replacement requested;
    runner_start: replacement requested -> runner ready; reconcile: log read + tool query.
    """

    detect: float
    teardown: float
    runner_start: float
    reconcile: float


def recovery_seconds(cal: Calibration, fixed: FixedCosts, *, domain: str, prefix_tokens: int, apc: bool) -> dict:
    """Predict fault -> first new token, split into components.

    inference: the GPU process is replaced, so the replay is always a cache miss.
    runner: the GPU process survives; with APC the replayed prefix is a cache hit.
    """
    if domain not in {"inference", "runner"}:
        raise ValueError("domain is inference or runner")
    cached = domain == "runner" and apc
    parts = {"detect": fixed.detect, "teardown": fixed.teardown,
             "startup": cal.worker_startup if domain == "inference" else fixed.runner_start,
             "reconcile": fixed.reconcile,
             "replay_prefill": cal.prefill(prefix_tokens, cached=cached),
             "first_token": cal.decode(1, prefix_tokens)}
    return {"domain": domain, "apc": apc, "prefix_tokens": prefix_tokens, "cache_hit": cached,
            "components": parts, "total": sum(parts.values())}


def fixed_costs_from_cell(rows: list[dict]) -> FixedCosts:
    ms = lambda key: median(r[key] for r in rows) / 1000.0
    return FixedCosts(detect=ms("detect_ms"), teardown=ms("teardown_ms"),
                      runner_start=ms("startup_ms"), reconcile=ms("reconcile_ms"))


def validate_on_spike(results: Path) -> dict:
    """Predict the three spike recovery cells and report error against their medians.

    Calibration uses only the standalone cells. Fixed costs come from the APC-off
    runner cell for both runner cells and from the inference cell's own detect /
    teardown / reconcile (the inference startup is predicted from the cold-start
    cell). The APC-on runner cell is therefore a held-out prediction: none of its
    measurements enter the model.
    """
    results = Path(results)
    cal = calibrate_from_spike(results)
    cells = json.loads((results / "recovery.json").read_text(encoding="utf-8"))
    by_name = {name.split("\n")[0] + (" " + name.split("\n")[1] if "\n" in name else ""): rows
               for name, rows in cells.items()}
    inference = next(rows for name, rows in by_name.items() if name.startswith("Inference kill"))
    runner_off = next(rows for name, rows in by_name.items() if name.startswith("Runner kill APC off"))
    runner_on = next(rows for name, rows in by_name.items() if name.startswith("Runner kill APC on"))

    runner_fixed = fixed_costs_from_cell(runner_off)
    inference_fixed = fixed_costs_from_cell(inference)
    cases = [("inference kill, APC off", "inference", False, inference, inference_fixed, "fixed costs in-cell; startup from coldstart"),
             ("runner kill, APC off", "runner", False, runner_off, runner_fixed, "fixed costs in-cell"),
             ("runner kill, APC on", "runner", True, runner_on, runner_fixed, "held out: fixed costs from the APC-off cell")]
    report = []
    for label, domain, apc, rows, fixed, basis in cases:
        prefix = int(median(r["prefix_tokens"] for r in rows))
        predicted = recovery_seconds(cal, fixed, domain=domain, prefix_tokens=prefix, apc=apc)
        measured = median(r["fault_to_next_token_ms"] for r in rows) / 1000.0
        report.append({"cell": label, "basis": basis, "prefix_tokens": prefix, "n": len(rows),
                       "measured_s": measured, "predicted_s": predicted["total"],
                       "relative_error": (predicted["total"] - measured) / measured,
                       "components_s": predicted["components"]})
    return {"calibration": asdict(cal), "runner_fixed_costs": asdict(runner_fixed), "cells": report}


# ------------------------------------------------------------------ whole-task S and H


@dataclass(frozen=True)
class TaskShape:
    """Token budget of one scripted task; pilot runs record these per step."""

    steps: int = 20
    initial_prefix: int = 2048  # instruction + padding before step 1
    step_prompt: int = 40  # "Execute next request ..." tokens added per step
    generated: int = 30  # tool-call JSON tokens per step
    tool_result: int = 20  # "Tool result: ..." tokens per step
    tool_seconds: float = 0.002

    @classmethod
    def from_pilot(cls, prefix_lengths: list[dict], *, tool_seconds: float = 0.002) -> "TaskShape":
        """Measured shape from a pilot's actual-prefix-lengths.json (one row per completed step)."""
        if not prefix_lengths:
            raise ValueError("pilot recorded no steps")
        first = prefix_lengths[0]
        generated = [r["generated_tokens"] for r in prefix_lengths]
        # committed = prompt + generated + tool result; prompt = previous committed + step prompt
        results = [r["committed_prefix_tokens"] - r["prompt_tokens"] - r["generated_tokens"] for r in prefix_lengths]
        step_prompts = [later["prompt_tokens"] - earlier["committed_prefix_tokens"]
                        for earlier, later in zip(prefix_lengths, prefix_lengths[1:])]
        step_prompt = round(median(step_prompts)) if step_prompts else 40
        return cls(steps=len(prefix_lengths), initial_prefix=first["prompt_tokens"] - step_prompt,
                   step_prompt=step_prompt, generated=round(median(generated)), tool_result=round(median(results)),
                   tool_seconds=tool_seconds)

    @classmethod
    def from_matrix(cls, rows: list[dict], *, generated: int = 30, tool_result: int = 19) -> "TaskShape":
        """Per-step token growth from matrix runs (initial and final prefix of complete runs).

        The matrix records only the total growth, so the split between step prompt,
        generated call and tool result is assumed; only `generated` affects decode time.
        """
        growth = [(int(r["final_prefix_tokens"]) - int(r["initial_prefix_tokens"])) / 20 for r in rows
                  if r.get("final_prefix_tokens") and r.get("initial_prefix_tokens")]
        if not growth:
            raise ValueError("no run records both initial and final prefix")
        per_step = round(median(growth))
        return cls(steps=20, step_prompt=per_step - generated - tool_result, generated=generated, tool_result=tool_result)

    def prefix_after(self, completed: int) -> int:
        return self.initial_prefix + completed * (self.step_prompt + self.generated + self.tool_result)


def step_seconds(cal: Calibration, task: TaskShape, index: int, *, apc: bool) -> float:
    """Generate and execute step `index` (0-based) in a run whose earlier history is warm."""
    history = task.prefix_after(index)
    prompt = history + task.step_prompt
    if apc and index > 0:  # history is cached from the previous step; only new tokens miss
        prefill = cal.prefill(history, cached=True) + (cal.prefill(prompt) - cal.prefill(history))
    else:
        prefill = cal.prefill(prompt)
    return prefill + cal.decode(task.generated, prompt) + task.tool_seconds


@dataclass(frozen=True)
class LoggingCost:
    """Durable prepare + commit per step: base + per_token x committed prefix after the step.

    Each CONTEXT_COMMITTED stores the full token prefix, so the cost grows with it.
    """

    base: float
    per_token: float = 0.0

    def step(self, committed_prefix: int) -> float:
        return self.base + self.per_token * committed_prefix

    def steps(self, task: "TaskShape", first: int, last: int) -> float:
        """Cost of steps first..last-1 (0-based)."""
        return sum(self.step(task.prefix_after(i + 1)) for i in range(first, last))

    def task_seconds(self, task: "TaskShape") -> float:
        return self.steps(task, 0, task.steps)


def fit_logging_cost(h_by_prefix: dict[int, float], shape: "TaskShape") -> LoggingCost:
    """Fit base and per_token to whole-task fault-free overheads H measured at several initial prefixes."""
    rows, targets = [], []
    for prefix, overhead in sorted(h_by_prefix.items()):
        task = TaskShape(steps=shape.steps, initial_prefix=prefix, step_prompt=shape.step_prompt,
                         generated=shape.generated, tool_result=shape.tool_result)
        rows.append([task.steps, sum(task.prefix_after(i + 1) for i in range(task.steps))])
        targets.append(overhead)
    if len(rows) < 2:
        raise ValueError("need H at two or more prefixes")
    base, per_token = _least_squares(rows, targets)
    return LoggingCost(base, per_token)


def one_fault_saving(cal: Calibration, fixed: FixedCosts, task: TaskShape, *, progress: float, domain: str,
                     apc: bool, restart_reset_seconds: float = 0.0, logging: LoggingCost | None = None) -> dict:
    """S = T_restart,one-fault - T_OutcomeServe,one-fault for a fault during step k+1.

    Both methods pay detection, teardown, process start and the remaining
    generation; those cancel. Restart redoes the k completed steps and resets
    tool state; OutcomeServe reconciles and replays the committed prefix once
    instead, and (v2) keeps paying durable logging on steps k+1..N, which the
    restart baseline does not. With logging=None this is the v1 model.
    """
    if not 0 <= progress < 1:
        raise ValueError("progress must be in [0, 1)")
    completed = int(task.steps * progress)
    redo = sum(step_seconds(cal, task, i, apc=apc) for i in range(completed))
    replay = recovery_seconds(cal, fixed, domain=domain, prefix_tokens=task.prefix_after(completed), apc=apc)
    resume = replay["components"]["reconcile"] + replay["components"]["replay_prefill"]
    # After a restart, step k+1 sees a history that restart rebuilt itself; after a
    # resume, its history was just replayed. Either way it is warm, so it cancels.
    after_fault = logging.steps(task, completed, task.steps) if logging else 0.0
    return {"completed_steps": completed, "restart_redo_s": redo, "restart_reset_s": restart_reset_seconds,
            "outcomeserve_resume_s": resume, "outcomeserve_logging_after_fault_s": after_fault,
            "S": redo + restart_reset_seconds - resume - after_fault}


def measure_log_overhead(steps: int = 20, repeats: int = 5) -> dict:
    """Failure-free durability cost per step on this host: prepare append + atomic commit.

    Uses the real DurableLog and an in-process C2 ToolService for COMMITTED
    evidence; the tool call itself is excluded because both methods make it.
    """
    from outcomeserve.contracts import request_fingerprint
    from outcomeserve.log import DurableLog
    from outcomeserve.storage import StateStore
    from outcomeserve.tools import ToolService
    from outcomeserve.context import context_spec_id, scripted_context
    from workloads import make_inventory_workload

    samples = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory() as directory:
            workload = make_inventory_workload(f"overhead-{repeat}", seed=33, steps=steps)
            task = workload["task_id"]
            store = StateStore(Path(directory) / "runtime.sqlite")
            store.create_task(task, workload["initial_inventory"])
            spec = scripted_context(workload["model_revision"])
            context_id = context_spec_id(spec)
            store.bind_context(task, spec)
            log, tools = DurableLog(store), ToolService(store)
            prefix = [1000]
            for step in workload["steps"]:
                fingerprint = request_fingerprint(step["tool_name"], step["args"])
                identity = {"task_id": task, "step_id": step["step_id"], "op_id": step["op_id"], "owner_epoch": 1}
                body = {"schema_version": "1.1.0", "context_spec_id": context_id, **identity,
                        "record_id": f"{step['step_id']}:prepare", "model_revision": workload["model_revision"],
                        "record_type": "STEP_PREPARED", "payload": {
                            "tool_name": step["tool_name"], "args": step["args"], "request_fingerprint": fingerprint,
                            "generated_token_ids": step["token_ids"], "generation_prompt_token_ids": prefix}}
                sequence = len(log.read(task))
                started = time.perf_counter()
                prepared = log.append(task, body["record_id"], sequence, 1, body)["record"]
                logged = time.perf_counter() - started
                response = tools.submit(**identity, tool_name=step["tool_name"], args=step["args"],
                                        request_fingerprint=fingerprint)
                prefix = prefix + step["token_ids"]
                started = time.perf_counter()
                status = log.commit_step(prepared, response, prefix, owner_epoch=1, expected_sequence=sequence + 1,
                                         commit_id=f"{step['step_id']}:commit")["status"]
                logged += time.perf_counter() - started
                if status != "COMMITTED":
                    raise RuntimeError(f"commit failed: {status}")
                samples.append(logged)
    return {"per_step_s": median(samples), "n": len(samples), "steps": steps, "repeats": repeats,
            "note": "host-specific SQLite WAL/FULL fsync cost; rerun on the GPU host for reported H"}


def matrix_predictions(cal: Calibration, fixed: FixedCosts, *, logging: LoggingCost | float,
                       domain: str = "inference", apc: bool = False, prefixes=(2048, 8192, 16384),
                       progresses=(0.1, 0.5, 0.9), shape: TaskShape | None = None) -> list[dict]:
    """Predicted S, H and break-even q for each planned matrix cell.

    A float `logging` is v1: a constant per-step cost that only enters H.
    """
    shape = shape or TaskShape()
    v1 = not isinstance(logging, LoggingCost)
    cost = LoggingCost(float(logging)) if v1 else logging
    cells = []
    for prefix in prefixes:
        task = TaskShape(steps=shape.steps, initial_prefix=prefix, step_prompt=shape.step_prompt,
                         generated=shape.generated, tool_result=shape.tool_result, tool_seconds=shape.tool_seconds)
        overhead = cost.task_seconds(task)
        for progress in progresses:
            saving = one_fault_saving(cal, fixed, task, progress=progress, domain=domain, apc=apc,
                                      logging=None if v1 else cost)
            curve = break_even_curve(overhead, saving["S"]) if saving["S"] > 0 else None
            cells.append({"initial_prefix": prefix, "progress": progress, "domain": domain, "apc": apc,
                          **saving, "H": overhead,
                          "break_even_q": curve["break_even_q"] if curve else None})
    return cells


# ------------------------------------------------------------------ check against the GPU matrix


def matrix_measurements(runs_csv: Path) -> dict:
    """Mean paired S (fault -> verified) and H (fault-free task) per cell from a matrix runs.csv."""
    with open(runs_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs = defaultdict(dict)
    for row in rows:
        if row["valid"] == "True" and row["passed"] == "True":
            pairs[row["cell"]].setdefault(row["seed"], {})[row["method"]] = row
    cells = {}
    for cell, seeds in pairs.items():
        complete = [m for m in seeds.values() if set(m) == {"restart", "outcomeserve"}]
        if "nofault" in cell:
            diffs = [float(m["outcomeserve"]["task_to_verified_seconds"]) - float(m["restart"]["task_to_verified_seconds"])
                     for m in complete]
        else:
            diffs = [float(m["restart"]["fault_to_verified_seconds"]) - float(m["outcomeserve"]["fault_to_verified_seconds"])
                     for m in complete]
        first_token = [float(m["outcomeserve"]["fault_to_first_new_token_seconds"]) for m in complete
                       if m["outcomeserve"].get("fault_to_first_new_token_seconds")]
        cells[cell] = {"pairs": len(diffs), "mean": sum(diffs) / len(diffs) if diffs else None,
                       "outcomeserve_fault_to_first_token_median": median(first_token) if first_token else None}
    return {"rows": rows, "cells": cells}


def _cell(prefix: int, progress: float | None) -> str:
    return f"p{prefix // 1024}k-" + ("nofault" if progress is None else f"f{round(progress * 100)}")


def validate_on_matrix(spike: Path, runs_csv: Path) -> dict:
    """Predict the measured matrix from spike calibration plus the fault-free H cells.

    The one-fault cells are out of sample for both versions: v1 uses only spike
    calibration; v2 adds a logging cost fitted to the separate fault-free (1H)
    runs. H itself is checked leave-one-prefix-out.
    """
    cal = calibrate_from_spike(spike)
    cells = json.loads((Path(spike) / "recovery.json").read_text(encoding="utf-8"))
    inference_fixed = fixed_costs_from_cell(next(rows for name, rows in cells.items() if name.startswith("Inference kill")))
    measured = matrix_measurements(runs_csv)
    shape = TaskShape.from_matrix([r for r in measured["rows"] if r["method"] == "outcomeserve"])
    prefixes = (2048, 8192, 16384)
    h = {p: measured["cells"][_cell(p, None)]["mean"] for p in prefixes}
    logging = fit_logging_cost(h, shape)

    def task(prefix):
        return TaskShape(steps=shape.steps, initial_prefix=prefix, step_prompt=shape.step_prompt,
                         generated=shape.generated, tool_result=shape.tool_result)

    h_check = []
    for held in prefixes:
        fitted = fit_logging_cost({p: v for p, v in h.items() if p != held}, shape)
        h_check.append({"prefix": held, "measured_H": h[held], "predicted_H": fitted.task_seconds(task(held)),
                        "basis": "leave this prefix out"})

    s_check = []
    for prefix in prefixes:
        for progress in (0.1, 0.5, 0.9):
            cell = measured["cells"][_cell(prefix, progress)]
            v1 = one_fault_saving(cal, inference_fixed, task(prefix), progress=progress, domain="inference", apc=False)
            v2 = one_fault_saving(cal, inference_fixed, task(prefix), progress=progress, domain="inference", apc=False,
                                  logging=logging)
            replay = recovery_seconds(cal, inference_fixed, domain="inference",
                                      prefix_tokens=task(prefix).prefix_after(v2["completed_steps"]), apc=False)
            s_check.append({"cell": _cell(prefix, progress), "pairs": cell["pairs"], "measured_S": cell["mean"],
                            "v1_S": v1["S"], "v2_S": v2["S"], "v2_logging_after_fault": v2["outcomeserve_logging_after_fault_s"],
                            "measured_first_token": cell["outcomeserve_fault_to_first_token_median"],
                            "predicted_first_token": replay["total"] - replay["components"]["detect"]
                            - replay["components"]["teardown"]})
    mae = lambda key: sum(abs(c[key] - c["measured_S"]) for c in s_check) / len(s_check)
    return {"task_shape": shape.__dict__, "logging_cost": asdict(logging), "H": h_check, "S": s_check,
            "mean_abs_error_s": {"v1": mae("v1_S"), "v2": mae("v2_S")},
            "notes": "first-token prediction excludes detect/teardown: the matrix supervisor kills on its barrier, "
                     "so FAULT_APPLIED -> FIRST_NEW_TOKEN starts at the kill"}

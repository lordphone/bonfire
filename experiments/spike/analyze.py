# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.7", "jsonschema==4.26.0"]
# ///
"""Aggregate spike results into summary.md and figures.

    uv run experiments/spike/analyze.py experiments/spike/results/<id>

Recovery intervals come only from TelemetryEvent boundaries (measurement contract);
cold-start sub-phases come from vLLM's own log lines and are labelled as such.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import jsonschema
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, light mode, fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OTHER = "#b5b4af"  # unattributed time between boundaries; neutral, not a series hue


def events(*paths: Path) -> list[dict]:
    evs = []
    for p in paths:
        if p.exists():
            evs += [json.loads(line) for line in p.open()]
    return sorted(evs, key=lambda e: e["monotonic_ns"])


def first(evs, etype, after=0, attempt=None):
    for e in evs:
        if e["event_type"] == etype and e["monotonic_ns"] >= after and (
                attempt is None or e["worker"]["attempt"] == attempt):
            return e["monotonic_ns"]
    raise KeyError(etype)


def ms(a, b):
    return (b - a) / 1e6


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{st.median(xs):.1f} (σ {st.stdev(xs):.1f}, n={len(xs)})"


def med(xs):
    return st.median(xs) if xs else 0.0


SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "telemetry-event.schema.json"


def validate(out: Path) -> tuple[int, int]:
    """Validate every event, and that each file's sequence is gap-free and monotonic_ns ordered."""
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))
    n = 0
    for p in sorted(out.rglob("*.jsonl")):
        if p.name == "durable-log.jsonl":
            continue
        prev_seq, prev_mono = 0, -1
        for line in p.open():
            e = json.loads(line)
            validator.validate(e)
            if e["sequence"] != prev_seq + 1:
                raise ValueError(f"{p}: sequence gap at {e['sequence']}")
            if p.name != "gpu.jsonl" and e["monotonic_ns"] < prev_mono:
                raise ValueError(f"{p}: monotonic_ns out of order at {e['sequence']}")
            prev_seq, prev_mono = e["sequence"], e["monotonic_ns"]
            n += 1
    return n, sum(1 for p in out.rglob("*.jsonl") if p.name != "durable-log.jsonl")


# --------------------------------------------------------------------------- recovery


def recovery_inference(base: Path):
    rows = []
    for task in sorted(p for p in base.glob("agent-*") if p.is_dir()):
        evs = events(task / "runner.jsonl")
        f = first(evs, "FAULT_INJECTED")
        c = first(evs, "CRASH_DETECTED", f)
        s = first(evs, "WORKER_START_REQUESTED", c)
        ready = first(evs, "WORKER_READY", s)
        r0, r1 = first(evs, "RECONCILIATION_STARTED", ready), first(evs, "RECONCILIATION_FINISHED", ready)
        p0, p1 = first(evs, "PREFILL_STARTED", r1), first(evs, "PREFILL_FINISHED", r1)
        nt = first(evs, "NEXT_TOKEN_EMITTED", p1)
        pref = next(e for e in evs if e["event_type"] == "PREFILL_STARTED")["details"]["committed_prefix_tokens"]
        rows.append({"task": task.name, "detect_ms": ms(f, c), "teardown_ms": ms(c, s), "startup_ms": ms(s, ready),
                     "reconcile_ms": ms(r0, r1), "prefill_ms": ms(p0, p1), "crash_to_next_token_ms": ms(c, nt),
                     "fault_to_next_token_ms": ms(f, nt), "prefix_tokens": pref})
    return rows


def recovery_runner(base: Path):
    rows = []
    for task in sorted(p for p in base.glob("agent-*") if p.is_dir()):
        evs = events(task / "supervisor.jsonl", task / "runner-attempt-1.jsonl", task / "runner-attempt-2.jsonl")
        f = first(evs, "FAULT_INJECTED")
        c = first(evs, "CRASH_DETECTED", f)
        s = first(evs, "WORKER_START_REQUESTED", c, attempt=2)
        up = first(evs, "WORKER_PROCESS_STARTED", s, attempt=2)
        r0, r1 = first(evs, "RECONCILIATION_STARTED", up), first(evs, "RECONCILIATION_FINISHED", up)
        p0, p1 = first(evs, "PREFILL_STARTED", r1), first(evs, "PREFILL_FINISHED", r1)
        nt = first(evs, "NEXT_TOKEN_EMITTED", p1)
        pref = next(e for e in evs if e["event_type"] == "PREFILL_STARTED")["details"]["committed_prefix_tokens"]
        rows.append({"task": task.name, "detect_ms": ms(f, c), "teardown_ms": ms(c, s), "startup_ms": ms(s, up),
                     "reconcile_ms": ms(r0, r1), "prefill_ms": ms(p0, p1), "crash_to_next_token_ms": ms(c, nt),
                     "fault_to_next_token_ms": ms(f, nt), "prefix_tokens": pref})
    return rows


# --------------------------------------------------------------------------- figures


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_recovery(groups: dict[str, list[dict]], path: Path):
    """Two panels with their own scales: a cold inference restart is ~100x a runner restart,
    so one axis would flatten the runner bars to nothing."""
    parts = [("detect_ms", "Crash detection"), ("teardown_ms", "Teardown"), ("startup_ms", "Worker startup"),
             ("reconcile_ms", "Reconcile"), ("prefill_ms", "Re-prefill + first token")]
    panels = [[n for n in groups if n.startswith("Inference") and groups[n]],
              [n for n in groups if n.startswith("Runner") and groups[n]]]
    panels = [ns for ns in panels if ns]
    # sized for one column of a two-column paper (3.3 in), so text stays legible
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4, 2.7), dpi=300,
                             gridspec_kw={"width_ratios": [len(ns) for ns in panels]})
    axes = axes if len(panels) > 1 else [axes]
    for ax, names, unit, scale in zip(axes, panels, ("s", "ms"), (1000, 1)):
        bottoms = [0.0] * len(names)
        for i, (k, label) in enumerate(parts):
            vals = [med([r[k] for r in groups[n]]) / scale for n in names]
            ax.bar(names, vals, bottom=bottoms, color=SERIES[i], label=label, width=0.55,
                   edgecolor="white", linewidth=1)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        # Bars total the median fault → next token; the remainder is time between boundaries.
        totals = [med([r["fault_to_next_token_ms"] for r in groups[n]]) / scale for n in names]
        ax.bar(names, [max(t - b, 0) for t, b in zip(totals, bottoms)], bottom=bottoms, color=OTHER,
               label="Other", width=0.55, edgecolor="white", linewidth=1)
        for x, t in enumerate(totals):
            ax.text(x, t, f"{t:.1f} {unit}" if unit == "s" else f"{t:.0f} {unit}",
                    ha="center", va="bottom", color=INK, fontsize=6.5)
        ax.margins(y=0.12)
        _style(ax, f"Fault → next token ({unit})")
        ax.yaxis.label.set_size(7)
        ax.tick_params(axis="both", labelsize=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, loc="upper center", ncol=3,
               handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.8)
    fig.savefig(path)
    plt.close(fig)


def fig_prefill(rows: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    lengths = sorted({r["tokens"] for r in rows})
    series = [("APC off", lambda r: not r["apc"]),
              ("APC on, first request", lambda r: r["apc"] and r["pass"] == "first"),
              ("APC on, same prefix again", lambda r: r["apc"] and r["pass"] == "repeat")]
    # APC on with a cold cache matches APC off, so that line is dashed with hollow markers to stay visible
    styles = [dict(linestyle="-", marker="o"), dict(linestyle="--", marker="o", markerfacecolor="white"),
              dict(linestyle="-", marker="s")]
    for i, (label, pred) in enumerate(series):
        ys = [med([r["prefill_s"] * 1000 for r in rows if pred(r) and r["tokens"] == L]) for L in lengths]
        ax.plot(lengths, ys, markersize=6, linewidth=2, color=SERIES[i], label=label, **styles[i])
    ax.set_xticks(lengths, [f"{L // 1024}K" for L in lengths])
    ax.set_xlabel("Prefix length (tokens)", color=INK2)
    _style(ax, "Median prefill time (ms)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- report

THERMAL_OR_POWER = 0x4 | 0x8 | 0x20 | 0x40 | 0x80  # SW power cap, HW slowdown, SW/HW thermal, power brake


def gpu_section(out: Path) -> list[str]:
    md = ["## GPU (GPU_SAMPLE every 200 ms)", "",
          "| Experiment | Samples | Median SM clock at ≥90% util (MHz) | Min SM clock at ≥90% util (MHz) | Max temp (°C) | Throttle samples |",
          "| --- | --- | --- | --- | --- | --- |"]
    for p in sorted(out.glob("*/gpu.jsonl")):
        d = [json.loads(line)["details"] for line in p.open()]
        busy = [x["sm_clock_mhz"] for x in d if x["utilization_pct"] >= 90]
        if "clock_event_reasons" in d[0]:
            thr = str(sum(1 for x in d if x["clock_event_reasons"] & THERMAL_OR_POWER))
        else:
            thr = "not logged"
        md.append(f"| {p.parent.name} | {len(d)} | {med(busy):.0f} | {min(busy, default=0):.0f} | "
                  f"{max(x['temperature_c'] for x in d):.0f} | {thr} |")
    return md + ["", "Throttle samples count power-cap, slowdown, or thermal clock event reasons; "
                 "older runs did not log the reasons.", ""]



def main(out: Path):
    n_events, n_files = validate(out)
    md = [f"# Spike results: `{out.name}`", "",
          f"{n_events} telemetry events in {n_files} files, all valid against `telemetry-event.schema.json` "
          "with gap-free sequences.", ""]
    hw = next(iter(events(*out.glob("*/supervisor.jsonl"))), {}).get("hardware", {})
    md += [f"GPU {hw.get('gpu_model')} (driver {hw.get('driver_version')}), host `{hw.get('host_id')}`. "
           "Values are median (σ, n). Times in ms unless noted.", ""]

    cs = json.loads((out / "coldstart" / "summary.json").read_text()) if (out / "coldstart").exists() else []
    jit_inferred = any("driver_jit_cache" not in r for r in cs)
    for i, r in enumerate(cs):
        # Runs before the harness recorded it: the driver JIT cache mount was new at the start of
        # run 20261103-rtx5090, so only its first start ran with an empty driver cache.
        r.setdefault("driver_jit_cache", "empty" if i == 0 else "warm")
    if cs:
        md += ["## Cold start (APC off)", "",
               "| Weights | Driver JIT cache | Startup (s) | Weights load (s) | torch.compile (s) | CUDA graphs (s) | Engine init (s) |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for jit in ("empty", "warm"):
            for w in ("nvme", "page_cache"):
                g = [r for r in cs if r["weights"] == w and r["driver_jit_cache"] == jit]
                if g:
                    md.append(f"| {w} | {jit} | " + " | ".join(
                        stats([r.get(k) for r in g]) for k in
                        ("startup_s", "weights_load_s", "torch_compile_s", "cuda_graph_s", "engine_init_s")) + " |")
        md += ["", "Sub-phases are vLLM's self-reported log values (diagnostic); startup is "
               "`WORKER_START_REQUESTED` → `WORKER_READY`. The vLLM torch.compile cache was warm for every start."]
        if jit_inferred:
            md += ["The driver JIT cache state was not recorded in this run; it is inferred (only the first "
                   "start followed a new cache mount, and its CUDA graph capture took 70 s instead of 5 s). "
                   "The empty-cache row is a single first-start observation (n=1), diagnostic only."]
        md.append("")

    pre = []
    for apc in ("off", "on"):
        p = out / f"prefill-apc-{apc}" / "summary.json"
        if p.exists():
            pre += json.loads(p.read_text())
    if pre:
        md += ["## Prefill (time to first token, max_tokens=1)", "",
               "| Tokens | APC off (ms) | APC on, first (ms) | APC on, repeat (ms) | APC-off throughput (tok/s) |",
               "| --- | --- | --- | --- | --- |"]
        for L in sorted({r["tokens"] for r in pre}):
            off = [r["prefill_s"] * 1000 for r in pre if not r["apc"] and r["tokens"] == L]
            on1 = [r["prefill_s"] * 1000 for r in pre if r["apc"] and r["pass"] == "first" and r["tokens"] == L]
            on2 = [r["prefill_s"] * 1000 for r in pre if r["apc"] and r["pass"] == "repeat" and r["tokens"] == L]
            md.append(f"| {L} | {stats(off)} | {stats(on1)} | {stats(on2)} | {L / (med(off) / 1000):,.0f} |")
        md.append("")
        fig_prefill(pre, out / "prefill.png")

    dec = json.loads((out / "decode" / "summary.json").read_text()) if (out / "decode").exists() else []
    if dec:
        md += ["## Decode", "", "| Mode | Context / thinking | Decode (tok/s) | Generated tokens | TTFT (ms) |",
               "| --- | --- | --- | --- | --- |"]
        for L in (2048, 16384):
            g = [r for r in dec if r["kind"] == "fixed" and r["context_tokens"] == L]
            md.append(f"| fixed 512 tokens | {L} ctx | {stats([r['decode_tok_s'] for r in g])} | 512 | – |")
        for th in (False, True):
            g = [r for r in dec if r["kind"] == "chat" and r["thinking"] == th]
            md.append(f"| chat, agent prompts | thinking {'on' if th else 'off'} | "
                      f"{stats([r['decode_tok_s'] for r in g])} | {stats([r['gen_tokens'] for r in g])} | "
                      f"{stats([r['ttft_s'] * 1000 for r in g])} |")
        md.append("")

    groups = {}
    if (out / "kill-inference").exists():
        groups["Inference kill\n(cold restart)"] = recovery_inference(out / "kill-inference")
    for apc in ("off", "on"):
        b = out / f"kill-runner-apc-{apc}"
        if b.exists():
            groups[f"Runner kill\nAPC {apc}"] = recovery_runner(b)
    if groups:
        md += ["## Recovery after `kill -9` mid-generation (step 6 of 10)", "",
               "| Fault domain | Detect | Teardown | Startup | Reconcile | Re-prefill + 1st token | Fault → next token | Crash → next token | Prefix tokens |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, rows in groups.items():
            md.append(f"| {name.replace(chr(10), ' ')} | " + " | ".join(stats([r[k] for r in rows]) for k in (
                "detect_ms", "teardown_ms", "startup_ms", "reconcile_ms", "prefill_ms",
                "fault_to_next_token_ms", "crash_to_next_token_ms", "prefix_tokens")) + " |")
        md += ["", "Fault → next token (`FAULT_INJECTED` → `NEXT_TOKEN_EMITTED`) is the headline; the phases "
               "do not sum to it exactly, because short gaps between boundaries are unattributed. "
               "For the inference kill, reconcile is only the two fsynced boundary events: the runner "
               "survives, so there is no log to replay.", ""]
        fig_recovery(groups, out / "recovery.png")
        (out / "recovery.json").write_text(json.dumps(groups, indent=1))

    md += gpu_section(out)
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

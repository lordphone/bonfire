"""Week 1 spike on a local RTX 5090: cold start, prefill, decode, and kill tests.

Every timing boundary is a TelemetryEvent (schemas/telemetry-event.schema.json)
appended to a JSONL file and fsynced. vLLM log lines are kept only as
diagnostics for the cold-start phase breakdown, never as interval boundaries.

Stdlib only, so runner start-up time is not inflated by imports.

    python3 experiments/spike/spike.py all --out experiments/spike/results/<id>
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
IMAGE = "vllm/vllm-openai@sha256:f99aa709b3f873aa1e1aadf5c1db35970119e8af9b36fe15d2a667aae0ed005c"
MODEL = {"id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "dtype": "bfloat16"}
MODEL_DIR = Path.home() / "models" / "Qwen3-4B"
COMPILE_CACHE = Path.home() / ".cache" / "bonfire-spike"  # written by the root container
COMPAT_DIR = Path(__file__).resolve().parent / "compat"  # see compat/torchcodec/__init__.py
MAX_MODEL_LEN = 32768
SEED = 20261026
PORT = 8000
EXPERIMENT_ID = "spike-5090"
# Ordinary Qwen3 vocabulary ids; special tokens start at 151643.
TOKEN_RANGE = (1000, 150000)


# --------------------------------------------------------------------------- telemetry


def _gpu_identity() -> dict[str, str | None]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    name, uuid, driver = (s.strip() for s in out.split(","))
    return {"host_id": socket.gethostname(), "gpu_model": name, "gpu_uuid": uuid, "driver_version": driver}


def _clock_id() -> str:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return f"{socket.gethostname()}-boot-{boot}"


def configuration_id(apc: bool) -> str:
    return f"qwen3-4b-vllm-0.32.0-rtx5090-mml{MAX_MODEL_LEN}-apc-{'on' if apc else 'off'}"


def worker(lineage: str, attempt: int) -> dict:
    return {"lineage_id": lineage, "instance_id": f"{lineage}-attempt-{attempt}",
            "attempt": attempt, "owner_epoch": attempt}


class Emitter:
    """Append-only JSONL writer. One file per process, so sequence is gap-free per file;
    CLOCK_MONOTONIC is host-wide, so files from one run merge on monotonic_ns."""

    _hardware: dict | None = None

    def __init__(self, path: Path, run_id: str, task_id: str, apc: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.seq = sum(1 for _ in path.open()) if path.exists() else 0
        self.f = path.open("a")
        self.lock = threading.Lock()
        if Emitter._hardware is None:
            env = os.environ.get("BONFIRE_HARDWARE")
            Emitter._hardware = json.loads(env) if env else _gpu_identity()
            os.environ["BONFIRE_HARDWARE"] = json.dumps(Emitter._hardware)
        self.base = {
            "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "run_id": run_id,
            "task_id": task_id, "configuration_id": configuration_id(apc), "clock_id": _clock_id(),
            "hardware": Emitter._hardware, "model": MODEL,
        }

    def emit(self, event_type: str, wk: dict, step: str = "none", details: dict | None = None,
             injection_id: str | None = None, mono: int | None = None) -> int:
        with self.lock:
            mono = time.monotonic_ns() if mono is None else mono
            self.seq += 1
            evt = {**self.base, "event_id": f"{self.base['run_id']}:{self.path.stem}:{self.seq:06d}",
                   "event_type": event_type, "step_id": step, "injection_id": injection_id,
                   "sequence": self.seq, "wall_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                   "monotonic_ns": mono, "worker": wk, "accepted_state_mutation": False,
                   "details": details or {}}
            self.f.write(json.dumps(evt) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())
            return mono


class GpuSampler:
    """GPU_SAMPLE every 200 ms (measurement contract), plus SM clock for consumer-card drift."""

    def __init__(self, em: Emitter, wk: dict):
        self.em = Emitter(em.path.with_name("gpu.jsonl"), em.base["run_id"], em.base["task_id"],
                          apc=em.base["configuration_id"].endswith("apc-on"))
        self.wk = wk
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        for line in self.proc.stdout:
            try:
                util, mem, power, temp, sm = (float(x) for x in line.split(","))
            except ValueError:
                continue
            self.em.emit("GPU_SAMPLE", self.wk, details={
                "utilization_pct": util, "memory_used_bytes": int(mem * 1024 * 1024),
                "power_watts": power, "temperature_c": temp, "sm_clock_mhz": sm})

    def stop(self):
        self.proc.terminate()
        self.proc.wait()


# --------------------------------------------------------------------------- vLLM server


def _http(method: str, path: str, body: dict | None = None, port: int = PORT, timeout: float = 600):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, json.dumps(body) if body is not None else None,
                 {"Content-Type": "application/json"})
    return conn, conn.getresponse()


def healthy(port: int = PORT) -> bool:
    try:
        conn, r = _http("GET", "/health", port=port, timeout=1)
        r.read()
        conn.close()
        return r.status == 200
    except OSError:
        return False


def evict_page_cache(root: Path) -> None:
    """Drop the weights from the page cache without root, so a cold start reads from NVMe."""
    for p in root.rglob("*"):
        if p.is_file():
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


LOG_PHASES = {
    "weights_load_s": re.compile(r"Loading weights took ([\d.]+) seconds"),
    "model_load_s": re.compile(r"Model loading took [\d.]+ GiB (?:memory )?and ([\d.]+) seconds"),
    "torch_compile_s": re.compile(r"torch\.compile took ([\d.]+) s in total"),
    "cuda_graph_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
    "engine_init_s": re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"),
}


def _cache_state(d: Path) -> str:
    try:
        return "warm" if any(d.iterdir()) else "empty"
    except OSError:
        return "empty"


class Server:
    """One vLLM container per worker attempt; SIGKILL of the container is the inference-process crash."""

    def __init__(self, out: Path, apc: bool, em: Emitter, lineage: str = "vllm"):
        self.out, self.apc, self.em, self.lineage = out, apc, em, lineage
        self.attempt = 0
        self.name = None
        self.logp = None

    def start(self, step: str = "none", evict: bool = False) -> dict:
        self.attempt += 1
        wk = worker(self.lineage, self.attempt)
        self.name = f"bonfire-spike-{os.getpid()}-{self.attempt}"
        if evict:
            evict_page_cache(MODEL_DIR)
        for d in ("vllm", "triton", "nv"):
            (COMPILE_CACHE / d).mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "run", "-d", "--rm", "--name", self.name, "--runtime=nvidia", "--gpus", "all",
               "--ipc=host", "-p", f"{PORT}:8000",
               "-v", f"{MODEL_DIR}:/model:ro", "-v", f"{COMPILE_CACHE}/vllm:/root/.cache/vllm",
               "-v", f"{COMPILE_CACHE}/triton:/root/.triton",
               # CUDA driver PTX JIT cache: the image's kernels are JIT-compiled for sm_120
               "-v", f"{COMPILE_CACHE}/nv:/root/.nv", "-e", "CUDA_CACHE_MAXSIZE=4294967296",
               "-v", f"{COMPAT_DIR}:/opt/bonfire-compat:ro", "-e", "PYTHONPATH=/opt/bonfire-compat",
               "--entrypoint", "vllm", IMAGE,
               "serve", "/model", "--served-model-name", "qwen3-4b", "--dtype", "bfloat16",
               "--max-model-len", str(MAX_MODEL_LEN), "--tensor-parallel-size", "1", "--seed", str(SEED),
               "--gpu-memory-utilization", "0.85",
               "--enable-prompt-tokens-details",  # reports cached_tokens, i.e. APC hits
               "--enable-prefix-caching" if self.apc else "--no-enable-prefix-caching"]
        caches = {"compile_cache": _cache_state(COMPILE_CACHE / "vllm" / "torch_compile_cache"),
                  "driver_jit_cache": _cache_state(COMPILE_CACHE / "nv")}
        t0 = self.em.emit("WORKER_START_REQUESTED", wk, step,
                          {"container": self.name, "weights_evicted": evict, **caches})
        subprocess.run(cmd, check=True, capture_output=True)
        self.em.emit("WORKER_PROCESS_STARTED", wk, step, {"container": self.name})
        self.logp = self.out / "vllm-logs" / f"{self.name}.log"
        self.logp.parent.mkdir(parents=True, exist_ok=True)
        self._logproc = subprocess.Popen(["docker", "logs", "-f", self.name],
                                         stdout=self.logp.open("w"), stderr=subprocess.STDOUT)
        while not healthy():
            if subprocess.run(["docker", "inspect", self.name], capture_output=True).returncode != 0:
                raise RuntimeError(f"vLLM container exited before ready; see {self.logp}")
            time.sleep(0.05)
        t1 = self.em.emit("WORKER_READY", wk, step, {"container": self.name})
        time.sleep(0.5)  # let the log follower flush
        phases = self.log_phases()
        return {"attempt": self.attempt, "startup_s": (t1 - t0) / 1e9, **caches, **phases}

    def log_phases(self) -> dict:
        text = self.logp.read_text(errors="replace") if self.logp else ""
        res = {}
        for k, rx in LOG_PHASES.items():
            m = [float(x) for x in rx.findall(text)]
            if m:
                res[k] = sum(m) if k == "cuda_graph_s" else m[-1]
        return res

    def kill(self) -> None:
        subprocess.run(["docker", "kill", "-s", "KILL", self.name], capture_output=True)

    def stop(self) -> None:
        if self.name:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            while healthy():
                time.sleep(0.1)
            # wait for the port and GPU memory to be released before the next start
            while subprocess.run(["docker", "inspect", self.name], capture_output=True).returncode == 0:
                time.sleep(0.1)


# --------------------------------------------------------------------------- requests


def rand_tokens(n: int, seed) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(*TOKEN_RANGE) for _ in range(n)]


def stream_completion(prompt_ids: list[int], max_tokens: int, on_token=None, port: int = PORT):
    """Stream a completion from token ids. Returns (token_times_ns, token_ids or None, usage).

    on_token(i) is called after the i-th streamed chunk; it may raise to abort."""
    body = {"model": "qwen3-4b", "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    conn, r = _http("POST", "/v1/completions", body, port=port)
    if r.status != 200:
        raise RuntimeError(f"completion failed {r.status}: {r.read()[:300]!r}")
    times, ids, usage = [], [], None
    try:
        while True:
            line = r.readline()
            if not line:
                raise ConnectionError("stream closed before [DONE]")
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                tid = ch.get("token_ids")
                n = len(tid) if tid else (1 if ch.get("text") else 0)
                if n:
                    times.append(time.monotonic_ns())
                    if tid:
                        ids.extend(tid)
                    if on_token:
                        on_token(len(times))
    finally:
        conn.close()
    return times, (ids or None), usage


def chat(messages, enable_thinking: bool, max_tokens: int, seed: int):
    body = {"model": "qwen3-4b", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            # Qwen3 recommended sampling for each mode
            **({"temperature": 0.6, "top_p": 0.95, "top_k": 20} if enable_thinking
               else {"temperature": 0.7, "top_p": 0.8, "top_k": 20})}
    conn, r = _http("POST", "/v1/chat/completions", body)
    times, usage, finish = [], None, None
    while True:
        raw = r.readline()
        if not raw:
            break
        line = raw.strip()
        if not line.startswith(b"data: "):
            continue
        if line[6:] == b"[DONE]":
            break
        chunk = json.loads(line[6:])
        usage = chunk.get("usage") or usage
        for ch in chunk.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                times.append(time.monotonic_ns())
            finish = ch.get("finish_reason") or finish
    conn.close()
    return times, usage, finish


def warmup(lengths=(512, 2048, 8192, 16384)):
    """Greedy requests at every measured length and sampled chat requests, so kernel JIT
    (Triton, and the driver's PTX JIT on sm_120) happens before anything is measured."""
    for L in lengths:
        stream_completion(rand_tokens(L, f"warmup-{L}"), 16)
    chat([{"role": "user", "content": "Say hi."}], False, 16, SEED)
    chat([{"role": "user", "content": "Say hi."}], True, 16, SEED)


# --------------------------------------------------------------------------- experiments


def exp_coldstart(out: Path, reps: int):
    """Startup to readiness, APC off. First attempt may populate the compile cache; it is
    tagged and reported separately. Each rep is run with weights evicted (NVMe) and cached (RAM)."""
    em = Emitter(out / "coldstart" / "supervisor.jsonl", "coldstart", "coldstart", apc=False)
    srv = Server(out / "coldstart", apc=False, em=em)
    rows = []
    for rep in range(reps):
        for evict in (True, False):
            r = srv.start(evict=evict)
            srv.stop()
            r |= {"rep": rep, "weights": "nvme" if evict else "page_cache"}
            rows.append(r)
            print("coldstart", r, flush=True)
    _dump(out / "coldstart" / "summary.json", rows)


def exp_prefill(out: Path, reps: int, apc: bool, lengths=(2048, 8192, 16384)):
    em = Emitter(out / f"prefill-apc-{'on' if apc else 'off'}" / "supervisor.jsonl",
                 f"prefill-apc-{'on' if apc else 'off'}", "prefill", apc=apc)
    srv = Server(out / f"prefill-apc-{'on' if apc else 'off'}", apc=apc, em=em)
    srv.start()
    wk = worker("bench-client", 1)
    sampler = GpuSampler(em, wk)
    rows = []
    try:
        warmup()
        for rep in range(reps):
            for L in lengths:
                ids = rand_tokens(L, f"prefill-{L}-{rep}")
                passes = ("first", "repeat") if apc else ("first",)
                for p in passes:
                    step = f"L{L}-r{rep}-{p}"
                    t0 = time.monotonic_ns()
                    times, _, usage = stream_completion(ids, 1)
                    t1 = times[0]
                    em.emit("PREFILL_STARTED", wk, step, {"committed_prefix_tokens": L}, mono=t0)
                    em.emit("PREFILL_FINISHED", wk, step, {"committed_prefix_tokens": L}, mono=t1)
                    row = {"rep": rep, "tokens": L, "pass": p, "apc": apc, "prefill_s": (t1 - t0) / 1e9,
                           "prompt_tokens": (usage or {}).get("prompt_tokens"),
                           "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
                    rows.append(row)
                    print("prefill", row, flush=True)
    finally:
        sampler.stop()
        srv.stop()
    _dump(out / f"prefill-apc-{'on' if apc else 'off'}" / "summary.json", rows)


AGENT_PROMPTS = [
    "You are an order-management agent. Tools: get_order(id), list_inventory(sku), create_order(sku, qty), "
    "decrement_inventory(sku, qty). A customer wants 3 units of SKU A-17 but inventory shows 2. "
    "Plan the tool calls you would make and explain how you would handle the shortage.",
    "You are an order-management agent with tools get_order, refund_order, and send_email. Order 4471 was "
    "charged twice. Describe step by step which tools you call and with what arguments.",
    "As a retail support agent, a user asks to change the shipping address of order 9021 after it has "
    "shipped. Reason about the policy options and the tool calls needed.",
    "You manage inventory for a small store. Given restock lead times of 5 days and daily demand of 12 "
    "units with 40 units in stock, decide whether to reorder and which tool calls to make.",
    "A customer wants to cancel one item from a three-item order where one item already shipped. "
    "Walk through the tool calls and the final state of the order.",
]


def exp_decode(out: Path, reps: int):
    em = Emitter(out / "decode" / "supervisor.jsonl", "decode", "decode", apc=False)
    srv = Server(out / "decode", apc=False, em=em)
    srv.start()
    wk = worker("bench-client", 1)
    sampler = GpuSampler(em, wk)
    rows = []
    try:
        warmup()
        # Fixed-length decode: pure decode speed at two context lengths.
        for rep in range(reps):
            for L in (2048, 16384):
                times, _, _ = stream_completion(rand_tokens(L, f"decode-{L}-{rep}"), 512)
                dt = (times[-1] - times[0]) / 1e9
                row = {"kind": "fixed", "rep": rep, "context_tokens": L, "gen_tokens": len(times),
                       "decode_tok_s": (len(times) - 1) / dt, "ttft_s": None}
                rows.append(row)
                print("decode", row, flush=True)
        # Natural generation with thinking on and off: typical lengths for the agent workload.
        for rep in range(reps):
            for think in (False, True):
                msgs = [{"role": "user", "content": AGENT_PROMPTS[rep % len(AGENT_PROMPTS)]}]
                t0 = time.monotonic_ns()
                times, usage, finish = chat(msgs, think, 8192, SEED + rep)
                n = (usage or {}).get("completion_tokens", len(times))
                dt = (times[-1] - times[0]) / 1e9 if len(times) > 1 else float("nan")
                row = {"kind": "chat", "rep": rep, "thinking": think, "gen_tokens": n, "finish": finish,
                       "decode_tok_s": (len(times) - 1) / dt, "ttft_s": (times[0] - t0) / 1e9,
                       "total_s": (times[-1] - t0) / 1e9}
                rows.append(row)
                print("decode", row, flush=True)
    finally:
        sampler.stop()
        srv.stop()
    _dump(out / "decode" / "summary.json", rows)


# --------------------------------------------------------------------------- agent loop


class InferenceCrash(Exception):
    pass


def run_agent(log: Path, em: Emitter, wk: dict, task: str, steps: int, prefix_tokens: int, gen_tokens: int,
              tool_tokens: int, fault: str, fault_step: int, fault_after: int, server: Server | None = None,
              results: list | None = None):
    """Scripted 10-step agent loop over token ids with a durable JSONL log.

    Each step: generate gen_tokens (decode), then append a synthetic tool result of tool_tokens.
    CONTEXT_COMMITTED is fsynced after each step. On start the log is replayed (reconcile);
    the first generation after recovery is timed as re-prefill and next token."""
    t_rec0 = None
    committed: list[int] = []
    done = 0
    if log.exists():
        # the pending step is unknown until the log is read
        t_rec0 = em.emit("RECONCILIATION_STARTED", wk, "recovery")
        for line in log.open():
            rec = json.loads(line)
            if rec["type"] == "PREFIX":
                committed = list(rec["token_ids"])
            elif rec["type"] == "CONTEXT_COMMITTED":
                committed += rec["gen_ids"] + rec["tool_ids"]
                done = rec["step"]
        # A crash during generation leaves no STEP_PREPARED: nothing was dispatched.
        em.emit("RECONCILIATION_FINISHED", wk, f"step-{done + 1:02d}",
                {"outcome": "NOT_COMMITTED", "operation_id": f"{task}:step-{done + 1:02d}"})
    else:
        committed = rand_tokens(prefix_tokens, f"{task}-prefix")
        _append(log, {"type": "PREFIX", "token_ids": committed})

    recovering = t_rec0 is not None
    step = done + 1
    while step <= steps:
        sid = f"step-{step:02d}"
        inj = f"{task}:{sid}:{fault}"

        def on_token(i, sid=sid, inj=inj, step=step):
            if fault != "none" and step == fault_step and i == fault_after:
                em.emit("FAULT_INJECTED", wk, sid, {"boundary": "DURING_GENERATION", "action": "TERMINATE_PROCESS",
                                                    "target": fault}, injection_id=inj)
                if fault == "kill-runner":
                    os.kill(os.getpid(), signal.SIGKILL)
                server.kill()

        if recovering:
            em.emit("PREFILL_STARTED", wk, sid, {"committed_prefix_tokens": len(committed)})
        try:
            times, ids, _ = stream_completion(committed, gen_tokens, on_token)
        except (ConnectionError, OSError, http.client.HTTPException) as e:
            if fault != "kill-inference" or server is None:
                raise
            em.emit("CRASH_DETECTED", wk, sid, {"error": type(e).__name__, "target": "inference"})
            # Supervisor role: wait for the dead container to be reaped, then start a fresh worker.
            server.stop()
            server.start(step=sid)
            if results is not None:
                results.append({"phase": "restart", "phases": server.log_phases()})
            t_rec0 = em.emit("RECONCILIATION_STARTED", wk, sid)
            em.emit("RECONCILIATION_FINISHED", wk, sid,
                    {"outcome": "NOT_COMMITTED", "operation_id": f"{task}:{sid}"})
            recovering = True
            fault = "none"
            continue
        if recovering:
            em.emit("PREFILL_FINISHED", wk, sid, {"committed_prefix_tokens": len(committed)}, mono=times[0])
            em.emit("GENERATION_RESUMED", wk, sid, mono=times[0])
            em.emit("NEXT_TOKEN_EMITTED", wk, sid, mono=times[0])
            recovering = False
        gen = ids if ids and len(ids) == gen_tokens else rand_tokens(gen_tokens, f"{task}-gen-{step}")
        tool = rand_tokens(tool_tokens, f"{task}-tool-{step}")
        _append(log, {"type": "CONTEXT_COMMITTED", "step": step, "gen_ids": gen, "tool_ids": tool})
        committed += gen + tool
        step += 1


def _append(log: Path, rec: dict):
    with log.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


AGENT = dict(steps=10, prefix_tokens=8192, gen_tokens=128, tool_tokens=256, fault_step=6, fault_after=64)


def exp_kill_inference(out: Path, reps: int):
    base = out / "kill-inference"
    em = Emitter(base / "supervisor.jsonl", "kill-inference", "agent", apc=False)
    srv = Server(base, apc=False, em=em)
    srv.start()
    warmup()
    sampler = GpuSampler(em, worker("bench-client", 1))
    try:
        for rep in range(reps):
            task = f"agent-{rep:02d}"
            run_em = Emitter(base / task / "runner.jsonl", f"kill-inference-{rep:02d}", task, apc=False)
            srv.em = run_em
            run_agent(base / task / "durable-log.jsonl", run_em, worker("runner", 1), task, fault="kill-inference",
                      server=srv, **AGENT)
            print("kill-inference done", rep, flush=True)
    finally:
        sampler.stop()
        srv.stop()


def exp_kill_runner(out: Path, reps: int, apc: bool):
    base = out / f"kill-runner-apc-{'on' if apc else 'off'}"
    em = Emitter(base / "supervisor.jsonl", base.name, "agent", apc=apc)
    srv = Server(base, apc=apc, em=em)
    srv.start()
    warmup()
    sampler = GpuSampler(em, worker("bench-client", 1))
    try:
        for rep in range(reps):
            task = f"agent-{rep:02d}"
            run_id = f"{base.name}-{rep:02d}"
            sup = Emitter(base / task / "supervisor.jsonl", run_id, task, apc=apc)
            for attempt in (1, 2):
                wk = worker("runner", attempt)
                cmd = [sys.executable, __file__, "runner", "--out", str(base / task), "--run-id", run_id,
                       "--task", task, "--attempt", str(attempt), "--apc", str(int(apc)),
                       "--fault", "kill-runner" if attempt == 1 else "none"]
                sup.emit("WORKER_START_REQUESTED", wk, details={"target": "runner"})
                p = subprocess.Popen(cmd)
                rc = p.wait()
                if attempt == 1:
                    sup.emit("PROCESS_EXITED", wk, details={"returncode": rc})
                    sup.emit("CRASH_DETECTED", wk, details={"target": "runner", "returncode": rc})
                    if rc != -signal.SIGKILL:
                        raise RuntimeError(f"runner attempt 1 exited {rc}, expected SIGKILL")
                elif rc != 0:
                    raise RuntimeError(f"runner attempt 2 failed {rc}")
            print("kill-runner done", apc, rep, flush=True)
    finally:
        sampler.stop()
        srv.stop()


def cmd_runner(a):
    out = Path(a.out)
    wk = worker("runner", a.attempt)
    em = Emitter(out / f"runner-attempt-{a.attempt}.jsonl", a.run_id, a.task, apc=bool(a.apc))
    em.emit("WORKER_PROCESS_STARTED", wk, details={"pid": os.getpid()})
    run_agent(out / "durable-log.jsonl", em, wk, a.task, fault=a.fault, **AGENT)


def _dump(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("coldstart", "prefill", "decode", "kill-inference", "kill-runner", "all"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--reps", type=int, default=5)
    r = sub.add_parser("runner")
    for k in ("--out", "--run-id", "--task", "--fault"):
        r.add_argument(k, required=True)
    r.add_argument("--attempt", type=int, required=True)
    r.add_argument("--apc", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "runner":
        return cmd_runner(a)
    out = Path(a.out)
    if a.cmd in ("coldstart", "all"):
        exp_coldstart(out, a.reps)
    if a.cmd in ("prefill", "all"):
        exp_prefill(out, a.reps, apc=False)
        exp_prefill(out, a.reps, apc=True)
    if a.cmd in ("decode", "all"):
        exp_decode(out, a.reps)
    if a.cmd in ("kill-runner", "all"):
        exp_kill_runner(out, a.reps, apc=False)
        exp_kill_runner(out, a.reps, apc=True)
    if a.cmd in ("kill-inference", "all"):
        exp_kill_inference(out, a.reps)


if __name__ == "__main__":
    main()

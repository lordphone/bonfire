"""Independent-process CPU/RSS sampling, with optional real NVML GPU samples."""

import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def process_sample(pid):
    """Return lifetime CPU seconds, current RSS and a PID-reuse-safe identity."""
    try:
        if os.name != "nt":
            data = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return {"pid": pid, "identity": f"{pid}:{data[19]}",
                    "cpu_seconds": (int(data[11]) + int(data[12])) / os.sysconf("SC_CLK_TCK"),
                    "rss_bytes": int(data[21]) * os.sysconf("SC_PAGE_SIZE")}
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "paged_peak", "paged",
                                                   "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(FileTime)] * 4
        memory_api = ctypes.WinDLL("psapi", use_last_error=True)
        memory_api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            created, exited, system, user = FileTime(), FileTime(), FileTime(), FileTime()
            memory = Memory()
            memory.cb = ctypes.sizeof(memory)
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                return None
            if not memory_api.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                return None
            ticks = lambda value: (value.high << 32) | value.low
            return {"pid": pid, "identity": f"{pid}:{ticks(created)}",
                    "cpu_seconds": (ticks(system) + ticks(user)) / 10_000_000,
                    "rss_bytes": int(memory.working)}
        finally:
            kernel.CloseHandle(handle)
    except (OSError, ValueError, IndexError):
        return None


def _monitor(queue, output, interval, gpu_indices, start_ns, ready):
    import queue as queue_module
    output = Path(output)
    watched, last, peak_rss, count, missing, baselines, identities = {}, {}, 0, 0, set(), {}, {}
    gpu, gpu_error, peak_gpu, gpu_identity = None, None, None, []
    if gpu_indices:
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu = [(index, pynvml.nvmlDeviceGetHandleByIndex(index)) for index in gpu_indices]
            def value(item):
                return item.decode() if isinstance(item, bytes) else item
            gpu_identity = [{"index": index, "name": value(pynvml.nvmlDeviceGetName(handle)),
                             "uuid": value(pynvml.nvmlDeviceGetUUID(handle)),
                             "driver": value(pynvml.nvmlSystemGetDriverVersion())} for index, handle in gpu]
        except Exception as exc:
            gpu_error = str(exc)
    stopped = False
    ready.set()
    while not stopped:
        try:
            message = queue.get(timeout=interval)
            if message["action"] == "stop":
                stopped, stop_ns = True, message["stop_ns"]
            else:
                watched[message["pid"]] = message["role"]
                identities.pop(message["pid"], None)
                if message.get("baseline"):
                    baselines[message["baseline"]["identity"]] = message["baseline"]["cpu_seconds"]
        except queue_module.Empty:
            pass
        if stopped:
            break  # keep last in-window samples; never sample after verification
        samples = []
        from outcomeserve.processes import owned_descendants
        for pid in list(watched):
            current = process_sample(pid)
            if current is not None and pid in identities and current["identity"] != identities[pid]:
                del watched[pid]
        watched = owned_descendants(watched)
        for pid, role in list(watched.items()):
            sample = process_sample(pid)
            if sample:
                if pid in identities and sample["identity"] != identities[pid]:
                    continue
                identities[pid] = sample["identity"]
                sample["role"] = role
                last[sample["identity"]] = sample
                samples.append(sample)
            elif not any(value["pid"] == pid for value in last.values()):
                missing.add(pid)
        missing.difference_update(value["pid"] for value in samples)
        peak_rss = max(peak_rss, sum(value["rss_bytes"] for value in samples))
        gpu_samples = []
        if gpu is not None:
            try:
                for index, handle in gpu:
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_samples.append({"index": index, "memory_used_bytes": memory.used,
                                        "utilization_pct": utilization.gpu})
                peak_gpu = max(peak_gpu or 0, sum(sample["memory_used_bytes"] for sample in gpu_samples))
            except Exception as exc:
                gpu_error = str(exc)
        with (output / "resource-samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"monotonic_ns": time.perf_counter_ns(), "processes": samples, "gpus": gpu_samples}) + "\n")
        count += 1
    summary = {"schema_version": "1.0.0", "sampling_interval_seconds": interval,
               "clock_source": "perf_counter", "clock_resolution_seconds": time.get_clock_info("perf_counter").resolution,
               "window_start_ns": start_ns, "window_stop_ns": stop_ns,
               "sample_count": count, "process_cpu_seconds": sum(max(0, value["cpu_seconds"] - baselines.get(identity, 0)) for identity, value in last.items()),
               "peak_summed_rss_bytes": peak_rss, "peak_gpu_memory_bytes": peak_gpu,
               "gpu_allocation_seconds": len(gpu_indices) * (stop_ns - start_ns) / 1e9 if gpu_indices else None,
               "gpu_error": gpu_error, "unsampled_pids": sorted(missing), "last_process_samples": list(last.values()),
               "gpu_identity": gpu_identity,
               "measurement_valid": bool(watched) and not missing and gpu_error is None,
               "limitations": ["last samples may miss CPU before abrupt exit", "summed RSS may double-count shared pages",
                               "allocation-seconds are reserved time, not active GPU cycles"]}
    (output / "resources.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if gpu is not None:
        pynvml.nvmlShutdown()


class ResourceMonitor:
    def __init__(self, output, *, interval=0.05, gpu_indices=(), start_ns=None):
        if interval <= 0 or len(set(gpu_indices)) != len(gpu_indices):
            raise ValueError("positive interval and distinct exclusive GPU indices required")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.context = mp.get_context("spawn")
        self.queue = self.context.Queue()
        ready = self.context.Event()
        self.start_ns = start_ns if start_ns is not None else time.perf_counter_ns()
        self.process = self.context.Process(target=_monitor, args=(self.queue, str(self.output), interval, tuple(gpu_indices), self.start_ns, ready))
        self.process.start()
        if not ready.wait(15):
            self.process.terminate()
            self.process.join(timeout=5)
            raise RuntimeError("resource monitor did not become ready")

    def watch(self, pid, role, *, existing=False):
        self.queue.put({"action": "watch", "pid": pid, "role": role,
                        "baseline": process_sample(pid) if existing else None})

    def stop(self, *, stop_ns=None):
        self.queue.put({"action": "stop", "stop_ns": stop_ns if stop_ns is not None else time.perf_counter_ns()})
        self.process.join(timeout=15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            raise RuntimeError("resource monitor failed to stop")
        self.queue.close()
        self.queue.join_thread()
        if self.process.exitcode != 0:
            raise RuntimeError("resource monitor failed")
        return json.loads((self.output / "resources.json").read_text(encoding="utf-8"))

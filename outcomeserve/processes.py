"""Inspect/terminate only process trees created by the experiment supervisor."""

import ctypes
import os
from pathlib import Path
import subprocess


def process_parents():
    if os.name != "nt":
        result = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    result[int(entry.name)] = int(fields[1])
                except (OSError, ValueError, IndexError):
                    pass
        return result
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD),
                    ("exe", wintypes.WCHAR * 260)]
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError("cannot enumerate owned process descendants")
    result, entry = {}, Entry()
    entry.size = ctypes.sizeof(entry)
    try:
        success = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while success:
            result[int(entry.pid)] = int(entry.parent)
            success = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return result


def owned_descendants(roots):
    result = dict(roots)
    parents = process_parents()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result[pid] = result[parent]
                changed = True
    return result


def kill_owned(process):
    if not isinstance(process, subprocess.Popen):
        raise TypeError("an owned Popen handle is required")
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Windows venv python.exe may be a redirector; killing only its PID
        # leaves the actual interpreter alive. taskkill /T targets this owned tree.
        killer = subprocess.Popen(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, error = killer.communicate(timeout=10)
        if killer.returncode and process.poll() is None:
            raise RuntimeError(error.decode(errors="replace"))
    else:
        # CPU workers are single-process on Unix. Owned inference descendants are
        # terminated before their parent; never signal the caller's process group.
        for pid in reversed(list(owned_descendants({process.pid: "owned"}))):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
    process.wait(timeout=10)

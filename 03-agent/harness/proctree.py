"""Start a process so that its whole tree can be killed, even if the controller dies first (phase 4).

Run 1 of the agent test lost its watchdog together with the gateway and the agent kept running. Here:
  * Windows: the child is put in a job object with KILL_ON_JOB_CLOSE, so the tree dies when this process dies as well, and
    `kill()` terminates the job (every descendant, however deep). If the job cannot be used, `taskkill /T` is the fallback.
  * Linux/macOS: the child leads its own session and process group; `kill()` signals the group.
Only the tree that this object started is ever touched; nothing is looked up by image name.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID]

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class _Extended(ctypes.Structure):
        _fields_ = [("Basic", _Basic), ("Io", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _Accounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64), ("ThisPeriodTotalUserTime", ctypes.c_int64),
                    ("ThisPeriodTotalKernelTime", ctypes.c_int64), ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]


class ProcTree:
    def __init__(self, argv: list[str], cwd: str, env: dict[str, str], stdout: Any, stderr: Any = subprocess.STDOUT):
        kw: dict[str, Any] = {"cwd": cwd, "env": env, "stdin": subprocess.DEVNULL, "stdout": stdout, "stderr": stderr}
        self.job = None
        self.job_ok = False
        if IS_WINDOWS:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        self.proc = subprocess.Popen(argv, **kw)
        if IS_WINDOWS:
            self._assign_job()

    def _assign_job(self) -> None:
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _Extended()
        info.Basic.LimitFlags = 0x2000                     # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if _k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) and \
                _k32.AssignProcessToJobObject(job, int(self.proc._handle)):  # noqa: SLF001
            self.job, self.job_ok = job, True
        else:
            _k32.CloseHandle(job)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> int | None:
        return self.proc.poll()

    def active_processes(self) -> int | None:
        """How many processes of the tree are still alive (Windows job accounting), None where not known."""
        if IS_WINDOWS and self.job_ok:
            acc = _Accounting()
            if _k32.QueryInformationJobObject(self.job, 1, ctypes.byref(acc), ctypes.sizeof(acc), None):
                return int(acc.ActiveProcesses)
        return None

    def wait(self, timeout: float, poll: float = 1.0) -> bool:
        """True if the main process exited within `timeout` seconds."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                return True
            time.sleep(min(poll, max(0.0, end - time.monotonic())))
        return self.proc.poll() is not None

    def kill(self) -> None:
        """Kill the whole tree (and wait briefly for the main process)."""
        if IS_WINDOWS:
            if self.job_ok:
                _k32.TerminateJobObject(self.job, 1)
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True, timeout=30)
        else:
            import signal
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        if IS_WINDOWS and self.job:
            _k32.CloseHandle(self.job)
            self.job = None

#!/usr/bin/env python3
"""Host driver for the REA benchmark (phase 6 part C): one `forge serve` on config A, then arm R and arm O against it, then stop.

    python rea_grid.py --arms R,O --out C:\\Users\\arthu\\llm\\phase6\\results-rea --repo C:\\Users\\arthu\\llm\\forge\\repo --log phase6-rea.log

The server is started through the planner exactly as in phase 3/4 (config A: tuned layers, 16K, thinking off); only its own process tree is
killed. While the arms run, a sampler appends free RAM, pages/sec, page faults/sec, pagefile use and the working sets of llama-server, java and the VM to `<out>/mem.log` every 30 s (WMI), because
REA's Ghidra (a JVM) runs next to the model server on a 16 GB machine. Stop between tasks with a file named STOP in the output directory.
The log ends with an `exit=<code> <timestamp>` line, which the other host jobs read as "finished".
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quality"))
import qgrid  # noqa: E402

IS_WINDOWS = os.name == "nt"


MEM_PS = ('$m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; $pf = Get-CimInstance Win32_PageFileUsage; '
          'function W($n) { $p = Get-Process $n -ErrorAction SilentlyContinue; if ($p) { [int](($p | Measure-Object WorkingSet64 -Sum).Sum / 1MB) } else { 0 } }; '
          '"avail_mb=$($m.AvailableMBytes) pages_s=$($m.PagesPersec) faults_s=$($m.PageFaultsPersec) pagefile_used_mb=$($pf.CurrentUsage) '
          'llama_ws_mb=$(W \'llama-server\') java_ws_mb=$(W \'java\') vm_ws_mb=$(W \'VirtualBoxVM\')"')


def sample_memory(path: Path, stop: threading.Event, every: float = 30.0) -> None:
    """One line per sample: free RAM, pages/sec, page faults/sec, pagefile use, working sets of llama-server, java (Ghidra) and the VM. WMI, not
    typeperf: the host is pt-BR and its performance counters have localized names."""
    while not stop.is_set():
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", MEM_PS], capture_output=True, text=True, timeout=60)
            out = r.stdout.strip().splitlines()
            line += out[-1] if out and "avail_mb" in out[-1] else "no sample"
        except (OSError, subprocess.SubprocessError):
            line += "sampler failed"
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        stop.wait(every)


def runner_cmd(arm: str, a: argparse.Namespace, out: Path) -> list[str]:
    cmd = [a.python, "-u", str(HERE / "rea_bench.py"), "run", "--arm", arm, "--endpoint", f"http://127.0.0.1:{a.port}/v1", "--out", str(out / f"{arm}.jsonl"),
           "--runs", str(a.runs), "--time-limit", str(a.time_limit), "--stop-file", str(out / "STOP"), "--rea-root", str(a.rea_root),
           "--work-dir", str(a.work_dir), "--python", a.python, "--opencode", a.opencode, "--note", f"phase 6 REA benchmark, arm {arm}, config A"]
    if a.binary:
        cmd += ["--binary", str(a.binary)]
    if a.gguf:
        cmd += ["--gguf", str(a.gguf)]
    if a.tasks:
        cmd += ["--tasks", a.tasks]
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", default="R,O")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=HERE.parent.parent)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--health-timeout", type=float, default=600.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--binary", type=Path, default=None)
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--rea-root", type=Path, default=Path(r"C:\Users\arthu\llm\rea"))
    ap.add_argument("--opencode", default=r"C:\Users\arthu\.opencode\bin\opencode.exe")
    ap.add_argument("--work-dir", type=Path, default=Path(r"C:\Users\arthu\llm\phase6\work-rea"))
    ap.add_argument("--log", type=Path)
    a = ap.parse_args(argv)
    qgrid.LOG_PATH = a.log
    a.out.mkdir(parents=True, exist_ok=True)
    if qgrid.running("llama-server") or qgrid.running("llama-bench"):
        qgrid.log("ABORT: a llama process is running")
        return 3
    qgrid.unload_ollama()
    arm_cfg = {"id": "A", "serve": [], "title": "config A"}
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    serve_out = a.out / "A.serve.out"
    cmd = qgrid.forge_command(arm_cfg, a.port, a.python, a.health_timeout)
    qgrid.log(f"forge: {' '.join(cmd)}")
    kw: dict = {"cwd": str(a.repo), "env": env, "stdin": subprocess.DEVNULL}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    worst = 0
    stop = threading.Event()
    with serve_out.open("wb") as fo:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, **kw)
        try:
            deadline = time.monotonic() + a.health_timeout * 3
            info = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    qgrid.log(f"forge serve exited with {proc.returncode} before it was healthy")
                    break
                info = qgrid.parse_serve_json(serve_out.read_text(encoding="utf-8", errors="replace"))
                if info and qgrid.http_ok(f"http://127.0.0.1:{a.port}/health"):
                    break
                time.sleep(2)
            else:
                info = None
            if not info:
                qgrid.log("server FAILED to start")
                return 4
            import json
            (a.out / "A.serve.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            qgrid.log(f"server up: ngl={info['config']['ngl']} ctx={info['config']['ctx']} ub={info['config']['ubatch']} source={info['source']}")
            sampler = threading.Thread(target=sample_memory, args=(a.out / "mem.log", stop), daemon=True)
            sampler.start()
            for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
                rc_cmd = runner_cmd(arm, a, a.out)
                qgrid.log(f"ARM {arm}: {' '.join(rc_cmd)}")
                logf = a.log.open("ab") if a.log else None
                rc = subprocess.run(rc_cmd, env=env, stdin=subprocess.DEVNULL, stdout=logf or None, stderr=subprocess.STDOUT).returncode
                if logf:
                    logf.close()
                qgrid.log(f"ARM {arm} rc={rc}")
                worst = max(worst, rc)
        finally:
            stop.set()
            qgrid.kill_tree(proc)
            time.sleep(3)
            qgrid.log(f"stopped; llama-server processes on the host now, any owner: {qgrid.running('llama-server')}")
    return worst


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        qgrid.emit(f"exit={code} {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    raise SystemExit(code)

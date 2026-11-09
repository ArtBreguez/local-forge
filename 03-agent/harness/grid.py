#!/usr/bin/env python3
"""Run the phase 4 grid on the machine that holds the model (the Windows host): servers, arms, tasks, runs. Stdlib only.

    python grid.py --stages "O+T@1;T-norepair@1;O+T@2;T-ngram@1" --out-dir C:\\...\\results --work-dir C:\\...\\work --repo C:\\...\\forge\\repo

A stage is `arm+arm@run` (all tasks, in a fixed shuffled order, one arm after the other for each task, so the arms see the same
machine state). `forge serve` is started once per distinct server configuration (config A, or A + `--spec ngram-mod`) and killed by
its own PID tree when the configuration changes and at the end; the runner starts one agent at a time. Ends with an `exit=<code>` line,
which is also what the other host jobs read as "the GPU is free". A run already in `<out-dir>/<arm>.jsonl` is skipped, so the same
command resumes after a kill. `--budget-min` stops launching new runs after that many minutes; `--stop-file` stops the same way when the
file appears (a gentle brake: the running agent finishes).
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "quality"))
import qgrid  # noqa: E402  (parse_serve_json, kill_tree, running, http_ok, unload_ollama)
import runner  # noqa: E402
import taskbench  # noqa: E402

IS_WINDOWS = sys.platform.startswith("win")
LOG: Path | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG is not None:
        with LOG.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def parse_stages(text: str) -> list[tuple[list[str], int]]:
    stages = []
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        arms, _, run = part.partition("@")
        names = [x.strip() for x in arms.split("+") if x.strip()]
        bad = [n for n in names if n not in runner.ARMS]
        if bad or not run.isdigit():
            raise SystemExit(f"bad stage {part!r}: arms {sorted(runner.ARMS)}, '@<run number>'")
        stages.append((names, int(run)))
    return stages


class Server:
    def __init__(self, a: argparse.Namespace, env: dict[str, str]):
        self.a, self.env = a, env
        self.proc: subprocess.Popen | None = None
        self.flags: list[str] | None = None

    def healthy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and qgrid.http_ok(f"http://127.0.0.1:{self.a.port}/health")

    def start(self, flags: list[str]) -> bool:
        self.stop()
        out_dir: Path = self.a.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "A" if not flags else "A+" + "".join(flags).replace("--", "").replace("-", "")
        if qgrid.running("llama-server"):
            log("ABORT: a llama-server is already running (not ours to touch)")
            return False
        cmd = qgrid.forge_command({"serve": flags}, self.a.port, self.a.python, self.a.health_timeout)
        log(f"server {tag}: {' '.join(cmd)}")
        kw: dict = {"cwd": str(self.a.repo), "env": self.env, "stdin": subprocess.DEVNULL}
        if IS_WINDOWS:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        serve_out = out_dir / f"serve-{tag}.out"
        with serve_out.open("wb") as fo:
            self.proc = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, **kw)
        deadline = time.monotonic() + self.a.health_timeout * 3
        info = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                log(f"server {tag}: forge serve exited with {self.proc.returncode} before it was healthy")
                return False
            info = qgrid.parse_serve_json(serve_out.read_text(encoding="utf-8", errors="replace"))
            if info and qgrid.http_ok(f"http://127.0.0.1:{self.a.port}/health"):
                break
            time.sleep(2)
        else:
            log(f"server {tag}: not healthy in time; see {serve_out}")
            return False
        c = info["config"]
        log(f"server {tag} up: ngl={c['ngl']} ctx={c['ctx']} fa={c['fa']} kv={c['kv']} ub={c['ubatch']} spec={c.get('spec_type')} source={info['source']}")
        (out_dir / f"serve-{tag}.json").write_text(__import__("json").dumps(info, indent=2), encoding="utf-8")
        self.flags = flags
        return True

    def stop(self) -> None:
        if self.proc is not None:
            qgrid.kill_tree(self.proc)           # forge serve and its llama-server child, by PID tree; never by image name
            time.sleep(3)
            log(f"server stopped; llama-server processes on the host now, any owner: {qgrid.running('llama-server')}")
        self.proc, self.flags = None, None


def runner_cmd(a: argparse.Namespace, arm: str, task: str, run: int) -> list[str]:
    cmd = [a.python, str(HERE / "runner.py"), "--arm", arm, "--tasks", task, "--runs", str(run), "--endpoint", f"http://127.0.0.1:{a.port}/v1",
           "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir), "--time-limit", str(a.time_limit), "--python", a.python,
           "--shuffle-seed", "0", "--note", a.note]
    if a.node:
        cmd += ["--node", a.node]
    if a.opencode:
        cmd += ["--opencode", a.opencode]
    if a.oc_cache_seed:
        cmd += ["--oc-cache-seed", a.oc_cache_seed]
    return cmd


def main(argv: list[str] | None = None) -> int:
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", required=True)
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=HERE.parent.parent)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--node", default=None)
    ap.add_argument("--opencode", default=None)
    ap.add_argument("--oc-cache-seed", default=None)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--health-timeout", type=float, default=600.0)
    ap.add_argument("--time-limit", type=float, default=runner.DEFAULT_LIMIT_S)
    ap.add_argument("--shuffle-seed", type=int, default=4)
    ap.add_argument("--budget-min", type=float, default=0.0, help="stop launching runs after this many minutes (0: no limit)")
    ap.add_argument("--stop-file", type=Path, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--log", type=Path)
    a = ap.parse_args(argv)
    LOG = a.log
    stages = parse_stages(a.stages)
    tasks = [t.id for t in taskbench.load_tasks(ids=[x for x in a.tasks.split(",") if x] or None)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for k in ("FORGE_HOME", "FORGE_LLAMA_DIR", "FORGE_MODEL"):
        if not env.get(k):
            log(f"warning: {k} is not set")
    if qgrid.running("llama-server") or qgrid.running("llama-bench"):
        log("ABORT: a llama process is running")
        return 3
    qgrid.unload_ollama()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    srv = Server(a, env)
    t0 = time.monotonic()
    worst = 0
    skipped: list[str] = []
    try:
        for si, (arms, run) in enumerate(stages):
            order = list(tasks)
            if a.shuffle_seed:
                random.Random(a.shuffle_seed + run).shuffle(order)
            for task in order:
                for arm in arms:
                    flags = runner.ARMS[arm].get("serve", [])
                    stop = (a.budget_min and (time.monotonic() - t0) / 60 > a.budget_min) or (a.stop_file and a.stop_file.exists())
                    if stop:
                        skipped.append(f"{arm}:{task}:{run}")
                        continue
                    if srv.flags != flags or not srv.healthy():
                        if not srv.start(flags):
                            log("server did not start; stopping the grid")
                            return 4
                    log(f"RUN {arm} {task} run {run} (stage {si + 1}/{len(stages)}, {(time.monotonic() - t0) / 60:.0f} min in)")
                    cmd = runner_cmd(a, arm, task, run)
                    if LOG is not None:
                        with LOG.open("ab") as lf:
                            rc = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT).returncode
                    else:
                        rc = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL).returncode
                    log(f"RUN {arm} {task} run {run} runner rc={rc}")
                    worst = max(worst, rc)
    finally:
        srv.stop()
        if skipped:
            log(f"NOT RUN (budget or stop file): {len(skipped)} runs: {', '.join(skipped[:60])}")
    return worst


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        line = f"exit={code} {time.strftime('%Y-%m-%dT%H:%M:%S')}"
        print(line, flush=True)
        if LOG is not None:
            with LOG.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
    raise SystemExit(code)

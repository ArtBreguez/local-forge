#!/usr/bin/env python3
"""Run the arms of `arms.json` one after another on the machine that holds the model (the Windows host).

For every arm: start a fresh `python -m forge serve <arm flags>` (so the planner places the layers for that arm), wait until the
server is healthy, save what `forge serve` printed (the exact llama-server command line, the chosen config) to `<arm>.serve.json`,
run `run_quality.py` against it, then kill the whole process tree and check that no llama-server is left.

    python qgrid.py --arms A,B,C --out C:\\Users\\arthu\\llm\\phase3\\results --repo C:\\Users\\arthu\\llm\\forge\\repo

Environment for forge: FORGE_HOME, FORGE_LLAMA_DIR, FORGE_MODEL (as in `forge serve`). Meant to run inside a scheduled task; it logs
every step to stdout and ends with an `exit=<code> <timestamp>` line, which is also what the other host jobs use as "finished".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import qbench

HERE = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform.startswith("win")


LOG_PATH: Path | None = None       # set by --log: every line is also appended there (UTF-8), and the runner's output goes there too


def emit(line: str) -> None:
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def log(msg: str) -> None:
    emit(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}")


def load_arms(path: Path = HERE / "arms.json") -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    return {a["id"]: {**defaults, **a} for a in data["arms"]}


def parse_serve_json(text: str) -> dict[str, Any] | None:
    """The JSON object `forge serve` prints once the server is healthy: it starts at the last line that is exactly `{` with no
    indentation (the nested objects inside it, e.g. the `attempts` entries, are indented `{` lines of their own)."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].rstrip() == "{":
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                return None
    return None


REASONING_MARGIN = 128      # tokens on top of the reasoning budget for the injected message and the end-of-thinking tag


def arm_server_env(arm: dict[str, Any]) -> dict[str, str]:
    """Environment variables for the llama-server of this arm. llama-server reads its flags from LLAMA_ARG_* variables as well, and
    `forge serve` has no pass-through for flags it does not plan: `--reasoning-budget N` is LLAMA_ARG_THINK_BUDGET, and its message
    LLAMA_ARG_THINK_BUDGET_MESSAGE (injected before the end-of-thinking tag when the budget runs out)."""
    env: dict[str, str] = {}
    if arm.get("reasoning_budget"):
        env["LLAMA_ARG_THINK_BUDGET"] = str(arm["reasoning_budget"])
        if arm.get("reasoning_budget_message"):
            env["LLAMA_ARG_THINK_BUDGET_MESSAGE"] = arm["reasoning_budget_message"]
    return env


def forge_command(arm: dict[str, Any], port: int, python: str, health_timeout: float = 600.0) -> list[str]:
    return [python, "-m", "forge", "serve", "--port", str(port), "--health-timeout", f"{health_timeout:g}", *arm["serve"]]


def runner_command(arm: dict[str, Any], port: int, python: str, out_dir: Path, node: str | None, max_problems: int = 0, problems: str = "") -> list[str]:
    cmd = [python, str(HERE / "run_quality.py"), "--endpoint", f"http://127.0.0.1:{port}/v1", "--arm", arm["id"],
           "--out", str(out_dir / f"{arm['id']}.jsonl"), "--thinking", arm["thinking"],
           "--think-budget", str(arm["reasoning_budget"] + REASONING_MARGIN if arm.get("reasoning_budget") else arm["think_budget"]),
           "--temperature", str(arm["temperature"]), "--samples", str(arm["samples"]), "--serve-json", str(out_dir / f"{arm['id']}.serve.json"),
           "--note", arm["title"], "--prompt-variant", arm.get("prompt_variant", qbench.DEFAULT_VARIANT)]
    if arm.get("reasoning_budget_message"):
        cmd += ["--budget-message", arm["reasoning_budget_message"]]
    if node:
        cmd += ["--node", node]
    if max_problems:
        cmd += ["--max-problems", str(max_problems)]
    if problems:
        cmd += ["--problems", problems]
    return cmd


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=60)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def running(image: str) -> int:
    """How many processes with this image name are running (llama-server.exe / llama-server)."""
    try:
        if IS_WINDOWS:
            r = subprocess.run(["tasklist", "/NH", "/FO", "CSV", "/FI", f"IMAGENAME eq {image}.exe"], capture_output=True, text=True, timeout=60)
            return sum(1 for line in r.stdout.splitlines() if line.lower().startswith(f'"{image}.exe"'))
        r = subprocess.run(["pgrep", "-x", image], capture_output=True, text=True, timeout=30)
        return len(r.stdout.split())
    except (OSError, subprocess.SubprocessError):
        return -1


def kill_image(image: str) -> None:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/IM", f"{image}.exe"], capture_output=True, timeout=60)
        else:
            subprocess.run(["pkill", "-9", "-x", image], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local URL
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def unload_ollama() -> None:
    """One model at a time: ask Ollama to drop whatever it holds (keep_alive 0). Silent when Ollama is not running."""
    base = "http://127.0.0.1:11434"
    try:
        with urllib.request.urlopen(base + "/api/ps", timeout=5) as r:  # noqa: S310
            models = json.loads(r.read().decode("utf-8")).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return
    for m in models:
        body = json.dumps({"model": m["name"], "keep_alive": 0}).encode("utf-8")
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=60).read()  # noqa: S310
            log(f"unloaded Ollama model {m['name']}")
        except (urllib.error.URLError, OSError):
            log(f"could not unload Ollama model {m['name']}")


def run_arm(arm: dict[str, Any], a: argparse.Namespace, env: dict[str, str]) -> int:
    out_dir: Path = a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    serve_out = out_dir / f"{arm['id']}.serve.out"
    log(f"ARM {arm['id']} start: {arm['title']}")
    if running("llama-server"):
        log(f"ARM {arm['id']} ABORT: a llama-server is already running")
        return 3
    cmd = forge_command(arm, a.port, a.python, a.health_timeout)
    log(f"ARM {arm['id']} forge: {' '.join(cmd)}")
    server_env = {**env, **arm_server_env(arm)}
    if arm_server_env(arm):
        log(f"ARM {arm['id']} server environment: {arm_server_env(arm)}")
    kw: dict[str, Any] = {"cwd": str(a.repo), "env": server_env, "stdin": subprocess.DEVNULL}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    rc = 1
    with serve_out.open("wb") as fo:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, **kw)
        try:
            deadline = time.monotonic() + a.health_timeout * 3
            info = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    log(f"ARM {arm['id']} forge serve exited with {proc.returncode} before it was healthy")
                    break
                info = parse_serve_json(serve_out.read_text(encoding="utf-8", errors="replace"))
                if info and http_ok(f"http://127.0.0.1:{a.port}/health"):
                    break
                time.sleep(2)
            else:
                info = None
            if not info:
                log(f"ARM {arm['id']} FAILED to start; see {serve_out}")
                return 4
            info["server_env"] = arm_server_env(arm)
            (out_dir / f"{arm['id']}.serve.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            log(f"ARM {arm['id']} server up: ngl={info['config']['ngl']} ctx={info['config']['ctx']} fa={info['config']['fa']} "
                f"kv={info['config']['kv']} ub={info['config']['ubatch']} spec={info['config'].get('spec_type')} source={info['source']}")
            log(f"ARM {arm['id']} llama-server: {' '.join(info['command'])}")
            rcmd = runner_command(arm, a.port, a.python, out_dir, a.node, a.max_problems, a.problems)
            log(f"ARM {arm['id']} runner: {' '.join(rcmd)}")
            if LOG_PATH is not None:
                with LOG_PATH.open("ab") as lf:
                    rc = subprocess.run(rcmd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(rcmd, env=env, stdin=subprocess.DEVNULL).returncode
            log(f"ARM {arm['id']} runner rc={rc}")
        finally:
            kill_tree(proc)
            kill_image("llama-server")
            time.sleep(3)
            left = running("llama-server")
            log(f"ARM {arm['id']} stopped; llama-server processes left: {left}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", required=True, help="comma-separated arm ids from arms.json")
    ap.add_argument("--out", required=True, type=Path, help="directory for <arm>.jsonl, <arm>.serve.json, <arm>.serve.out")
    ap.add_argument("--repo", type=Path, default=HERE.parent.parent, help="checkout that holds the forge package (cwd of `forge serve`)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--node", default=None)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--health-timeout", type=float, default=600.0)
    ap.add_argument("--max-problems", type=int, default=0, help="smoke test: only the first N problems")
    ap.add_argument("--problems", default="", help="smoke test: comma-separated problem ids or prefixes (passed to the runner)")
    ap.add_argument("--log", type=Path, help="also append every line (and the runner's progress) to this file; it ends with an `exit=` line")
    a = ap.parse_args(argv)
    global LOG_PATH
    LOG_PATH = a.log
    arms = load_arms()
    wanted = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in wanted if x not in arms]
    if unknown:
        log(f"unknown arms: {unknown}; known: {list(arms)}")
        return 2
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for k in ("FORGE_HOME", "FORGE_LLAMA_DIR", "FORGE_MODEL"):
        if not env.get(k):
            log(f"warning: {k} is not set")
    if running("llama-server") or running("llama-bench"):
        log("ABORT: a llama process is running")
        return 3
    unload_ollama()
    worst = 0
    for arm_id in wanted:
        rc = run_arm(arms[arm_id], a, env)
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        emit(f"exit={code} {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    raise SystemExit(code)

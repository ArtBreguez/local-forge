#!/usr/bin/env python3
"""Run one agent on one task from a clean copy and judge the result (phase 4). Stdlib only, Windows and Linux.

    python runner.py --arm T --tasks py-wrap,js-lru --runs 1 --endpoint http://127.0.0.1:8081/v1 --out-dir results/phase4

Per (task, arm, run):
  1. a clean copy of the starting repository in a fresh directory; the visible test files are hashed;
  2. the agent starts as the root of a process tree (`proctree.ProcTree`: a job object on Windows, a process group elsewhere);
     the time limit is enforced here, on that tree, and the tree is killed at the limit, so a dead controller cannot leave an agent running;
  3. when the agent has ended (or been killed) the runner judges it itself: it lays the hidden tests over a copy of the directory and runs
     exactly the task's visible and hidden test files, re-hashes the visible test files, and decides.
Success = every test passes AND the visible test files are byte-identical AND the agent finished inside the time limit.

Arms (see ARMS): O = OpenCode alone, T = the thin orchestrator, T-norepair = T without the Python indentation repair, T-ngram = T on a
server started with `--spec ngram-mod` (the server is the grid driver's business, the runner only needs the endpoint).
One JSON object per line is appended to `<out-dir>/<arm>.jsonl`; a (task, arm, run) already in the file is skipped, so a killed grid resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ORCH = HERE.parent / "orchestrator" / "orchestrate.py"
sys.path.insert(0, str(HERE))
import meter as metermod  # noqa: E402
import proctree  # noqa: E402
import taskbench  # noqa: E402

RUNNER_VERSION = 1
DEFAULT_LIMIT_S = 900.0
ARMS: dict[str, dict[str, Any]] = {
    "O": {"kind": "opencode", "title": "OpenCode alone on the forge endpoint, config A"},
    "T": {"kind": "orchestrator", "title": "thin orchestrator, config A"},
    "T-norepair": {"kind": "orchestrator", "no_repair": True, "title": "T without the Python indentation repair"},
    "T-ngram": {"kind": "orchestrator", "serve": ["--spec", "ngram-mod"], "title": "T on a server with --spec ngram-mod"},
}
OPENCODE_MODEL = "forge/dsv4"


def opencode_config(base_url: str, ctx: int = 16384, out: int = 4096) -> dict[str, Any]:
    return {"$schema": "https://opencode.ai/config.json",
            "provider": {"forge": {"npm": "@ai-sdk/openai-compatible", "name": "forge (llama-server, config A)",
                                   "options": {"baseURL": base_url, "apiKey": "none"},
                                   "models": {"dsv4": {"name": "DeepSeek-V4-Pro-Qwen3.5-9B heretic (forge)", "tool_call": True, "reasoning": False,
                                                       "limit": {"context": ctx, "output": out}}}}},
            "model": OPENCODE_MODEL, "autoupdate": False, "share": "disabled",
            "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"}}


def rmtree(path: Path) -> None:
    def onerr(func, p, _exc):          # Windows: read-only files
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerr, ignore_errors=False) if path.exists() else None


def tail_text(path: Path, n: int = 3000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-n:].decode("utf-8", "replace")


def done_keys(path: Path) -> set[tuple[str, str, int]]:
    keys = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "run":
                keys.add((r["task"], r["arm"], r["run"]))
    return keys


def agent_env(a: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", NODE_NO_WARNINGS="1")
    dirs = [str(Path(a.python or sys.executable).parent)]
    node = a.node or taskbench.qbench.find_node()
    if node and Path(node).is_absolute():
        dirs.append(str(Path(node).parent))
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    return env


def start_agent(arm: dict[str, Any], t: taskbench.Task, work: Path, repo: Path, a: argparse.Namespace, mt: metermod.Meter | None,
                log: Any) -> tuple[proctree.ProcTree, Path | None]:
    env = agent_env(a)
    if arm["kind"] == "orchestrator":
        result = work / "orch-result.json"
        argv = [a.python or sys.executable, "-u", str(ORCH), "--endpoint", a.endpoint, "--repo", str(repo), "--task-file", str(work / "prompt.md"),
                "--test-cmd", json.dumps(taskbench.expand(t.test_cmd, a.python, a.node)), "--lang", t.lang, "--result", str(result),
                "--deadline-s", str(max(60.0, a.time_limit - a.grace_s)), "--max-calls", str(a.max_calls)]
        for p in t.visible_tests:
            argv += ["--protect", p]
        if arm.get("no_repair"):
            argv.append("--no-repair")
        return proctree.ProcTree(argv, str(repo), env, log), result
    # OpenCode: private server inside our process tree (--standalone), config and data isolated under the run directory
    oc = work / "oc"
    for d in ("config/opencode", "data", "state", "cache"):
        (oc / d).mkdir(parents=True, exist_ok=True)
    (oc / "config" / "opencode" / "opencode.json").write_text(json.dumps(opencode_config(mt.url if mt else a.endpoint), indent=2), encoding="utf-8")
    env.update({"XDG_CONFIG_HOME": str(oc / "config"), "XDG_DATA_HOME": str(oc / "data"), "XDG_STATE_HOME": str(oc / "state"),
                "XDG_CACHE_HOME": str(oc / "cache"), "OPENCODE_DISABLE_AUTOUPDATE": "1"})
    if a.oc_cache_seed and Path(a.oc_cache_seed).is_dir():
        shutil.copytree(a.oc_cache_seed, oc / "cache", dirs_exist_ok=True)
    argv = [a.opencode, "run", "--standalone", "--auto", "--format", "json", "--model", OPENCODE_MODEL, t.prompt]
    return proctree.ProcTree(argv, str(repo), env, log), None


def run_one(t: taskbench.Task, arm_id: str, run_idx: int, a: argparse.Namespace, ref_totals: dict[str, int | None]) -> dict[str, Any]:
    arm = ARMS[arm_id]
    base = Path(a.work_dir) / f"{t.id}__{arm_id}__{run_idx}"
    if base.exists():
        rmtree(base)
    base.mkdir(parents=True)
    repo = taskbench.materialize(t, base / "repo")
    (base / "prompt.md").write_text(t.prompt + "\n", encoding="utf-8")
    protected = t.visible_tests
    before = taskbench.hash_files(repo, protected)
    start_copy = base / "start"
    taskbench.copy_tree(repo, start_copy)
    mt = metermod.Meter(a.endpoint).start() if arm["kind"] == "opencode" else None
    log_path = base / "agent.log"
    rec: dict[str, Any] = {"type": "run", "task": t.id, "arm": arm_id, "run": run_idx, "lang": t.lang, "kind": t.kind, "difficulty": t.difficulty,
                           "limit_s": a.time_limit, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.monotonic()
    timed_out = False
    agent_rc: int | None = None
    leftover: int | None = None
    result_path: Path | None = None
    start_error = ""
    try:
        with open(log_path, "wb") as log:
            try:
                tree, result_path = start_agent(arm, t, base, repo, a, mt, log)
            except OSError as e:             # e.g. the OpenCode executable is missing: a recorded failure, not a crashed grid
                tree, start_error = None, f"{type(e).__name__}: {e}"[:300]
            if tree is None:
                agent_rc = -1
                log.write(start_error.encode("utf-8"))
            else:
                rec["job_object"] = tree.job_ok if proctree.IS_WINDOWS else None
                exited = tree.wait(a.time_limit)
                if not exited:
                    timed_out = True
                agent_rc = tree.poll()
                tree.kill()                  # the agent is done or out of time: nothing it started may outlive the run
                leftover = tree.active_processes()
                tree.close()
    finally:
        wall = time.monotonic() - t0
        if mt:
            mt.stop()
    rec["wall_s"] = round(min(wall, a.time_limit) if timed_out else wall, 1)
    rec["timed_out"] = timed_out
    rec["agent_rc"] = agent_rc
    rec["leftover_processes"] = leftover
    # ---- the judgement, made by the runner and not by the agent
    changed = taskbench.changed_files(before, repo)
    check = taskbench.run_check(t, repo, a.python, a.node, timeout=a.test_timeout)
    passed, total = taskbench.score(t, check, ref_totals.get(t.id))
    unchanged_tree = taskbench.diff_summary(start_copy, repo, 100000) == ""
    rec.update({"tests_passed": passed, "tests_total": total, "check_rc": check.rc, "modified_tests": changed})
    rec["success"] = bool(check.ok and not changed and not timed_out)
    if rec["success"]:
        rec["reason"] = "ok"
    elif changed:
        rec["reason"] = "test_modified"
    elif timed_out:
        rec["reason"] = "timeout" + (" (tests pass at the end)" if check.ok else "")
    elif start_error:
        rec["reason"] = "agent_start_failed"
    elif unchanged_tree:
        rec["reason"] = "no_change"
    elif check.rc is None:
        rec["reason"] = "tests_hang"
    elif not check.ok and (check.total is None or check.failed is None):
        rec["reason"] = "tests_error"
    else:
        rec["reason"] = "tests_failed"
    rec["detail"] = check.detail[:300]
    # ---- what the agent did
    stats: dict[str, Any] = {}
    if mt:
        stats = mt.totals()
        rec["steps"], rec["tool_calls"] = stats["steps"], stats["tool_calls"]
        rec["prompt_tokens"], rec["completion_tokens"] = stats["prompt_tokens"], stats["completion_tokens"]
        rec["model_seconds"], rec["api_errors"], rec["context_overflow"] = stats["model_seconds"], stats["errors"], stats["context_overflow"]
        rec["requests"] = stats["requests"]
        rec["trace"] = [{k: r.get(k) for k in ("t", "status", "tools", "tool_calls", "prompt_tokens", "completion_tokens", "finish", "secs")}
                        for r in mt.records][-60:]
    elif result_path is not None and result_path.is_file():
        try:
            o = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError:
            o = {}
        rec["steps"], rec["tool_calls"] = o.get("steps"), o.get("tool_calls")
        rec["prompt_tokens"], rec["completion_tokens"] = o.get("prompt_tokens"), o.get("completion_tokens")
        rec["model_seconds"], rec["edits"] = o.get("model_seconds"), o.get("edits")
        rec["orchestrator"] = {k: o.get(k) for k in ("stop_reason", "repairs", "rescues", "syntax_rejects", "no_edit", "blocked_paths", "reviews",
                                                    "test_runs", "restored_best", "repair_enabled")}
        rec["trace"] = o.get("trace")
    else:
        rec["steps"] = rec["tool_calls"] = rec["prompt_tokens"] = rec["completion_tokens"] = None
    rec["diff"] = taskbench.diff_summary(start_copy, repo, a.diff_chars)
    rec["agent_log_tail"] = tail_text(log_path, a.log_chars)
    if not a.keep_work:
        rmtree(base)
    return rec


def write_record(path: Path, rec: dict[str, Any], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write(json.dumps({"type": "meta", **meta}) + "\n")
        f.write(json.dumps(rec) + "\n")


def reference_totals(tasks: list[taskbench.Task], a: argparse.Namespace) -> dict[str, int | None]:
    """How many tests each task has (visible + hidden), counted once on the reference solution; needed to score a run that crashed early."""
    out: dict[str, int | None] = {}
    with tempfile.TemporaryDirectory(prefix="p4ref-") as tmp:
        for t in tasks:
            d = taskbench.materialize(t, Path(tmp) / t.id)
            taskbench.copy_tree(t.solution_dir, d)
            out[t.id] = taskbench.run_check(t, d, a.python, a.node, timeout=a.test_timeout).total
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--runs", default="1", help="how many runs per task, or a comma list of run numbers (e.g. 2)")
    ap.add_argument("--endpoint", required=True, help="the forge endpoint, e.g. http://127.0.0.1:8081/v1")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "p4work")
    ap.add_argument("--time-limit", type=float, default=DEFAULT_LIMIT_S, help="seconds per run, enforced on the whole process tree")
    ap.add_argument("--grace-s", type=float, default=30.0, help="the orchestrator is told to stop this long before the limit")
    ap.add_argument("--test-timeout", type=float, default=120.0)
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--python", default=None)
    ap.add_argument("--node", default=None)
    ap.add_argument("--opencode", default=r"C:\Users\arthu\.opencode\bin\opencode.exe")
    ap.add_argument("--oc-cache-seed", default=None, help="directory copied into OpenCode's cache dir (e.g. its models.json)")
    ap.add_argument("--shuffle-seed", type=int, default=4, help="the order of the tasks is shuffled with this seed (0: the suite order)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--diff-chars", type=int, default=3000)
    ap.add_argument("--log-chars", type=int, default=2500)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    tasks = taskbench.load_tasks(ids=[x for x in a.tasks.split(",") if x] or None)
    runs = list(range(1, int(a.runs) + 1)) if a.runs.isdigit() else [int(x) for x in a.runs.split(",")]
    if a.shuffle_seed:
        random.Random(a.shuffle_seed).shuffle(tasks)
    out = a.out_dir / f"{a.arm}.jsonl"
    done = done_keys(out)
    refs = reference_totals(tasks, a)
    meta = {"runner_version": RUNNER_VERSION, "arm": a.arm, "arm_title": ARMS[a.arm]["title"], "endpoint": a.endpoint, "time_limit_s": a.time_limit,
            "platform": sys.platform, "python": sys.version.split()[0], "node": taskbench.qbench.find_node(a.node), "note": a.note,
            "max_calls": a.max_calls, "temperature": 0, "seed": 1, "thinking": False, "reference_totals": refs, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta["job_objects"] = proctree.IS_WINDOWS
    code = 0
    for run_idx in runs:
        for t in tasks:
            if (t.id, a.arm, run_idx) in done:
                print(f"skip {t.id} {a.arm} run {run_idx}: already in {out}", flush=True)
                continue
            print(f"{time.strftime('%H:%M:%S')} start {t.id} arm={a.arm} run={run_idx}", flush=True)
            rec = run_one(t, a.arm, run_idx, a, refs)
            write_record(out, rec, meta)
            print(f"{time.strftime('%H:%M:%S')} done  {t.id} arm={a.arm} run={run_idx}: success={rec['success']} reason={rec['reason']} "
                  f"tests={rec['tests_passed']}/{rec['tests_total']} wall={rec['wall_s']}s steps={rec.get('steps')}", flush=True)
            if rec.get("leftover_processes"):
                print(f"WARNING: {rec['leftover_processes']} processes of the agent tree were still alive after the kill", flush=True)
                code = max(code, 6)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

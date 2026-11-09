#!/usr/bin/env python3
"""Phase 6, part C of issue #23: can the local model use REA? Fifteen small reverse-engineering tasks with known answers
(`tasks.json`, ground truth committed before any run; the sources are `src/bench.c` and the committed `bin/bench.exe`), two arms:

* **R** - a fixed recipe: the HARNESS picks the REA commands for the task (`recipe` in tasks.json), runs them, and makes one model call
  that reads the (trimmed) output and answers. The model never chooses a tool.
* **O** - free tool use: OpenCode with five REA tools (`rea_mcp.py`); the MODEL chooses tools, arguments and order and decides when to stop.
  Built-in write and shell tools are denied, so every action is a REA call.

Both arms ask for the answer as a last line `ANSWER: <answer>`; scoring is automatic (exact constant, name, choice, yes/no). Measured:
success, invalid tool calls (arm O: unknown tool, bad arguments, a file that does not exist), REA errors, wall time, tokens.

    python rea_bench.py run --arm R --endpoint http://127.0.0.1:8081/v1 --out results.jsonl --binary bin\\bench.exe --gguf <model.gguf>
    python rea_bench.py run --arm O --endpoint ... --opencode C:\\Users\\arthu\\.opencode\\bin\\opencode.exe ...
    python rea_bench.py report results.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import stat
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "03-agent" / "harness"))
sys.path.insert(0, str(ROOT / "03-agent" / "quality"))
import meter as metermod  # noqa: E402
import proctree  # noqa: E402
import qbench  # noqa: E402
import reacli  # noqa: E402
import run_quality  # noqa: E402

RUNNER_VERSION = 1
OPENCODE_MODEL = "forge/dsv4"
KNOWN_TOOLS = {"rea_decompile", "rea_decompile_all", "rea_search", "rea_imports", "rea_model_header"}
SYSTEM = ("You are a reverse engineer. You are given the output of REA tools (a reverse-engineering toolkit) and a question about a binary or a file. "
          "Answer from that output only. Keep it short. End your reply with one line: ANSWER: <your answer>")
FREE_PROMPT = ("You are reverse engineering {target}. Use the rea_* tools (they run REA, a reverse-engineering toolkit) to find out; do not guess. "
               "Question: {question}\nWhen you know, end your final message with one line: ANSWER: <short answer>")


# ------------------------------------------------------------------------------------------------ tasks and scoring

def load_tasks(path: Path = HERE / "tasks.json") -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_NUM = re.compile(r"0[xX][0-9a-fA-F]+|\b\d+\b|\b[0-9a-fA-F]{2,}\b")
_STRIP = "`*_\"' \t."


def parse_number(text: str, decimal_only: bool = False) -> int | None:
    """The first number in an answer: `0x...` is hex, digits alone are decimal, a bare token with a-f is hex (not with decimal_only)."""
    for tok in _NUM.findall(text):
        if tok[:2].lower() == "0x":
            return None if decimal_only else int(tok, 16)
        if tok.isdigit():
            return int(tok)
        if not decimal_only:
            return int(tok, 16)
    return None


def score(task: dict[str, Any], answer: str | None) -> tuple[bool, str]:
    """(success, why). `answer` is the text after the last `ANSWER:` line, or None when the model never wrote one."""
    if answer is None:
        return False, "no ANSWER line"
    s = task["scoring"]
    kind, expected = s["type"], s["expected"]
    if kind == "hex":
        got = parse_number(answer, s.get("decimal_only", False))
        return (got == expected, "ok" if got == expected else f"got {answer[:60]!r}")
    if kind == "exact":
        got = answer.strip(_STRIP)
        return (got == expected, "ok" if got == expected else f"got {got[:60]!r}")
    low = answer.lower().strip(_STRIP)
    if kind == "yesno":
        first = re.split(r"\W+", low, maxsplit=1)[0]
        return (first == expected, "ok" if first == expected else f"got {answer[:60]!r}")
    if kind == "name":
        names = set(re.findall(r"\brb_[a-z0-9_]+", low))
        return (names == {expected}, "ok" if names == {expected} else f"got {sorted(names)} from {answer[:60]!r}")
    if kind == "choice":
        text = low.replace(s.get("strip_suffix", "\0"), "")
        found = [c for c in s["choices"] if re.search(rf"(?<![a-z0-9_-]){re.escape(c)}(?![a-z0-9_-])", text)]
        return (found == [expected], "ok" if found == [expected] else f"got {found} from {answer[:60]!r}")
    raise ValueError(f"unknown scoring type {kind!r}")


def question_for(task: dict[str, Any], files: dict[str, str]) -> str:
    return task["question"]


def target_for(task: dict[str, Any], files: dict[str, str]) -> str:
    kind = "the Windows executable" if task["file"] == "bench" else "the GGUF model file"
    return f"{kind} {files[task['file']]}"


# ------------------------------------------------------------------------------------------------ arm R: fixed recipe

def run_recipe_steps(task: dict[str, Any], rea: reacli.Rea, files: dict[str, str]) -> list[tuple[str, reacli.ReaResult]]:
    out = []
    for step in task["recipe"]:
        call, arg = step["call"], step.get("arg", "")
        path = files[task["file"]]
        if call == "decompile":
            res = rea.decompile(path, arg)
        elif call == "decompile_all":
            res = rea.decompile_all(path, arg)
        elif call == "imports":
            res = rea.imports(path)
        elif call == "model_header":
            res = rea.model_header(path)
        else:
            raise ValueError(f"unknown recipe call {call!r}")
        out.append((f"{call} {arg}".strip(), res))
    return out


def run_arm_r(task: dict[str, Any], rea: reacli.Rea, files: dict[str, str], a: argparse.Namespace, run_idx: int) -> dict[str, Any]:
    t0 = time.monotonic()
    steps = run_recipe_steps(task, rea, files)
    tool_s = sum(r.secs for _, r in steps)
    errors = sum(1 for _, r in steps if not r.ok)
    body = "\n\n".join(f"REA `{name}`:\n{r.text}" for name, r in steps)
    user = f"Question: {task['question']}\n\n{body}\n\nEnd with one line: ANSWER: <short answer>"
    payload = {"model": "local", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "temperature": 0.0,
               "seed": a.seed + run_idx, "max_tokens": 600, "stream": False, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    rec: dict[str, Any] = {"tool_calls": len(steps), "invalid_calls": 0, "tool_errors": errors, "tool_s": round(tool_s, 1), "steps": 1,
                           "recipe": [n for n, _ in steps]}
    t1 = time.monotonic()
    text = ""
    try:
        resp = run_quality.chat(a.endpoint, payload, a.request_timeout)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        text = qbench.strip_thinking(msg.get("content") or "")[0]
        usage = resp.get("usage") or {}
        rec.update({"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "reply": text[-600:]})
    except RuntimeError as e:
        rec["api_error"] = str(e)[:300]
    rec["model_s"] = round(time.monotonic() - t1, 1)
    answer = reacli.final_answer(text)
    ok, why = score(task, answer)
    rec.update({"answer": answer, "success": ok, "why": "api_error" if "api_error" in rec else why, "wall_s": round(time.monotonic() - t0, 1), "timed_out": False})
    return rec


# ------------------------------------------------------------------------------------------------ arm O: OpenCode + REA tools

def opencode_config(base_url: str, mcp_cmd: list[str], ctx: int = 16384, out: int = 4096) -> dict[str, Any]:
    return {"$schema": "https://opencode.ai/config.json",
            "provider": {"forge": {"npm": "@ai-sdk/openai-compatible", "name": "forge (llama-server, config A)",
                                   "options": {"baseURL": base_url, "apiKey": "none"},
                                   "models": {"dsv4": {"name": "DeepSeek-V4-Pro-Qwen3.5-9B heretic (forge)", "tool_call": True, "reasoning": False,
                                                       "limit": {"context": ctx, "output": out}}}}},
            "model": OPENCODE_MODEL, "autoupdate": False, "share": "disabled",
            "permission": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
            "mcp": {"servers": {"rea": {"type": "local", "command": mcp_cmd, "codemode": False}}}}


def parse_events(log_text: str) -> dict[str, Any]:
    """From OpenCode's `--format json` lines: the final text and every tool call (name, status)."""
    final, tools = "", []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        part = d.get("part") or {}
        if d.get("type") == "text" and part.get("text"):
            final = part["text"]
        if part.get("type") == "tool":
            state = part.get("state") or {}
            if state.get("status") in ("completed", "error"):
                tools.append({"tool": part.get("tool"), "status": state.get("status")})
    return {"final_text": final, "tools": tools}


def read_calls(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def rmtree(path: Path) -> None:
    def onerr(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=onerr)


def run_arm_o(task: dict[str, Any], files: dict[str, str], a: argparse.Namespace, run_idx: int) -> dict[str, Any]:
    base = Path(a.work_dir) / f"{task['id']}__O__{run_idx}"
    rmtree(base)
    base.mkdir(parents=True)
    calls_log = base / "mcp-calls.jsonl"
    mt = metermod.Meter(a.endpoint, seed=a.seed + run_idx).start()
    mcp_cmd = [a.python or sys.executable, str(HERE / "rea_mcp.py"), "--log", str(calls_log), "--rea-root", str(a.rea_root)]
    if a.rea_cmd:
        mcp_cmd += ["--rea-cmd", a.rea_cmd]
    oc = base / "oc"
    for d in ("config/opencode", "data", "state", "cache"):
        (oc / d).mkdir(parents=True, exist_ok=True)
    (oc / "config" / "opencode" / "opencode.json").write_text(json.dumps(opencode_config(mt.url, mcp_cmd), indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", OPENCODE_DISABLE_AUTOUPDATE="1",
               XDG_CONFIG_HOME=str(oc / "config"), XDG_DATA_HOME=str(oc / "data"), XDG_STATE_HOME=str(oc / "state"), XDG_CACHE_HOME=str(oc / "cache"))
    prompt = FREE_PROMPT.format(target=target_for(task, files), question=question_for(task, files))
    oc_cmd = json.loads(a.opencode) if a.opencode.lstrip().startswith("[") else [a.opencode]       # a JSON list lets tests run a fake
    argv = [*oc_cmd, "run", "--standalone", "--auto", "--format", "json", "--model", OPENCODE_MODEL, prompt]
    work = base / "work"
    work.mkdir()
    log_path = base / "agent.log"
    rec: dict[str, Any] = {}
    t0 = time.monotonic()
    timed_out, leftover, rc = False, None, None
    try:
        with open(log_path, "wb") as log:
            try:
                tree = proctree.ProcTree(argv, str(work), env, log)
            except OSError as e:
                tree = None
                log.write(f"{type(e).__name__}: {e}".encode("utf-8"))
            if tree is not None:
                if not tree.wait(a.time_limit):
                    timed_out = True
                rc = tree.poll()
                tree.kill()
                leftover = tree.active_processes()
                tree.close()
    finally:
        wall = time.monotonic() - t0
        mt.stop()
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    ev = parse_events(text)
    calls = read_calls(calls_log)
    stats = mt.totals()
    unknown = sum(1 for t in ev["tools"] if t["tool"] not in KNOWN_TOOLS and t["status"] == "error")
    builtin = sum(1 for t in ev["tools"] if t["tool"] not in KNOWN_TOOLS)
    bad_args = sum(1 for c in calls if not c["valid"])
    answer = reacli.final_answer(ev["final_text"])
    ok, why = score(task, answer)
    rec.update({"tool_calls": len(ev["tools"]), "rea_calls": len(calls), "builtin_calls": builtin, "invalid_calls": bad_args + unknown,
                "invalid_args": bad_args, "unknown_tool_calls": unknown, "tool_errors": sum(1 for c in calls if c["valid"] and not c["ok"]),
                "tool_s": round(sum(c["secs"] for c in calls), 1), "model_s": stats["model_seconds"], "steps": stats["steps"], "requests": stats["requests"],
                "prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"], "api_errors": stats["errors"],
                "context_overflow": stats["context_overflow"], "answer": answer, "success": bool(ok and not timed_out),
                "why": "timeout" if timed_out else why, "wall_s": round(min(wall, a.time_limit) if timed_out else wall, 1), "timed_out": timed_out,
                "agent_rc": rc, "leftover_processes": leftover, "rea_tools_used": [c["tool"] for c in calls], "final_text": ev["final_text"][-600:],
                "log_tail": text[-1500:] if not ok else ""})
    if not a.keep_work:
        rmtree(base)
    return rec


# ------------------------------------------------------------------------------------------------ driver and report

def done_keys(path: Path) -> set[tuple[str, str, int]]:
    keys = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("type") == "run":
                    keys.add((r["task"], r["arm"], r["run"]))
    return keys


def cmd_run(a: argparse.Namespace) -> int:
    data = load_tasks()
    tasks = data["tasks"]
    if a.tasks:
        wanted = [w.strip() for w in a.tasks.split(",") if w.strip()]
        tasks = [t for t in tasks if t["id"] in wanted]
    files = {"bench": str(a.binary), "gguf": str(a.gguf)}
    if Path(a.binary).is_file():
        sha = qbench_sha(Path(a.binary))
        if sha != data["binary"]["sha256"]:
            print(f"the binary {a.binary} is not the one the ground truth is for (sha256 {sha})", file=sys.stderr)
            return 2
    order = list(range(len(tasks)))
    random.Random(a.shuffle_seed).shuffle(order)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(a.out)
    rea = reacli.Rea(json.loads(a.rea_cmd) if a.rea_cmd else None, None, a.rea_timeout, a.rea_root)
    if not a.out.exists() or a.out.stat().st_size == 0:
        meta = {"type": "meta", "runner_version": RUNNER_VERSION, "arm": a.arm, "endpoint": a.endpoint, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "time_limit_s": a.time_limit, "seed": a.seed, "binary_sha256": data["binary"]["sha256"], "gguf_header_sha256": data["gguf"]["header_sha256"],
                "argv": sys.argv[1:], "note": a.note}
        with a.out.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(meta) + "\n")
    for run_idx in range(1, a.runs + 1):
        for i in order:
            t = tasks[i]
            if (t["id"], a.arm, run_idx) in seen:
                continue
            if a.stop_file and a.stop_file.exists():
                print("stop file found: stopping", flush=True)
                return 0
            rec = run_arm_r(t, rea, files, a, run_idx) if a.arm == "R" else run_arm_o(t, files, a, run_idx)
            rec = {"type": "run", "task": t["id"], "category": t["category"], "arm": a.arm, "run": run_idx, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "expected": t["scoring"]["expected"], **rec}
            with a.out.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"[{a.arm}] {t['id']} run{run_idx} {'OK' if rec['success'] else 'FAIL ' + str(rec['why'])} calls={rec['tool_calls']} invalid={rec['invalid_calls']} "
                  f"errors={rec['tool_errors']} wall={rec['wall_s']}s", flush=True)
    return 0


def qbench_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("type") == "run":
                    arms.setdefault(r["arm"], []).append(r)
    return arms


def build_report(arms: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["| arm | success, first run per task (Wilson 95%) | all runs | invalid tool calls (runs with one) | REA errors | tool calls, median per run | "
             "wall min per task, median (mean) | of which REA, median min | model tokens out, median |", "|---|---|---|---|---|---|---|---|---|"]
    first: dict[str, dict[str, bool]] = {}
    for arm, runs in sorted(arms.items()):
        seen: dict[str, bool] = {}
        for r in sorted(runs, key=lambda r: r["run"]):
            seen.setdefault(r["task"], bool(r["success"]))
        first[arm] = seen
        k, n = sum(seen.values()), len(seen)
        lo, hi = qbench.wilson(k, n)
        allk = sum(1 for r in runs if r["success"])
        inv = sum(r["invalid_calls"] for r in runs)
        inv_runs = sum(1 for r in runs if r["invalid_calls"])
        wall = [r["wall_s"] for r in runs]
        lines.append(f"| {arm} | {k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%) | {allk}/{len(runs)} | {inv} ({inv_runs} runs) | {sum(r['tool_errors'] for r in runs)} "
                     f"| {statistics.median([r['tool_calls'] for r in runs]):g} | {statistics.median(wall) / 60:.1f} ({statistics.mean(wall) / 60:.1f}) "
                     f"| {statistics.median([r.get('tool_s') or 0 for r in runs]) / 60:.1f} | {statistics.median([r.get('completion_tokens') or 0 for r in runs]):g} |")
    out = ["\n".join(lines)]
    if "R" in first and "O" in first:
        common = sorted(set(first["R"]) & set(first["O"]))
        only_r = [t for t in common if first["R"][t] and not first["O"][t]]
        only_o = [t for t in common if first["O"][t] and not first["R"][t]]
        out.append(f"\nR against O on the same {len(common)} tasks (first runs): only R {len(only_r)} {only_r}, only O {len(only_o)} {only_o}, "
                   f"exact McNemar p = {qbench.mcnemar_exact(len(only_r), len(only_o)):.3f}")
    cats: dict[str, dict[str, list[int]]] = {}
    for arm, runs in arms.items():
        seen_t: set[str] = set()
        for r in sorted(runs, key=lambda r: r["run"]):
            if r["task"] in seen_t:
                continue
            seen_t.add(r["task"])
            c = cats.setdefault(r.get("category", "?"), {}).setdefault(arm, [0, 0])
            c[0] += bool(r["success"])
            c[1] += 1
    arm_names = sorted(arms)
    rows = ["| category | " + " | ".join(arm_names) + " |", "|---|" + "---|" * len(arm_names)]
    for cat in sorted(cats):
        rows.append(f"| {cat} | " + " | ".join(f"{cats[cat].get(a, [0, 0])[0]}/{cats[cat].get(a, [0, 0])[1]}" for a in arm_names) + " |")
    out.append("\n" + "\n".join(rows))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["R", "O"], required=True)
    r.add_argument("--endpoint", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--binary", type=Path, default=HERE / "bin" / "bench.exe")
    r.add_argument("--gguf", type=Path, default=Path(load_tasks()["gguf"]["host_path"]))
    r.add_argument("--tasks", default="")
    r.add_argument("--runs", type=int, default=1)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--shuffle-seed", type=int, default=6)
    r.add_argument("--time-limit", type=float, default=600.0, help="arm O: seconds per task, enforced on the whole process tree")
    r.add_argument("--request-timeout", type=float, default=900.0)
    r.add_argument("--rea-root", type=Path, default=reacli.DEFAULT_ROOT)
    r.add_argument("--rea-cmd", default=None, help="JSON list: argv of the rea command (tests use a fake)")
    r.add_argument("--rea-timeout", type=float, default=600.0)
    r.add_argument("--opencode", default=r"C:\Users\arthu\.opencode\bin\opencode.exe")
    r.add_argument("--python", default=None)
    r.add_argument("--work-dir", default=str(Path(os.environ.get("TEMP", "/tmp")) / "rea-bench"))
    r.add_argument("--keep-work", action="store_true")
    r.add_argument("--stop-file", type=Path, default=None)
    r.add_argument("--note", default="")
    rp = sub.add_parser("report")
    rp.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    print(build_report(load_runs(a.files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

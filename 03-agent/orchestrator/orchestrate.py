#!/usr/bin/env python3
"""A thin orchestrator for a small local model (phase 4, arm T). Stdlib only, Windows and Linux.

The model never gets a tool loop. The orchestrator is the loop, and it owns every decision that does not need language:

    brief       one task text, the repository files and the visible tests go into every request
    step        the model answers with the full new text of at most two files (`FILE: path` + a fenced block): one or two edits
    check       before a file is written: a deterministic Python indentation repair (`pyindent`, switch off with --no-repair), then a
                syntax check; a file that does not parse is NOT written, the error goes back to the model
    verify      after every edit the orchestrator runs the tests itself; the model's own claims are never read
    feedback    only the failing output of the first failing test goes back (trimmed), not the whole log
    review      when the visible tests pass, one more request compares the files with the task text, because the hidden tests check
                more than the visible ones; a reply `DONE` ends it, edits start the loop again
    budget      --max-calls model requests, --retries rounds without progress, --deadline-s seconds; at the end the best state seen
                (most visible tests passing) is restored

Usage:
    python orchestrate.py --endpoint http://127.0.0.1:8081/v1 --repo DIR --task-file prompt.md \
        --test-cmd '["python", "-m", "unittest"]' --result result.json [--protect tests/test_a.py] [--no-repair]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pyindent  # noqa: E402

TEXT_EXT = {".py", ".js", ".mjs", ".cjs", ".json", ".md", ".txt"}
TEST_DIRS = {"tests", "test", "__tests__"}
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}
FILES_BUDGET = 14000          # characters of repository files in a request (about 4K tokens)
TESTS_BUDGET = 8000
MAX_FILES_PER_REPLY = 2

SYSTEM = """You are a careful programmer working in a small repository. You get the task, the files of the repository, the tests (read only) and the output of the failing test. Reply with the complete new content of every file you change (at most two files per reply) and nothing else, in exactly this format:

FILE: path/relative/to/the/repository/root
```python
<the whole file>
```

Rules: write the whole file, never a fragment or a diff. Keep every part of a file that does not need to change exactly as it is. Indent Python with 4 spaces per level. Never edit the tests. Do not explain. If a task needs more than two files, do the most important two now; you will be asked again.

Example of a reply:

FILE: src/example.py
```python
def add(a, b):
    if not isinstance(a, int):
        raise TypeError("a must be an int")
    return a + b
```"""


# ------------------------------------------------------------------------------------------------ repository helpers

def is_test_path(rel: str, protect: set[str]) -> bool:
    p = Path(rel)
    return (rel in protect or any(part in TEST_DIRS for part in p.parts[:-1])
            or p.name.startswith("test_") or p.name.endswith(("_test.py", ".test.js", ".test.mjs", ".spec.js")))


def list_repo(repo: Path) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for n in sorted(filenames):
            rel = (Path(dirpath) / n).relative_to(repo).as_posix()
            if Path(n).suffix in TEXT_EXT or n == "package.json":
                out.append(rel)
    return out


def read(repo: Path, rel: str) -> str:
    return (repo / rel).read_text(encoding="utf-8", errors="replace")


def safe_path(repo: Path, rel: str) -> str | None:
    rel = rel.strip().strip("`'\"").replace("\\", "/").lstrip("./") if rel.strip().startswith("./") else rel.strip().strip("`'\"").replace("\\", "/")
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in Path(rel).parts:
        return None
    if Path(rel).suffix not in TEXT_EXT:
        return None
    if any(part in SKIP_DIRS for part in Path(rel).parts):
        return None
    return rel


# ------------------------------------------------------------------------------------------------ tests

class TestResult:
    def __init__(self, rc: int | None, total: int | None, failed: int | None, blocks: list[tuple[str, str]], raw: str):
        self.rc, self.total, self.failed, self.blocks, self.raw = rc, total, failed, blocks, raw

    @property
    def ok(self) -> bool:
        return self.rc == 0 and bool(self.total) and not self.failed

    @property
    def passed(self) -> int:
        if self.rc is None or self.total is None:
            return 0
        return max(0, self.total - (self.failed or 0))


def _trim_traceback(block: str, limit: int = 1500) -> str:
    lines = [ln for ln in block.splitlines() if "unittest" not in ln and "/lib/python" not in ln.replace("\\", "/")
             and "node:internal" not in ln]
    text = "\n".join(lines).strip()
    return text if len(text) <= limit else text[:limit // 3] + "\n...\n" + text[-(2 * limit // 3):]


def parse_python(rc: int | None, err: str) -> TestResult:
    m = re.search(r"^Ran (\d+) tests?", err, re.M)
    total = int(m.group(1)) if m else None
    blocks: list[tuple[str, str]] = []
    for part in re.split(r"^={60,}\n", err, flags=re.M)[1:]:
        head, _, body = part.partition("\n")
        blocks.append((head.strip(), _trim_traceback(head + "\n" + body.split("\n" + "-" * 60)[0])))
    failed = None
    if re.search(r"^FAILED \(", err, re.M):
        failed = len(blocks) or 1
    elif re.search(r"^OK", err, re.M):
        failed = 0
    if failed is None and rc:
        blocks = blocks or [("run", _trim_traceback(err[-1500:]))]
    return TestResult(rc, total, failed, blocks, err)


def parse_node(rc: int | None, out: str, err: str) -> TestResult:
    def num(key: str) -> int | None:
        m = re.search(rf"^# {key} (\d+)", out, re.M)
        return int(m.group(1)) if m else None
    lines = out.splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)not ok \d+ - (.*)$", lines[i])
        if m:
            indent = len(m.group(1))
            j = i + 1
            body = [lines[i]]
            while j < len(lines) and not re.match(rf"^\s{{0,{indent}}}(not ok|ok) \d+", lines[j]) and not lines[j].startswith("# "):
                body.append(lines[j])
                j += 1
            text = "\n".join(ln for ln in body if not re.match(r"\s*(duration_ms|type|location|\.\.\.|---)\b", ln))
            blocks.append((m.group(2).strip(), _trim_traceback(text)))
            i = j
        else:
            i += 1
    total, failed = num("tests"), num("fail")
    if not blocks and rc:
        blocks = [("run", _trim_traceback((err or out)[-1500:]))]
    return TestResult(rc, total, failed, blocks, out + err)


def run_tests(repo: Path, cmd: list[str], lang: str, timeout: float = 120.0) -> TestResult:
    argv = list(cmd)
    if lang == "js" and "--test" in argv and not any(a.startswith("--test-reporter") for a in argv):
        argv.append("--test-reporter=tap")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", NODE_NO_WARNINGS="1")
    kw: dict[str, Any] = {"cwd": str(repo), "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "env": env}
    if sys.platform.startswith("win"):
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(argv, **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
        rc: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, err = proc.communicate()
        rc = None
    o, e = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    res = parse_python(rc, e) if lang == "py" else parse_node(rc, o, e)
    if rc is None:
        res.blocks = [("timeout", f"the tests did not finish in {timeout:g}s (an endless loop?)")]
    return res


def _kill(proc: subprocess.Popen) -> None:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=30)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def syntax_error(lang: str, rel: str, text: str, node: str | None) -> str | None:
    """None if the file parses, else a one-line message."""
    if lang == "py" and rel.endswith(".py"):
        try:
            compile(text, rel, "exec")
        except (SyntaxError, ValueError) as e:
            return f"{type(e).__name__}: {e}"[:300]
    elif rel.endswith((".js", ".mjs", ".cjs")) and node:
        return None            # checked after writing (node --check needs the package.json that decides the module type)
    return None


def node_check(repo: Path, rel: str, node: str) -> str | None:
    try:
        r = subprocess.run([node, "--check", rel], cwd=str(repo), capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
        return "SyntaxError: " + " | ".join(lines[:4])[:300]
    return None


# ------------------------------------------------------------------------------------------------ model

class Llm:
    def __init__(self, endpoint: str, model: str, temperature: float, seed: int, max_tokens: int, thinking: bool = False):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.seed, self.max_tokens, self.thinking = model, temperature, seed, max_tokens, thinking
        self.prompt_tokens = self.completion_tokens = self.calls = 0
        self.seconds = 0.0

    def chat(self, system: str, user: str, timeout: float) -> tuple[str, dict[str, Any]]:
        body = {"model": self.model, "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens, "stream": False,
                "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": self.thinking},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=max(10.0, timeout)) as r:  # noqa: S310 - the local forge endpoint
            data = json.loads(r.read().decode("utf-8"))
        secs = time.monotonic() - t0
        usage = data.get("usage") or {}
        self.calls += 1
        self.seconds += secs
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        return text, {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "seconds": round(secs, 1),
                      "finish": choice.get("finish_reason")}


FILE_RE = re.compile(r"(?:^|\n)[ \t]*(?:#+\s*|\*\*)?(?:FILE|File|file)\s*:?\s*`?([^\s`*]+)`?\**[ \t]*\n+[ \t]*(`{3,}|~{3,})[^\n]*\n(.*?)\n[ \t]*\2", re.S)


def parse_edits(reply: str, single_target: str | None) -> list[tuple[str, str]]:
    """[(path, new content)] from a reply. Without any FILE marker, one fenced block is taken for `single_target` if there is one."""
    edits = [(m.group(1), m.group(3)) for m in FILE_RE.finditer(reply)]
    if edits:
        return edits
    blocks = re.findall(r"(?:^|\n)[ \t]*(`{3,})[^\n]*\n(.*?)\n[ \t]*\1", reply, re.S)
    if len(blocks) == 1 and single_target:
        return [(single_target, blocks[0][1])]
    return []


# ------------------------------------------------------------------------------------------------ prompts

def build_prompt(task: str, repo: Path, protect: set[str], failing: TestResult | None, focus_files: list[str], history: list[str],
                 mode: str, first_failure_only: bool = True) -> str:
    rels = list_repo(repo)
    src = [r for r in rels if not is_test_path(r, protect) and r != "package.json" or r == "package.json"]
    tests = [r for r in rels if is_test_path(r, protect) and Path(r).name != "__init__.py"]
    src.sort(key=lambda r: (r not in focus_files, len(read(repo, r)), r))
    parts = ["TASK\n" + task.strip(), "REPOSITORY FILES (you may change these and add new files)"]
    used = 0
    for r in src:
        body = read(repo, r)
        if used + len(body) > FILES_BUDGET and used:
            parts.append(f"=== {r} === (not shown: too large, {len(body)} characters)")
            continue
        used += len(body)
        parts.append(f"=== {r} ===\n{body.rstrip()}\n")
    parts.append("TESTS (read only, do not change)")
    used = 0
    for r in tests:
        body = read(repo, r)
        if used + len(body) > TESTS_BUDGET and used:
            continue
        used += len(body)
        parts.append(f"=== {r} ===\n{body.rstrip()}\n")
    if mode == "review":
        parts.append("RESULT\nThe visible tests pass. Hidden tests check the requirements of the task text more strictly. Compare the files with the task "
                     "text point by point. If any requirement is not implemented yet, reply with the FILE blocks that implement it. If everything "
                     "is implemented, reply with exactly: DONE")
    elif failing is not None and failing.blocks:
        total = failing.total if failing.total is not None else "?"
        names = [n for n, _ in failing.blocks]
        parts.append(f"RESULT\n{len(failing.blocks)} test(s) fail ({failing.passed} of {total} pass). Failing: " + "; ".join(names[:6])
                     + f"\nOutput of the first failure:\n{failing.blocks[0][1]}")
    else:
        parts.append("RESULT\nThe visible tests pass, but the task is not necessarily finished: do what the task text asks.")
    if history:
        parts.append("EARLIER ROUNDS\n" + "\n".join(history[-3:]))
    parts.append("Reply with the FILE blocks of the next change.")
    return "\n\n".join(parts)


def failure_files(res: TestResult | None, repo: Path, protect: set[str]) -> list[str]:
    if res is None:
        return []
    found: list[str] = []
    for _, text in res.blocks:
        for m in re.finditer(r'(?:File "|\(|at |^\s*)([\w./\\:-]+\.(?:py|js|mjs|cjs))', text, re.M):
            p = Path(m.group(1).replace("\\", "/"))
            cand = [str(p)] + [str(Path(*p.parts[i:])) for i in range(1, len(p.parts))]
            for c in cand:
                if (repo / c).is_file() and not is_test_path(c, protect) and c not in found:
                    found.append(Path(c).as_posix())
                    break
    return found


# ------------------------------------------------------------------------------------------------ main loop

class Run:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.repo = Path(a.repo).resolve()
        self.cmd: list[str] = json.loads(a.test_cmd)
        self.lang = a.lang or ("js" if Path(self.cmd[0]).stem.startswith("node") else "py")
        self.protect = set(a.protect or [])
        self.task = Path(a.task_file).read_text(encoding="utf-8")
        self.t0 = time.monotonic()
        self.llm = Llm(a.endpoint, a.model, a.temperature, a.seed, a.max_tokens)
        self.node = self.cmd[0] if self.lang == "js" else None
        self.trace: list[dict[str, Any]] = []
        self.stats = {"edits": 0, "tool_calls": 0, "test_runs": 0, "repairs": 0, "rescues": 0, "syntax_rejects": 0, "no_edit": 0,
                      "blocked_paths": 0, "reviews": 0}
        self.history: list[str] = []
        self.best: tuple[int, dict[str, str | None]] | None = None
        self.written: set[str] = set()
        self.stop_reason = "running"

    def left(self) -> float:
        return self.a.deadline_s - (time.monotonic() - self.t0)

    def snapshot(self) -> dict[str, str | None]:
        return {r: (read(self.repo, r) if (self.repo / r).is_file() else None) for r in sorted(self.written)}

    def restore(self, snap: dict[str, str | None]) -> None:
        for r, text in snap.items():
            p = self.repo / r
            if text is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8", newline="\n")

    def tests(self) -> TestResult:
        self.stats["test_runs"] += 1
        self.stats["tool_calls"] += 1
        return run_tests(self.repo, self.cmd, self.lang, timeout=min(120.0, max(20.0, self.left())))

    def save(self, final: bool = False) -> None:
        out = {"steps": self.llm.calls, "prompt_tokens": self.llm.prompt_tokens, "completion_tokens": self.llm.completion_tokens,
               "model_seconds": round(self.llm.seconds, 1), "wall_s": round(time.monotonic() - self.t0, 1), "stop_reason": self.stop_reason,
               "final": final, "repair_enabled": not self.a.no_repair, **self.stats, "trace": self.trace}
        tmp = Path(str(self.a.result) + ".tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, self.a.result)

    def apply(self, edits: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        """Write the edits that pass the checks. Returns (applied paths, messages for the model about the rest)."""
        applied: list[str] = []
        notes: list[str] = []
        for raw_path, text in edits[:MAX_FILES_PER_REPLY]:
            rel = safe_path(self.repo, raw_path)
            if rel is None or is_test_path(rel, self.protect):
                self.stats["blocked_paths"] += 1
                notes.append(f"{raw_path}: not allowed (tests and paths outside the repository cannot be changed)")
                continue
            text = text.replace("\r\n", "\n")
            if not text.endswith("\n"):
                text += "\n"
            if self.lang == "py" and rel.endswith(".py") and not self.a.no_repair:
                fixed, changed, repaired = pyindent.repair(text)
                if repaired:
                    self.stats["repairs"] += 1
                    self.stats["rescues"] += 1
                    text = fixed
            err = syntax_error(self.lang, rel, text, self.node)
            if err:
                self.stats["syntax_rejects"] += 1
                notes.append(f"{rel}: NOT written, it does not parse: {err}. Send the whole file again with the error fixed (4-space indentation).")
                continue
            p = self.repo / rel
            before = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
            if self.node and rel.endswith((".js", ".mjs", ".cjs")):
                jerr = node_check(self.repo, rel, self.node)
                if jerr:
                    if before is None:
                        p.unlink(missing_ok=True)
                    else:
                        p.write_text(before, encoding="utf-8", newline="\n")
                    self.stats["syntax_rejects"] += 1
                    notes.append(f"{rel}: NOT written, it does not parse: {jerr}")
                    continue
            if before == text:
                notes.append(f"{rel}: identical to the current file (no change)")
                continue
            self.written.add(rel)
            self.stats["edits"] += 1
            self.stats["tool_calls"] += 1
            applied.append(rel)
        if len(edits) > MAX_FILES_PER_REPLY:
            notes.append(f"only the first {MAX_FILES_PER_REPLY} files were taken")
        return applied, notes

    def run(self) -> int:
        a = self.a
        res: TestResult | None = self.tests()
        self.record_best(res)
        mode = "fix"
        reviewed_green = 0
        no_progress = 0
        last_passed = res.passed if res else 0
        rounds = 0
        while True:
            if res is not None and res.ok and rounds > 0 and mode != "review":
                if reviewed_green >= a.reviews:
                    self.stop_reason = "tests pass"
                    break
                mode, reviewed_green = "review", reviewed_green + 1
                self.stats["reviews"] += 1
            if self.llm.calls >= a.max_calls:
                self.stop_reason = "call budget used up"
                break
            if self.left() < 90:
                self.stop_reason = "deadline"
                break
            if no_progress >= a.retries:
                self.stop_reason = "no progress"
                break
            if res is not None and res.ok and rounds == 0:
                mode = "fix"
            prompt = build_prompt(self.task, self.repo, self.protect, None if mode == "review" else res,
                                  failure_files(res, self.repo, self.protect), self.history, mode)
            try:
                reply, meta = self.llm.chat(SYSTEM, prompt, self.left() - 20)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
                self.stop_reason = f"model error: {type(e).__name__}: {str(e)[:120]}"
                break
            rounds += 1
            single = None
            non_test = [r for r in list_repo(self.repo) if not is_test_path(r, self.protect) and r.endswith((".py", ".js", ".mjs", ".cjs"))]
            if len(non_test) == 1:
                single = non_test[0]
            ff = failure_files(res, self.repo, self.protect)
            if ff:
                single = ff[0] if len(ff) == 1 else single
            edits = parse_edits(reply, single)
            entry: dict[str, Any] = {"round": rounds, "mode": mode, **meta, "files": [p for p, _ in edits]}
            if mode == "review" and re.fullmatch(r"\W*DONE\W*", reply.strip(), re.I) and not edits:
                entry["result"] = "review: DONE"
                self.trace.append(entry)
                self.stop_reason = "reviewed"
                break
            if not edits:
                self.stats["no_edit"] += 1
                no_progress += 1
                entry["result"] = "no FILE block in the reply"
                self.history.append(f"Round {rounds}: your reply had no FILE block, so nothing changed. Use the format FILE: path then a fenced block.")
                self.trace.append(entry)
                if mode == "review":
                    self.stop_reason = "review gave no edit"
                    break
                self.save()
                continue
            applied, notes = self.apply(edits)
            entry["applied"], entry["notes"] = applied, notes
            if not applied:
                no_progress += 1
                self.history.append(f"Round {rounds}: nothing was written. " + " ".join(notes))
                entry["result"] = "nothing written"
                self.trace.append(entry)
                if mode == "review":
                    self.stop_reason = "review edit rejected"
                    break
                self.save()
                continue
            res = self.tests()
            entry["passed"], entry["total"], entry["ok"] = res.passed, res.total, res.ok
            entry["result"] = "tests pass" if res.ok else f"{len(res.blocks)} failing"
            self.trace.append(entry)
            self.history.append(f"Round {rounds}: you changed {', '.join(applied)}; afterwards {res.passed} of {res.total} tests pass."
                                + (" " + " ".join(notes) if notes else ""))
            self.record_best(res)
            if res.passed > last_passed or res.ok:
                no_progress = 0
            else:
                no_progress += 1
            last_passed = max(last_passed, res.passed)
            mode = "fix" if not res.ok else mode
            if res.ok and mode == "review":
                mode = "fix"
            self.save()
        # restore the best state seen if the last one is worse
        if self.best and res is not None and res.passed < self.best[0]:
            self.restore(self.best[1])
            self.stats["restored_best"] = 1
        self.save(final=True)
        return 0

    def record_best(self, res: TestResult) -> None:
        if self.best is None or res.passed > self.best[0] or (res.ok and res.passed >= self.best[0]):
            self.best = (res.passed, self.snapshot())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8081/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--test-cmd", required=True, help="JSON list: the test command, run in the repository")
    ap.add_argument("--lang", choices=["py", "js"])
    ap.add_argument("--protect", action="append", default=[], help="repository-relative path of a file that must not change (repeatable)")
    ap.add_argument("--result", required=True, help="JSON file rewritten after every round")
    ap.add_argument("--max-calls", type=int, default=10, help="model requests")
    ap.add_argument("--retries", type=int, default=3, help="rounds in a row without more tests passing")
    ap.add_argument("--reviews", type=int, default=1, help="comparisons with the task text once the visible tests pass")
    ap.add_argument("--deadline-s", type=float, default=840.0, help="stop starting new requests when this many seconds are used")
    ap.add_argument("--max-tokens", type=int, default=2200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-repair", action="store_true", help="skip the deterministic Python indentation repair (arm T-norepair)")
    a = ap.parse_args(argv)
    run = Run(a)
    try:
        return run.run()
    except KeyboardInterrupt:
        run.stop_reason = "interrupted"
        run.save(final=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

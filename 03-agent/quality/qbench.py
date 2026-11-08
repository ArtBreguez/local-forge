"""Phase 3 quality benchmark: problems, code extraction, hidden-test execution and statistics. Stdlib only; Windows and Linux.

A problem is a prompt (`problems/<id>/prompt.md`) plus hidden tests (`problems/<id>/tests.py|tests.js`). The model sees the
prompt only. `run_tests` writes the extracted code to a temp dir next to the hidden tests and runs them with
`python -m unittest` or `node --test`, with a timeout that kills the whole process tree.

The code in the answers is model-written and is executed WITHOUT a sandbox beyond a temp working directory, a timeout and a closed
stdin: run this only on a machine where that is acceptable.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform.startswith("win")

SYSTEM_PROMPT = ("You are an expert programmer. Reply with one complete solution in a single fenced code block and nothing else: "
                 "no explanation, no tests, no example calls, no reading from standard input.")
LANG_NOTE = {
    "py": "Language: Python 3 (standard library only). Define the requested function(s) or class at module level.",
    "js": "Language: JavaScript for Node.js (CommonJS, no dependencies). Define the requested function(s) or class and "
          "export them with `module.exports = { ... };`.",
}
# Prompt variants. v1 is the first wording. v2 adds one sentence for Python: in the first smoke run this model sometimes wrote a
# whole function with one-space (or no) indentation, which is an IndentationError and says nothing about the logic.
PROMPT_VARIANTS = {"v1": LANG_NOTE, "v2": {**LANG_NOTE, "py": LANG_NOTE["py"] + " Indent with 4 spaces per level."}}
FENCE_LANGS = {"py": {"python", "py", "python3", ""}, "js": {"javascript", "js", "node", "nodejs", "cjs", ""}}
CODE_HINTS = {"py": ("def ", "class "), "js": ("function", "=>", "class ")}
OUTPUT_LIMIT = 2000


# ------------------------------------------------------------------------------------------------ problems

@dataclass(frozen=True)
class Problem:
    id: str
    lang: str                       # py | js
    entries: tuple[str, ...]        # names the tests import
    difficulty: str
    max_tokens: int                 # budget for the answer itself (thinking arms add a thinking budget on top)
    tags: tuple[str, ...]
    subtle: bool
    root: Path = HERE

    @property
    def prompt(self) -> str:
        return (self.root / "problems" / self.id / "prompt.md").read_text(encoding="utf-8").strip()

    @property
    def tests_path(self) -> Path:
        return self.root / "problems" / self.id / ("tests.py" if self.lang == "py" else "tests.js")

    def ref_path(self, kind: str) -> Path:
        return self.root / "_ref" / self.id / f"{kind}.{self.lang}"

    @property
    def ext(self) -> str:
        return "py" if self.lang == "py" else "js"


def load_problems(root: Path = HERE) -> list[Problem]:
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return [Problem(d["id"], d["lang"], tuple(d["entries"]), d["difficulty"], int(d["max_tokens"]), tuple(d["tags"]), bool(d["subtle"]), root) for d in data]


def build_messages(p: Problem, variant: str = "v1") -> list[dict[str, str]]:
    """The same two messages for every arm. Thinking is switched by the template kwargs of the request, not by the text."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{PROMPT_VARIANTS[variant][p.lang]}\n\n{p.prompt}"}]


# ------------------------------------------------------------------------------------------------ extraction

_FENCE = re.compile(r"(?m)^[ \t]*```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.S)
_OPEN_FENCE = re.compile(r"(?m)^[ \t]*```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove reasoning from `text`. Returns (answer, unfinished).

    The generation prompt of this model's template ends in `<think>\\n`, so the reasoning arrives as text that CLOSES with `</think>`
    and never opens with a tag when the server does not split it off. Text that opens a `<think>` and never closes it is an
    unfinished thought (the token cap hit while thinking): there is no answer in it.
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return "", True
    return text, False


def extract_code(text: str, lang: str, thinking: bool = False, reasoning_split: bool = False) -> tuple[str | None, str]:
    """The code of an answer, or (None, why). Policy: the first fenced block whose tag fits the language (or has no tag),
    else the first fenced block, else an unterminated fence (answer cut by the token cap), else the whole answer if it looks like code.

    thinking=True with no `</think>` in the text (and no server-side split of the reasoning) means the reasoning never finished.
    """
    if thinking and not reasoning_split and "</think>" not in text:
        return None, "thinking did not finish"
    answer, unfinished = strip_thinking(text)
    if unfinished:
        return None, "thinking did not finish"
    answer = answer.strip()
    if not answer:
        return None, "empty answer"
    blocks = [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(answer)]
    if blocks:
        for tag, body in blocks:
            if tag in FENCE_LANGS[lang]:
                return body.strip("\n") + "\n", "fenced block"
        return blocks[0][1].strip("\n") + "\n", "first fenced block (language tag did not match)"
    m = _OPEN_FENCE.search(answer)
    if m:
        return answer[m.end():].strip("\n") + "\n", "unterminated fence"
    if any(h in answer for h in CODE_HINTS[lang]):
        return answer + "\n", "whole answer"
    return None, "no code in the answer"


_EXPORT_KW = re.compile(r"(?m)^([ \t]*)export[ \t]+(?=(?:async[ \t]+)?(?:function|class|const|let|var)\b)")
_EXPORT_LIST = re.compile(r"(?m)^[ \t]*export[ \t]*\{[^}]*\}[ \t]*;?[ \t]*$")
_EXPORT_DEFAULT = re.compile(r"(?m)^([ \t]*)export[ \t]+default[ \t]+")


def normalize_js(code: str, entries: tuple[str, ...]) -> tuple[str, bool]:
    """Make an ES-module style answer loadable by `require`: drop `export` keywords and add the CommonJS export line if the answer
    has none. Returns (code, changed). The prompt asks for CommonJS; this only forgives the module syntax, never the logic."""
    new = _EXPORT_DEFAULT.sub(r"\1", _EXPORT_LIST.sub("", _EXPORT_KW.sub(r"\1", code)))
    if not re.search(r"\bmodule\.exports\b|\bexports\.\w+\s*=", new):
        new = new.rstrip("\n") + "\n\nmodule.exports = { " + ", ".join(entries) + " };\n"
    return new, new != code


# ------------------------------------------------------------------------------------------------ running tests

@dataclass
class TestOutcome:
    passed: bool
    kind: str | None                # None when passed | syntax | test_fail | timeout
    detail: str = ""
    tests_total: int | None = None
    tests_failed: int | None = None
    seconds: float = 0.0
    normalized: bool = False


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if IS_WINDOWS:
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


def run_limited(cmd: list[str], cwd: str, timeout: float, env: dict[str, str] | None = None) -> tuple[int | None, str, str]:
    """Run `cmd`, return (returncode or None on timeout, stdout, stderr). On timeout the whole process tree is killed
    (`node --test` runs each file in a child process that a plain kill of the parent would leave spinning)."""
    kw: dict[str, Any] = {"cwd": cwd, "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "env": env}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
        rc: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
        rc = None
    return rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def find_node(explicit: str | None = None) -> str | None:
    for cand in (explicit, os.environ.get("QBENCH_NODE"), shutil.which("node"), r"C:\Program Files\nodejs\node.exe", "/usr/bin/node"):
        if cand and (shutil.which(cand) or Path(cand).is_file()):
            return cand
    return None


def _first_python_failure(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("FAIL: ", "ERROR: ")):
            block: list[str] = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("=" * 20):
                    break
                block.append(nxt)
            msg = next((b for b in reversed(block) if b.strip() and not b.startswith("-" * 20) and not b.startswith("Ran ")), "")
            return f"{line} :: {msg.strip()}"[:400]
    return ""


def _parse_python(rc: int | None, err: str) -> tuple[int | None, int | None, str]:
    m = re.search(r"^Ran (\d+) tests?", err, re.M)
    total = int(m.group(1)) if m else None
    failed = None
    fm = re.search(r"^FAILED \((.*)\)", err, re.M)
    if fm:
        failed = sum(int(n) for n in re.findall(r"=(\d+)", fm.group(1)))
    elif re.search(r"^OK", err, re.M):
        failed = 0
    detail = _first_python_failure(err)
    if not detail and rc:
        detail = "\n".join(err.strip().splitlines()[-4:])[:400]
    return total, failed, detail


def _parse_node(rc: int | None, out: str, err: str) -> tuple[int | None, int | None, str]:
    def num(key: str) -> int | None:
        m = re.search(rf"^# {key} (\d+)", out, re.M)
        return int(m.group(1)) if m else None
    total, failed = num("tests"), num("fail")
    detail = ""
    m = re.search(r"^\s*not ok \d+ - (.*)$", out, re.M)
    if m:
        detail = f"not ok - {m.group(1)}"
        em = re.search(r"^\s*error: ['\"]?(.*?)['\"]?$", out[m.end():], re.M)
        if em:
            detail += f" :: {em.group(1)}"
    if not detail and rc:
        detail = "\n".join((err or out).strip().splitlines()[-4:])
    return total, failed, detail[:400]


def run_tests(p: Problem, code: str, *, timeout: float = 30.0, python: str | None = None, node: str | None = None) -> TestOutcome:
    """Write `code` + the hidden tests into a temp dir and run them."""
    t0 = time.monotonic()
    normalized = False
    if p.lang == "js":
        code, normalized = normalize_js(code, p.entries)
    with tempfile.TemporaryDirectory(prefix="qbench-") as tmp:
        solution = Path(tmp) / f"solution.{p.ext}"
        solution.write_text(code, encoding="utf-8", newline="\n")
        shutil.copyfile(p.tests_path, Path(tmp) / f"test_hidden.{p.ext}")
        if p.lang == "js":      # the nearest package.json decides CommonJS vs ES module: a stray {"type": "module"} in a parent directory
            (Path(tmp) / "package.json").write_text('{"private": true, "type": "commonjs"}\n', encoding="utf-8")   # (the host has one in C:\\Users\\arthu) would break every require()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", NODE_NO_WARNINGS="1")
        if p.lang == "py":
            try:
                compile(code, "solution.py", "exec")
            except (SyntaxError, ValueError) as e:
                return TestOutcome(False, "syntax", f"{type(e).__name__}: {e}"[:300], seconds=time.monotonic() - t0, normalized=normalized)
            rc, out, err = run_limited([python or sys.executable, "-m", "unittest", "-v", "test_hidden"], tmp, timeout, env)
            total, failed, detail = _parse_python(rc, err)
        else:
            exe = find_node(node)
            if exe is None:
                raise RuntimeError("node not found (use --node or $QBENCH_NODE)")
            rc, out, err = run_limited([exe, "--check", "solution.js"], tmp, min(timeout, 30.0), env)
            if rc != 0:
                why = "timeout during --check" if rc is None else (err.strip().splitlines() or ["syntax error"])[-1]
                return TestOutcome(False, "syntax" if rc is not None else "timeout", why[:300], seconds=time.monotonic() - t0, normalized=normalized)
            rc, out, err = run_limited([exe, "--test", "--test-reporter=tap", "test_hidden.js"], tmp, timeout, env)
            total, failed, detail = _parse_node(rc, out, err)
    secs = time.monotonic() - t0
    if rc is None:
        return TestOutcome(False, "timeout", f"tests did not finish in {timeout:g}s", total, failed, secs, normalized)
    if rc == 0:
        return TestOutcome(True, None, "", total, 0 if failed is None else failed, secs, normalized)
    return TestOutcome(False, "test_fail", detail, total, failed, secs, normalized)


# ------------------------------------------------------------------------------------------------ statistics

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b: only the baseline passed, c: only the other arm passed)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021) from n samples of which c passed."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


@dataclass
class ArmSummary:
    arm: str
    n: int = 0
    passed: int = 0
    kinds: dict[str, int] = field(default_factory=dict)
    cap_hits: int = 0
    median_tok_s: float | None = None
    median_tokens: float | None = None
    total_wall_s: float = 0.0


def read_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") == "meta":
            meta = d
        elif d.get("type") == "result":
            rows.append(d)
    return meta, rows


def summarize(arm: str, rows: list[dict[str, Any]]) -> ArmSummary:
    s = ArmSummary(arm, n=len(rows), passed=sum(1 for r in rows if r["pass"]))
    for r in rows:
        if not r["pass"]:
            s.kinds[r["kind"]] = s.kinds.get(r["kind"], 0) + 1
    s.cap_hits = sum(1 for r in rows if r.get("cap_hit"))
    s.median_tok_s = median([r["tok_s"] for r in rows if r.get("tok_s")])
    s.median_tokens = median([r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None])
    s.total_wall_s = sum(r.get("wall_s") or 0.0 for r in rows)
    return s

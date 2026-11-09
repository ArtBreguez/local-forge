#!/usr/bin/env python3
"""Phase 4 task suite: loader, validators and the test check shared by the runner and the orchestrator.

A task is a directory under `03-agent/tasks/<id>/`:

    task.json     id, lang (py|js), kind, difficulty, title, test_cmd, visible_tests, hidden_tests, start_state
    prompt.md     what the agent is told (identical for every arm)
    repo/         the starting repository, including the visible tests
    hidden/       the hidden tests (paths relative to the repository root), never copied into the agent's directory
    solution/     files that, laid over repo/, make every test pass
    broken/       files that, laid over repo/, still fail at least one hidden test

`test_cmd` is the command the agent is told to use ({python} and {node} are filled in). The *check* after a run is stricter:
it runs exactly the visible and hidden test files of the task (not whatever tests the agent added) and reads the counts.

    python taskbench.py validate [--task ID]     # reference passes, broken and the starting repo fail, as promised
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE.parent / "tasks"
sys.path.insert(0, str(HERE.parent / "quality"))
import qbench  # noqa: E402  (process-tree kill, test output parsers, Wilson / McNemar, median)

REQUIRED = ("id", "lang", "kind", "difficulty", "title", "test_cmd", "visible_tests", "hidden_tests", "start_state")
LANGS = ("py", "js")
KINDS = ("function", "bugfix", "two-file", "refactor", "codebase")
START_STATES = ("failing", "green")
IGNORED_NAMES = {"__pycache__", "node_modules", ".git"}


class TaskError(ValueError):
    pass


@dataclass
class Task:
    id: str
    lang: str
    kind: str
    difficulty: str
    title: str
    test_cmd: list[str]
    visible_tests: list[str]
    hidden_tests: list[str]
    start_state: str
    root: Path = field(default=Path("."))

    @property
    def prompt(self) -> str:
        return (self.root / "prompt.md").read_text(encoding="utf-8").strip()

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def hidden_dir(self) -> Path:
        return self.root / "hidden"

    @property
    def solution_dir(self) -> Path:
        return self.root / "solution"

    @property
    def broken_dir(self) -> Path:
        return self.root / "broken"

    @property
    def test_files(self) -> list[str]:
        """The files the check runs: visible then hidden, without the package marker `__init__.py`."""
        return [p for p in self.visible_tests + self.hidden_tests if Path(p).name != "__init__.py"]


def load_task(root: Path) -> Task:
    data = json.loads((root / "task.json").read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise TaskError(f"{root.name}: task.json lacks {missing}")
    return Task(**{k: data[k] for k in REQUIRED}, root=root)


def load_tasks(root: Path = TASKS_DIR, ids: list[str] | None = None) -> list[Task]:
    tasks = [load_task(p.parent) for p in sorted(root.glob("*/task.json"))]
    if ids:
        known = {t.id for t in tasks}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise TaskError(f"unknown tasks {unknown}; known: {sorted(known)}")
        tasks = [t for t in tasks if t.id in ids]
    return tasks


def list_files(root: Path) -> list[str]:
    """Relative POSIX paths of every file under root, without caches."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_NAMES)
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            out.append((Path(dirpath) / name).relative_to(root).as_posix())
    return out


def validate_task(t: Task) -> list[str]:
    """Static problems of the task description (an empty list means it is well formed); no test is run."""
    problems: list[str] = []
    if t.root.name != t.id:
        problems.append(f"directory name {t.root.name!r} differs from id {t.id!r}")
    if t.lang not in LANGS:
        problems.append(f"lang {t.lang!r} not in {LANGS}")
    if t.kind not in KINDS:
        problems.append(f"kind {t.kind!r} not in {KINDS}")
    if t.start_state not in START_STATES:
        problems.append(f"start_state {t.start_state!r} not in {START_STATES}")
    if not t.test_cmd or not any("{python}" in a or "{node}" in a for a in t.test_cmd[:1]):
        problems.append("test_cmd must start with {python} or {node}")
    if t.lang == "py" and not t.id.startswith("py-") or t.lang == "js" and not t.id.startswith("js-"):
        problems.append("the id must start with the language (py- / js-)")
    if not (t.root / "prompt.md").is_file() or len(t.prompt) < 80:
        problems.append("prompt.md is missing or too short")
    for d in (t.repo_dir, t.hidden_dir, t.solution_dir, t.broken_dir):
        if not d.is_dir():
            problems.append(f"missing directory {d.name}/")
    if problems:
        return problems
    repo_files = set(list_files(t.repo_dir))
    hidden_files = set(list_files(t.hidden_dir))
    for p in t.visible_tests:
        if p not in repo_files:
            problems.append(f"visible test {p} is not in repo/")
    for p in t.hidden_tests:
        if p not in hidden_files:
            problems.append(f"hidden test {p} is not in hidden/")
        if p in repo_files:
            problems.append(f"hidden test {p} is also in repo/ (the agent would see it)")
    extra_hidden = hidden_files - set(t.hidden_tests)
    if extra_hidden:
        problems.append(f"files in hidden/ that are not listed in hidden_tests: {sorted(extra_hidden)}")
    if not t.test_files:
        problems.append("no test file")
    if not any(Path(p).name != "__init__.py" for p in t.hidden_tests):
        problems.append("no hidden test file")
    protected = set(t.visible_tests)
    for name in ("solution", "broken"):
        overlay = set(list_files(getattr(t, f"{name}_dir")))
        if overlay & protected:
            problems.append(f"{name}/ overwrites protected test files {sorted(overlay & protected)}")
        if not overlay:
            problems.append(f"{name}/ is empty")
    if t.lang == "js" and "package.json" not in repo_files:
        problems.append("a JS task needs its own package.json (the nearest one decides CommonJS vs ESM)")
    if t.prompt.count("`") % 2:
        problems.append("prompt.md has an unbalanced backtick")
    leak = [p for p in t.hidden_tests if Path(p).name in t.prompt]
    if leak:
        problems.append(f"the prompt names a hidden test file: {leak}")
    return problems


# ------------------------------------------------------------------------------------------------ files

def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*IGNORED_NAMES, "*.pyc"))


def materialize(t: Task, dest: Path) -> Path:
    """A clean copy of the starting repository at dest (which must not exist or be empty)."""
    if dest.exists() and any(dest.iterdir()):
        raise TaskError(f"{dest} is not empty")
    copy_tree(t.repo_dir, dest)
    return dest


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def hash_files(root: Path, rels: list[str]) -> dict[str, str | None]:
    return {r: sha256(root / r) for r in rels}


def changed_files(before: dict[str, str | None], root: Path) -> list[str]:
    """Which of the hashed files are now different or gone."""
    return [r for r, h in before.items() if sha256(root / r) != h]


def diff_summary(a: Path, b: Path, limit: int = 3000) -> str:
    """Unified diff of two trees, truncated; what the agent changed."""
    import difflib
    out: list[str] = []
    names = sorted(set(list_files(a)) | set(list_files(b)))
    for n in names:
        ta = (a / n).read_text(encoding="utf-8", errors="replace").splitlines() if (a / n).is_file() else []
        tb = (b / n).read_text(encoding="utf-8", errors="replace").splitlines() if (b / n).is_file() else []
        if ta != tb:
            out.extend(difflib.unified_diff(ta, tb, f"a/{n}", f"b/{n}", lineterm="", n=1))
    text = "\n".join(out)
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more characters]"


# ------------------------------------------------------------------------------------------------ running tests

def expand(argv: list[str], python: str | None = None, node: str | None = None) -> list[str]:
    py = python or sys.executable
    nd = node or qbench.find_node() or "node"
    return [a.replace("{python}", py).replace("{node}", nd) for a in argv]


def check_argv(t: Task, python: str | None = None, node: str | None = None, files: list[str] | None = None) -> list[str]:
    """The command of the check: exactly the task's visible and hidden test files (or the given subset)."""
    files = t.test_files if files is None else files
    base = expand(t.test_cmd, python, node)
    if t.lang == "py":
        return base + [Path(p).with_suffix("").as_posix().replace("/", ".") for p in files]
    return base + ["--test-reporter=tap"] + files


@dataclass
class TestRun:
    rc: int | None            # None: timed out
    total: int | None
    failed: int | None
    detail: str
    seconds: float
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.rc == 0 and bool(self.total) and self.failed == 0


def run_argv(argv: list[str], cwd: Path, lang: str, timeout: float = 120.0) -> TestRun:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", NODE_NO_WARNINGS="1")
    t0 = time.monotonic()
    rc, out, err = qbench.run_limited(argv, str(cwd), timeout, env)
    secs = time.monotonic() - t0
    if lang == "py":
        total, failed, detail = qbench._parse_python(rc, err)
    else:
        total, failed, detail = qbench._parse_node(rc, out, err)
    if rc is None:
        detail = f"tests did not finish in {timeout:g}s"
    text = (out + "\n" + err).strip()
    return TestRun(rc, total, failed, detail, secs, text[-6000:])


def run_check(t: Task, workdir: Path, python: str | None = None, node: str | None = None, timeout: float = 120.0,
              hidden: bool = True) -> TestRun:
    """Run the task's test files on a COPY of workdir with the hidden tests laid over it (so the agent's own directory stays
    as it left it)."""
    with tempfile.TemporaryDirectory(prefix="p4check-") as tmp:
        scratch = Path(tmp) / "repo"
        copy_tree(workdir, scratch)
        if hidden:
            copy_tree(t.hidden_dir, scratch)
        if t.lang == "js" and not (scratch / "package.json").is_file():
            (scratch / "package.json").write_text('{"private": true, "type": "commonjs"}\n', encoding="utf-8")
        files = None if hidden else [p for p in t.test_files if p in set(t.visible_tests)]
        argv = check_argv(t, python, node, files)
        return run_argv(argv, scratch, t.lang, timeout)


def score(t: Task, run: TestRun, reference_total: int | None) -> tuple[int, int]:
    """(passed, total) of a check run. A test module that cannot even be loaded counts as failures of the tests it holds."""
    total = max(run.total or 0, reference_total or 0)
    if run.rc is None:
        return 0, total
    passed = max(0, (run.total or 0) - (run.failed or 0))
    if run.failed is None and run.rc != 0:
        passed = 0
    return min(passed, total), total


# ------------------------------------------------------------------------------------------------ reference checks

@dataclass
class Validation:
    task: str
    problems: list[str]
    reference_total: int | None = None


def validate_behaviour(t: Task, python: str | None = None, node: str | None = None) -> Validation:
    """Run the three promises of a task: the reference passes everything, the broken one fails a hidden test, and the starting repo
    is in the stated state (visible tests failing, or green for a refactor) and fails the hidden tests."""
    v = Validation(t.id, validate_task(t))
    if v.problems:
        return v
    with tempfile.TemporaryDirectory(prefix="p4val-") as tmp:
        tmp_p = Path(tmp)
        start = materialize(t, tmp_p / "start")
        vis = run_check(t, start, python, node, hidden=False)
        if t.start_state == "failing" and vis.ok:
            v.problems.append("the visible tests already pass in the starting repo")
        if t.start_state == "green" and not vis.ok:
            v.problems.append(f"start_state is green but the visible tests fail: {vis.detail}")
        full = run_check(t, start, python, node)
        if full.ok:
            v.problems.append("the starting repo already passes the hidden tests")
        sol = materialize(t, tmp_p / "sol")
        copy_tree(t.solution_dir, sol)
        ref = run_check(t, sol, python, node)
        if not ref.ok:
            v.problems.append(f"the reference solution fails: {ref.detail or ref.output[-300:]}")
        else:
            v.reference_total = ref.total
        sol_vis = run_check(t, sol, python, node, hidden=False)
        if not sol_vis.ok:
            v.problems.append(f"the reference solution fails the visible tests: {sol_vis.detail}")
        bad = materialize(t, tmp_p / "bad")
        copy_tree(t.broken_dir, bad)
        brk = run_check(t, bad, python, node)
        if brk.ok:
            v.problems.append("the broken solution passes every hidden test")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="run the reference, broken and starting states of every task")
    v.add_argument("--task", action="append", default=[])
    v.add_argument("--python")
    v.add_argument("--node")
    sub.add_parser("list", help="print the suite")
    a = ap.parse_args(argv)
    tasks = load_tasks(ids=getattr(a, "task", None) or None)
    if a.cmd == "list":
        for t in tasks:
            print(f"{t.id:22} {t.lang} {t.kind:9} {t.difficulty:7} {t.title}")
        return 0
    bad = 0
    for t in tasks:
        res = validate_behaviour(t, a.python, a.node)
        print(f"{'ok  ' if not res.problems else 'FAIL'} {t.id:22} reference tests: {res.reference_total}")
        for p in res.problems:
            print(f"     - {p}")
        bad += bool(res.problems)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

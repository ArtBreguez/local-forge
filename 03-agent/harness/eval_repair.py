#!/usr/bin/env python3
"""Offline measure of the Python indentation repair (phase 4) on the answers phase 3 recorded.

Takes every distinct Python answer in `results/phase3/*.jsonl` that did not even compile (kind `syntax`), runs `pyindent.repair` on it and
runs the problem's hidden tests on the result. Also flattens the 20 Python reference files of the phase 4 tasks to one space per line
(the shape phase 3 saw) and checks how many compile again, how many come back byte for byte, and how many tasks pass their tests.

    python eval_repair.py > results/phase4/repair-offline.txt
"""
from __future__ import annotations

import glob
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for sub in ("quality", "orchestrator", "harness"):
    sys.path.insert(0, str(HERE.parent / sub))
import pyindent  # noqa: E402
import qbench  # noqa: E402
import taskbench  # noqa: E402


def flatten(code: str) -> str:
    return "\n".join((" " + ln.strip()) if ln.startswith((" ", "\t")) else ln for ln in code.split("\n"))


def phase3_answers() -> list[dict]:
    seen, out = set(), []
    for f in sorted(glob.glob(str(ROOT / "results" / "phase3" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") == "result" and r.get("lang") == "py" and r.get("kind") == "syntax" and (r["problem"], r.get("code")) not in seen:
                seen.add((r["problem"], r.get("code")))
                out.append({**r, "file": Path(f).name})
    return out


def main() -> int:
    probs = {p.id: p for p in qbench.load_problems()}
    answers = phase3_answers()
    comp = passed = 0
    print("== phase 3 answers that did not compile (distinct):", len(answers))
    for r in answers:
        fixed, _, ok = pyindent.repair(r.get("code") or "")
        res = None
        if ok:
            comp += 1
            res = qbench.run_tests(probs[r["problem"]], fixed, timeout=30).passed
            passed += bool(res)
        print(f"  {r['file']:12} {r['problem']:24} compiles after repair: {str(ok):5}  hidden tests pass: {res}")
    print(f"   -> {comp} of {len(answers)} compile after the repair, {passed} of them pass the hidden tests (before: 0 compile)")
    print("\n== phase 4 Python reference solutions flattened to one space per line")
    exact = compiled = total = tasks_ok = 0
    py_tasks = [t for t in taskbench.load_tasks() if t.lang == "py"]
    for t in py_tasks:
        with tempfile.TemporaryDirectory() as tmp:
            d = taskbench.materialize(t, Path(tmp) / "r")
            for rel in taskbench.list_files(t.solution_dir):
                code = (t.solution_dir / rel).read_text(encoding="utf-8")
                if rel.endswith(".py") and "\n    " in code:
                    total += 1
                    fixed, _, ok = pyindent.repair(flatten(code))
                    compiled += ok
                    exact += fixed == code
                    code = fixed
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text(code, encoding="utf-8")
            run = taskbench.run_check(t, d)
            tasks_ok += run.ok
            print(f"  {t.id:20} all tests pass after flatten + repair: {run.ok}")
    print(f"   -> files: {compiled}/{total} compile again, {exact}/{total} identical to the original; tasks: {tasks_ok}/{len(py_tasks)} pass all tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

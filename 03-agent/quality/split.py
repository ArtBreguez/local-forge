"""Phase 6: split every problem's tests into a VISIBLE part (used to choose among candidates) and a HIDDEN part.

The rule is fixed here, written to `split.json` and committed BEFORE any best-of-N run: the test functions of a problem, in file
order, alternate visible / hidden starting with visible (indices 0, 2, 4, ... are visible). Files list their tests roughly from the
plain cases to the edge cases, so alternating gives the visible part both kinds, and the hidden part still holds half of the edge
cases. Every problem has at least four test functions, so both parts are never empty.

A candidate answer is judged by the *whole* original test file (visible + hidden, the phase 3 definition, so numbers stay comparable
with phase 3's pass@1). Selection may only look at the visible part.

    python split.py --write      # regenerate split.json from problems/*/tests.*
    python split.py --check      # exit 1 when split.json is out of date

Stdlib only.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_JSON = HERE / "split.json"
RULE = "test functions in file order; indices 0, 2, 4, ... are visible, 1, 3, 5, ... are hidden"

_JS_TEST = re.compile(r"^test\(\s*(['\"`])(.*?)\1", re.S)
_JS_CLOSE = re.compile(r"^\}\);\s*$")


def py_tests(src: str) -> list[tuple[str, int, int]]:
    """(name, first line, last line), 1-based inclusive, of every `test_*` method of every top-level class (decorators included)."""
    out: list[tuple[str, int, int]] = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                    first = min([item.lineno] + [d.lineno for d in item.decorator_list])
                    out.append((f"{node.name}.{item.name}", first, item.end_lineno or item.lineno))
    return out


def js_tests(src: str) -> list[tuple[str, int, int]]:
    """(title, first line, last line) of every top-level `test('title', () => { ... });` (closing `});` in column 0)."""
    lines = src.splitlines()
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        m = _JS_TEST.match(lines[i])
        if m:
            j = i
            if not re.search(r"\}\);\s*$", lines[i]) or lines[i].count("{") != lines[i].count("}"):
                j = i + 1
                while j < len(lines) and not _JS_CLOSE.match(lines[j]):
                    j += 1
            out.append((m.group(2), i + 1, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def list_tests(lang: str, src: str) -> list[tuple[str, int, int]]:
    return py_tests(src) if lang == "py" else js_tests(src)


def visible_indices(n: int) -> list[int]:
    return list(range(0, n, 2))


def _is_test(item: ast.AST) -> bool:
    return isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_")


def build_part(lang: str, src: str, keep: set[int]) -> str:
    """The test file with every test function whose index is not in `keep` removed. Imports and helpers stay; a TestCase class that
    would be left without any test method is removed whole (a helper class that never had tests stays)."""
    tests = list_tests(lang, src)
    drop: set[int] = set()
    for idx, (_, first, last) in enumerate(tests):
        if idx not in keep:
            drop.update(range(first, last + 1))
    if lang == "py":
        idx = 0
        for node in ast.parse(src).body:
            if isinstance(node, ast.ClassDef):
                n = sum(1 for i in node.body if _is_test(i))
                if n and not any(j in keep for j in range(idx, idx + n)):
                    drop.update(range(min([node.lineno] + [d.lineno for d in node.decorator_list]), (node.end_lineno or node.lineno) + 1))
                idx += n
    text = "\n".join(ln for no, ln in enumerate(src.splitlines(), 1) if no not in drop) + "\n"
    if lang == "py":
        ast.parse(text)
    return text


def _problems(root: Path) -> list[dict]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def build_split(root: Path = HERE) -> dict:
    problems: dict[str, dict] = {}
    for d in _problems(root):
        lang = d["lang"]
        src = (root / "problems" / d["id"] / ("tests.py" if lang == "py" else "tests.js")).read_text(encoding="utf-8")
        names = [t[0] for t in list_tests(lang, src)]
        vis = visible_indices(len(names))
        problems[d["id"]] = {"lang": lang, "n_tests": len(names), "tests": names, "visible": vis,
                             "hidden": [i for i in range(len(names)) if i not in vis]}
    return {"rule": RULE, "problems": problems}


def load_split(path: Path = SPLIT_JSON) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def part_source(root: Path, problem_id: str, lang: str, part: str, split: dict | None = None) -> str:
    """Source text of the `visible` or `hidden` part of a problem's tests, built from the committed split."""
    split = split or load_split(root / "split.json")
    entry = split["problems"][problem_id]
    src = (root / "problems" / problem_id / ("tests.py" if lang == "py" else "tests.js")).read_text(encoding="utf-8")
    return build_part(lang, src, set(entry[part]))


def write_parts(root: Path, out_dir: Path, split: dict | None = None) -> dict[str, dict[str, Path]]:
    """Write `<id>.visible.<ext>` and `<id>.hidden.<ext>` for every problem into out_dir. Returns {id: {"visible": path, "hidden": path}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split = split or load_split(root / "split.json")
    paths: dict[str, dict[str, Path]] = {}
    for d in _problems(root):
        ext = "py" if d["lang"] == "py" else "js"
        paths[d["id"]] = {}
        for part in ("visible", "hidden"):
            path = out_dir / f"{d['id']}.{part}.{ext}"
            path.write_text(part_source(root, d["id"], d["lang"], part, split), encoding="utf-8", newline="\n")
            paths[d["id"]][part] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--write" in argv:
        data = build_split()
        SPLIT_JSON.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        n = sum(p["n_tests"] for p in data["problems"].values())
        v = sum(len(p["visible"]) for p in data["problems"].values())
        print(f"wrote {SPLIT_JSON.name}: {len(data['problems'])} problems, {n} test functions, {v} visible, {n - v} hidden")
        return 0
    if "--check" in argv:
        ok = load_split() == build_split()
        print("split.json is up to date" if ok else "split.json is OUT OF DATE (run split.py --write)")
        return 0 if ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

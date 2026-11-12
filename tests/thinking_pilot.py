"""What the tests of the thinking-pilot fixes share (tests/test_rea_thinking_*.py, tests/test_rea_kaggle_llama_bin.py): the frozen pilot replies (`tests/fixtures/thinking-pilot/`), the kernel and the kit's REA wrapper
loaded by path, and a reader's check for a part of an answer that is missing."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from distill import traces  # noqa: E402
from rea_bench import grammar, scoring  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "thinking-pilot"
KIT = ROOT / "distill" / "backends" / "kaggle"
THINKING_CONFIG = ROOT / "distill" / "configs" / "rea-thinking.json"


def load_by_path(name: str, path: pathlib.Path):
    """A module by its file, not by whichever directory another test put first on sys.path (`build_tasks` exists twice: the generic one and the REA wrapper)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(KIT))
run_teacher = load_by_path("pilot_run_teacher", KIT / "run_teacher.py")
sys.path.remove(str(KIT))
build_tasks = load_by_path("pilot_rea_build_tasks", ROOT / "tools" / "kaggle" / "build_tasks.py")


def pilot():
    """({task id: task}, [reply rows]) of the frozen pilot data; a row is {case, id, kind, sample, finish, tokens, text}."""
    tasks = {t["id"]: t for t in map(json.loads, (FIX / "tasks.jsonl").read_text().splitlines())}
    return tasks, [json.loads(line) for line in (FIX / "replies.jsonl").read_text().splitlines()]


TASKS, ROWS = pilot()


def rows(case: str) -> list:
    return [r for r in ROWS if r["case"] == case]


def reply_of(row: dict) -> str:
    """The reply as the scorer builds it from a pilot line (`traces.reply_text`, a thinking run)."""
    return traces.reply_text(row["text"], row["finish"], "", True)


def omitted_parts(final: str, t: dict) -> int:
    """How many parts of a multi-part answer the final line leaves out altogether (not: gets wrong): the numeric groups of the task's answer that have no number in the line, plus one when the answer has a
    name group (a technique, a cipher) and the line has no word that is not a function word or a unit. A reader's check, not a scorer: `score_strict` decides."""
    s = scoring.strict_task(t)
    if s["score"] != "tokens" or len(s["accept"]) < 2:
        return 0
    sp = grammar.Spec(s)
    toks = grammar.lex(grammar.normalise(final))
    numbers = [x for x in toks if x.kind == "num"]
    words = [x for x in toks if x.kind == "word" and x.text not in grammar.GLUE and x.text not in sp.units and x.text not in ("hex", "decimal", "hexadecimal")]
    return max(0, len(sp.numeric) - len(numbers)) + (1 if sp.name_groups and not words else 0)

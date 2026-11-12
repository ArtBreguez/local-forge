"""REA (reverse engineering of decompiled C) as a distillation subject: the adapter between `distill.domain.Domain` and the `rea_bench` library (tools/rea-bench).

What lives where. `rea_bench` is the subject's library and stays where it is: the task generators (families), the compile and fact gates, the strict answer grammar (`scoring.score_strict`), the truth templates
(`targets`), the held-out list. The Ghidra decompile stage is REA's too (`tools/modal/modal_rea_data.py --stage decompile`): it turns generated tasks into `decompiled-<tag>.jsonl`, which this adapter loads.
Nothing here decides anything; every method calls the function that decided it before the pipeline was split:

  verify           `scoring.score_strict`: the whitelist grammar on the text, for every family
  truth_target     `scoring.truth_target`: the canonical headline and one factual sentence rendered from the task's own parameters
  needs_agreement  the task's `chance` (the best constant guess within its variant) is `AGREE_CHANCE` or more
  messages         `SYSTEM_RECIPE` (plain; the text of modal_rea.py, which stays untouched) or `reasoning.SYSTEM_REASONING`, then the pseudocode and the question
  heldout_ids      the ids of seed 1001 (the held-out split: `<family>-1001-<n>`)
  task_keys        the content of a decompiled task: its normalised pseudocode with its question, and its question with its answer key (kind, question, `accept`); `normalize_prompt` is what makes two decompilations of
                   one function equal (the Ghidra names of two compiles differ in their addresses only). The ids never enter.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys

_RB = pathlib.Path(__file__).resolve().parents[3] / "tools" / "rea-bench"      # the repository's copy; in a Modal container rea_bench is mounted next to distill
if (_RB / "rea_bench").is_dir() and str(_RB) not in sys.path:
    sys.path.insert(0, str(_RB))

from rea_bench import dedupe, outputs, reasoning, scoring  # noqa: E402
from rea_bench.core import HELDOUT_SEED  # noqa: E402

from ...domain import Domain  # noqa: E402

AGREE_CHANCE = 0.05          # round 4 (m8): 0.06-0.063 variants (fsm/opcode_value, bounds/char_class, multi/limit_helper) kept one lucky sample of six in 31% of the tasks
# The plain system message of the sample stage: a copy of `SYSTEM_RECIPE` of tools/modal/modal_rea.py (which needs Modal to import and is not touched); a test compares the two.
SYSTEM_RECIPE = ("You are a reverse engineer. You are given decompiled C pseudocode of one function from a stripped binary and a "
                 "question. Answer in one short line with the concrete value or name asked for, then one sentence of justification.")


# Ghidra's names for what it cannot name: a function (FUN_), data (DAT_), a label (LAB_), a pointer (PTR_), a subroutine (SUB_), each followed by the address in hex. Two compiles of one source put them at other
# addresses, so the same function decompiles to texts that differ in these only. `PTR_DAT_00104010` is one name inside another: the inner one is collapsed, and its prefix stays.
GHIDRA_ADDRESS = re.compile(r"(?<![A-Za-z0-9])(FUN|DAT|LAB|PTR|SUB)_[0-9a-fA-F]+(?![A-Za-z0-9_])")


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:24]


class _HeldoutIds:
    """The held-out split is the tasks of seed 1001, whose ids end in `-1001-<n>`: a container for `id in heldout_ids()`, with no list to build."""

    def __contains__(self, task_id) -> bool:
        return isinstance(task_id, str) and re.search(rf"-{HELDOUT_SEED}-\d+$", task_id) is not None


class Rea(Domain):
    name = "rea"
    tasks_prefix = "decompiled-"

    def load_tasks(self, source) -> list:
        """The decompiled tasks (records with pseudocode) of `source`: a `decompiled-<tag>.jsonl` of the decompile stage, several of them, or a directory with one `<id>.json` per task (the Modal volume).
        Files: sorted by `<id>.json`, the order the sample stage reads the volume in, so that the records come out in its order; a file of the held-out seed 1001 is refused by its name."""
        return self._read(source, heldout=False)

    def load_heldout(self, source) -> list:
        """The same records for an evaluation: the files of the held-out seed 1001 (`decompiled-seed1001*.jsonl`) are the ones to read here, and are not refused."""
        return self._read(source, heldout=True)

    def _read(self, source, heldout: bool) -> list:
        files = [pathlib.Path(f) for f in ([source] if isinstance(source, (str, os.PathLike)) else source)]
        if len(files) == 1 and files[0].is_dir():
            tasks = [json.loads(p.read_text()) for p in sorted(files[0].glob("*.json"))]
            return [t for t in tasks if "pseudocode" in t]
        for f in files:
            seed = re.search(r"seed(\d+)", f.name)
            if seed and not heldout:
                dedupe.check_sample_seed(int(seed.group(1)))             # the held-out split is never turned into SFT pairs
        tasks = sorted(outputs.load_tasks(files), key=lambda t: f"{t['id']}.json")
        if not tasks:
            raise SystemExit(f"no decompiled task (a record with pseudocode) in {', '.join(map(str, files))}")
        return tasks

    def messages(self, task: dict, mode: str) -> list:
        if mode == "plain":
            return [{"role": "system", "content": SYSTEM_RECIPE}, {"role": "user", "content": reasoning.user_message(task)}]
        return reasoning.messages(task)

    def verify(self, final_answer: str, task: dict) -> bool:
        return scoring.score_strict(final_answer, task)

    def needs_agreement(self, task: dict) -> bool:
        c = scoring.strict_task(task).get("chance")                # a legacy task gets its `chance` from the conversion (skip: 0.14)
        return c is not None and c >= AGREE_CHANCE

    def heldout_ids(self):
        return _HeldoutIds()

    def truth_target(self, task: dict):
        return scoring.truth_target(task)["text"]

    def check_task(self, task: dict) -> None:
        scoring.check_mode(task)       # an unknown score mode must stop the run before the model download and the server start
        scoring.check_target(task)     # so must a task without a truth template, or one whose rendered target its own scorer rejects

    def contradicts(self, analysis: str, answer: str, task: dict) -> bool:
        return reasoning.contradicts(analysis, answer, task)

    def task_words(self, task: dict) -> set:
        return reasoning.task_words(task)

    def canon_word(self, word: str) -> str:
        return reasoning.value_word(word)

    def kinds(self):
        return reasoning.known_kinds()

    def normalize_prompt(self, text: str) -> str:
        return " ".join(GHIDRA_ADDRESS.sub(r"\1_ADDR", text).split())

    def task_keys(self, task: dict) -> dict:
        """{"code": sha of (the normalised pseudocode, the question), "answer": sha of (kind, the question, `accept`)}. The question is normalised as a prompt is. `refs` (the bytes the function references) is not
        part of "code": two tasks with the same code and question and other data are the same problem to be excluded (the stricter reading). `accept` is taken as stored. A rule the record cannot give (no
        pseudocode, no question or accept) is left out."""
        question, code, accept = task.get("question"), task.get("pseudocode"), task.get("accept")
        question = self.normalize_prompt(question) if isinstance(question, str) else ""
        keys = {}
        if question and isinstance(code, str) and code.strip():
            keys["code"] = _digest("code", self.normalize_prompt(code), question)
        if question and accept is not None and task.get("kind"):
            keys["answer"] = _digest("answer", task["kind"], question, accept)
        return keys


DOMAIN = Rea()

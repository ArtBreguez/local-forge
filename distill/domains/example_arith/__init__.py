"""A TEST FIXTURE, not a subject to train on: tiny arithmetic word problems with a code-computed answer. It exists to prove that the core of `distill/` does not depend on REA
(tests/test_distill_core.py runs the whole pipeline on it with no `rea_bench` import) and to show the smallest honest `Domain`. A real subject copies the shape, not the content."""
from __future__ import annotations

import json
import pathlib
import random
import re

from ...domain import Domain

HELDOUT_SEED = 9999
KINDS = ("add", "sub", "mul")
SYSTEM_PLAIN = "Answer with the number only."
SYSTEM_REASONING = "Work it out in at most three short lines, then write one last line `Answer: <the number>` and nothing after it."


def make_tasks(seed: int, n: int = 12) -> list:
    """`n` problems from `seed`, ids `arith-<seed>-<i>`; the answer is computed here from the operands, never by a model. `kind` cycles add, sub, mul."""
    rng, out = random.Random(f"arith/{seed}"), []
    for i in range(n):
        kind = KINDS[i % 3]
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        text = {"add": f"Ann has {a} apples and buys {b} more. How many apples does she have?", "sub": f"Bob has {a + b} marbles and loses {b}. How many are left?",
                "mul": f"A box holds {a} eggs. How many eggs are in {b} boxes?"}[kind]
        out.append({"id": f"arith-{seed}-{i:03d}", "kind": kind, "question": text, "answer": {"add": a + b, "sub": a, "mul": a * b}[kind]})
    return out


class Arith(Domain):
    name = "example_arith"

    def load_tasks(self, source) -> list:
        if isinstance(source, int):
            return make_tasks(source)                               # a seed
        files = [source] if isinstance(source, (str, pathlib.Path)) else source                           # a jsonl file of tasks, or several
        return [json.loads(line) for f in files for line in pathlib.Path(f).read_text().splitlines() if line.strip()]

    def messages(self, task: dict, mode: str) -> list:
        return [{"role": "system", "content": SYSTEM_PLAIN if mode == "plain" else SYSTEM_REASONING}, {"role": "user", "content": task["question"]}]

    def verify(self, final_answer: str, task: dict) -> bool:
        m = re.fullmatch(r"\s*(-?\d+)\s*\.?\s*", final_answer or "")
        return m is not None and int(m.group(1)) == task["answer"]      # the whole answer is one number, equal to the computed one

    def needs_agreement(self, task: dict) -> bool:
        return task["answer"] < 10                                  # a guess among the one-digit numbers is right often

    def truth_target(self, task: dict):
        return str(task["answer"])

    def heldout_ids(self) -> set:
        return {t["id"] for t in make_tasks(HELDOUT_SEED)}


DOMAIN = Arith()

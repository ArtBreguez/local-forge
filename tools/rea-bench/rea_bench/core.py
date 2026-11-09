"""Shared pieces of the REA task generator: the Task record, C headers, compiler/flag sampling, file output.

Every family module exposes `FAMILY` (its name) and `VARIANTS` (variant name -> function(rng) -> Task). `build()` runs the
variants round-robin; each task draws from its own `random.Random(f"{family}/{seed}/{i}")`, so adding or tuning one family
never shifts what another family (or another task of the same family) produces for a seed.
"""
from __future__ import annotations

import json
import pathlib
import random
import string
from dataclasses import dataclass, field

HDR = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdint.h>\n\n"
MAIN = '\n\nint main(int argc, char **argv) { (void)argc; (void)argv; puts("ok"); return 0; }\n'

OPTS = ["-O0", "-O1", "-O2", "-O3", "-Os"]
OPT_WEIGHTS = [1, 3, 4, 2, 2]
CCS = ["gcc", "clang"]
CC_WEIGHTS = [3, 2]


def rword(rng, n):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def hx(v: int, digits: int = 0) -> list:
    """Spellings of a hex answer: 0x1a2b, 1a2b and the zero-padded forms (the question says 'hex', so bare hex is fine)."""
    out = [f"0x{v:x}", f"{v:x}"]
    if digits:
        out += [f"0x{v:0{digits}x}", f"{v:0{digits}x}"]
    return list(dict.fromkeys(out))


def dec(v: int) -> list:
    """Spellings of a decimal answer; the 0x form of the same value is unambiguous so it is accepted too."""
    return [str(v), f"0x{v:x}"] if v >= 0 else [str(v)]


def c_bytes(bs) -> str:
    return ", ".join(f"0x{b:02x}" for b in bs)


def c_u32s(ws) -> str:
    return ", ".join(f"0x{w:08x}u" for w in ws)


@dataclass
class Task:
    """One generated task. `truth` is what the scorer sees; `params`, `wrong` and `check` are for the test-suite only."""
    kind: str                       # family name
    variant: str                    # structural form inside the family
    body: str                       # C text defining `proc` (plus types / helpers before it)
    question: str
    truth: dict                     # {"score": ..., "accept": ..., optional "reject": ...}
    id: str = ""
    opt: str = "-O2"
    cc: str | None = None           # None = legacy task (consumer assumes gcc)
    headers: str = HDR
    helpers: list = field(default_factory=list)   # extra functions to decompile with proc (multi-function family)
    params: dict = field(default_factory=dict)
    wrong: str = ""                 # a plausible wrong answer the scorer must reject
    check: tuple | None = None      # (C main, expected stdout): a harness that proves the truth

    def source(self, main: str | None = None) -> str:
        return self.headers + self.body + (MAIN if main is None else "\n\n" + main)

    def meta(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "question": self.question, "opt": self.opt}
        if self.cc:
            d["cc"] = self.cc
        d["function"] = "proc"
        if self.variant:
            d["variant"] = self.variant
        if self.helpers:
            d["helpers"] = list(self.helpers)
        d.update(self.truth)
        return d


def pick_flags(family: str, seed: int, i: int, cc: str = "mix") -> tuple:
    """Compiler and optimisation level for task i, from an rng that does not touch the template's own stream."""
    rng = random.Random(f"flags/{family}/{seed}/{i}")
    opt = rng.choices(OPTS, OPT_WEIGHTS)[0]
    chosen = rng.choices(CCS, CC_WEIGHTS)[0]
    return opt, (chosen if cc == "mix" else cc)


def build(family: str, variants: dict, n: int, seed: int, cc: str = "mix") -> list:
    names = list(variants)
    out = []
    for i in range(n):
        rng = random.Random(f"{family}/{seed}/{i}")
        t = variants[names[i % len(names)]](rng)
        t.id = f"{t.kind}-{seed}-{i:04d}"
        t.opt, t.cc = pick_flags(family, seed, i, cc)
        out.append(t)
    return out


def write_tasks(tasks, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out / f"{t.id}.c").write_text(t.source(), encoding="utf-8")
        (out / f"{t.id}.json").write_text(json.dumps(t.meta()), encoding="utf-8")

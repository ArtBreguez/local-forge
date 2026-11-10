"""Shared pieces of the REA task generator: the Task record, C headers, compiler/flag sampling, file output.

Every family module exposes `FAMILY` (its name) and `VARIANTS` (variant name -> function(rng) -> Task). `build()` runs the
variants round-robin; each task draws from its own `random.Random(f"{family}/{seed}/{i}")`, so adding or tuning one family
never shifts what another family (or another task of the same family) produces for a seed.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
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


def _bare_hex_ok(s: str) -> bool:
    """A hex spelling without 0x is only safe when no plain word or decimal number can be mistaken for it: all zeros, or 3+ digits
    holding both a letter a-f and a digit ('a', 'be', 'ad', 'bad', 'dead' are words; '10' is decimal ten, not 0x10)."""
    return set(s) == {"0"} or (len(s) >= 3 and any(c in "abcdef" for c in s) and any(c.isdigit() for c in s))


def hx(v: int, digits: int = 0) -> list:
    """Spellings of a hex answer: 0x1a2b and, where unambiguous (see _bare_hex_ok), 1a2b. The scorer compares 0x literals by value,
    so 0x08, 0x0008 and 0x8 are one spelling; `digits` only adds the zero-padded bare form."""
    forms = [f"0x{v:x}", f"{v:x}"] + ([f"{v:0{digits}x}"] if digits else [])
    return list(dict.fromkeys(f for f in forms if f.startswith("0x") or _bare_hex_ok(f)))


def dec(v: int) -> list:
    """Spellings of a decimal answer; the 0x form of the same value is unambiguous so it is accepted too."""
    return [str(v), f"0x{v:x}"] if v >= 0 else [str(v)]


DECOY_NAMES = ["pad", "tmp", "seed", "salt", "tick", "stat", "mark", "hint", "slot", "note", "gap", "spin"]


def decoys(rng, count: int, forbidden=()) -> str:
    """`count` harmless statements for the top of a function: volatile locals and static counters that survive every -O level, so
    the compiled function has extra, unrelated constants (real code is full of them). Each carries random constants, which is what
    keeps small templates (a standard CRC, a base64 encoder, ChaCha) from being the same program in every task. No constant equals
    a value in `forbidden` (the answer must not be matched by a decoy)."""
    out, names = [], rng.sample(DECOY_NAMES, count)
    for k, base in enumerate(names):
        n = f"{base}{k}"
        c = rng.randrange(0x1000, 0x10000)
        while c in forbidden:
            c = rng.randrange(0x1000, 0x10000)
        kind, s = rng.randrange(5), rng.randint(1, 7)
        if kind == 0:
            out += [f"volatile unsigned {n} = 0x{c:x}u;", f"{n} ^= {n} >> {s};"]
        elif kind == 1:
            out += [f"static volatile unsigned {n};", f"{n} += 0x{c:x}u;"]
        elif kind == 2:
            m = rng.randint(4, 16)
            out += [f"volatile unsigned char {n}[{m}] = {{0}};", f"{n}[{rng.randrange(m)}] = (unsigned char)({n}[{rng.randrange(m)}] + 0x{c & 0xff:02x});"]
        elif kind == 3:
            out += [f"volatile unsigned {n} = {c}u;", f"{n} = ({n} << {s}) | ({n} >> {32 - s});"]
        else:
            out += [f"static volatile unsigned short {n} = 0x{c:x};", f"{n} = (unsigned short)({n} + {rng.randint(1, 999)});"]
    return "".join(f"    {line}\n" for line in out)


_PROC_OPEN = re.compile(r"\bproc\s*\([^)]*\)\s*\{\n")


def with_decoys(body: str, rng, count: int, forbidden=()) -> str:
    """`body` with `count` decoy statements as the first lines of proc."""
    if not count:
        return body
    m = _PROC_OPEN.search(body)
    assert m, "no proc( ... ) { to put decoys in"
    return body[:m.end()] + decoys(rng, count, forbidden) + body[m.end():]


def near(v: int) -> list:
    """v and its neighbours: a bound the compiler may show one off (`v > 9` as `v >= 10`)."""
    return [v - 1, v, v + 1]


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
    check: tuple | None = None      # (C main, expected stdout[, preamble]): a harness that proves the truth
    facts: list = field(default_factory=list)   # groups of alternatives (int immediates / str data) the compiled code must still hold (survive.py)

    def source(self, main: str | None = None, preamble: str = "") -> str:
        """The generated file; with `main` a harness instead of the stub main, with `preamble` (macros that swap a libc call for a
        fake) placed after the headers and before the body."""
        return self.headers + preamble + self.body + (MAIN if main is None else "\n\n" + main)

    def meta(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "question": self.question, "opt": self.opt}
        if self.cc:
            d["cc"] = self.cc
        d["function"] = "proc"
        if self.variant:
            d["variant"] = self.variant
        if self.helpers:
            d["helpers"] = list(self.helpers)
        if self.facts:
            d["facts"] = self.facts
        d.update(self.truth)
        return d


def pick_flags(family: str, seed: int, i: int, cc: str = "mix", attempt: int = 0) -> tuple:
    """Compiler and optimisation level for task i, from an rng that does not touch the template's own stream."""
    rng = random.Random(f"flags/{family}/{seed}/{i}" + (f"/r{attempt}" if attempt else ""))
    opt = rng.choices(OPTS, OPT_WEIGHTS)[0]
    chosen = rng.choices(CCS, CC_WEIGHTS)[0]
    return opt, (chosen if cc == "mix" else cc)


MAX_TRIES = 60


def build(family: str, variants: dict, n: int, seed: int, cc: str = "mix", gate=None) -> list:
    """Task i is variant i mod len(variants), drawn from its own rng. With a `gate` (survive.Gate) a draw whose asked fact is not in
    the code compiled at its own cc/opt is thrown away and the task is drawn again (new parameters, new cc/opt) from the stream
    "<family>/<seed>/<i>/r<attempt>"; attempt 0 is the stream of the ungated generator, so a gate only changes the tasks it rejects."""
    names = list(variants)
    out = []
    for i in range(n):
        for attempt in range(MAX_TRIES):
            rng = random.Random(f"{family}/{seed}/{i}" + (f"/r{attempt}" if attempt else ""))
            t = variants[names[i % len(names)]](rng)
            t.id = f"{t.kind}-{seed}-{i:04d}"
            t.opt, t.cc = pick_flags(family, seed, i, cc, attempt)
            if gate is None or gate(t):
                break
        else:
            raise RuntimeError(f"{family}/{names[i % len(names)]}: no draw in {MAX_TRIES} keeps its fact in the code (task {i}, seed {seed})")
        out.append(t)
    return out


def write_tasks(tasks, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out / f"{t.id}.c").write_text(t.source(), encoding="utf-8")
        (out / f"{t.id}.json").write_text(json.dumps(t.meta()), encoding="utf-8")

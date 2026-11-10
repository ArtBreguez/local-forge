"""Shared pieces of the REA task generator: the Task record, C headers, compiler/flag sampling, file output.

Every family module exposes `FAMILY` (its name) and `VARIANTS` (variant name -> function(rng) -> Task). `build()` runs the
variants round-robin; each task draws from its own `random.Random(f"{family}/{seed}/{i}")`, so adding or tuning one family
never shifts what another family (or another task of the same family) produces for a seed.

The held-out seed (HELDOUT_SEED) draws from reserved parameter values: `pool()` splits a family's small discrete choices (standard
CRC algorithms, ChaCha/Salsa round counts, return codes, standard alphabets) into a training part and a held-out part, so the program
a held-out task asks about is never one a training seed can make, with or without decoys (round-2 review, M2).
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import string
from dataclasses import dataclass, field

HELDOUT_SEED = 1001

HDR = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdint.h>\n\n"
MAIN = '\n\nint main(int argc, char **argv) { (void)argc; (void)argv; puts("ok"); return 0; }\n'

OPTS = ["-O0", "-O1", "-O2", "-O3", "-Os"]
OPT_WEIGHTS = [1, 3, 4, 2, 2]
CCS = ["gcc", "clang"]
CC_WEIGHTS = [3, 2]


class TaskRng(random.Random):
    """The rng of one task. Same stream as random.Random(x); `held` is True for the held-out seed (see pool())."""
    held = False


def is_held(rng) -> bool:
    return bool(getattr(rng, "held", False))


def pool(rng, items, held_items) -> list:
    """The values this task may draw from: only `held_items` for the held-out seed, everything else for any other seed. Whole parameter
    values are reserved this way where a family has so few programs that decoys alone cannot keep a held-out program from a training
    seed (standard alphabets, standard CRC algorithms, cipher round counts)."""
    hs = list(held_items)
    return [x for x in items if (x in hs) == is_held(rng)]


def pick_int(rng, lo: int, hi: int, mod: int = 5) -> int:
    """randint(lo, hi) restricted by residue: multiples of `mod` for the held-out seed, all other values for training seeds."""
    return rng.choice([v for v in range(lo, hi + 1) if (v % mod == 0) == is_held(rng)])


def rword(rng, n):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def _bare_hex_ok(s: str) -> bool:
    """A hex spelling without 0x is only safe when no plain word or decimal number can be mistaken for it: all zeros, or 2+ characters
    holding both a letter a-f and a digit ('3f', 'a7', 'edb8'; 'a', 'be', 'ad', 'bad', 'dead' are words; '10' is decimal ten)."""
    return set(s) == {"0"} or (len(s) >= 2 and any(c in "abcdef" for c in s) and any(c.isdigit() for c in s))


def hx(v: int, digits: int = 0) -> list:
    """Spellings of a hex answer: 0x1a2b and, where unambiguous (see _bare_hex_ok), 1a2b. The scorer compares 0x literals by value,
    so 0x08, 0x0008 and 0x8 are one spelling; `digits` only adds the zero-padded bare form."""
    forms = [f"0x{v:x}", f"{v:x}"] + ([f"{v:0{digits}x}"] if digits else []) + ([f"x{v:02x}"] if digits == 2 else [])   # x3f: the C escape \\x3f
    return list(dict.fromkeys(f for f in forms if f.startswith(("0x", "x")) or _bare_hex_ok(f)))


def dec(v: int) -> list:
    """Spellings of a decimal answer; the 0x form of the same value is unambiguous so it is accepted too."""
    return [str(v), f"0x{v:x}"] if v >= 0 else [str(v)]


DECOY_NAMES = ["pad", "tmp", "seed", "salt", "tick", "stat", "mark", "hint", "slot", "note", "gap", "spin"]
DECOY_MARK = "// decoy"          # every decoy line ends with this: dedupe.strip_decoys removes them to get the program without decoys


def forms(a: int) -> set:
    """How the integer `a` prints in an instruction of `objdump -d`: its magnitude, and the 16/32/64-bit two's-complement pattern of
    its negation (`x - a` is `add edi, 0xffff5fff` for a = 0xa001); a negative `a` prints as its own pattern, also as an 8-bit one
    (`cmp al, 0x9c` for -100) when |a| < 256. A pattern of a value too wide for the width is a truncation of it, not a form of it, and
    is not offered: no low byte of a positive value, no 16-bit part of a 32-bit constant."""
    out = {abs(a)}
    for m in (0xffff, 0xffffffff, 0xffffffffffffffff):
        if abs(a) <= m:
            out.add((-a) & m if a > 0 else a & m)
    if a < 0 and -a < 256:
        out.add(a & 0xff)
    return out - {0}


def _decoy_statement(rng, n: str) -> tuple:
    """(two lines, the integer literals they use). A volatile local or a static counter, kept by every -O level."""
    c = rng.randrange(0x1000, 0x10000)
    kind, s = rng.randrange(5), rng.randint(1, 7)
    if kind == 0:
        return [f"volatile unsigned {n} = 0x{c:x}u;", f"{n} ^= {n} >> {s};"], {c, s}
    if kind == 1:
        return [f"static volatile unsigned {n};", f"{n} += 0x{c:x}u;"], {c}
    if kind == 2:
        m = rng.randint(4, 16)
        i, j = rng.randrange(m), rng.randrange(m)
        return [f"static volatile unsigned char {n}[{m}];", f"{n}[{i}] = (unsigned char)({n}[{j}] + 0x{c & 0xff:02x});"], {m, i, j, c & 0xff}
    if kind == 3:
        return [f"volatile unsigned {n} = {c}u;", f"{n} = ({n} << {s}) | ({n} >> {32 - s});"], {c, s, 32 - s}
    k = rng.randint(1, 999)
    return [f"static volatile unsigned short {n} = 0x{c:x};", f"{n} = (unsigned short)({n} + {k});"], {c, k}


def decoys(rng, count: int, forbidden=()) -> str:
    """`count` harmless statements for the top of a function: volatile locals and static counters that survive every -O level, so
    the compiled function has extra, unrelated constants (real code is full of them). Each carries random constants, which is what
    keeps small templates (a standard CRC, a base64 encoder, ChaCha) from being the same program in every task. NO literal of a
    statement (constants, shift counts, array sizes, indices) is a value in `forbidden` or one of its instruction forms: a decoy must
    never equal the key or a fact alternative, or the fact gate and a model reading the code could take it for the answer (m8). Every
    line ends with DECOY_MARK so the decoys can be stripped again."""
    ban = set()
    for v in forbidden:
        ban |= {v} | forms(v)
    out, names = [], rng.sample(DECOY_NAMES, count)
    for k, base in enumerate(names):
        for _ in range(500):
            lines, lits = _decoy_statement(rng, f"{base}{k}")
            if not lits & ban:
                break
        else:
            raise RuntimeError("no decoy statement avoids the forbidden values")
        out += lines
    return "".join(f"    {line} {DECOY_MARK}\n" for line in out)


def strip_decoys(body: str) -> str:
    """The body without its decoy lines."""
    return "\n".join(line for line in body.split("\n") if not line.rstrip().endswith(DECOY_MARK))


_PROC_OPEN = re.compile(r"\bproc\s*\([^)]*\)\s*\{\n")


def with_decoys(body: str, rng, count: int, forbidden=()) -> str:
    """`body` with `count` decoy statements as the first lines of proc. `forbidden`: every integer the question or the fact gate can
    ask for (the answer values and all fact alternatives)."""
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
    chance: float | None = None     # estimated probability that a guess is right (chance.py); None = legacy task

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
        if self.chance is not None:
            d["chance"] = self.chance
        d.update(self.truth)
        return d


FLAG_PATIENCE = 8     # a gated task is drawn again at the SAME compiler and -O level this many times before the flags change too


def pick_flags(family: str, seed: int, i: int, cc: str = "mix", attempt: int = 0) -> tuple:
    """Compiler and optimisation level for task i, from an rng that does not touch the template's own stream. A redraw keeps the flags of
    attempt 0 for the first FLAG_PATIENCE attempts (new parameters, same cc/opt) and only then draws flags of its own: if every redraw
    drew new flags, a variant whose facts vanish at one compiler or -O level would end up with the others (round-2 review, m1:
    bounds/char_class gcc 61% -> 38%, crc/bitwise_poly -O2 40% -> 18%)."""
    a = attempt if attempt >= FLAG_PATIENCE else 0
    rng = random.Random(f"flags/{family}/{seed}/{i}" + (f"/r{a}" if a else ""))
    opt = rng.choices(OPTS, OPT_WEIGHTS)[0]
    chosen = rng.choices(CCS, CC_WEIGHTS)[0]
    return opt, (chosen if cc == "mix" else cc)


MAX_TRIES = 60


def build(family: str, variants: dict, n: int, seed: int, cc: str = "mix", gate=None) -> list:
    """Task i is variant i mod len(variants), drawn from its own rng. With a `gate` (survive.Gate) a draw whose asked fact is not in
    the code compiled at its own cc/opt is thrown away and the task is drawn again (new parameters; the same cc/opt for the first
    FLAG_PATIENCE redraws, see pick_flags) from the stream "<family>/<seed>/<i>/r<attempt>"; attempt 0 is the stream of the ungated
    generator, so a gate only changes the tasks it rejects."""
    names = list(variants)
    out = []
    for i in range(n):
        for attempt in range(MAX_TRIES):
            t = draw(family, variants, i, seed, cc, attempt)
            if gate is None or gate(t):
                break
        else:
            raise RuntimeError(f"{family}/{names[i % len(names)]}: no draw in {MAX_TRIES} keeps its fact in the code (task {i}, seed {seed})")
        out.append(t)
    return out


def draw(family: str, variants: dict, i: int, seed: int, cc: str = "mix", attempt: int = 0) -> Task:
    """Draw `attempt` of task i. Which attempt a gate keeps depends on the compilers, the set of possible draws does not."""
    from .chance import CHANCE
    names = list(variants)
    rng = TaskRng(f"{family}/{seed}/{i}" + (f"/r{attempt}" if attempt else ""))
    rng.held = seed == HELDOUT_SEED
    t = variants[names[i % len(names)]](rng)
    t.id = f"{t.kind}-{seed}-{i:04d}"
    t.opt, t.cc = pick_flags(family, seed, i, cc, attempt)
    t.chance = CHANCE.get(f"{family}/{t.variant}")
    return t


def write_tasks(tasks, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out / f"{t.id}.c").write_text(t.source(), encoding="utf-8")
        (out / f"{t.id}.json").write_text(json.dumps(t.meta()), encoding="utf-8")

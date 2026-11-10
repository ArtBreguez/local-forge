"""The "fact survives" gate: is the thing a task asks for still in the compiled code?

A task's `facts` are groups of alternatives; every group needs one alternative present. An int must appear as a number in the
`objdump -d` of the compiled functions (every function but main: proc, its helpers, anything the compiler split off), compared by
magnitude (an offset prints as `[rdi+0x18]`, -500 as `+0x1f4`; also tried as its 8/16/32/64-bit two's-complement pattern, negated or not); a
string must appear in the object file (read-only data, or as immediates). The check runs at the task's own compiler and -O level.

This is a necessary condition, not proof that the decompiler shows it: a value below 0x10 is in most programs, a constant folded
into a lookup table or a magic multiply is gone. What it removes is the tasks the reviewer found unanswerable: TEA's round count
(the loop counter becomes a compare against DELTA*rounds), a CRC polynomial in a vectorised table loop, a character class that
became one bit-test against a 42-bit mask.

A "derived form" (hi - lo for a range tested as `v - lo <= width`, N/2 for a loop over double rounds) is a listed alternative of
its group, and the question of that variant says the compiler may show it that way.

    python3 tools/rea-bench survival --family all --n 120 --seed 5       survival rate per variant, gcc and clang, every -O level
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile

from .core import OPTS

OBJDUMP = shutil.which("objdump")
_HEX = re.compile(r"(?<![\w.])-?0x([0-9a-f]+)\b")
_DEC = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_FUNC = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")


def toolchain(cc: str):
    """Path of the compiler or None."""
    return shutil.which(cc)


def compile_object(src: str, cc: str, opt: str, out: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([cc, opt, "-fno-lto", "-w", "-c", "-o", str(out), "-x", "c", "-"], input=src, capture_output=True, text=True, timeout=60)


def numbers(disasm: str) -> set:
    """Magnitudes of every number in the instructions of the disassembly (functions other than main)."""
    out, keep = set(), False
    for line in disasm.splitlines():
        m = _FUNC.match(line.strip())
        if m:
            keep = m.group(1) != "main"
            continue
        if not keep or "\t" not in line:
            continue
        insn = re.sub(r"#.*|<[^>]*>", "", line.split("\t", 1)[1])
        out |= {int(h, 16) for h in _HEX.findall(insn)}
        out |= {int(d) for d in _DEC.findall(_HEX.sub(" ", insn))}
    return out


def _forms(a: int) -> set:
    """How `a` prints in an instruction: its magnitude, and its 8/16/32/64-bit two's-complement pattern, also negated (`x - lo` is
    `add edi, -lo` = `add edi, 0xfffffe0c`)."""
    return {abs(a)} | {v & m for v in (a, -a) for m in (0xff, 0xffff, 0xffffffff, 0xffffffffffffffff)} - {0}


def group_present(group: list, nums: set, blob: bytes) -> bool:
    for alt in group:
        if isinstance(alt, str):
            if alt.encode() in blob:
                return True
        elif _forms(alt) & nums:
            return True
    return False


def check(facts: list, src: str, cc: str, opt: str) -> tuple:
    """(True, "") when every fact group is present, else (False, what is missing). Raises RuntimeError when the toolchain is missing
    or the file does not compile."""
    if not OBJDUMP or not toolchain(cc):
        raise RuntimeError(f"the fact gate needs {cc} and objdump on PATH (gen --no-survive skips it)")
    with tempfile.TemporaryDirectory() as d:
        obj = pathlib.Path(d, "o.o")
        r = compile_object(src, cc, opt, obj)
        if r.returncode:
            raise RuntimeError(f"{cc} {opt} failed: {r.stderr.strip()[:200]}")
        dis = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn", "-M", "intel", str(obj)], capture_output=True, text=True, timeout=60).stdout
        nums, blob = numbers(dis), obj.read_bytes()
    missing = [g for g in facts if not group_present(g, nums, blob)]
    return (not missing, f"{cc} {opt}: none of {missing[0][:6]} in the code" if missing else "")


def check_task(t, cc: str | None = None, opt: str | None = None) -> tuple:
    """The gate for one task at its own compiler and -O level (or the ones given). Tasks without facts pass."""
    if not t.facts:
        return True, ""
    return check(t.facts, t.source(), cc or t.cc or "gcc", opt or t.opt)


class Gate:
    """Callable for `core.build`: True when the task's facts survive at its own cc/opt. Counts what it did."""

    def __init__(self):
        self.tried = self.rejected = 0
        self.by_variant: dict = {}

    def __call__(self, t) -> bool:
        if not t.facts:
            return True
        ok, _why = check_task(t)
        self.tried += 1
        self.rejected += not ok
        row = self.by_variant.setdefault(f"{t.kind}/{t.variant}", [0, 0])
        row[0] += 1
        row[1] += not ok
        return ok

    def summary(self) -> str:
        return f"fact gate: {self.rejected} of {self.tried} draws rejected and resampled"


def survival_rates(tasks, ccs=("gcc", "clang")) -> dict:
    """{(family/variant, cc): [present, total]} over every task and every -O level; tasks without facts are skipped."""
    out: dict = {}
    for t in tasks:
        if not t.facts:
            continue
        src = t.source()
        for cc in ccs:
            if not toolchain(cc):
                continue
            row = out.setdefault((f"{t.kind}/{t.variant}", cc), [0, 0])
            for opt in OPTS:
                row[0] += check(t.facts, src, cc, opt)[0]
                row[1] += 1
    return out

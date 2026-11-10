"""Verify generated tasks before they are sent anywhere expensive: the C compiles with no warning at every -O level, and the
harness a family ships with prints what the Python reference computed (so the truth the question rests on is proven).

    python3 tools/rea-bench check --family all --n 330 --seed 1001
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from . import scoring, survive
from .core import OPTS

CLANG = shutil.which("clang")
UBSAN_FLAGS = ("-fsanitize=undefined", "-fno-sanitize-recover=all")


# Replies with no answer in them, only common numbers (the reviewer's content-free fillers): no task may accept any of them.
FILLER_REPLIES = ["0 1 2", "1 2 3", "0x10 0x20", "0 1 2 3 4 8 16 32 64 255 256 0x10 0x20"]


def filler_passes(t) -> list:
    """The filler replies the task's scorer accepts (empty list = fine)."""
    return [r for r in FILLER_REPLIES if scoring.score(r, t.truth)]


def compile_check(src: str, opt: str, cc: str = "gcc") -> subprocess.CompletedProcess:
    """-Wall -Werror compile of the generated file alone (no link): any warning at this level fails. The object goes to a temp dir:
    `zig cc` (the clang wrapper on machines without one) ignores `-o /dev/null` for stdin input and leaves `-.o` in the cwd."""
    with tempfile.TemporaryDirectory() as d:
        return subprocess.run([cc, opt, "-Wall", "-Werror", "-c", "-o", str(pathlib.Path(d, "o.o")), "-x", "c", "-"], input=src,
                              capture_output=True, text=True, timeout=60)


def run_harness(src: str, opt: str, extra=(), cc: str = "gcc") -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as d:
        exe = pathlib.Path(d, "h")
        b = subprocess.run([cc, opt, "-Wall", *extra, "-o", str(exe), "-x", "c", "-"], input=src, capture_output=True, text=True, timeout=60)
        if b.returncode:
            return b
        return subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)


def ubsan_available() -> bool:
    r = subprocess.run(["gcc", *UBSAN_FLAGS, "-o", "/dev/null", "-x", "c", "-"], input="int main(void){return 0;}",
                       capture_output=True, text=True)
    return r.returncode == 0


def harness_runs(ubsan: bool) -> list:
    return [("-O0", ()), ("-O2", ())] + ([("-O1", UBSAN_FLAGS)] if ubsan else [])


def check_task(t, ubsan: bool = True, survive_check: bool = True) -> list:
    """Problems found with one task (empty list = fine): a filler reply scores, the C does not compile clean, a harness disagrees
    with the truth, or (survive_check) the asked constant is not in the code compiled at the task's own cc/opt."""
    problems = [f"{t.id} ({t.variant}) accepts the content-free reply {r!r}" for r in filler_passes(t)] if t.cc else []   # legacy: frozen, see cli
    if survive_check and t.facts and survive.OBJDUMP and survive.toolchain(t.cc or "gcc"):     # no compiler: the gate warned (survive.Gate)
        ok, why = survive.check_task(t)
        if not ok:
            problems.append(f"{t.id} ({t.variant}) fact does not survive: {why}")
    for cc in ["gcc"] + (["clang"] if CLANG else []):
        for opt in OPTS:
            r = compile_check(t.source(), opt, cc)
            if r.returncode:
                problems.append(f"{t.id} ({t.variant}) {cc} {opt}: {r.stderr.strip()[:300]}")
    if t.check:
        main_c, want, *pre = t.check
        src = t.source(main_c, pre[0] if pre else "")
        for opt, extra in harness_runs(ubsan):
            r = run_harness(src, opt, extra)
            if r.returncode or r.stdout != want:
                problems.append(f"{t.id} ({t.variant}) harness {opt} {' '.join(extra)}: exit {r.returncode}, "
                                f"{(r.stderr.strip() or 'output differs: ' + repr(r.stdout[:120]) + ' != ' + repr(want[:120]))[:300]}")
    return problems

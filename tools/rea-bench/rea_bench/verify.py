"""Verify generated tasks before they are sent anywhere expensive: the C compiles with no warning at every -O level, and the
harness a family ships with prints what the Python reference computed (so the truth the question rests on is proven).

    python3 tools/rea-bench check --family all --n 330 --seed 1001
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from .core import OPTS

CLANG = shutil.which("clang")
UBSAN_FLAGS = ("-fsanitize=undefined", "-fno-sanitize-recover=all")


def compile_check(src: str, opt: str, cc: str = "gcc") -> subprocess.CompletedProcess:
    """-Wall -Werror compile of the generated file alone (no link): any warning at this level fails."""
    return subprocess.run([cc, opt, "-Wall", "-Werror", "-c", "-o", "/dev/null", "-x", "c", "-"], input=src,
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


def check_task(t, ubsan: bool = True) -> list:
    """Problems found with one task (empty list = fine)."""
    problems = []
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

"""Round-2 review of the REA task generator, the CLI without clang (M5): legacy needs no compiler, the gate degrades with a warning."""
from __future__ import annotations

import collections
import contextlib
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from .helpers import ROOT

RB = ROOT / "tools" / "rea-bench"
sys.path.insert(0, str(RB))

from rea_bench import core, dedupe, families, scoring, survive, verify  # noqa: E402
from rea_bench.core import DECOY_MARK, strip_decoys  # noqa: E402

NEW = [f for f in families.modules() if f != "legacy"]
GCC_OBJDUMP = bool(shutil.which("gcc") and shutil.which("objdump"))
CLANG = bool(shutil.which("clang"))
LIT = re.compile(r"\b(0x[0-9a-fA-F]+|\d+)[uUlL]*\b")


def lits(text: str) -> set:
    return {int(x, 16) if x.lower().startswith("0x") else int(x) for x in re.findall(r"\b(0x[0-9a-fA-F]+|\d+)[uUlL]*\b", text)}


def mutate_asked_literal(t, new=None):
    """The task's source with the asked literal (the probe's anchor) changed: the compiled program no longer has the task's constant."""
    spec = next(s for g in t.facts for s in [survive.split_group(g)[1]] if s)
    src = t.source()
    m = re.search(spec["at"], src)
    old = m.group(1)
    if new is None:
        v = survive._parse_int(old)
        new = f"0x{v ^ 0x3c3c:0{max(len(old) - 2, 1)}x}" if old.lower().startswith("0x") else str(v + 13)
    return src[:m.start(1)] + new + src[m.end(1):]


class NoClang(unittest.TestCase):
    """M5: the CLI and the gate need no compiler for a family without facts, and degrade with a warning for clang draws."""

    def run_cli(self, *args, path="/usr/bin:/bin"):
        env = {**os.environ, "PATH": path}
        return subprocess.run([sys.executable, str(RB), *args], capture_output=True, text=True, env=env, timeout=600)

    def test_legacy_runs_without_any_compiler(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("--n", "3", "--seed", "1", "--out", d, path=d)           # a PATH with no gcc, clang or objdump at all
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = self.run_cli("gen", "--family", "legacy", "--n", "3", "--seed", "1", "--out", d + "/g", path=d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("needs", r.stdout + r.stderr)

    def test_the_gate_warns_once_and_counts_what_it_could_not_check(self):
        msgs = []
        gate = survive.Gate(warn=msgs.append)
        gate.available = lambda cc: cc == "gcc"
        t_clang = next(t for t in families.generate("crc", 40, 3) if t.cc == "clang" and t.facts)
        t_none = families.generate("structs", 1, 3)[0]
        t_none.facts = []
        self.assertTrue(gate(t_clang) and gate(t_clang) and gate(t_none))
        self.assertEqual(len(msgs), 1)
        self.assertIn("clang", msgs[0])
        self.assertEqual(gate.unchecked, {"clang": 2})
        self.assertIn("2 draws with clang not checked", gate.summary())
        self.assertEqual(gate.tried, 0)

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_a_machine_without_clang_gates_the_gcc_draws(self):
        with tempfile.TemporaryDirectory() as d:
            for tool in ("gcc", "objdump", "as", "ld"):
                if shutil.which(tool):
                    os.symlink(shutil.which(tool), os.path.join(d, tool))
            r = self.run_cli("gen", "--family", "crc,cipher", "--n", "12", "--seed", "3", "--out", d + "/o", path=d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("NOT checked", r.stderr)
            self.assertIn("not checked", r.stdout)


if __name__ == "__main__":
    unittest.main()

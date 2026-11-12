"""Round-2 review of the REA task generator, the fact gate (M1) and the decoys (m8): a changed constant is not the task's fact, a
truncation of a value is not the value, the compiler's folded constants and tables fail the probe, no decoy literal equals an answer."""
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
        alts = [a for g in t.facts for a in survive.split_group(g)[0] if isinstance(a, int)]
        if old.lower().startswith("0x"):
            new = f"0x{v ^ 0x3c3c:0{max(len(old) - 2, 1)}x}"
        else:       # a value that is not one of the group's own alternatives (a half-round loop bound + 13 can be the round count itself)
            new = str(next(v + d for d in (13, 29, 47, 71) if v + d not in alts))
    return src[:m.start(1)] + new + src[m.end(1):]


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class GateMutation(unittest.TestCase):
    """M1: change the asked constant in the C source, keep the facts, ask the gate: it must say no. (The reviewer's gate_mutation.py
    let ChaCha/Salsa through 40/40, crc/table_poly 20/40, ptrace 2/40 ...)"""

    def test_a_changed_constant_is_not_the_task_s_fact(self):
        seen = collections.Counter()
        for fam in ("cipher", "crc", "antidebug"):
            m = families.modules()[fam]
            for t in families.generate(fam, 3 * len(m.VARIANTS), 5):
                if not any(survive.split_group(g)[1] for g in t.facts):
                    continue
                for opt in ("-O1", "-O2"):
                    self.assertFalse(survive.check(t.facts, mutate_asked_literal(t), "gcc", opt)[0], f"{t.id} ({t.variant}) {opt}")
                seen[f"{fam}/{t.variant}"] += 1
        self.assertEqual({v for k in seen for v in [k.split("/")[1]]} & {"chacha", "salsa", "tea_enc", "xtea_dec", "bitwise_poly", "table_poly", "ptrace", "tracerpid", "timing_ns"},
                         {"chacha", "salsa", "tea_enc", "xtea_dec", "bitwise_poly", "table_poly", "ptrace", "tracerpid", "timing_ns"})

    def test_an_original_task_passes_at_its_own_flags(self):
        gate = survive.Gate()
        for fam in ("cipher", "crc", "antidebug"):
            for t in families.generate(fam, 14, 6, "gcc", gate):
                self.assertTrue(survive.check_task(t)[0], f"{t.id} ({t.variant}) {t.cc} {t.opt}")

    def test_the_round_count_of_chacha_is_a_value_the_function_returns(self):
        """A round count of 8 or 12 is a rotation constant of every ChaCha/Salsa; proc now returns the count it ran."""
        for alg in ("chacha", "salsa"):
            for style_seed in range(1, 40):
                t = next(t for t in families.generate("cipher", 7, style_seed) if t.variant == alg)
                self.assertIn("return done;", t.body)
                self.assertIn("done += 2;", t.body)
                self.assertTrue(t.check[1].endswith(f"{t.params['rounds']}\n"), "the harness checks the returned count")
        styles = {next(t for t in families.generate("cipher", 7, s) if t.variant == "chacha").params["style"] for s in range(1, 40)}
        self.assertEqual(styles, {0, 1, 2}, "all three loop shapes are probed (the `i++` one had a wrong anchor once)")

    def test_a_constant_the_compiler_folds_away_does_not_show_its_marker(self):
        """gcc -O2 turns a counting loop that only adds a constant into one product: the count (7) is gone from the code."""
        src = core.HDR + "unsigned proc(unsigned x) {\n    unsigned i, s = x;\n    for (i = 0; i < 300; i++) s += 0x9e3779b9u;\n    return s;\n}\n" + core.MAIN
        probe = {"at": r"i < (\d+);", "kind": "near"}
        self.assertFalse(survive.check([[299, 300, 301, probe]], src, "gcc", "-O2")[0])
        returned = (core.HDR + "unsigned proc(unsigned *out) {\n    unsigned i, s = *out;\n    for (i = 0; i < 300; i++) s += 0x9e3779b9u;\n"
                    "    *out = s;\n    return i;\n}\n" + core.MAIN)
        self.assertTrue(survive.check([[299, 300, 301, probe]], returned, "gcc", "-O2")[0], "a returned counter stays a constant")

    def test_a_crc_poly_fails_when_the_compiler_made_a_table(self):
        rows = ", ".join(f"0x{(i * 2654435761) & 0xffffffff:08x}u" for i in range(256))
        src = (core.HDR + f"static const unsigned T[256] = {{{rows}}};\nunsigned proc(const unsigned char *p, unsigned n) {{\n"
               "    unsigned crc = 0xffffffffu;\n    while (n--) crc = T[(crc ^ *p++) & 255u] ^ (crc >> 8) ^ 0xedb88320u;\n    return crc;\n}\n" + core.MAIN)
        probe = {"at": r"\^ (0x[0-9a-f]{8})u;", "kind": "exact", "hex": True, "bits": 32}
        self.assertTrue(survive.check([[0xedb88320, probe]], src, "gcc", "-O2")[0])
        ok, why = survive.check([[0xedb88320, {**probe, "no_table": True}]], src, "gcc", "-O2")
        self.assertFalse(ok)
        self.assertIn("table", why)

    def test_an_unknown_source_literal_is_not_accepted(self):
        src = core.HDR + "int proc(int v) { return v == 77 ? 200 : 0; }\n" + core.MAIN
        self.assertTrue(survive.check([[200, {"at": r"\? (\d+) :", "kind": "exact"}]], src, "gcc", "-O2")[0])
        self.assertFalse(survive.check([[201, {"at": r"\? (\d+) :", "kind": "exact"}]], src, "gcc", "-O2")[0])
        self.assertFalse(survive.check([[200, {"at": r"nowhere (\d+)", "kind": "exact"}]], src, "gcc", "-O2")[0])


class Decoys(unittest.TestCase):
    """The exclusion strips decoys by their marker; m8: no decoy literal is an answer or a fact alternative."""

    def test_every_decoy_line_is_marked_and_stripping_leaves_the_program(self):
        for fam in NEW:
            for t in families.generate(fam, 150, 4):
                marked = [l for l in t.body.split("\n") if l.rstrip().endswith(DECOY_MARK)]
                names = [l for l in t.body.split("\n") if re.search(r"\b(?:%s)\d\b" % "|".join(core.DECOY_NAMES), l)]
                self.assertEqual(marked, names, f"{t.id}: a decoy line without its marker (or a marker on a real line)")
                self.assertEqual(len(marked) % 2, 0)
                self.assertNotIn(DECOY_MARK, strip_decoys(t.body))
                self.assertEqual(strip_decoys(t.body).count("\n") + len(marked), t.body.count("\n"))

    def test_no_decoy_literal_equals_an_answer_or_a_fact_alternative(self):
        """m8 (reviewer: pktsum 5-13% of tasks with a decoy literal equal to the key, rc4 5%, chacha/salsa 4% and 28% equal to a fact alternative)."""
        bad, total = [], 0
        for fam in NEW:
            m = families.modules()[fam]
            for t in families.generate(fam, 250 * len(m.VARIANTS), 5):
                dl = "\n".join(l for l in t.body.split("\n") if l.rstrip().endswith(DECOY_MARK))
                if not dl:
                    continue
                total += 1
                dlits = lits(dl)
                keys = set()
                for g in t.truth["accept"]:
                    for a in (g if isinstance(g, list) else []):
                        v = scoring.num_value(a.lower())
                        if v is not None:
                            keys.add(v)
                facts = {a for g in t.facts for a in survive.split_group(g)[0] if isinstance(a, int)}
                if dlits & keys or dlits & facts:
                    bad.append((t.id, t.variant, sorted(dlits & (keys | facts))))
        self.assertGreater(total, 4000)
        self.assertEqual(bad[:5], [], f"{len(bad)} of {total} tasks have a decoy literal equal to an answer or a fact alternative")


if __name__ == "__main__":
    unittest.main()

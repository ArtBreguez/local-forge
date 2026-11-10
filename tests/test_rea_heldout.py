"""Round-2 review of the REA task generator, held-out (M2): the held-out seed is disjoint from training as a PROGRAM (decoys stripped),
the decoy marker the exclusion relies on, reserved parameter values, and the refusal of a seed-1001 run beyond the pinned range (m7)."""
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


class HeldOutProgram(unittest.TestCase):
    def test_decoys_differ_but_the_program_key_does_not(self):
        rng_a, rng_b = core.TaskRng("a"), core.TaskRng("b")
        body = "int proc(int v) {\n    return v + 1;\n}"
        a, b = core.with_decoys(body, rng_a, 3), core.with_decoys(body, rng_b, 3)
        self.assertNotEqual(a, b)
        self.assertEqual(strip_decoys(a), body)
        self.assertEqual(dedupe.normalize(strip_decoys(a)), dedupe.normalize(strip_decoys(b)))

        class T:
            def __init__(self, body):
                self.body, self.question = body, "q"
        self.assertEqual(dedupe.task_key(T(a)), dedupe.task_key(T(b)))
        self.assertNotEqual(dedupe.body_key(T(a)), dedupe.body_key(T(b)))


class HeldOutIsDisjoint(unittest.TestCase):
    """M2: held-out is disjoint as a PROGRAM (decoys stripped), not as a text: the reviewer found 60% of the held-out encoding tasks,
    32% of crc, 28% of cipher and 15% of antidebug with a training twin that differed only in its decoys."""

    def test_no_training_task_has_the_program_of_a_held_out_task(self):
        """Raw generation against the pinned list (the first draw of each held-out slot): 0 for every new family, without any filter."""
        held = dedupe.load_heldout()
        for seed in (7, 8):
            for fam in NEW:
                tasks = families.generate(fam, 3000, seed)
                self.assertEqual(sum(dedupe.task_key(t) in held for t in tasks), 0, f"{fam} seed {seed}")

    def test_the_pipeline_also_drops_the_later_draws_of_a_held_out_slot(self):
        """Against EVERY draw the gate could keep in a held-out slot (HELDOUT_ATTEMPTS per slot) a few raw training tasks coincide in the
        families with small program spaces (structs offsets); training_tasks drops exactly those."""
        every = dedupe.heldout_keys()
        kept, dropped = dedupe.training_tasks(NEW, 3000 * len(NEW), 7)
        self.assertEqual([t.id for t in kept if dedupe.task_key(t) in every], [])
        self.assertLess(len(dropped), 60, "a handful, not a share of the data")
        self.assertTrue(all(dedupe.task_key(t) in every for t in dropped))

    def test_no_held_out_task_has_a_training_twin(self):
        for fam in NEW:
            ho = {dedupe.task_key(t) for t in families.generate(fam, dedupe.HELDOUT_N, 1001)}
            for seed in (7, 8):
                tr = {dedupe.task_key(t) for t in families.generate(fam, 3000, seed)}
                self.assertEqual(len(ho & tr), 0, f"{fam} seed {seed}")

    def test_the_reserved_values_are_the_held_out_ones(self):
        from rea_bench import f_crc, f_cipher
        held_labels = {a[0] for a in f_crc.HELD_STANDARD}
        for t in families.generate("crc", 600, 1001):
            if t.params["label"] != "custom":
                self.assertIn(t.params["label"], held_labels)
        for t in families.generate("crc", 600, 7):
            self.assertNotIn(t.params["label"], held_labels)
        for t in families.generate("cipher", 700, 1001):
            if t.variant in ("chacha", "salsa"):
                self.assertIn(t.params["rounds"], f_cipher.HELD_ROUNDS)
        for t in families.generate("cipher", 700, 7):
            if t.variant in ("chacha", "salsa"):
                self.assertNotIn(t.params["rounds"], f_cipher.HELD_ROUNDS)
        for fam, key in (("antidebug", "rv"),):
            for seed, mult in ((1001, True), (7, False)):
                for t in families.generate(fam, 300, seed):
                    if key in t.params:
                        self.assertEqual(t.params[key] % 5 == 0, mult, t.id)
        for t in families.generate("encoding", 400, 1001):
            self.assertTrue(t.params["custom"], "the held-out seed draws shuffled alphabets only")
        self.assertTrue(any(not t.params["custom"] for t in families.generate("encoding", 400, 7)))

    def test_a_seed_1001_run_beyond_the_listed_range_is_refused(self):
        dedupe.check_heldout_range(["crc"], 120, 1001)
        dedupe.check_heldout_range(["crc", "cipher"], 240, 1001)
        dedupe.check_heldout_range(["legacy"], 5000, 7)
        with self.assertRaises(SystemExit) as cm:
            dedupe.check_heldout_range(["legacy"], 300, 1001)         # the default legacy run: 180 tasks the list does not cover
        self.assertIn("120", str(cm.exception))
        with self.assertRaises(SystemExit):
            dedupe.training_tasks(["crc", "cipher"], 250, 1001)
        self.assertEqual(dedupe.heldout_overflow([{"kind": "xor-key"}] * 121 + [{"kind": "crc"}] * 3), {"legacy": 121})
        self.assertEqual(dedupe.family_of_kind("crc"), "crc")
        self.assertEqual(dedupe.family_of_kind("skip"), "legacy")


if __name__ == "__main__":
    unittest.main()

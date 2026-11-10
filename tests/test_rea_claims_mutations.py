"""Round 6, M1: the claims table must reject a FALSE SENTENCE, not only a false fact.

The reviewer of round 6 made 79 one-edit false sentences (two numbers in swapped roles, a flipped direction word, a wrong name, a wrong unit, a structure
word) and found that the round-5 claims suite caught 2 of them and the whole suite 13: it checked the facts that rendered a sentence, nothing tied the words of
the sentence to them. They are ported here (tests/rea_mutations_data.py, adjusted where a round-6 rewording changed the text they edit) and applied one by one to
a temp copy of tools/rea-bench/rea_bench/targets.py, which is loaded as a module and rendered exactly as the suite renders the real one: every sentence the
mutation changed is rendered for a few tasks of the keys it touches, parsed with its pattern and its claims run (`test_rea_claims.exercise`). The claims
machinery alone must fail: the golden file (which sees any edit) is not consulted. A mutation that passes is a sentence that can be false and stay in the data.

EXTRA are mutations of the same kinds on sentences the reviewer did not touch (a unit, a request name, a field, a byte order, a count word, the two sentences
that were false in rounds 5 and 6).

    REA_MUT_IDS=S1,F2 python3 -m unittest tests.test_rea_claims_mutations     a subset
"""
from __future__ import annotations

import collections
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

from .helpers import ROOT
from . import test_rea_claims as T
from .rea_mutations_data import MUTATIONS

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, targets  # noqa: E402

SRC = pathlib.Path(targets.__file__)
TASKS_PER_KEY = 3
SEED = 4242

EXTRA = [
    ("X1", "pktsum", "quantifier", "The bytes from offset {n} up to the stored checksum are the bytes that the checksum is computed over.",
     "Changing any byte of the packet from offset {n} up to the stored checksum makes the checksum fail."),
    ("X2", "legacy", "false for NUL", "or {miss} for any other printable character.", "or {miss} when it is not there."),
    ("X3", "antidebug", "unit", "reports a debugger when it is above {thr} nanoseconds.", "reports a debugger when it is above {thr} microseconds."),
    ("X4", "antidebug", "unit", "The largest time between the two clock readings that proc still lets pass is {thr} microseconds.",
     "The largest time between the two clock readings that proc still lets pass is {thr} nanoseconds."),
    ("X5", "legacy", "direction", "A rotation to the left by {r} bits comes after the XOR with 0x{k:08x}.", "A rotation to the right by {r} bits comes after the XOR with 0x{k:08x}."),
    ("X6", "legacy", "byte order", "read little-endian, must be 0x{m:08x}", "read big-endian, must be 0x{m:08x}"),
    ("X7", "antidebug", "field", "proc reads the TracerPid line of /proc/self/status", "proc reads the Uid line of /proc/self/status"),
    ("X8", "antidebug", "request", "When ptrace(PTRACE_TRACEME) fails because", "When ptrace(PTRACE_CONT) fails because"),
    ("X9", "strobf", "count", "followed by a terminating NUL byte.", "followed by two terminating NUL bytes."),
    ("X10", "cipher", "name", "The rotation amounts are {rot[0]}, {rot[1]}, {rot[2]} and {rot[3]} bits, as in {alg}", "The rotation amounts are {rot[0]}, {rot[1]}, {rot[2]} and {rot[3]} bits, as in AES"),
    ("X11", "legacy", "number role", "the multiplier {a}, applied {r} times.", "the multiplier {r}, applied {a} times."),
    ("X12", "bounds", "direction", "A value below {lo} becomes {lo}", "A value above {lo} becomes {lo}"),
    ("X13", "fsm", "number role", "numbered 0 to {last}, and the machine starts in state 0.", "numbered 0 to {n}, and the machine starts in state 0."),
]
FAMILY_OF_ID = {"S": "structs", "F": "fsm", "C": "crc", "H": "cipher", "E": "encoding", "T": "strobf", "P": "pktsum", "B": "bounds", "A": "antidebug",
                "U": "multi", "L": "legacy"}
_TASKS: dict = {}


def load_mutated(old: str, new: str):
    """targets.py with `old` replaced by `new` (which must occur exactly once), loaded as a module of the package from a temp copy."""
    text = SRC.read_text()
    assert text.count(old) == 1, f"the text to edit occurs {text.count(old)} times: {old[:80]!r}"
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "targets.py"
        path.write_text(text.replace(old, new))
        spec = importlib.util.spec_from_file_location("rea_bench._targets_mutated", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def affected_sentences(mod) -> dict:
    """{key: {index of every sentence of the template whose text differs from the real one}}"""
    out = {}
    for key, spec in mod.SPECS.items():
        idx = {i for i, s in enumerate(spec.sentences) if s != targets.SPECS[key].sentences[i]}
        if idx:
            out[key] = idx
    return out


def tasks_of(key: str) -> list:
    fam = key.split("/")[0] if "/" in key else "legacy"
    if fam not in _TASKS:
        mod = families.modules()[fam]
        _TASKS[fam] = families.generate(fam, TASKS_PER_KEY * 2 * len(mod.VARIANTS), SEED)
    base = key.split("#")[0]
    return [t for t in _TASKS[fam] if targets.key_of(t.meta()).split("#")[0] == base][:TASKS_PER_KEY]


def caught_by(mod, affected: dict):
    """(task id, claim, why) of the first failure of the table on the mutated sentences, or None."""
    for key in affected:
        for task in tasks_of(key):
            for k, name, why in T.exercise(task, T=mod, affected=affected, stop=True):
                if why:
                    return task.id, name, why
    return None


class MutationsAreCaught(unittest.TestCase):
    """One test per mutation (generated below)."""


def _mutation_test(mid, fam, kind, old, new):
    def test(self):
        mod = load_mutated(old, new)
        aff = affected_sentences(mod)
        self.assertTrue(aff, f"{mid}: the edit changes no sentence")
        hit = caught_by(mod, aff)
        self.assertIsNotNone(hit, f"{mid} ({fam}, {kind}) is a false sentence that the claims table accepts: {old!r} -> {new!r}\n  keys: {sorted(aff)}")
    test.__name__ = f"test_{mid}_{fam}_{kind.replace(' ', '_').replace('/', '_')}"
    return test


_WANT = set(filter(None, os.environ.get("REA_MUT_IDS", "").split(",")))
for _row in list(MUTATIONS) + EXTRA:
    if not _WANT or _row[0] in _WANT:
        setattr(MutationsAreCaught, f"test_{_row[0]}", _mutation_test(*_row))


class MutationCorpus(unittest.TestCase):
    def test_there_are_79_reviewer_mutations_in_the_11_families_and_each_edits_the_text_once(self):
        self.assertEqual(len(MUTATIONS), 79)
        text = SRC.read_text()
        for mid, fam, kind, old, new in MUTATIONS + EXTRA:
            self.assertEqual(text.count(old), 1, f"{mid}: the text to edit does not occur exactly once")
            self.assertNotEqual(old, new, mid)
            self.assertEqual(FAMILY_OF_ID.get(mid[0]) if mid[0] != "X" else fam, fam, mid)
        per = collections.Counter(fam for _i, fam, *_r in MUTATIONS)
        self.assertEqual(sorted(per), sorted(FAMILY_OF_ID.values()))

    def test_the_machinery_accepts_the_real_templates(self):
        """The control: with no edit, no sentence of any of the families fails (a quick run over the same tasks)."""
        for key in ("structs/field_read", "bounds/clamp", "pktsum/fletcher16", "antidebug/timing_us", "hash"):
            task = tasks_of(key)[0]
            res = T.exercise(task, stop=True)
            self.assertEqual([r for r in res if r[2]], [], key)


if __name__ == "__main__":
    unittest.main()

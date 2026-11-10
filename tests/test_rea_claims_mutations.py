"""Rounds 6 and 7: the claims table must reject a FALSE SENTENCE, a FALSE HEADLINE and a FALSE FACT, not only the fact that rendered them.

Round 6: the reviewer made 79 one-edit false sentences (two numbers in swapped roles, a flipped direction word, a wrong name, a wrong unit, a structure word) and
found that the round-5 claims suite caught 2 of them and the whole suite 13: it checked the facts that rendered a sentence, nothing tied the words of the sentence to
them. Round 7: 109 more (89 sentences, fact functions and constants, 12 headlines, 8 facts that are wrong for about 20% of the tasks) and the claims machinery read
no headline at all: 2 of the 12 headline mutations passed every test but the golden file. Both sets are in tests/rea_mutations_data.py and are applied one by one
to a temp copy of tools/rea-bench/rea_bench/targets.py, which is loaded as a module and rendered exactly as the suite renders the real one: the sentences and
headlines that the mutation changes on the sampled tasks (compared render by render with the real module: text and facts) are parsed with their patterns and their
claims run (`test_rea_claims.exercise`). The claims machinery alone must fail: the golden file (which sees any edit of a template) is not consulted, and a fact or
a constant edit changes no template text at all. A mutation that passes is a sentence, a headline or a fact that can be false and stay in the data.

Round 7, m5: the sample is TASKS_PER_KEY (12) tasks per key or more: the first task of every code form of the key (test_rea_claims.form_keys, the forms that
`ClaimsHold` counts), then the next ones. With 3 tasks per key, 6 of the reviewer's 40 partial runs (a fact that is wrong for the tasks whose id has sum(ord) % 5 == r)
were not even exercised and the hash name and offset mutations (NL3, NL4) changed nothing on the 3 sampled hash tasks. The 8 partial mutations run at the 5 residues
r = 0..4: 40 runs.

EXTRA are mutations of the same kinds on sentences the reviewer did not touch (a unit, a request name, a field, a byte order, a count word, the two sentences
that were false in rounds 5 and 6).

    REA_MUT_IDS=S1,NHD1 python3 -m unittest tests.test_rea_claims_mutations     a subset
"""
from __future__ import annotations

import collections
import importlib.util
import os
import pathlib
import sys
import tempfile
import time
import unittest

from .helpers import ROOT
from . import test_rea_claims as T
from .rea_mutations_data import MUTATIONS, MUTATIONS_R7

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, targets  # noqa: E402

SRC = pathlib.Path(targets.__file__)
TASKS_PER_KEY = 12          # at least this many tasks per key (round 7, m5: 3 left 6 of 40 partial mutations unexercised); the sample also covers every code form of the key
POOL_PER_VARIANT = 60       # tasks drawn per variant to choose the sample from
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
FAMILIES = sorted(FAMILY_OF_ID.values())
ROWS = [(mid, fam, kind, None, old, new) for mid, fam, kind, old, new in list(MUTATIONS) + EXTRA] + list(MUTATIONS_R7)    # (id, family, kind, ctx, old, new)
# the partial mutations (NQ1-NQ8) are wrong for the tasks whose id has sum(ord) % 5 == 0; the other four residues are four more subsets of the draws (the reviewer's 40 runs)
PARTIAL = [(f"{mid}r{r}", fam, kind + f" (residue {r})", ctx, old, new.replace("% 5 == 0", f"% 5 == {r}")) for mid, fam, kind, ctx, old, new in MUTATIONS_R7 if fam == "partial"
           for r in (1, 2, 3, 4)]
ROWS += PARTIAL
_TASKS: dict = {}


def apply_edit(text: str, ctx, old: str, new: str) -> str:
    """`old` replaced by `new`: where it occurs exactly once (ctx None), or where it first occurs after `ctx` (which occurs exactly once)."""
    if ctx is None:
        assert text.count(old) == 1, f"the text to edit occurs {text.count(old)} times: {old[:80]!r}"
        return text.replace(old, new)
    assert text.count(ctx) == 1, f"the context occurs {text.count(ctx)} times: {ctx[:80]!r}"
    a = text.index(ctx)
    b = text.index(old, a)
    return text[:b] + new + text[b + len(old):]


def load_mutated(old: str, new: str, ctx=None):
    """targets.py with `old` replaced by `new`, loaded as a module of the package from a temp copy."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "targets.py"
        path.write_text(apply_edit(SRC.read_text(), ctx, old, new))
        spec = importlib.util.spec_from_file_location("rea_bench._targets_mutated", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def family_of(key: str) -> str:
    return key.split("/")[0] if "/" in key else "legacy"


def choose(tasks: list) -> list:
    """The sample of one key: the first task of every code form (the values of the generator's form parameters and the forms only the C shows, as `ClaimsHold` counts
    them: test_rea_claims.form_keys), then the next tasks until there are TASKS_PER_KEY. A sentence that is false for one form only (or for the draws of one parameter
    class) is rendered for a task of that form."""
    chosen, covered = [], set()
    for t in tasks:
        new = set(T.form_keys(t)) - covered
        if new:
            chosen.append(t)
            covered |= new
    for t in tasks:
        if len(chosen) >= TASKS_PER_KEY:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen


def sample(fam: str) -> dict:
    """{key: the tasks of the family that are of this key (`choose`)}: the sample that every mutation of the family is rendered over."""
    if fam not in _TASKS:
        mod = families.modules()[fam]
        pool = families.generate(fam, POOL_PER_VARIANT * len(mod.VARIANTS), SEED)
        by_key: dict = collections.defaultdict(list)
        for t in pool:
            by_key[targets.key_of(t.meta()).split("#")[0]].append(t)
        _TASKS[fam] = {k: choose(v) for k, v in by_key.items()}
    return _TASKS[fam]


def tasks_of(key: str) -> list:
    return sample(family_of(key)).get(key.split("#")[0], [])


def families_touched(mod, fam: str) -> list:
    """The families whose tasks a mutation can change: its own; a headline mutation: the families of the keys whose headlines differ; a partial one: all of them."""
    if fam == "head":
        return sorted({family_of(k) for k, s in mod.SPECS.items() if s.heads != targets.SPECS[k].heads})
    return FAMILIES if fam == "partial" else [fam]


def _render(T_, rec, j, k):
    try:
        r = T_.render(rec, j, k)
        return r["sentence"], r["headline"], r["facts"]
    except Exception as e:                                        # a template that no longer renders is a change
        return f"{type(e).__name__}: {e}", f"{type(e).__name__}: {e}", None


def affected_renders(mod, fam: str) -> dict:
    """{task id: (task, {key: {sentence indexes, "h<row>" for a headline row}})}: what the mutation changes on the sampled tasks, render by render (the text of the
    sentence or of the headline, or the facts that rendered it), compared with the real templates. A fact or a constant edit changes no template text."""
    out = {}
    for f in families_touched(mod, fam):
        for tasks in sample(f).values():
            for task in tasks:
                items: dict = collections.defaultdict(set)
                for rec in [task.meta()] + T.other_questions(task):
                    key = targets.key_of(rec)
                    try:
                        sentences = list(T.applicable(key, mod.SPECS[key].facts(rec), mod))
                        heads = T.head_slots(key, mod.SPECS[key].facts(rec), mod)
                    except Exception:                             # the facts no longer compute: every sentence and headline of the key is changed
                        sentences, heads = list(range(len(targets.SPECS[key].sentences))), [(r, r) for r in range(len(targets.SPECS[key].heads))]
                    for j, i in enumerate(sentences):
                        a, b = _render(mod, rec, j, 0), _render(targets, rec, j, 0)
                        if a[0] != b[0] or a[2] != b[2]:
                            items[key].add(i)
                    for row, k in heads:
                        a, b = _render(mod, rec, 0, k), _render(targets, rec, 0, k)
                        if a[1] != b[1] or a[2] != b[2]:
                            items[key].add(f"h{row}")
                if items:
                    out[task.id] = (task, dict(items))
    return out


def caught_by(mod, affected: dict):
    """(task id, claim, why) of the first failure of the claims machinery on the mutated sentences and headlines, or None."""
    for task, items in affected.values():
        for key, name, why in T.exercise(task, T=mod, affected=items, stop=True):
            if why:
                return task.id, name, why
    return None


class MutationsAreCaught(unittest.TestCase):
    """One test per mutation (generated below)."""


def _mutation_test(mid, fam, kind, ctx, old, new):
    def test(self):
        mod = load_mutated(old, new, ctx)
        aff = affected_renders(mod, fam)
        self.assertTrue(aff, f"{mid}: the edit changes no sentence, no headline and no fact on the {TASKS_PER_KEY} sampled tasks per key of its families")
        hit = caught_by(mod, aff)
        self.assertIsNotNone(hit, f"{mid} ({fam}, {kind}) is false and the claims table accepts it: {old!r} -> {new!r}\n  keys: {sorted({k for _t, it in aff.values() for k in it})}")
    test.__name__ = f"test_{mid}_{fam}_{kind.replace(' ', '_').replace('/', '_')}"
    return test


_WANT = set(filter(None, os.environ.get("REA_MUT_IDS", "").split(",")))
for _row in ROWS:
    if not _WANT or _row[0] in _WANT:
        setattr(MutationsAreCaught, f"test_{_row[0]}", _mutation_test(*_row))


class MutationCorpus(unittest.TestCase):
    def test_there_are_79_round_6_and_109_round_7_mutations_and_each_edits_the_text_once(self):
        self.assertEqual(len(MUTATIONS), 79)
        self.assertEqual(len(MUTATIONS_R7), 109)
        self.assertEqual(len({r[0] for r in ROWS}), len(ROWS), "a mutation id is used twice")
        self.assertEqual(len(PARTIAL), 32, "the partial mutations: 8 x 4 more residues of sum(ord(id)) % 5")
        text = SRC.read_text()
        for mid, fam, kind, ctx, old, new in ROWS:
            self.assertNotEqual(old, new, mid)
            self.assertNotEqual(apply_edit(text, ctx, old, new), text, mid)       # (apply_edit asserts that the edit applies exactly once)
        for mid, fam, *_r in MUTATIONS + EXTRA:
            self.assertEqual(FAMILY_OF_ID.get(mid[0]) if mid[0] != "X" else fam, fam, mid)
        self.assertEqual(sorted(collections.Counter(fam for _i, fam, *_r in MUTATIONS)), FAMILIES)
        per = collections.Counter(fam for _i, fam, *_r in MUTATIONS_R7)
        self.assertEqual(sorted(per), sorted(FAMILIES + ["head", "partial"]))
        self.assertEqual((per["head"], per["partial"]), (12, 8))

    def test_the_sample_has_at_least_the_minimum_per_key_and_covers_every_code_form_of_every_key(self):
        """Round 7, m5: a form that the sample does not render could hide a sentence that is false for it. Every value of every domain of test_rea_claims.FORM_DOMAINS
        (the code forms that `ClaimsHold` counts) is in the sample of its key, which has TASKS_PER_KEY tasks or more."""
        for fam in FAMILIES:
            for key, tasks in sample(fam).items():
                self.assertGreaterEqual(len(tasks), TASKS_PER_KEY, key)
                self.assertEqual(len({t.id for t in tasks}), len(tasks), key)
        seen = collections.defaultdict(set)
        for fam in FAMILIES:
            for tasks in sample(fam).values():
                for t in tasks:
                    for k, name, v in T.form_keys(t):
                        seen[(k, name)].add(v)
        for (k, name), domain in T.FORM_DOMAINS.items():
            self.assertTrue(domain <= seen[(k, name)], f"{k} {name}: the sample misses {sorted(domain - seen[(k, name)], key=str)}")

    def test_a_swap_of_the_rounds_and_the_multiplier_is_rejected_in_seconds(self):
        """Round 7, m4: L5 swaps the two numbers of the LCG sentence: the rounds become 22,695,477 or 1,103,515,245 and the claim looped in Python for 7 s to 6 minutes per
        task instead of failing. It now rejects a number outside the rounds the generator draws, with that reason, in a moment."""
        row = next(r for r in ROWS if r[0] == "L5")
        t0 = time.time()
        mod = load_mutated(row[4], row[5], row[3])
        hit = caught_by(mod, affected_renders(mod, "legacy"))
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], "lcg_reference")
        self.assertIn("the rounds", hit[2])
        self.assertLess(time.time() - t0, 30, "the claim must fail, not run for minutes")

    def test_the_headline_mutations_edit_headlines_only(self):
        """NHD1-NHD12: the sentences of every template are the real ones and some headline differs, so that the only way to see them is to read a headline."""
        rows = [r for r in MUTATIONS_R7 if r[1] == "head"]
        self.assertEqual(len(rows), 12)
        for mid, fam, kind, ctx, old, new in rows:
            mod = load_mutated(old, new, ctx)
            self.assertEqual({k: s.sentences for k, s in mod.SPECS.items()}, {k: s.sentences for k, s in targets.SPECS.items()}, mid)
            self.assertTrue([k for k, s in mod.SPECS.items() if s.heads != targets.SPECS[k].heads], f"{mid}: no headline differs")

    def test_the_machinery_accepts_the_real_templates(self):
        """The control: with no edit, no sentence and no headline of any of the families fails (a quick run over the same tasks)."""
        for key in ("structs/field_read", "bounds/clamp", "pktsum/fletcher16", "antidebug/timing_us", "hash", "multi/decode_secret", "cipher/tea_dec"):
            task = tasks_of(key)[0]
            res = T.exercise(task, stop=True)
            self.assertEqual([r for r in res if r[2]], [], key)
            self.assertTrue(any(r[1] in ("clamp_interval", "timing_threshold", "returns_one_exactly_in_range", "pkt_range_start", "hash_final_xor", "accepts_only_the_secret",
                                         "tea_reference", "field_read_offset") for r in res), key)

    def test_the_comparison_sees_a_fact_edit_that_changes_no_template_text(self):
        """NB3 (`"hi": _val(...) + 1`) edits a fact function: the sentence and the headline texts change on the sampled tasks while the templates stay the same."""
        row = next(r for r in MUTATIONS_R7 if r[0] == "NB3")
        mod = load_mutated(row[4], row[5], row[3])
        self.assertEqual({k: s.sentences for k, s in mod.SPECS.items()}, {k: s.sentences for k, s in targets.SPECS.items()})
        aff = affected_renders(mod, "bounds")
        self.assertTrue(aff)
        self.assertTrue(any(any(isinstance(i, int) for i in items) and any(isinstance(i, str) for i in items) for _t, it in aff.values() for items in it.values()),
                        "both the sentences and the headlines of a bounds key change")


if __name__ == "__main__":
    unittest.main()

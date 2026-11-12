"""Round 8, what is stored, asked and accepted: M1, M2 (7), m2, m3, m4, m5, m7, m17, m18, m20.

M1   the antidebug/env sentence states what the code does with the variable, not a real-world class of it.
M2   (7) the question text of every variant is pinned in tests/golden/rea_questions.txt, with a failure message of its own.
m2   a timing headline has no unit. m3 the timing accept group is the task's own API. m4 the legacy `skip` sentence is true for b != 0.
m5   the machines of the states question have no two states that behave alike.
m7   the `magic` question names the byte order (and nothing else of the legacy output changed).
m17  a ChaCha/Salsa name carries the task's own round count. m18 the CRC `init` question says what the code loads. m20 the count questions name the N-1 shapes.

    REA_QUESTIONS_UPDATE=1 python3 -m unittest tests.test_rea_round8_text.Questions      rewrite tests/golden/rea_questions.txt (review the diff; then `rea-bench heldout --write`)
"""
from __future__ import annotations

import collections
import difflib
import hashlib
import itertools
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_r8 import tasks_of

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, f_crc, f_fsm, families, legacy, scoring, targets  # noqa: E402

GOLDEN_Q = ROOT / "tests" / "golden" / "rea_questions.txt"


# ---- M1 -----------------------------------------------------------------------------------------------------------------------------

class EnvSentence(unittest.TestCase):
    def test_no_sentence_or_headline_of_antidebug_env_classifies_the_variable(self):
        """Round 8, M1: `The function tests the environment variable X as a debugging variable.` was stored for LD_LIBRARY_PATH, GLIBC_TUNABLES and
        MALLOC_ARENA_MAX: the code only tests the variable, and none of those is a debugging variable. Only what proc does with the variable is stated."""
        spec = targets.SPECS["antidebug/env"]
        for text in spec.sentences + spec.heads:
            self.assertNotRegex(text, r"(?i)debug", text)
        self.assertIn("The value that proc returns depends on the environment variable {NAME}.", spec.sentences)

    def test_every_phrasing_for_a_variable_that_is_no_debugging_variable_is_true_of_the_code(self):
        quiet = [t for t in tasks_of("antidebug", "env", per=120) if t.params["name"] in ("LD_LIBRARY_PATH", "GLIBC_TUNABLES", "MALLOC_ARENA_MAX", "LD_SHOW_AUXV", "LD_ORIGIN_PATH")]
        self.assertGreater(len(quiet), 3)
        for t in quiet:
            for i in range(3):
                s = targets.render(t.meta(), index=i)["sentence"]
                self.assertNotRegex(s.replace(t.params["name"], ""), r"(?i)debug", s)


# ---- M2 (7): the question text -----------------------------------------------------------------------------------------------------------

_NUM = re.compile(r"0x[0-9a-fA-F]+|\d+")


def question_lines(seeds=(4241, core.HELDOUT_SEED), per=240, max_values=8) -> list:
    """`key | question` for every question TEXT of every variant (legacy: its kind). A number that takes more than `max_values` values among the generated tasks of that
    text (a constant, an argument) is `<N>`; a number that takes a few (a CRC width and the all-ones value built from it, the target state of an fsm) stays, so a
    wrong hint inside the question is a different line."""
    out = set()
    for fam, m in families.modules().items():
        nv = len(getattr(m, "VARIANTS", ())) or 1
        groups = collections.defaultdict(list)
        for seed in seeds:
            for t in families.generate(fam, per * nv, seed):
                key = f"{t.kind}/{t.variant}" if t.variant else t.kind
                groups[(key, _NUM.sub("\0", t.question))].append(tuple(_NUM.findall(t.question)))
        for (key, skel), rows in groups.items():
            cols = list(zip(*rows)) if rows and rows[0] else []
            vary = [len(set(c)) > max_values for c in cols]
            for combo in {tuple("<N>" if v else x for x, v in zip(r, vary)) for r in rows}:
                it = iter(combo)
                out.add(f"{key} | " + re.sub("\0", lambda _m: next(it), skel))
    return sorted(out)


class Questions(unittest.TestCase):
    """Round 8, M2: nothing but the held-out list noticed a change of the question text: a false hint (`an all-ones value is 0xfffe`), `a double round counts as 3`,
    offsets counted from 1, `smallest elapsed time`, `SHORTEST length`, exclusive bounds, a decimal question over a hex key each changed every task of a variant."""

    def test_the_question_text_of_every_variant_is_the_pinned_one(self):
        now = question_lines()
        if os.environ.get("REA_QUESTIONS_UPDATE"):
            GOLDEN_Q.write_text("# the question text of every variant (a number that takes more than 8 values is <N>); tests/test_rea_round8.py Questions\n" + "\n".join(now) + "\n")
        self.assertTrue(GOLDEN_Q.exists(), "run once with REA_QUESTIONS_UPDATE=1")
        old = [l for l in GOLDEN_Q.read_text().splitlines() if l and not l.startswith("#")]
        if old != now:
            diff = "\n".join(l for l in difflib.unified_diff(old, now, "pinned", "generated", lineterm="", n=0) if not l.startswith(("---", "+++", "@@")))
            self.fail("THE QUESTION TEXT OF A VARIANT CHANGED. The question is what the stored answer answers: check that the hint, the unit, the base, the bounds "
                      "(inclusive or not) and the counting origin it states are the ones the key uses, then regenerate with REA_QUESTIONS_UPDATE=1 python3 -m unittest "
                      "tests.test_rea_round8_text.Questions and `python3 tools/rea-bench heldout --write` (the question is part of the held-out key). Changed lines:\n" + diff)

    def test_the_pinned_list_has_every_variant(self):
        keys = {l.split(" | ")[0] for l in GOLDEN_Q.read_text().splitlines() if " | " in l}
        want = {f"{t.kind}/{t.variant}" if t.variant else t.kind for fam in families.modules() for t in families.generate(fam, 66, 7)}
        self.assertEqual(sorted(want - keys), [])


# ---- m2, m3 ----------------------------------------------------------------------------------------------------------------------------

class Timing(unittest.TestCase):
    def test_a_timing_headline_states_no_unit_because_the_question_says_no_other_unit(self):
        """Round 8, m2: 3 of the 4 headlines were `timing, 190000000 ns` over a question that says `Answer with a plain integer in nanoseconds: no separators, no other unit`."""
        for variant in ("timing_ns", "timing_us"):
            ts = tasks_of("antidebug", variant, per=20)
            self.assertIn("no other unit", ts[0].question)
            for t in ts:
                seen = set()
                for k in range(4):
                    r = targets.render(t.meta(), head=k)
                    self.assertRegex(r["headline"], r"^timing( check)?[,:] \d+$", r["headline"])
                    self.assertTrue(scoring.score_strict(r["text"], t.meta()), r["text"])
                    seen.add(r["headline"])
                self.assertEqual(len(seen), 4)

    def test_the_accept_group_is_the_clock_call_the_task_makes(self):
        """Round 8, m3: `rdtsc, 190000000 ns` scored true on a clock_gettime task, and gettimeofday on both variants."""
        for variant, own, other in (("timing_ns", "clock_gettime", "gettimeofday"), ("timing_us", "gettimeofday", "clock_gettime")):
            for t in tasks_of("antidebug", variant, per=10):
                m, thr = t.meta(), t.params["thr"]
                self.assertIn(own + "(", t.body)
                for good in ("timing", "time-based", own):
                    self.assertTrue(scoring.score_strict(f"{good}, {thr}", m), (t.id, good))
                for bad in (other, "rdtsc", "ptrace", "env"):
                    self.assertFalse(scoring.score_strict(f"{bad}, {thr}", m), (t.id, bad))
                    self.assertFalse(scoring.score_strict(f"{bad}, {thr} ns", m), (t.id, bad))


# ---- m4, m7 ----------------------------------------------------------------------------------------------------------------------------

class LegacySkip(unittest.TestCase):
    def test_no_skip_sentence_says_a_byte_value_does_not_count(self):
        """Round 8, m4: `Exactly one byte value does not count in the sum` is false for b != 0 (a byte 0x00 adds 0 as well); the test leaves one value out."""
        for s in targets.SPECS["skip"].sentences:
            self.assertNotRegex(s, r"(?i)does not count|not count")
        self.assertIn("The test leaves exactly one byte value out of the sum, and it is 0x{b:02x}.", targets.SPECS["skip"].sentences)

    def test_the_sentence_is_about_the_comparison_that_the_source_has(self):
        for t in [t for t in families.generate("legacy", 300, 7) if t.kind == "skip"]:
            self.assertRegex(t.body, rf"p\[i\] != {t.truth['accept'][0]}\)")


class LegacyMagic(unittest.TestCase):
    OLD = "What 32-bit magic value must the header start with (hex), and what is the highest accepted version?"
    ROUND7 = {1001: "97beddfa668013b6a82176c4bf4108b3726b2d9a1c1ba3714d4aaef24a9b2418", 7: "9505981ce3199e87db6490807a798603b08c84a391ee579b5c9f25c0b91a592a"}

    @staticmethod
    def digest(seed: int) -> str:
        h = hashlib.sha256()
        with tempfile.TemporaryDirectory() as d:
            core.write_tasks(families.generate("legacy", 60, seed), pathlib.Path(d))
            for p in sorted(pathlib.Path(d).glob("*")):
                h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
        return h.hexdigest()

    def test_the_question_says_which_byte_order_the_key_is_in(self):
        """Round 8, m7: `What 32-bit magic value must the header start with` has two readings (the bytes in file order, or the little-endian word the code compares)."""
        ts = [t for t in families.generate("legacy", 300, 7) if t.kind == "magic"]
        self.assertGreater(len(ts), 20)
        for t in ts:
            self.assertIn("little-endian word", t.question)
            self.assertIn("(hex", t.question)

    def test_nothing_else_of_the_legacy_output_changed(self):
        """The round-7 pins are reproduced when the old question text is put back: only the `magic` question differs."""
        orig = legacy.t_magic

        def old(rng):
            body, q, truth, kind = orig(rng)
            assert "little-endian" in q
            return body, self.OLD, truth, kind

        templates = [old if f is orig else f for f in legacy.TEMPLATES]
        with mock.patch.object(legacy, "TEMPLATES", templates):
            for seed, want in self.ROUND7.items():
                self.assertEqual(self.digest(seed), want, f"the legacy output of seed {seed} differs from round 7 by more than the magic question")


# ---- m5 --------------------------------------------------------------------------------------------------------------------------------

def _distinguishable(delta: list, alphabet_size: int) -> set:
    """Pairs of states that some input tells apart, the accept state (the last) being the only observation: the fixpoint of 'one accepts and the other does not, or a
    letter leads to a pair that is already told apart'. Written independently of f_fsm.minimal."""
    s = len(delta)
    told = {(a, b) for a in range(s) for b in range(s) if (a == s - 1) != (b == s - 1)}
    changed = True
    while changed:
        changed = False
        for a, b in itertools.product(range(s), repeat=2):
            if (a, b) not in told and any((delta[a][c], delta[b][c]) in told for c in range(alphabet_size)):
                told.add((a, b))
                changed = True
    return told


class FsmStates(unittest.TestCase):
    def test_no_two_states_of_a_states_machine_behave_alike(self):
        """Round 8, m5: 20% of `table_states` and 14% of `switch_states` tasks had two states with one row (both {3, 4}): an automaton-minimising reader counts fewer."""
        n = 0
        for seed in (2027, 4049, 1001):
            for variant in ("table_states", "switch_states"):
                for t in tasks_of("fsm", variant, seed=seed, per=50):
                    delta, s = t.params["delta"], len(t.params["delta"])
                    n += 1
                    self.assertEqual(len({tuple(r) for r in delta}), s, f"{t.id}: two states have the same row")
                    told = _distinguishable(delta, len(t.params["alphabet"]))
                    self.assertTrue(all((a, b) in told for a in range(s) for b in range(s) if a != b), f"{t.id}: two states cannot be told apart")
        self.assertGreater(n, 250)

    def test_the_old_task_with_two_states_that_had_one_row_is_redrawn(self):
        """`fsm-2027-0032`: states 0 and 1 both {3, 4}."""
        t = families.generate("fsm", 36, 2027)[32]
        self.assertEqual(t.variant, "table_states")
        self.assertEqual(len({tuple(r) for r in t.params["delta"]}), len(t.params["delta"]))

    def test_minimal_agrees_with_the_independent_check_and_the_input_variants_are_unchanged(self):
        same_rows = [[1, 2], [3, 4], [3, 4], [3, 3], [4, 4]]                           # states 1 and 2 have one row
        moore_only = [[1, 2], [3, 3], [4, 4], [5, 4], [5, 3], [5, 5]]                  # all rows differ, but states 3 and 4 act alike, so 1 and 2 do too
        fine = [[1, 2], [3, 0], [2, 4], [4, 1], [4, 4]]
        want = {"same_rows": False, "moore_only": False, "fine": True}
        for name, delta in (("same_rows", same_rows), ("moore_only", moore_only), ("fine", fine)):
            s = len(delta)
            ok = len({tuple(r) for r in delta}) == s and all((a, b) in _distinguishable(delta, 2) for a in range(s) for b in range(s) if a != b)
            self.assertEqual(ok, want[name], name)
            self.assertEqual(f_fsm.minimal(delta), ok, name)


# ---- m17, m18, m20 ---------------------------------------------------------------------------------------------------------------------

class StreamNames(unittest.TestCase):
    def test_a_chacha_or_salsa_name_with_a_round_count_is_the_tasks_own(self):
        """Round 8, m17: the name group was chacha, chacha20, chacha12, chacha8, chacha-20 for every task: `ChaCha8, 20` was an answer for ChaCha20."""
        seen = collections.Counter()
        for variant, name in (("chacha", "ChaCha"), ("salsa", "Salsa")):
            for t in tasks_of("cipher", variant, per=60):
                r, m = t.params["rounds"], t.meta()
                seen[r] += 1
                for good in (f"{name}, {r}", f"{name}{r}, {r}", f"{name}-{r}, {r}", f"{name.lower()}{r}: {r} rounds"):
                    self.assertTrue(scoring.score_strict(good, m), (t.id, good))
                for x in {8, 12, 20, r + 2, r + 4} - {r}:
                    self.assertFalse(scoring.score_strict(f"{name}{x}, {r}", m), (t.id, x))
                    self.assertFalse(scoring.score_strict(f"{name}-{x}, {r}", m), (t.id, x))
                self.assertFalse(scoring.score_strict(f"{name}{r}, {r + 2}", m))
        self.assertGreater(len(seen), 4, "several round counts were drawn")


class CrcInitQuestion(unittest.TestCase):
    def test_the_init_question_says_the_code_loads_the_value_and_no_sentence_says_initial_value_of_the_crc(self):
        """Round 8, m18: for a reflected CRC the catalogue `init` is reflect(literal): `the initial value of the CRC register` read either way."""
        for variant in ("bitwise_init", "table_init"):
            for t in tasks_of("crc", variant, per=8):
                self.assertIn("loads into the CRC register", t.question)
                self.assertIn("exactly as the code writes it", t.question)
        for key in ("crc/bitwise_init", "crc/table_init"):
            for s in targets.SPECS[key].sentences:
                self.assertNotRegex(s, r"(?i)initial value of this")
                self.assertRegex(s, r"register")

    def test_a_reflected_crc_with_an_init_that_is_not_its_own_reflection_is_in_the_sample(self):
        ts = [t for t in tasks_of("crc", "bitwise_init", per=150) if t.params["refl"]]
        odd = [t for t in ts if t.params["init"] != f_crc.reflect(t.params["init"], t.params["width"])]
        self.assertGreater(len(odd), 5)
        for t in odd:
            m = t.meta()
            self.assertTrue(scoring.score_strict(f"0x{t.params['init']:x}", m), t.id)
            self.assertFalse(scoring.score_strict(f"0x{f_crc.reflect(t.params['init'], t.params['width']):x}", m), f"{t.id}: the catalogue reading of init is not the key")


class CountHints(unittest.TestCase):
    def test_the_count_questions_name_the_n_minus_1_shapes(self):
        """Round 8, m20: RC4 showed its key length only as `cmp ecx, 0xb; cmove` (11 for 12) and TEA at -O0 as `i <= 31`; the hint named `>` / `>=` only."""
        rc4 = tasks_of("cipher", "rc4", per=3)[0].question
        self.assertIn("`== N - 1`", rc4)
        self.assertIn("`<= N - 1`", rc4)
        self.assertIn("already moved on", rc4)
        for v in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec"):
            self.assertIn("`i <= N - 1` for `i < N`", tasks_of("cipher", v, per=3)[0].question)
        self.assertIn("the bound may show one off", tasks_of("cipher", "chacha", per=3)[0].question)


class EncodingReject(unittest.TestCase):
    def test_radix64_is_not_a_rejected_name_of_the_base64_variants(self):
        """Round 8, m8: `radix64` (OpenPGP's name for base64) was in the reject list of every encoding task, so a right base64 answer that mentions it lost the evaluation
        scorers' verdict. The strict grammar does not read `reject`, and a name that is in no accept group is a leftover there anyway."""
        n = 0
        for variant in ("base64_encode", "base64_decode", "base32_encode", "base58_decode"):
            for t in tasks_of("encoding", variant, per=12):
                if t.truth["score"] == "tokens":
                    n += 1
                    self.assertNotIn("radix64", t.truth["reject"], t.id)
                    self.assertFalse(scoring.score_strict("radix64, encode", t.meta()) or scoring.score_strict("radix64, decode", t.meta()), t.id)
        self.assertGreater(n, 10)


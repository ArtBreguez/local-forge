"""Round-2 review of the REA task generator, scorer part: B1, B2, M3, M4 and the yield items m2, m3, m5, m10.

The reviewer's attack scripts (repro_scorer.py, s_attack.py, s_attack2.py) are ported in `tests/rea_attack_matrix.py` and run here on
generated tasks of every new family: every wrong shape must be rejected, every right format must pass.
"""
from __future__ import annotations

import sys
import unittest

from . import rea_attack_matrix as am
from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, families, scoring, sft  # noqa: E402


def task(fam: str, variant: str, seed: int = 99, n: int = 40):
    m = families.modules()[fam]
    return next(t for t in families.generate(fam, n * len(m.VARIANTS), seed) if t.variant == variant)


class AttackMatrix(unittest.TestCase):
    """B1/B2/M3: nothing but the key passes; the review's 28 right formats and every other right format keep passing."""

    def run_seed(self, seed: int, per_variant: int):
        res, ex = am.matrix(am.tasks_for(seed, per_variant))
        s = am.summary(res)
        accepted = {n: (a, b, ex[(n, e)]) for (n, e), (a, b) in res.items() if e is False and a}
        rejected = {n: (a, b, ex[(n, e)]) for (n, e), (a, b) in res.items() if e is True and a < b}
        self.assertEqual(accepted, {}, "wrong replies accepted")
        self.assertEqual(rejected, {}, "right replies rejected")
        self.assertGreater(s["wrong_total"], 5000)
        self.assertEqual(s["formats_passed"], s["formats_total"])
        self.assertEqual({n for (n, e) in res if n in am.RIGHT_FORMATS}, set(am.RIGHT_FORMATS), "all 28 review formats were exercised")

    def test_seed_99(self):
        self.run_seed(99, 12)

    def test_seed_1001(self):
        self.run_seed(1001, 12)

    def test_the_reviewers_repros_by_hand(self):
        s = scoring.score
        head = task("multi", "decode_secret").truth
        k = head["accept"][0]
        for r in (f"{k} or {k[:-1]}x", f"{k[:-1]}x or {k}", f"not {k}", f"{k}7", f"x{k}", f"/etc/{k}", k.upper()):
            self.assertFalse(s(r, head), r)
        self.assertTrue(s(k, head) and s(f"`{k}`", head) and s(f"The string is {k}.", head) and s(f'"{k}"', head))
        xtea = task("cipher", "xtea_enc").truth
        n = xtea["accept"][2][0]
        for r in (f"TEA (not XTEA), encrypt, {n}", f"TEA, not XTEA, encrypt, {n}", f"XTEA or TEA, encrypt, {n}", f"XTEA, not encrypt, {n}"):
            self.assertFalse(s(r, xtea), r)
        self.assertTrue(s(f"XTEA, encrypt, {n}", xtea))
        b64 = task("encoding", "base64_encode", n=60)
        tok = next(t for t in families.generate("encoding", 120, 99) if t.variant == "base64_encode" and t.truth["score"] == "tokens").truth
        for r in ("base64 encode or base36 encode", "base64, not encode", "base64 encode (not base45)", "base64, base36 encode"):
            self.assertFalse(s(r, tok), r)
        self.assertTrue(s("base64 encode", tok) and not s("not 64", {"score": "tokens", "accept": [core.dec(64)]}))
        self.assertIsNotNone(b64)


class HeadlineRule(unittest.TestCase):
    def test_a_lead_in_over_a_list_is_not_an_answer(self):
        """M3: a skipped lead-in must be followed by exactly one answer line before prose or the end."""
        t = {"score": "tokens", "accept": [core.hx(8)]}
        s = scoring.score
        self.assertFalse(s("Possible offsets:\n0x8\n0x10\n0x18", t))
        self.assertFalse(s("Candidates:\n- 0x8\n- 0x10", t))
        self.assertFalse(s("Candidates:\n0x10\n0x8", t))
        self.assertFalse(s("Answer:", t))
        self.assertEqual(scoring.headline("Possible offsets:\n0x8\n0x10"), "")
        self.assertTrue(s("Answer:\n0x8", t))
        self.assertTrue(s("Answer:\n\n`0x8`\nIt is the pointer to the next node, read from the third word.", t))
        self.assertTrue(s("```\n0x8\n```\nThe next pointer sits after the tag.", t))

    def test_hedges_arithmetic_and_candidates_fail_in_tokens_mode(self):
        t = {"score": "tokens", "accept": [core.dec(64)]}
        s = scoring.score
        for r in ("not 64", "64 or 32", "either 64 or 128", "64 maybe", "probably 64", "it isn't 64", "64, 65", "64 + 64", "64 == 64", "could be 64",
                  "64 vs 32", "64 rather than 32", "instead of 32, 64", "perhaps 64"):
            self.assertFalse(s(r, t), r)
        self.assertTrue(s("64", t) and s("It is 64.", t) and s("0x40 (64)", t))

    def test_units_and_standard_names_are_not_candidates(self):
        """m3: numbers that belong to a unit or a standard name are ignored; a plain second number still fails."""
        s = scoring.score
        poly = {"score": "tokens", "accept": [core.hx(0xedb88320, 8), ["reflected"]], "reject": ["normal"]}
        self.assertTrue(s("0xEDB88320, reflected (CRC-32)", poly))
        xtea = {"score": "tokens", "accept": [["xtea"], ["decrypt"], core.dec(32)], "reject": ["tea"], "allow": ["extended tea"]}
        self.assertTrue(s("XTEA, decrypt, 32 rounds (128-bit key)", xtea))
        rc4 = {"score": "tokens", "accept": [["rc4"], core.dec(11)]}
        self.assertTrue(s("RC4, 11 bytes (88 bits)", rc4))
        off = {"score": "tokens", "accept": [core.hx(0x28)]}
        self.assertTrue(s("40 bytes on x86-64", off) and s("0x28 bytes (8-byte aligned)", off))
        self.assertTrue(s("ChaCha20, 20 rounds (256-bit key)", {"score": "tokens", "accept": [["chacha20"], core.dec(20)]}))
        self.assertFalse(s("RC4, 88 bits", rc4), "88 bits is not the byte count")
        self.assertFalse(s("RC4, 11 bytes, 12", rc4), "a plain second number is a candidate")
        self.assertFalse(s("RC4, 12-bit", rc4))

    def test_any_line_control_is_format_insensitive_and_never_stricter(self):
        t = {"score": "tokens", "accept": [core.dec(32)]}
        thinking = "Let me read the loop first.\nThe loop runs while i < 32.\n32"
        self.assertFalse(scoring.score(thinking, t))
        self.assertTrue(scoring.score_any_line(thinking, t))
        self.assertTrue(scoring.score_any_line("32", t))
        self.assertFalse(scoring.score_any_line("the loop runs 16 or 32 times", t), "a hedge is still a hedge on its own line")
        legacy = {"score": "exact", "accept": ["ab"]}
        self.assertEqual(scoring.score_any_line("x\n0xab", legacy), scoring.score("x\n0xab", legacy))

    def test_no_accepted_spelling_is_a_hedge_word(self):
        """The hedge check looks at the whole headline: it is only sound while no key spelling is itself one of the words."""
        for seed in (3, 99, 1001):
            for fam in am.NEW_FAMILIES:
                for t in families.generate(fam, 400, seed):
                    words = [a for g in t.truth["accept"] for a in (g if isinstance(g, list) else [g])]
                    if t.truth["score"] != "tokens":
                        continue
                    for w in words:
                        self.assertIsNone(scoring.HEDGE.search(w), f"{t.id}: accepted spelling {w!r} contains a hedge word")
                        self.assertFalse(scoring._arithmetic(w), t.id)


class StoredTarget(unittest.TestCase):
    """M4: the stored text is the headline and at most one justification sentence, and it never carries a contradiction."""

    def setUp(self):
        self.t = {"score": "tokens", "accept": [core.hx(8)]}

    def test_the_reviewers_contradiction_is_not_stored(self):
        self.assertTrue(scoring.score("0x8\nWait, actually the next pointer is at 0x70.", self.t), "the headline alone is right")
        self.assertIsNone(scoring.sft_target("0x8\nWait, actually the next pointer is at 0x70.", self.t))

    def test_truncation_keeps_headline_and_first_sentence(self):
        r = scoring.sft_target("**0x8**\n\nThe next pointer is read at offset 0x8. It then follows the pointer. The loop ends at NULL.", self.t)
        self.assertEqual(r["text"], "**0x8**\nThe next pointer is read at offset 0x8.")
        self.assertTrue(r["truncated"])
        r = scoring.sft_target("0x8\nThe next pointer is read at offset 0x8.", self.t)
        self.assertFalse(r["truncated"])
        self.assertEqual(scoring.sft_target("0x8", self.t)["text"], "0x8")
        self.assertEqual(scoring.sft_target("Answer:\n0x8\nIt is the first field.", self.t)["text"], "Answer:\n0x8\nIt is the first field.")
        self.assertEqual(scoring.sft_target("0x8\n- It is the first field.\n- It is read once.", self.t)["text"], "0x8\nIt is the first field.")
        self.assertEqual(scoring.sft_target("0x8\nThe loop, e.g. in the second block, reads it. More text.", self.t)["sentence"],
                         "The loop, e.g. in the second block, reads it.")

    def test_a_justification_that_names_another_value_hedges_or_changes_its_mind_is_not_stored(self):
        for body in ("The next pointer is at 0x70.", "Offset 0x8, not 0x10.", "It is 0x8 or maybe 0x10.", "However the field after it is at 12.",
                     "The first guess was 0x4 but the code reads 0x8.", "Hmm, that is the tag.", "Correction: it is at 0x8.",
                     "It cannot be 0x8 for a packed struct."):
            self.assertIsNone(scoring.sft_target(f"0x8\n{body}", self.t), body)
        for body in ("The next pointer sits after the 4-byte tag.", "It is the field read through p->next, at 0x8 in the struct.",
                     "The struct is 8-byte aligned (x86-64)."):
            self.assertIsNotNone(scoring.sft_target(f"0x8\n{body}", self.t), body)

    def test_names_and_strings(self):
        xtea = {"score": "tokens", "accept": [["xtea"], ["encrypt"], core.dec(32)], "reject": ["tea", "xxtea", "aes"], "allow": ["extended tea"]}
        self.assertIsNotNone(scoring.sft_target("XTEA, encrypt, 32\nIt is the eXtended TEA with a key-dependent index.", xtea))
        self.assertIsNone(scoring.sft_target("XTEA, encrypt, 32\nUnlike TEA it indexes the key with the sum.", xtea))
        self.assertIsNone(scoring.sft_target("XTEA, encrypt, 32\nIt uses the constant 0x9e3779b9.", xtea), "another number in the sentence")
        exact = {"score": "exact_head", "accept": ["admin_login42"]}
        self.assertIsNotNone(scoring.sft_target("admin_login42\nEach byte is XORed with 0x37.", exact))
        self.assertIsNone(scoring.sft_target("admin_login42\nIt could also read admin_login43.", exact))
        self.assertIsNone(scoring.sft_target("admin_login42\nThe string is built from admin and login, i.e. admin.login42.", exact))
        self.assertIsNone(scoring.sft_target("nope\nadmin_login42", exact))

    def test_the_stored_text_always_scores(self):
        for fam in am.NEW_FAMILIES:
            for t in families.generate(fam, 70, 5):
                for reply in (f"{scoring.canonical(t.truth)}\nThis follows from the code.", f"Answer:\n{scoring.canonical(t.truth)}",
                              f"**{scoring.canonical(t.truth)}**\nShort. Another sentence."):
                    r = scoring.sft_target(reply, t.truth)
                    self.assertIsNotNone(r, (t.id, reply))
                    self.assertTrue(scoring.score(r["text"], t.truth), (t.id, r["text"]))

    def test_legacy_targets_are_cut_and_must_still_score(self):
        t = {"score": "all", "accept": ["7", "1f"]}
        r = scoring.sft_target("seed 7 and 0x1f.\nBecause the loop shifts it. More text here.", t)
        self.assertEqual(r["text"], "seed 7 and 0x1f.\nBecause the loop shifts it.")
        self.assertTrue(r["truncated"])
        late = scoring.sft_target("Looking at the code.\nIt is seed 7 and 0x1f. Done.", t)
        self.assertEqual(late["text"], "Looking at the code.\nIt is seed 7 and 0x1f.")
        self.assertIsNone(scoring.sft_target("Looking at the code.\nStill reading.\nseed 7 and 0x1f", t), "the answer was cut off: nothing stored")
        self.assertIsNone(scoring.sft_target("seed 7 and 0x1f\nWait, the seed is 8.", t))
        self.assertIsNone(scoring.sft_target("seed 7 and 0x1f\nProbably right.", t))


class Selection(unittest.TestCase):
    """m5: a task with chance >= 0.2 keeps answers only when at least two of the K samples are right."""

    def test_agreement_only_above_the_chance_threshold(self):
        easy = {"score": "tokens", "accept": [core.dec(5)], "chance": 0.5, "kind": "crc"}
        hard = {**easy, "chance": 0.05}
        legacy = {"score": "exact", "accept": ["ab"], "kind": "legacy"}
        one = [("5", True), ("6", False), ("7", False)]
        two = [("5", True), ("5\nThe loop stops at five.", True), ("7", False)]
        self.assertEqual(sft.select(one, easy), ([], "agreement"))
        self.assertEqual([r["text"] for r in sft.select(two, easy)[0]], ["5", "5\nThe loop stops at five."])
        self.assertEqual(sft.select(one, hard)[1], "kept")
        self.assertEqual(sft.select([("0xab", True)], legacy)[1], "kept", "no chance recorded: no agreement rule")
        self.assertEqual(sft.select([("6", False)], hard), ([], "no_correct"))
        self.assertEqual(sft.select([("5\nBut maybe 6.", True)], hard), ([], "vetoed"))
        dup = sft.select([("5", True), ("5", True)], hard)[0]
        self.assertEqual(len(dup), 1, "identical stored texts are kept once")

    def test_every_variant_above_the_threshold_is_known(self):
        from rea_bench import chance
        high = sorted(k for k, v in chance.CHANCE.items() if v >= sft.AGREE_CHANCE)
        self.assertIn("crc/bitwise_init", high)
        self.assertIn("pktsum/xor8", high)
        self.assertNotIn("cipher/xtea_enc", high)

    def test_stats_are_split_by_family(self):
        st = sft.Stats()
        t1 = {"kind": "crc", "score": "tokens", "accept": [["1"]], "chance": 0.5}
        t2 = {"kind": "bounds", "score": "tokens", "accept": [["1"]], "chance": 0.01}
        o1, o2 = [("1", True), ("2", False)], [("2", False)]
        st.add(t1, o1, *sft.select(o1, t1))
        st.add(t2, o2, *sft.select(o2, t2))
        d = st.as_dict()
        self.assertEqual((d["tasks"], d["answers"], d["correct"], d["kept_tasks"], d["agreement"], d["no_correct"]), (2, 3, 1, 0, 1, 1))
        self.assertEqual(d["by_kind"]["crc"]["agreement"], 1)
        self.assertEqual(d["by_kind"]["bounds"]["no_correct"], 1)


if __name__ == "__main__":
    unittest.main()

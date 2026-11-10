"""The scorer after round 3: the whitelist grammar (M1, M2, M3, m2, m9), the stored target (m1), the data path against the evaluation path (m10).

Earlier rounds' attack scripts are in `tests/rea_attack_matrix.py` (round 2: right and wrong formats, plus the "yield" class of right
answers the grammar does not parse) and `tests/rea_attacks_r3.py` (round 3: the reviewer's rows, which must all be rejected); the property
test is `tests/rea_property.py`. The grammar itself is unit-tested in `tests/test_rea_grammar.py`.
"""
from __future__ import annotations

import sys
import unittest

from . import rea_attack_matrix as am
from . import rea_attacks_r3 as ra
from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, families, scoring, sft  # noqa: E402


def task(fam: str, variant: str, seed: int = 99, n: int = 40):
    m = families.modules()[fam]
    return next(t for t in families.generate(fam, n * len(m.VARIANTS), seed) if t.variant == variant)


class AttackMatrix(unittest.TestCase):
    """Round 2's shapes on the whitelist grammar: no wrong reply passes, every right format passes, the yield class is rejected."""

    def run_seed(self, seed: int, per_variant: int):
        res, ex = am.matrix(am.tasks_for(seed, per_variant) + am.legacy_tasks(seed, per_variant * 10))
        s = am.summary(res)
        accepted = {n: (a, b, ex[(n, e)]) for (n, e), (a, b) in res.items() if e is False and a}
        rejected = {n: (a, b, ex[(n, e)]) for (n, e), (a, b) in res.items() if e is True and a < b}
        yielded = {n: (a, b, ex[(n, e)]) for (n, e), (a, b) in res.items() if e == "yield" and a}
        self.assertEqual(accepted, {}, "wrong replies accepted")
        self.assertEqual(rejected, {}, "right replies rejected")
        self.assertEqual(yielded, {}, "a yield-class format is accepted now: move it to the right formats")
        self.assertGreater(s["wrong_total"], 10000)
        self.assertEqual(s["formats_passed"], s["formats_total"])
        self.assertEqual({n for (n, e) in res if n in am.RIGHT_FORMATS}, set(am.RIGHT_FORMATS), "all 27 right formats of the review were exercised")
        rates = {fam: rate for fam, _a, _n, _y, rate in am.family_table()}
        self.assertEqual(set(rates), set(families.modules()), "a right-format rate for every family, legacy included")
        self.assertTrue(all(r > 0.7 for r in rates.values()), rates)

    def test_seed_99(self):
        self.run_seed(99, 12)

    def test_seed_1001(self):
        self.run_seed(1001, 12)

    def test_the_reviewers_repros_by_hand(self):
        s = scoring.score
        head = task("multi", "decode_secret").meta()
        k = head["accept"][0]
        for r in (f"{k} or {k[:-1]}x", f"{k[:-1]}x or {k}", f"not {k}", f"{k}7", f"x{k}", f"/etc/{k}", k.upper()):
            self.assertFalse(s(r, head), r)
        self.assertTrue(s(k, head) and s(f"`{k}`", head) and s(f"The string is {k}.", head) and s(f'"{k}"', head))
        xtea = task("cipher", "xtea_enc").meta()
        n = xtea["accept"][2][0]
        for r in (f"TEA (not XTEA), encrypt, {n}", f"TEA, not XTEA, encrypt, {n}", f"XTEA or TEA, encrypt, {n}", f"XTEA, not encrypt, {n}"):
            self.assertFalse(s(r, xtea), r)
        self.assertTrue(s(f"XTEA, encrypt, {n}", xtea))
        tok = next(t for t in families.generate("encoding", 120, 99) if t.variant == "base64_encode" and t.truth["score"] == "tokens").meta()
        for r in ("base64 encode or base36 encode", "base64, not encode", "base64 encode (not base45)", "base64, base36 encode"):
            self.assertFalse(s(r, tok), r)
        self.assertTrue(s("encode, base64", tok) and not s("not 64", {"score": "tokens", "accept": [core.dec(64)]}))


class RoundThreeAttacks(unittest.TestCase):
    """The reviewer's round-3 rows (M1 words and symbols, M2 exclusive bounds, M3 exempt numbers, number words, second names, exact strings,
    a right headline over a bad second line), ported: not one of them passes, under the evaluation scorer and the data scorer."""

    def run_seed(self, seed: int, n: int):
        res = ra.run(seed, n)
        self.assertEqual(res["accepted"], [])
        self.assertEqual(res["stored_bad"], [], "a bad second line reached the stored text")
        self.assertGreater(res["rows"], 50000)
        self.assertEqual(set(res["by_family"]), set(families.modules()))

    def test_seed_99(self):
        self.run_seed(99, 60)

    def test_seed_1001(self):
        self.run_seed(1001, 60)


class HeadlineRule(unittest.TestCase):
    def test_a_lead_in_over_a_list_is_not_an_answer(self):
        """M3 of round 2: a skipped lead-in must be followed by one answer line and then prose or nothing."""
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

    def test_two_answer_lines_are_a_list_whatever_they_hold(self):
        """The rows the round-3 review listed as `two-lines`, `codeblock-two`, `lead-two`: a second line that is not a sentence is a candidate list."""
        t = {"score": "tokens", "accept": [["40", "0x28"]], "kind": "structs"}
        for r in ("40\n43", "```\n40\n43\n```", "Answer:\n40\n43", "40\n0x28", "40\nor 43", "40\nmaybe", "40\n- 43"):
            self.assertFalse(scoring.score(r, t), r)
        self.assertTrue(scoring.score("40\nIt is read as one record of the array.", t))
        self.assertTrue(scoring.score("40\nJustification:\nIt is read as one record of the array.", t))
        self.assertEqual(scoring.headline("40. The stride of the array is 40 bytes."), "40.")
        self.assertEqual(scoring.headline("0x8\n0x10"), "", "no headline: the reply is a list")

    def test_the_text_after_the_first_sentence_of_the_line_is_the_justification(self):
        t = {"score": "tokens", "accept": [core.hx(8)]}
        r = scoring.sft_target("0x8. The next pointer is read at offset 0x8 of the node.", t)
        self.assertEqual((r["headline"], r["sentence"]), ("0x8.", "The next pointer is read at offset 0x8 of the node."))
        self.assertIsNone(scoring.sft_target("0x8. Or maybe 0x10.", t), "a short second sentence is a second candidate, not a justification")

    def test_the_any_line_control_is_format_insensitive_and_never_stricter(self):
        t = {"score": "tokens", "accept": [core.dec(32)]}
        thinking = "Let me read the loop first.\nThe loop runs while i < 32.\n32"
        self.assertFalse(scoring.score(thinking, t))
        self.assertTrue(scoring.score_any_line(thinking, t))
        self.assertTrue(scoring.score_any_line("32", t))
        self.assertFalse(scoring.score_any_line("the loop runs 16 or 32 times", t), "a hedge is still a hedge on its own line")
        legacy = {"score": "exact", "accept": ["ab"]}
        self.assertEqual(scoring.score_any_line("x\n0xab", legacy), scoring.score("x\n0xab", legacy))

    def test_no_accepted_spelling_is_a_word_of_the_grammar(self):
        """The grammar is only sound while a key's own spelling is never one of its function words, units or separators (it would be both a hit
        and a free filler), and while every spelling is ONE token of the lexer (or it could never match)."""
        from rea_bench import grammar
        vocabulary = grammar.GLUE | grammar.GENERIC | grammar.NOUNS | grammar.LO_LABELS | grammar.HI_LABELS | grammar.NEUTRAL_LABELS | {"and", "to"}
        for seed in (3, 99, 1001):
            for fam in am.NEW_FAMILIES:
                for t in families.generate(fam, 300, seed):
                    m = t.meta()
                    if m["score"] != "tokens":
                        continue
                    units = grammar.Spec(m).units
                    for g in m["accept"]:
                        for w in g:
                            self.assertNotIn(w.lower(), vocabulary | units, f"{t.id}: accepted spelling {w!r} is also a function word")
                            toks = grammar.lex(grammar.normalise(w).lower(), letter_runs=bool(m.get("cand")))
                            self.assertEqual(len(toks), 1, f"{t.id}: spelling {w!r} lexes as {toks}")
                            self.assertEqual(toks[0].kind, "num" if grammar.num_value(w.lower()) is not None else "word", (t.id, w))

    def test_no_function_word_is_a_name_of_the_family(self):
        """The reverse: a word the grammar lets through for free must not be a name any task of the family rejects or accepts elsewhere."""
        from rea_bench import grammar
        free = grammar.GLUE | grammar.GENERIC | grammar.NOUNS | grammar.ANY_UNITS | {w for u in grammar.UNITS.values() for w in u}
        for fam in am.NEW_FAMILIES:
            names = set()
            for t in families.generate(fam, 300, 5):
                m = t.meta()
                if m["score"] == "tokens":
                    names |= {a.lower() for g in m["accept"] for a in g} | {r.lower() for r in m.get("reject", [])}
            self.assertEqual(names & free, set(), fam)


class StoredTarget(unittest.TestCase):
    """What is stored is the headline and at most one justification sentence; a sentence that is not safe is dropped, never stored."""

    def setUp(self):
        self.t = {"score": "tokens", "accept": [core.hx(8)], "kind": "structs", "question": "At which byte offset (hex, like 0x1c) of the node is the pointer?"}

    def test_the_reviewers_contradiction_is_not_stored(self):
        self.assertTrue(scoring.score("0x8\nWait, actually the next pointer is at 0x70.", self.t), "the headline alone is right")
        r = scoring.sft_target("0x8\nWait, actually the next pointer is at 0x70.", self.t)
        self.assertEqual((r["text"], r["sentence"], r["sentence_dropped"]), ("0x8", "", True))

    def test_truncation_keeps_headline_and_first_sentence(self):
        r = scoring.sft_target("**0x8**\n\nThe next pointer is read at offset 0x8. It then follows the pointer. The loop ends at NULL.", self.t)
        self.assertEqual(r["text"], "0x8\nThe next pointer is read at offset 0x8.")
        self.assertTrue(r["truncated"] and not r["sentence_dropped"])
        r = scoring.sft_target("0x8\nThe next pointer is read at offset 0x8.", self.t)
        self.assertFalse(r["truncated"])
        self.assertEqual(scoring.sft_target("0x8", self.t)["text"], "0x8")
        self.assertEqual(scoring.sft_target("Answer:\n0x8\nIt is the first field of the node.", self.t)["text"], "Answer:\n0x8\nIt is the first field of the node.")
        self.assertEqual(scoring.sft_target("0x8\n- It is the first field of the node.\n- It is read once.", self.t)["text"], "0x8\nIt is the first field of the node.")
        self.assertEqual(scoring.sft_target("0x8\nThe loop, e.g. in the second block, reads it. More text.", self.t)["sentence"], "The loop, e.g. in the second block, reads it.")

    def test_a_justification_with_a_foreign_number_name_or_hesitation_is_dropped(self):
        for body in ("The next pointer is at 0x70 in the node.", "It is stored at offset 0x8, not at the offset 0x10.", "It is 0x8 or maybe 0x10 in the node.",
                     "However the field after it is at 12 in the node.", "The first guess was 0x4 but the code reads 0x8.", "Hmm, that is the tag of the node.",
                     "Correction: it is at 0x8 in the node.", "It cannot be 0x8 for a packed struct.", "The struct is 8-byte aligned (x86-64) on this target.",
                     "One field comes first and the pointer follows it."):
            r = scoring.sft_target(f"0x8\n{body}", self.t)
            self.assertIsNotNone(r, body)
            self.assertEqual((r["text"], r["sentence_dropped"]), ("0x8", True), body)
        for body in ("The next pointer sits after the tag of the node.", "It is the field read through p->next, at 0x8 in the struct.",
                     "The pointer that proc follows to reach the next node is read from the second word."):
            r = scoring.sft_target(f"0x8\n{body}", self.t)
            self.assertIsNotNone(r, body)
            self.assertFalse(r["sentence_dropped"], body)

    def test_names_and_strings(self):
        xtea = {"score": "tokens", "accept": [["xtea"], ["encrypt"], core.dec(32)], "reject": ["tea", "xxtea", "aes"], "allow": ["extended tea"], "kind": "cipher",
                "variant": "xtea_enc", "question": "This function transforms an 8-byte block with a 128-bit key: which cipher, direction and how many iterations"}
        self.assertFalse(scoring.sft_target("XTEA, encrypt, 32\nIt is the eXtended TEA with a key-dependent index of the sum.", xtea)["sentence_dropped"])
        self.assertTrue(scoring.sft_target("XTEA, encrypt, 32\nUnlike TEA it indexes the key with the sum of the delta.", xtea)["sentence_dropped"])
        self.assertTrue(scoring.sft_target("XTEA, encrypt, 32\nIt uses the constant 0x9e3779b9 in every round of the loop.", xtea)["sentence_dropped"], "another number")
        self.assertFalse(scoring.sft_target("XTEA, encrypt, 32\nThe loop runs 32 times over a 128-bit key and an 8-byte block.", xtea)["sentence_dropped"])
        exact = {"score": "exact_head", "accept": ["admin_login42"], "kind": "strobf"}
        self.assertFalse(scoring.sft_target("admin_login42\nEach byte is XORed with a constant before it is stored.", exact)["sentence_dropped"])
        self.assertTrue(scoring.sft_target("admin_login42\nEach byte is XORed with 0x37 before it is stored.", exact)["sentence_dropped"])
        self.assertTrue(scoring.sft_target("admin_login42\nIt could also read admin_login43 in a variant.", exact)["sentence_dropped"])
        self.assertTrue(scoring.sft_target("admin_login42\nThe string is built from admin and login, i.e. admin.login42 in the end.", exact)["sentence_dropped"])
        self.assertIsNone(scoring.sft_target("nope\nadmin_login42", exact))

    def test_m2_exact_keys_without_a_separator_and_a_lead_in_over_two_candidates(self):
        key = {"score": "exact_head", "accept": ["adminlogin42"], "kind": "strobf"}
        for body in ("The buffer then holds adminlogin4x after the loop.", "It may start with adminlogon42 in some builds."):
            self.assertTrue(scoring.sft_target(f"adminlogin42\n{body}", key)["sentence_dropped"], body)
        word = {"score": "exact_head", "accept": ["retries"], "kind": "keycount"}
        self.assertTrue(scoring.sft_target("retries\nThe buffer then holds retriex after the loop.", word)["sentence_dropped"], "a near copy of a plain-word key")
        self.assertFalse(scoring.sft_target("retries\nThe function counts the lines that start with this key.", word)["sentence_dropped"])
        url = {"score": "exact_head", "accept": ["http://login.local/driver"], "kind": "strobf"}
        self.assertFalse(scoring.score("Answer:\nhttp://login.local/driver\nhttp://login.local/drivex", url), "the second candidate has four alphabetic runs and no spaces")
        self.assertFalse(scoring.score("http://login.local/driver\nhttp://login.local/drivex", url))
        self.assertTrue(scoring.score("Answer:\nhttp://login.local/driver\nIt is built from the bytes of the table.", url))

    def test_m1_a_sentence_is_never_cut_at_an_abbreviation_a_number_or_an_open_parenthesis(self):
        """sft_trunc.py of the review: the stored sentence ends where the sentence ends, or the sentence is dropped."""
        for body, kept in (("Offset is 0x8 (approx. the second word) of the node.", "Offset is 0x8 (approx. the second word) of the node."),
                           ("Offset is 0x8, see Fig. 3 of the notes. More text follows here.", None),
                           ("No. 5 is the pointer to the payload. Offset 8 is next.", None),
                           ("U.S. English text is used here. Next node at 8.", "U.S. English text is used here."),
                           ('He says "it is 0x8." Then it moves on. And more.', 'He says "it is 0x8."'),
                           ("The value is 0.5 of the node size, i.e. half of it. Done.", None),
                           ("It sits after the tag, e.g. after the data pointer. The rest is padding.", "It sits after the tag, e.g. after the data pointer.")):
            r = scoring.sft_target(f"0x8\n{body}", self.t)
            self.assertIsNotNone(r, body)
            if kept is None:
                self.assertTrue(r["sentence_dropped"] or r["sentence"].endswith((".", '."')), (body, r))
            else:
                self.assertEqual(r["sentence"], kept, body)
            if r["sentence"]:
                self.assertEqual(r["sentence"].count("("), r["sentence"].count(")"), r)
                self.assertTrue(r["sentence"].endswith((".", '."', "!", "?")), r)
        self.assertEqual(scoring._first_sentence("It is the pointer (approx. the second word) of the node. Then it moves on.")[0],
                         "It is the pointer (approx. the second word) of the node.")
        self.assertEqual(scoring._first_sentence("It reads 0x8. The loop ends.")[0], "It reads 0x8.", "hex followed by a full stop ends a sentence at a capital")

    def test_the_stored_text_always_scores(self):
        for fam in am.NEW_FAMILIES:
            for t in families.generate(fam, 70, 5):
                m = t.meta()
                for reply in (f"{scoring.canonical(m)}\nThis follows from the code of the function.", f"Answer:\n{scoring.canonical(m)}",
                              f"**{scoring.canonical(m)}**\nShort but with enough words. Another sentence."):
                    r = scoring.sft_target(reply, m)
                    self.assertIsNotNone(r, (t.id, reply))
                    self.assertTrue(scoring.score_strict(r["text"], m), (t.id, r["text"]))
                    self.assertTrue(scoring.score(r["text"], m), (t.id, r["text"]))

    def test_a_failed_request_is_never_stored(self):
        self.assertIsNone(scoring.sft_target("[error TimeoutError('x')]", self.t))
        skip = {"score": "any", "accept": ["0x0a", "0a", "10"]}
        err = "[error <HTTPError 500: 'Internal Server Error'>]"
        self.assertFalse(scoring.score_strict("[error TimeoutError('x')]", skip))
        self.assertTrue("a" in err and scoring._legacy_score(err, skip), "the frozen matcher alone would take the letter a in the error text")
        self.assertFalse(scoring.score(err, skip) or scoring.score_any_line(err, skip), "neither scorer accepts the text of a failed request")

    def test_a_right_headline_is_always_stored_and_a_short_first_sentence_is_dropped(self):
        """A stored line that is no sentence would read as a list when the text is scored again: it is dropped instead, the headline stays."""
        for body in ("U.S. English. Next node at 8.", "It is 0x8. Actually the field at 0x10 is the real one.", "Yes. The pointer follows the payload in the node."):
            r = scoring.sft_target(f"0x8\n{body}", self.t)
            self.assertIsNotNone(r, body)
            self.assertEqual((r["text"], r["sentence_dropped"], r["truncated"]), ("0x8", True, True), body)
            self.assertTrue(scoring.score_strict(r["text"], self.t))


class DataPathAgainstEvaluation(unittest.TestCase):
    """m10: the frozen whole-reply substring scorer is for modal_rea.py's evaluation; the data (sample stage, sft_target) uses the grammar for every family."""

    LEGACY_KINDS = ("xor-key", "password", "magic", "lcg", "hash", "skip", "keycount", "alphabet", "rotate", "threshold")

    def test_the_evaluation_scorer_is_still_the_frozen_one(self):
        skip = {"id": "skip-7-0055", "kind": "skip", "score": "any", "accept": ["0x0a", "0a", "10"], "question": "The checksum skips bytes with one specific value. Which value (hex)?"}
        self.assertTrue(scoring.score("the letter a", skip), "the substring scorer accepts any reply with the letter a: frozen on purpose")
        self.assertTrue(scoring.score("magic 0xDEADBEEF version 3", {"score": "all", "accept": ["deadbeef", "3"]}))

    def test_the_data_scorer_does_not_accept_what_the_substring_scorer_does(self):
        skip = {"id": "skip-7-0055", "kind": "skip", "score": "any", "accept": ["0x0a", "0a", "10"], "question": "The checksum skips bytes with one specific value. Which value (hex)?"}
        for r in ("the letter a", "a", "I do not know, maybe 0 1 2 3 a b c d e f", "10 or 11", "not 0x0a", "10", "0x10"):
            self.assertFalse(scoring.score_strict(r, skip), r)         # `10` is 0x10 to a model that answers in hex (round 4, m1)
        for r in ("0x0a", "0A", "0a", "The value is 0x0a.", "0x0a\nThe checksum skips every byte with this value."):
            self.assertTrue(scoring.score_strict(r, skip), r)
        self.assertIsNone(scoring.sft_target("the letter a", skip))

    def test_content_free_replies_pass_the_evaluation_scorer_on_some_legacy_tasks_and_the_data_scorer_on_none(self):
        fillers = ["0 1 2", "1 2 3", "0x10 0x20", "0 1 2 3 4 8 16 32 64 255 256 0x10 0x20", "a b c d e f", "the key is a value"]
        lax = strict = 0
        for seed in (7, 99, 1001):
            for t in families.generate("legacy", 300, seed):
                m = t.meta()
                lax += any(scoring.score(r, m) for r in fillers)
                strict += any(scoring.score_strict(r, m) for r in fillers)
        self.assertGreater(lax, 0, "the frozen scorer does accept some content-free replies (documented)")
        self.assertEqual(strict, 0)

    def test_every_legacy_template_converts_and_its_canonical_answer_passes(self):
        seen = set()
        for t in families.generate("legacy", 200, 1001):
            m = t.meta()
            st = scoring.strict_task(m)
            seen.add(m["kind"])
            self.assertIn(st["score"], scoring.HEADLINE_MODES)
            self.assertTrue(scoring.score_strict(scoring.canonical(st), m), (t.id, scoring.canonical(st)))
            self.assertTrue(scoring.score(scoring.canonical(m), m), "the evaluation scorer accepts the canonical answer of the original truth")
        self.assertEqual(seen, set(self.LEGACY_KINDS))

    def test_legacy_hash_names_and_values(self):
        fnv = next(t.meta() for t in families.generate("legacy", 600, 7) if t.meta()["kind"] == "hash" and t.meta()["accept"][0][0] == "fnv")
        h = fnv["accept"][1][0]
        for r in (f"FNV-1a, 0x{h}", f"fnv1a and {h}", f"fnv, 0x{h.upper()}"):
            self.assertTrue(scoring.score_strict(r, fnv), r)
        for r in (f"djb2, 0x{h}", f"fnv, 0x{h[:-1]}{'0' if h[-1] != '0' else '1'}", "fnv", f"fnv or djb2, 0x{h}"):
            self.assertFalse(scoring.score_strict(r, fnv), r)


class Selection(unittest.TestCase):
    """A task with a high chance keeps answers only when at least two of the K samples are right; every reply is scored again."""

    def test_agreement_only_above_the_chance_threshold(self):
        easy = {"score": "tokens", "accept": [core.dec(5)], "chance": 0.5, "kind": "crc"}
        hard = {**easy, "chance": 0.01}
        legacy = {"score": "exact", "accept": ["ab"], "kind": "xor-key"}
        one = [("5", True), ("6", False), ("7", False)]
        two = [("5", True), ("5\nThe loop stops at five times the stride.", True), ("7", False)]
        self.assertEqual(sft.select(one, easy), ([], "agreement"))
        self.assertEqual([r["text"] for r in sft.select(two, easy)[0]], ["5", "5\nThe loop stops at five times the stride."])
        self.assertEqual(sft.select(one, hard)[1], "kept")
        self.assertEqual(sft.select([("ab", True)], legacy)[1], "kept", "no chance recorded: no agreement rule")
        self.assertEqual(sft.select([("6", False)], hard), ([], "no_correct"))
        self.assertEqual(sft.select([("5\nBut maybe 6 is right.", True)], hard)[1], "kept", "the headline is right: the sentence is dropped, not the reply")
        dup = sft.select([("5", True), ("5", True)], hard)[0]
        self.assertEqual(len(dup), 1, "identical stored texts are kept once")

    def test_the_count_is_made_of_storable_headlines_and_flags_are_ignored(self):
        """m3: two replies whose headlines are right agree even when one has a hedging body; a True flag on a wrong reply is not a vote."""
        easy = {"score": "tokens", "accept": [core.hx(0xffff, 4)], "chance": 0.55, "kind": "crc"}
        rows = [("0xffff", True), ("0xffff\nBut maybe it is 0x0 for some inputs of the table.", True)]
        recs, why = sft.select(rows, easy)
        self.assertEqual((why, [r["sentence_dropped"] for r in recs]), ("kept", [False]), "both agree; identical stored texts are one record")
        self.assertEqual(sft.select([("0xffff", True), ("0x0", True)], easy), ([], "agreement"), "a stale True flag on a wrong reply is not an agreeing sample")
        self.assertEqual(sft.select([("0xffff", False), ("0xffff", False)], easy)[1], "kept", "a False flag on a right reply does not hide it")

    def test_assemble_scores_every_reply_again(self):
        """m3 (3): a checkpoint written by an older scorer (a True flag on an error text, a right reply flagged False) is not trusted."""
        t = {"id": "s-1", "kind": "skip", "score": "any", "accept": ["0x0a", "0a", "10"], "chance": 0.01}
        msgs = lambda task: []
        results = {"s-1": {"outs": [["[error TimeoutError('x')]", True], ["the letter a", True], ["0x0a", False]]}}
        kept, st = sft.assemble([t], results, "m", msgs)
        self.assertEqual([r["messages"][-1]["content"] for r in kept], ["0x0a"])
        self.assertEqual((st["errors"], st["correct"], st["kept_records"]), (1, 1, 1))

    def test_every_variant_above_the_threshold_is_known(self):
        from rea_bench import chance
        high = sorted(k for k, v in chance.CHANCE.items() if v >= sft.AGREE_CHANCE)
        for v in ("crc/bitwise_init", "pktsum/xor8", "cipher/xtea_dec", "cipher/tea_enc", "cipher/chacha", "structs/linked_next", "encoding/base64_decode"):
            self.assertIn(v, high)
        for v in ("strobf/xor_const", "multi/decode_secret", "bounds/clamp", "antidebug/timing_ns", "bounds/index_limit", "bounds/loop_iterations"):
            self.assertNotIn(v, high)
        for v in ("fsm/opcode_value", "bounds/char_class", "multi/limit_helper", "crc/bitwise_poly", "crc/table_poly"):
            self.assertIn(v, high, "round 4 (m8): a variant a guess solves 6% of the time kept one lucky sample of six in 31% of its tasks")

    def test_chance_is_at_least_the_best_constant_guess_of_the_variant(self):
        """m3 (2): a model that reads the cipher from the code and guesses the round count is right on 28-32% of TEA/XTEA tasks."""
        from rea_bench import chance
        ts = [t for t in families.generate("cipher", 6000, 7) if t.variant == "xtea_dec"]
        best = max(__import__("collections").Counter(chance._answer(t) for t in ts).values()) / len(ts)
        self.assertGreater(best, 0.25)
        self.assertGreater(chance.CHANCE["cipher/xtea_dec"], 0.2, "the table is no longer the 4.5% of a guesser that samples the answers")

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

    def test_dropped_sentences_are_counted_and_marked(self):
        t = {"id": "a", "kind": "structs", "score": "tokens", "accept": [core.hx(8)], "chance": 0.01}
        kept, st = sft.assemble([t], {"a": {"outs": [["0x8\nWait, actually it is 0x70 in some builds of it.", True], ["0x8\nIt is the pointer to the next node.", True]]}}, "m",
                                lambda task: [])
        self.assertEqual(sorted(r["sentence_dropped"] for r in kept), [False, True])
        self.assertEqual(st["sentence_dropped"], 1)
        self.assertEqual(st["by_kind"]["structs"]["sentence_dropped"], 1)


if __name__ == "__main__":
    unittest.main()

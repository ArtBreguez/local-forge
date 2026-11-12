"""The whitelist grammar of a headline (round 3: M1, M2, M3, m2, m9) and the property test of the scorer.

Every wrong reply in the review failed for one reason: something was left over after the answer. These tests pin that rule from the grammar's side
(what it parses, what it leaves over) and the property test checks it on 2,000 random tasks per family.
"""
from __future__ import annotations

import sys
import unittest

from . import rea_property as rp
from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, families, grammar, scoring  # noqa: E402

S = scoring.score


def one(*spellings, **extra):
    """A task whose answer is a single number: `core.dec`/`core.hx` style spelling list."""
    return {"score": "tokens", "accept": [list(spellings)], **extra}


class Normalisation(unittest.TestCase):
    def test_unicode_quotes_apostrophes_dashes_minus_and_spaces_become_ascii(self):
        n = grammar.normalise
        self.assertEqual(n("isn’t ‘x’ “y” „z“"), "isn't 'x' \"y\" \"z\"")
        self.assertEqual(n("a b　c d"), "a b c d")
        self.assertEqual(n("−40 –40 —40 ‐40"), "-40 -40 -40 -40")
        self.assertEqual(n("＋４０ ＂ａ＂"), "+40 \"a\"")
        self.assertEqual(n(r"\x3f"), "x3f")
        self.assertEqual(n("≤ ≥ ≠ …"), "<= >= != ...")

    def test_a_minus_sign_is_the_key_only_for_a_negative_key(self):
        pos, neg = one("40", "0x28"), one("-50")
        for r in ("−40", "–40", "-40", "−40."):
            self.assertFalse(S(r, pos), r)
        for r in ("−50", "–50", "-50", "- 50 "):
            self.assertEqual(S(r, neg), r != "- 50 ", r)
        self.assertTrue(S("40", pos) and S("０x28", pos) and S("４０", pos), "full-width digits are digits")

    def test_non_ascii_leftovers_reject_and_unicode_digits_are_not_numbers(self):
        t = one("40", "0x28")
        for r in ("٤٠", "४०", "⁴⁰", "40²", "4​0", "40∎", "40→", "40 ≈", "≈40", "4０ ０"):
            self.assertFalse(S(r, t), r)


class M1(unittest.TestCase):
    """Any word, symbol or number that is not part of the answer rejects; the key's own spellings accept."""

    t = one("40", "0x28", kind="structs", question="What is the size in bytes (decimal) of one structure?")

    def test_negation_and_limit_forms(self):
        for r in ("≠ 40", "!= 40", "<> 40", "!40", "−40", "–40", "no 40", "No, 40", "never 40", "neither 40", "N/A 40", "unknown, 40", "wrong: 40",
                  "incorrect, 40", "false: 40", "avoid 40", "none of 40", "other than 40", "anything but 40", "all but 40", "except 40", "without 40",
                  "besides 40", "excluding 40", "non-40", "at least 40", "at most 40", "less than 40", "more than 40", "up to 40", "over 40", "under 40",
                  "above 40", "below 40", "> 40", "< 40", "isnt 40", "isn´t 40", "isnʼt 40", "isn′t 40", "isn＇t 40", "isn't 40", "isn’t 40", "¬40", "≤ 40",
                  "≥ 40", "~40", "±40", "40?", "40...", "approximately 40", "about 40", "roughly 40", "40 or so", "40 I think", "maybe 40", "probably 40",
                  "40 (not 41)", "40 (unsure)", "== 40", "40 + 1", "40 - 1", "40 * 2", "- 40", "+ 40 - 1", "#40", "$40", "@40", "&40"):
            self.assertFalse(S(r, self.t), r)

    def test_number_words_and_a_second_candidate(self):
        for r in ("40, forty-one", "40 and forty-one", "40 / forty-one", "forty-one, 40", "40; 41", "40 | 41", "40 (41)", "40, 0x29", "40 0x29", "forty, forty-one",
                  "40, Rabbit", "40 and Rabbit"):
            self.assertFalse(S(r, self.t), r)
        for r in ("40", "0x28", "0X28", "forty", "Forty", "40 (0x28)", "0x28 (40)", "0x28, 40", "40 bytes", "It is 40.", "The size is 40 bytes.", "size: 40",
                  "`40`", "**40**", "(40)", '"40"', "40,", "40;", "40:", "40!", "exactly 40", "40 / 0x28", "answer: 40"):
            self.assertTrue(S(r, self.t), r)

    def test_a_word_that_is_not_in_the_grammar_rejects(self):
        for r in ("40 and the loop runs", "40 because", "40, which is the size", "the answer might be 40", "40 on x86-64", "40 (x86-64)", "40 aligned"):
            self.assertFalse(S(r, self.t), r)

    def test_names(self):
        bb = {"score": "tokens", "accept": [["bbb", "bbh"]], "cand": "[bhx]{2,}", "kind": "fsm", "variant": "table_input",
              "question": "Give a shortest input string (letters only) for which proc returns 7."}
        for r in ("bbb", "b b b", "b, b, b", "Input: bbb", "The input is bbb.", "bbb bbh", '"bbb"'):
            self.assertTrue(S(r, bb), r)
        for r in ("bbb or bbh", "not bbb", "bbb, hhh", "bb", "bbbb", "xbbb", "bbb?", "b b b b", "bbb.bbh", "b b b h h h"):
            self.assertFalse(S(r, bb), r)

    cip = {"score": "tokens", "accept": [["xtea"], ["decrypt"], core.dec(32)], "allow": ["extended tea"], "kind": "cipher", "variant": "xtea_dec",
           "question": "This function transforms an 8-byte block with a 128-bit key: cipher, direction, iterations?"}

    def test_the_name_groups_come_in_any_order_the_numbers_keep_theirs(self):
        cip = self.cip
        for r in ("XTEA, decrypt, 32", "xtea decrypt 32 rounds", "XTEA (decrypt), 32 iterations", "XTEA, decrypt, 32 (128-bit key)", "extended TEA: XTEA, decrypt, 32",
                  "XTEA, decrypt, thirty-two",
                  "decrypt, XTEA, 32", "decrypt XTEA 32 rounds", "Decrypt, XTEA: 32", "decrypt (XTEA), 32 iterations", "decrypt with XTEA, 32 rounds.",
                  "decrypt and XTEA and 32", "decrypt, XTEA (extended TEA), 32", "decrypt, XTEA, 0x20", "decrypt / XTEA / 32 (128-bit key)"):
            self.assertTrue(S(r, cip), r)
        for r in ("32, XTEA, decrypt", "XTEA, 32, decrypt", "decrypt, 32, XTEA", "32, decrypt, XTEA", "32 rounds, decrypt XTEA", "XTEA, decrypt",
                  "decrypt, 32", "XTEA, 32", "XTEA, encrypt, 32", "encrypt, XTEA, 32", "TEA, decrypt, 32", "decrypt, TEA, 32", "XTEA, decrypt, 33", "decrypt, XTEA, 33",
                  "XTEA, decrypt, 32 bytes", "decrypt, XTEA, 32 bits", "decrypt, XTEA, 32 (64-bit key)", "XTEA decrypt or encrypt, 32", "decrypt or encrypt, XTEA, 32",
                  "XTEA, decrypt, 32, 32.0", "decrypt, XTEA, 32, 32.0"):
            self.assertFalse(S(r, cip), r)

    def test_every_group_exactly_once_and_nothing_left_over_in_any_name_order(self):
        cip = self.cip
        for r in ("XTEA, XTEA, 32", "decrypt, decrypt, 32", "decrypt, XTEA, decrypt, 32", "XTEA, decrypt, XTEA, 32", "decrypt, XTEA, 32, decrypt", "decrypt, XTEA, 32, XTEA",
                  "decrypt, XTEA, XTEA decrypt, 32", "decrypt, XTEA, not 32", "not decrypt, XTEA, 32", "decrypt, not XTEA, 32", "decrypt, XTEA (not TEA), 32",
                  "decrypt, XTEA, 32 or 33", "decrypt or XTEA, 32", "decrypt, XTEA; maybe 32", "decrypt, XTEA, 32?", "decrypt, XTEA, 32...", "decrypt, XTEA, rabbit, 32",
                  "decrypt, XTEA, encrypt, 32", "decrypt, TEA, XTEA, 32", "decrypt/encrypt, XTEA, 32"):
            self.assertFalse(S(r, cip), r)
        for r in ("decrypt decrypt, XTEA, 32", "XTEA XTEA decrypt 32"):
            self.assertTrue(S(r, cip), r)                  # a group written twice in a row is one group in two spellings, as before

    def test_a_block_is_a_run_of_name_groups_a_number_is_a_block_of_its_own(self):
        sp = grammar.Spec(self.cip)
        self.assertEqual(sp.block, [0, 0, 1])
        enc = {"score": "tokens", "accept": [["encode", "encodes"], ["base64", "base-64"]]}
        self.assertEqual(grammar.Spec(enc).block, [0, 0])
        mixed = {"score": "tokens", "accept": [["tea"], core.dec(8), ["encrypt"], ["xtea"], core.dec(9)], "kind": "x"}
        self.assertEqual(grammar.Spec(mixed).block, [0, 1, 2, 2, 3])
        for r in ("tea, 8, encrypt, xtea, 9", "tea, 8, xtea, encrypt, 9", "tea 8 xtea encrypt 9"):
            self.assertTrue(grammar.headline_ok(r, mixed), r)
        for r in ("encrypt, 8, tea, xtea, 9", "tea, encrypt, 8, xtea, 9", "8, tea, encrypt, xtea, 9", "tea, 8, encrypt, 9, xtea", "tea, 9, encrypt, xtea, 8",
                  "tea, 8, encrypt, xtea", "tea, 8, encrypt, xtea, 8"):
            self.assertFalse(grammar.headline_ok(r, mixed), r)

    def test_name_groups_of_a_block_never_share_a_spelling_in_the_families(self):
        """A token that two unwritten name groups both spell could be given to the wrong one; none of the generated tasks has such a pair."""
        for fam in families.modules():
            for t in families.generate(fam, 300, 7):
                m = scoring.strict_task(t.meta())
                if m["score"] != "tokens":
                    continue
                sp = grammar.Spec(m)
                for a in range(len(sp.groups)):
                    for b in range(a):
                        if a not in sp.numeric and b not in sp.numeric and sp.block[a] == sp.block[b]:
                            self.assertFalse(sp.groups[a][1] & sp.groups[b][1], (fam, m["accept"]))


class M2Bounds(unittest.TestCase):
    """A closed interval is written with inclusive forms only; every exclusive or half-open notation rejects."""

    t = {"score": "tokens", "accept": [core.dec(1570), core.dec(1751)], "kind": "bounds", "variant": "range_chain",
         "question": "Give the lowest and the highest such value (inclusive, decimal)."}

    def test_inclusive_forms(self):
        for r in ("1570 to 1751", "1570..1751", "1570-1751", "1570 - 1751", "[1570, 1751]", "from 1570 to 1751", "1570 <= x <= 1751", "1570 ≤ v ≤ 1751", "1570 and 1751",
                  "1570, 1751", "lowest 1570, highest 1751", "lower bound is 1570, upper 1751", "1570 to 1751 inclusive", "0x622 to 0x6d7", "[0x622, 1751]",
                  "Lowest: 1570; highest: 1751.", "1,570 to 1,751"):
            self.assertTrue(S(r, self.t), r)

    def test_exclusive_half_open_and_crossed_forms(self):
        for r in ("x > 1570 and x < 1751", "> 1570 and < 1751", "(1570, 1751)", "[1570, 1751)", "(1570, 1751]", "1570 <= x < 1751", "1570 < x <= 1751",
                  "1570 inclusive, 1751 exclusive", "strictly between 1570 and 1751", "between 1570 and 1751", "greater than 1570, less than 1751", "1570 to 1751 exclusive",
                  "above 1570, below 1751", "v > 1570 && v < 1751", "1751, 1570", "1751 to 1570", "highest 1751, lowest 1570", "lowest 1751, highest 1570",
                  "highest 1570, lowest 1751", "lowest 1570", "1570", "1570 to", "1570 to 1570", "1570 - 1570", "1570, to 1751", "1570, - 1751", "1570, --1751",
                  "1570 1751 1752", "1571 to 1750", "1570 to 1751, 1752", "from 1570", "1570 and 1751 and 1752"):
            self.assertFalse(S(r, self.t), r)

    def test_negative_bounds(self):
        t = {**self.t, "accept": [["-50"], core.dec(41)]}
        for r in ("-50 to 41", "-50..41", "-50-41", "−50 to 41", "[-50, 41]", "-50 <= x <= 41", "-50, 41", "lowest -50, highest 41"):
            self.assertTrue(S(r, t), r)
        for r in ("50 to 41", "-50 to -41", "-50 > x", "(-50, 41)", "-50, -41", "50, 41", "−50 to 41 exclusive"):
            self.assertFalse(S(r, t), r)

    def test_an_interval_form_needs_an_interval_task(self):
        """`to` and `-` between two numbers are ranges only where the question asks for an interval (bounds, threshold)."""
        magic = {"score": "tokens", "accept": [core.dec(1570), core.dec(7)], "kind": "magic"}
        self.assertTrue(S("1570, 7", magic))
        for r in ("1570 to 7", "1570-7", "[1570, 7]", "lowest 1570, highest 7", "1570..7", "x 1570, 7"):
            self.assertFalse(S(r, magic), r)
        self.assertTrue(S("(1570, 7)", magic), "a pair in parentheses is a plain pair where the two numbers are not bounds")


class M3Context(unittest.TestCase):
    """A number inside a unit or a standard name is never a hit; it is allowed only where the question itself states that quantity."""

    cip = {"score": "tokens", "accept": [["rc4"], core.dec(11), ], "kind": "cipher", "variant": "rc4",
           "question": "This function transforms a buffer in place. Which cipher is it, and how many bytes long is its key (decimal)?"}
    tea = {"score": "tokens", "accept": [["tea"], ["encrypt"], core.dec(48)], "kind": "cipher", "variant": "tea_enc",
           "question": "This function transforms an 8-byte block (two 32-bit words) with a 128-bit key and returns a number. Which cipher is it, does it encrypt or decrypt, "
                       "and how many iterations does its main loop run (decimal)?"}
    crc = {"score": "tokens", "accept": [core.hx(0xedb88320, 8), ["reflected"]], "kind": "crc", "variant": "bitwise_poly",
           "question": "This function computes a 32-bit CRC. Give the polynomial constant exactly as it appears in the code (hex), and say whether it is 'reflected' or 'normal'."}

    def test_the_exempt_number_is_not_a_hit(self):
        for r in ("CRC-11", "AES-11", "SHA-11", "MD-11", "RC-11", "DES-11", "x86-11", "x64-11", "base-11", "11-bit", "11-bits", "11-byte aligned", "11-bytes", "(11 bit)", "(11 bits)",
                  "RC4 (11-bit)", "11-byte"):
            self.assertFalse(S(r, self.cip), r)
        self.assertFalse(S("48-bit", self.tea) or S("TEA, encrypt, (48 bits)", self.tea) or S("CRC-48", self.tea))

    def test_a_wrong_side_claim_next_to_a_right_answer_rejects(self):
        for r in ("RC4, 11 bytes (99 bits)", "RC4, 11 bytes (64-bit key)", "RC4, 11 (12 bits)", "RC4, 11 bytes, 8-byte aligned"):
            self.assertFalse(S(r, self.cip), r)
        for r in ("TEA, encrypt, 48 (64-bit key)", "TEA, encrypt, 48 iterations (a 64-bit block)", "TEA, encrypt, 48 (256-bit key)", "TEA, encrypt, 48 (8-bit)",
                  "TEA, encrypt, 48 (CRC-32)", "TEA, encrypt, 48 on x86-64"):
            self.assertFalse(S(r, self.tea), r)

    def test_a_number_the_question_states_is_allowed_beside_a_right_answer(self):
        for r in ("RC4, 11 bytes (88 bits)", "RC4, 11 (88 bits)", "RC4: 11 bytes", "Key: RC4, 11 bytes", "RC4 cipher, 11 bytes", "It is the RC4 cipher, 11 bytes."):
            self.assertTrue(S(r, self.cip), r)
        for r in ("TEA, encrypt, 48 (128-bit key)", "TEA, encrypt, 48 iterations (a 128-bit key)", "TEA, encrypt, 48 (an 8-byte block)", "TEA, encrypt, 48 (128 bits)",
                  "TEA, encrypt, 48 rounds (32-bit)"):
            self.assertTrue(S(r, self.tea), r)
        self.assertTrue(S("0xEDB88320, reflected (CRC-32)", self.crc) and S("0xedb88320 reflected, 32-bit", self.crc))
        self.assertFalse(S("0xEDB88320, reflected (CRC-16)", self.crc) or S("0xEDB88320, reflected (16-bit)", self.crc))

    def test_a_noun_after_a_group_cannot_label_the_wrong_number(self):
        """A noun that names the quantity leads in; behind a group it could put `key` on the round count (`TEA, encrypt, key 48`)."""
        for r in ("TEA, encrypt, key 48", "TEA, encrypt, size 48", "TEA, encrypt, offset 48", "RC4, key length 11 bytes", "RC4, key 11", "It is RC4 with a key length of 11 bytes.",
                  "TEA, encrypt, 48, a block cipher"):
            self.assertFalse(S(r, self.tea if "TEA" in r else self.cip), r)
        self.assertTrue(S("RC4, 11 bytes (88-bit key)", self.cip), "the label of a context number follows it")

    def test_returns_and_variable_are_allowed_where_the_question_uses_them(self):
        ad = {"score": "tokens", "accept": [["tracerpid"], core.dec(144)], "kind": "antidebug", "variant": "tracerpid",
              "question": "Which anti-debugging technique does proc implement, and what does proc return when a tracer is attached (decimal)?"}
        for r in ("tracerpid, returns 144", "tracerpid check, returns 144", "tracerpid: it returns 144"):
            self.assertTrue(S(r, ad), r)
        no_return = {**ad, "question": "Which anti-debugging technique does proc implement, and above how many nanoseconds does it report?"}
        self.assertFalse(S("tracerpid, returns 144", no_return), "`returns` labels a number only where the question asks for a return value")
        env = {"score": "tokens", "accept": [["env", "environment", "getenv"], ["rt_verbose"]], "cand": "[a-z0-9]*_[a-z0-9_]*", "kind": "antidebug", "variant": "env"}
        for r in ("env, RT_VERBOSE", "environment variable RT_VERBOSE", "env check, variable name RT_VERBOSE", "getenv, 'RT_VERBOSE'"):
            self.assertTrue(S(r, env), r)
        for r in ("env, RT_VERBOSE_X", "env, RT_OTHER", "env or RT_VERBOSE", "environment variable"):
            self.assertFalse(S(r, env), r)

    def test_the_question_decides(self):
        """The same reply passes or fails with the question's own words: without a question there is nothing to exempt."""
        bare = {k: v for k, v in self.tea.items() if k != "question"}
        self.assertFalse(S("TEA, encrypt, 48 (128-bit key)", bare))
        self.assertTrue(S("TEA, encrypt, 48 (128-bit key)", self.tea))


class M9NumberWords(unittest.TestCase):
    def test_number_words_are_numbers_in_both_directions(self):
        t = one("32", "0x20", kind="cipher", variant="chacha", question="how many rounds")
        for r in ("thirty-two", "Thirty-Two rounds", "thirty-two (32)", "0x20 (thirty-two)"):
            self.assertTrue(S(r, t), r)
        for r in ("32, thirty-three", "thirty-two or thirty-three", "thirty-two, one", "thirty three", "thirty-two-ish", "a hundred and thirty-two"):
            self.assertFalse(S(r, t), r)
        self.assertEqual(grammar._NUMWORD["ninety-nine"], 99)
        self.assertEqual(len(grammar._NUMWORD), 100)


class Units(unittest.TestCase):
    def test_a_unit_is_the_questions_own_word(self):
        by = {"structs": ("bytes", "byte"), "fsm/table_states": ("states", "state"), "antidebug/timing_ns": ("ns", "nanoseconds"),
              "antidebug/timing_us": ("us", "microseconds", "µs"), "cipher": ("rounds", "iterations"), "fsm/opcode_count": ("opcodes", "cases")}
        for key, units in by.items():
            kind, _, variant = key.partition("/")
            t = {"score": "tokens", "accept": [["7", "0x7"]], "kind": kind, "variant": variant}
            for u in units:
                self.assertTrue(S(f"7 {u}", t), (key, u))
            for u in ("bits", "bit", "kilobytes", "seconds", "ms"):
                self.assertFalse(S(f"7 {u}", t), (key, u))
        crc = {"score": "tokens", "accept": [["0x7"]], "kind": "crc", "variant": "bitwise_init"}
        self.assertFalse(S("0x7 bytes", crc) or S("0x7 rounds", crc))
        states = {"score": "tokens", "accept": [["5", "0x5"]], "kind": "fsm", "variant": "table_states"}
        self.assertFalse(S("5 bytes", states), "a wrong unit is not decoration")

    def test_a_unit_belongs_to_the_last_number_of_a_pair(self):
        rot = {"score": "tokens", "accept": [["0x1234abcd", "1234abcd"], ["7"]], "kind": "rotate", "variant": ""}
        self.assertTrue(S("0x1234abcd, 7 bits", rot))
        self.assertFalse(S("0x1234abcd bits, 7", rot))


class ExactStrings(unittest.TestCase):
    def test_the_headline_is_the_key_exactly(self):
        t = {"score": "exact_head", "accept": ["admin status 26"], "kind": "strobf"}
        for r in ("admin status 26", '"admin status 26"', "`admin status 26`", "The string is: admin status 26", "It writes admin status 26.", "**admin status 26**",
                  "Plaintext: admin status 26", "‘admin status 26’", "“admin status 26”"):
            self.assertTrue(S(r, t), r)
        for r in ("admin status 27", "Admin status 26", "ADMIN STATUS 26", "adminstatus26", "admin status  26x", "admin status 26 and more", "-admin status 26",
                  ":admin status 26", "> admin status 26", "< admin status 26", "<> admin status 26", "x admin status 26", "key admin status 26 (or 27)", "admin status 26?",
                  "admin status 26...", "admin status 26 or admin status 27", "not admin status 26", "no admin status 26", "= admin status 26 =", "admin status 26\nadmin status 27"):
            self.assertFalse(S(r, t), r)

    def test_alphabets_are_case_and_space_significant(self):
        a = {"score": "exact_case", "accept": ["aBcD+/"], "kind": "encoding"}
        self.assertTrue(S("alphabet: `aBcD+/`", a) and S("aBcD+/", a) and S('"aBcD+/".', a))
        for r in ("abcd+/", "AbCd+/", "a B c D + /", "aBcD+/ aBcD+/", "aBcD+", "aBcD+/x", "aBcD/+"):
            self.assertFalse(S(r, a), r)


class PropertyMutations(unittest.TestCase):
    """2,000 random tasks per family: the canonical answer with one number moved, a name swapped, a second candidate added or a negation / limit /
    operator template put around it, on the canonical order and again on a base whose name groups are permuted. Not one mutated reply is accepted,
    under the evaluation scorer and the data scorer (families.py, all eleven families); every permutation of the name groups is accepted and every
    number moved relative to the names or swapped with another number is rejected. `python3 -m tests.rea_property 2000 31337 12` runs it with every separator (1.5 million replies, about 2 minutes); `... 100 31337 0` puts
    every template at every position."""

    def test_nothing_mutated_is_accepted(self):
        res = rp.run(2000, 31337, templates=6, separators=3)
        self.assertEqual(set(res), set(families.modules()))
        for fam, row in res.items():
            self.assertEqual(row["tasks"], 2000, fam)
            self.assertEqual(row["accepted"], [], fam)
            self.assertEqual(row["permuted_rejected"], [], fam)
            self.assertEqual(row["misordered_accepted"], [], fam)
            self.assertGreater(row["rows"], 20000, fam)
            self.assertTrue({"second", "token"} <= set(row["by_kind"]), fam)
        self.assertGreater(sum(r["rows"] for r in res.values()), 400000)
        for fam in ("cipher", "encoding", "antidebug"):            # the families with two name groups: permutations were tried, all accepted
            self.assertGreater(res[fam]["permuted"], 1000, fam)
        self.assertGreater(sum(r["misordered"] for r in res.values()), 2000)

    def test_every_template_at_every_position_on_a_sample(self):
        res = rp.run(60, 4242, templates=0, separators=0)
        for fam, row in res.items():
            self.assertEqual(row["accepted"], [], fam)
            self.assertEqual(row["permuted_rejected"], [], fam)
            self.assertEqual(row["misordered_accepted"], [], fam)
        self.assertGreater(sum(r["rows"] for r in res.values()), 150000)
        self.assertGreater(sum(r["permuted"] for r in res.values()), 500)

    def test_the_permutation_check_has_teeth(self):
        """The ordered grammar (every name group once, in the order of `accept`) rejects the permuted right replies the property test sends, and a
        grammar that frees the numbers too accepts the misordered ones: both checks can fail."""
        res = rp.run(30, 4242, templates=1, separators=1)
        self.assertEqual(sum(len(r["permuted_rejected"]) for r in res.values()), 0)
        orig = grammar.Spec.__init__

        def flat(self, t):
            orig(self, t)
            self.block = [0] * len(self.groups)

        grammar.Spec.__init__ = flat
        try:
            loose = rp.run(30, 4242, templates=1, separators=1)
        finally:
            grammar.Spec.__init__ = orig
        self.assertGreater(sum(len(r["misordered_accepted"]) for r in loose.values()), 0)


if __name__ == "__main__":
    unittest.main()

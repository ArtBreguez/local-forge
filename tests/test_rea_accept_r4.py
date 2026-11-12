"""Round 4, the ACCEPTANCE side of the data path (the teacher's reply is a filter now: a task enters the data only if its headline is accepted).

B1   legacy conversion: no bare hex digits for a decimal key, no all-digit text for a hex constant      (`strict_task`)
M1   labels bound to the quantity: context labels, quantity nouns, `hex`/`decimal`, `returns`, units     (`grammar.headline_ok`)
m2   a number asked in hex is not read as a decimal
M2   legacy hash aliases of THAT algorithm only
M4   exact strings: no normalisation, no noun in front of the key, a closed list of lead-in verbs
m1   legacy `skip`: no decimal numeral, a `chance`, so agreement applies
m3   a huge run of digits cannot crash the scorer;  m5  digit separators only in groups
"""
from __future__ import annotations

import sys
import unittest

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, grammar, scoring, sft  # noqa: E402

S = scoring.score_strict


def variant(fam: str, name: str, seed: int = 99, n: int = 30) -> dict:
    m = families.modules()[fam]
    return next(t for t in families.generate(fam, n * len(m.VARIANTS), seed) if t.variant == name).meta()


def legacy(kind: str, seed: int = 7, **want) -> dict:
    for t in families.generate("legacy", 400, seed):
        m = t.meta()
        if m["kind"] == kind and all(m["accept"][i] == v for i, v in want.items()):
            return m
    raise LookupError(kind)


def lcg(a: str, rounds: str) -> dict:
    return {"id": "lcg-7-0013", "kind": "lcg", "score": "all", "accept": [a, rounds],
            "question": "This is a linear congruential generator. Give the multiplier (decimal) and the number of rounds."}


class B1LegacyConversion(unittest.TestCase):
    def test_the_hex_digits_of_a_decimal_answer_are_another_number(self):
        t = lcg("1284865837", "39")
        for r in ("1284865837, 27", "1284865837, 0x27", "4c9d3d2d, 39"):
            self.assertEqual(S(r, t), r == "1284865837, 0x27", r)
        self.assertTrue(S("1284865837, 39", t) and S("0x4c957f2d, 39", t) and S("1284865837, 39 rounds", t))
        thr = {"kind": "threshold", "score": "all", "accept": ["801", "892"], "question": "What are the lower and upper bounds (inclusive) for which this returns 0?"}
        for r in ("321, 892", "801, 37c", "321 to 892", "321, 37c"):
            self.assertFalse(S(r, thr), r)
        for r in ("801, 892", "801 to 892", "0x321, 0x37c", "801, 0x37c"):
            self.assertTrue(S(r, thr), r)
        neg = {**thr, "accept": ["-402", "1411"]}
        self.assertTrue(S("-402, 1411", neg) and not S("-402, 583", neg))
        rot = {"kind": "rotate", "score": "all", "accept": ["3e7d1bfb", "25"], "question": "Give the XOR constant (hex) and the left-rotation amount."}
        self.assertFalse(S("0x3e7d1bfb, 19", rot))
        self.assertTrue(S("0x3e7d1bfb, 25", rot) and S("3e7d1bfb, 25", rot) and S("0x3e7d1bfb, 25 bits", rot))

    def test_the_digits_of_an_all_digit_hex_constant_are_not_a_decimal_spelling(self):
        for kind, acc, q in (("magic", ["30312932", "6"], "What 32-bit magic value must the header start with (hex), and what is the highest accepted version?"),
                             ("rotate", ["30312932", "7"], "Give the XOR constant (hex) and the left-rotation amount.")):
            t = {"kind": kind, "score": "all", "accept": acc, "question": q}
            n = acc[1]
            for r in (f"0x1ce89e4, {n}", f"30312932, {n}", f"{int('30312932')}, {n}"):
                self.assertFalse(S(r, t), r)
            self.assertTrue(S(f"0x30312932, {n}", t) and S(f"0x30312932 (808528178), {n}", t), kind)
        h = {"kind": "hash", "score": "all_groups", "accept": [["fnv"], ["30312932"]],
             "question": "Which well-known hash is this based on, and what constant is XORed at the end (hex)?"}
        for r in ("fnv, 0x1ce89e4", "fnv, 30312932", "fnv, 808528178"):
            self.assertFalse(S(r, h), r)
        self.assertTrue(S("fnv, 0x30312932", h))

    def test_every_legacy_decimal_group_rejects_its_own_bare_hex_digits_on_600_tasks(self):
        """The review's repro_B1 shapes over 600 tasks of two seeds: nothing wrong is accepted (it was 12-14% of the tasks), the canonical reply passes."""
        wrong = right = 0
        for seed in (7, 8):
            for t in families.generate("legacy", 600, seed):
                m = t.meta()
                if m["kind"] not in ("lcg", "rotate", "threshold", "magic", "hash"):
                    continue
                st = scoring.strict_task(m)
                self.assertTrue(S(scoring.canonical(st), m), m["id"])
                right += 1
                for gi, g in enumerate(st["accept"]):
                    base = (st.get("bases") or [None] * len(st["accept"]))[gi]
                    first = g[0]
                    v = grammar.num_value(first.lower())
                    if v is None or v < 10:
                        continue
                    others = [x[0] for x in st["accept"]]
                    for bad in {format(v, "x") if base == "dec" else str(v) if base == "hex" else None,
                                f"0x{int(format(v, 'x')):x}" if base == "hex" and format(v, "x").isdigit() else None} - {None}:
                        if bad == first or grammar.num_value(bad.lower()) == v and base == "dec":
                            continue
                        reply = ", ".join(bad if j == gi else o for j, o in enumerate(others))
                        wrong += S(reply, m)
        self.assertEqual(wrong, 0)
        self.assertGreater(right, 400)


class M1Labels(unittest.TestCase):
    def test_a_context_label_is_the_questions_own(self):
        xtea = variant("cipher", "xtea_dec")
        n = xtea["accept"][2][0]
        for r in (f"XTEA, decrypt, {n} rounds, 128-bit key", f"XTEA, decrypt, {n} rounds, 8-byte block", f"XTEA, decrypt, {n} iterations (a 128-bit key)",
                  f"XTEA, decrypt, {n} (32-bit words)", f"XTEA, decrypt, {n}, 128-bit"):
            self.assertTrue(S(r, xtea), r)
        for r in (f"XTEA, decrypt, {n} rounds, 128-bit block", f"XTEA decrypts {n} rounds with a 32-bit key", f"XTEA, decrypt, {n} rounds, 8-byte key",
                  f"XTEA, decrypt, {n}, 64-bit key", f"XTEA, decrypt, {n}, block 128-bit", f"XTEA, decrypt, {n}, 8-byte width", f"XTEA, decrypt, {n}, 128-bit size"):
            self.assertFalse(S(r, xtea), r)
        tea = variant("cipher", "tea_enc")
        self.assertFalse(S(f"tea, encrypt, {tea['accept'][2][0]}, 8-byte key", tea))
        chacha = variant("cipher", "chacha")
        c = chacha["accept"][1][0]
        self.assertTrue(S(f"ChaCha, {c} rounds (256-bit key, 64-byte block)", chacha))
        for r in (f"chacha, {c}, 64-byte key", f"chacha, {c}, 256-bit block", f"chacha, {c}, 32-bit key"):
            self.assertFalse(S(r, chacha), r)
        crc = variant("crc", "bitwise_poly")
        k = crc["accept"][0][0]
        w = 16 if "16-bit" in crc["question"] else 32
        self.assertTrue(S(f"{k}, {crc['accept'][1][0]}, {w}-bit crc", crc), "the question's own label")
        for r in (f"{k}, {crc['accept'][1][0]}, {w}-bit key", f"{k}, {crc['accept'][1][0]}, {w}-bit width"):
            self.assertFalse(S(r, crc), r)

    def test_a_quantity_noun_leads_in_only_when_the_question_uses_the_word(self):
        nxt = variant("structs", "linked_next")
        k = nxt["accept"][0][0]
        for r in (f"offset {k}", f"Offset: {k}", f"byte offset {k}", f"{k} bytes", f"The offset is {k}.", f"Answer:\n{k}"):
            self.assertTrue(S(r, nxt), r)
        for noun in ("size", "length", "count", "key", "alphabet", "poly", "string", "number", "field", "record", "register", "constant", "text", "secret", "input"):
            self.assertFalse(S(f"{noun} {k}", nxt), noun)
        for r in (f"The size of the key is:\n{k}", f"The input length is:\n{k}", f"Length of the string:\n{k}", f"The constant equals:\n{k}"):
            self.assertFalse(S(r, nxt), r)
            self.assertIsNone(scoring.sft_target(r, nxt), r)
        self.assertFalse(S(f"The value is:\n{k}", nxt), "`value` is not a word of this question")
        field = variant("structs", "field_read")
        self.assertTrue(S(f"The value is:\n{field['accept'][0][0]}", field), "`value` is a word of the field_read question")
        crc = variant("crc", "bitwise_init")
        k = crc["accept"][0][0]
        self.assertTrue(S(f"initial value {k}", crc) and S(f"init {k}", crc), "initial -> init is the same word")
        self.assertFalse(S(f"offset {k}", crc) or S(f"size {k}", crc))

    def test_hex_and_decimal_stand_beside_a_number_written_that_way(self):
        t = {"score": "tokens", "accept": [["0x12"]], "kind": "structs", "variant": "field_read",
             "question": "At which byte offset (hex, like 0x1c) of the structure is the value that proc returns?"}
        for r in ("0x12", "hex 0x12", "0x12 (hex)", "hex: 0x12", "hexadecimal 0x12", "0x12 (18)", "18 (0x12)", "0x12, hex"):
            self.assertTrue(S(r, t), r)
        for r in ("hex 18", "18 (hex)", "hexadecimal 18", "18", "12", "(18)", "It is 18", "hex 12", "18u", "hex, 18"):
            self.assertFalse(S(r, t), r)
        crc = {"score": "tokens", "accept": [["0xe3e6", "e3e6"], ["normal"]], "kind": "crc", "variant": "bitwise_poly",
               "question": "This function computes a 16-bit CRC. Give the polynomial constant exactly as it appears in the code (hex), and say whether it is 'reflected' or 'normal'."}
        self.assertTrue(S("0xe3e6, normal", crc) and S("hex e3e6, normal", crc) and S("hex 0xe3e6, normal", crc))
        for r in ("hex 58342, normal", "58342, normal", "0xe3e6 (decimal), normal", "hex normal 0xe3e6", "hex, 0xe3e6 normal x"):
            self.assertFalse(S(r, crc), r)
        dec = {"score": "tokens", "accept": [["ptrace"], ["111", "0x6f"]], "kind": "antidebug", "variant": "ptrace",
               "question": "Which anti-debugging technique does proc implement, and what does proc return when the call shows that a debugger is attached (decimal)?"}
        for r in ("ptrace, 111", "ptrace, decimal 111", "ptrace, 111 (decimal)", "ptrace, 0x6f", "ptrace, returns 111", "ptrace, 0x6f (111)"):
            self.assertTrue(S(r, dec), r)
        for r in ("ptrace, decimal 0x6f", "decimal ptrace, 0x6f", "ptrace, 0x6f (decimal)", "ptrace, hex 111", "ptrace, hex 0x6f"):
            self.assertFalse(S(r, dec), r)

    def test_a_single_digit_reads_the_same_in_both_bases(self):
        t = {"score": "tokens", "accept": [["0x7", "7"]], "kind": "structs", "variant": "field_read", "question": "At which byte offset (hex, like 0x1c) is it?"}
        self.assertTrue(S("7", t) and S("0x7", t) and S("0x07", t) and S("07", t))
        zero = {"score": "tokens", "accept": [["0x0", "0"]], "kind": "crc", "variant": "bitwise_xorout", "question": "What value is XORed (hex)? Answer 0 if nothing is XORed."}
        self.assertTrue(S("0", zero) and S("0x0", zero) and S("0x00", zero))
        self.assertFalse(S("0x10", zero) or S("10", zero))

    def test_returns_only_where_the_question_asks_for_a_return_value_and_labels_that_number(self):
        ptrace = variant("antidebug", "ptrace")
        n = ptrace["accept"][1][0]
        for r in (f"ptrace, returns {n}", f"ptrace: it returns {n}", f"ptrace check, return value {n}", f"ptrace, returns: {n}"):
            self.assertTrue(S(r, ptrace), r)
        for r in (f"returns ptrace, {n}", f"ptrace returns, {n} bytes", f"ptrace, {n} returns"):
            self.assertFalse(S(r, ptrace), r)
        states = variant("fsm", "table_states")
        k = states["accept"][0][0]
        for r in (f"returns {k}", f"return value {k}", f"It returns {k}", f"returned {k}"):
            self.assertFalse(S(r, states), r)
        self.assertTrue(S(f"{k} states", states) and S(f"{k}", states))
        field = variant("structs", "field_read")
        self.assertFalse(S(f"returns {field['accept'][0][0]}", field), "proc returns another value there: the answer is an offset")

    def test_a_unit_is_a_word_of_the_question_and_main_loop_is_a_phrase(self):
        xtea = variant("cipher", "xtea_dec")
        n = xtea["accept"][2][0]
        for r in (f"XTEA, decrypt, {n} iterations of the main loop", f"XTEA, decrypt, {n} rounds", f"XTEA, decrypt, {n} iterations"):
            self.assertTrue(S(r, xtea), r)
        for r in (f"XTEA decrypt {n} main", f"XTEA decrypt {n} times", f"XTEA decrypt {n} loops", f"XTEA decrypt {n} cycles", f"XTEA decrypt {n} main loops"):
            self.assertFalse(S(r, xtea), r)

    def test_a_name_noun_stands_beside_a_name(self):
        xtea = variant("cipher", "xtea_dec")
        n = xtea["accept"][2][0]
        for r in (f"XTEA cipher, decrypt, {n}", f"XTEA block cipher, decrypt, {n}", f"It is the XTEA block cipher, decrypt, {n}", f"cipher XTEA, decrypt, {n}"):
            self.assertTrue(S(r, xtea), r)
        for r in (f"XTEA, decrypt, {n}, a block cipher", f"block {n}, XTEA, decrypt", f"XTEA, decrypt, block {n}"):
            self.assertFalse(S(r, xtea), r)
        nxt = variant("structs", "linked_next")
        k = nxt["accept"][0][0]
        self.assertFalse(S(f"block {k}", nxt) or S(f"hash {k}", nxt) or S(f"order {k}", nxt) or S(f"name {k}", nxt))


class M2LegacyHashNames(unittest.TestCase):
    def test_the_aliases_are_the_algorithms_own_names(self):
        fnv = next(t.meta() for t in families.generate("legacy", 600, 7) if t.meta()["kind"] == "hash" and t.meta()["accept"][0][0] == "fnv")
        h = fnv["accept"][1][0]
        for r in ("fnv, 0x%s" % h, f"FNV-1a, 0x{h}", f"fnv1a, 0x{h}", f"FNV-32, 0x{h}", f"fnv32, 0x{h}"):
            self.assertTrue(S(r, fnv), r)
        for r in (f"FNV-1, 0x{h}", f"FNV1, 0x{h}", f"fnv-1, 0x{h}"):
            self.assertFalse(S(r, fnv), r)
        for stem, good, bad in (("murmur", ("murmur", "MurmurHash", "murmur2", "MurmurHash2", "murmur-ish", "Murmur-style"), ("MurmurHash3", "murmur3", "murmur64")),     # "based on": a reply may say MurmurHash2; the stored name is Murmur-style (round 5, B3)
                                ("djb2", ("djb2", "djb", "Bernstein", "djb-2"), ("djb2a", "djb2x", "djb1"))):
            t = next(t.meta() for t in families.generate("legacy", 800, 7) if t.meta()["kind"] == "hash" and t.meta()["accept"][0][0] == stem)
            h = t["accept"][1][0]
            for name in good:
                self.assertTrue(S(f"{name}, 0x{h}", t), name)
            for name in bad:
                self.assertFalse(S(f"{name}, 0x{h}", t), name)


class M4ExactStrings(unittest.TestCase):
    key = {"id": "strobf-99-0004", "kind": "strobf", "score": "exact_head", "accept": ["admin status 26"], "question": "What string does proc write into its argument? Give it as plain ASCII text."}

    def test_the_reply_is_not_normalised(self):
        for r in ("admin  status  26", "admin\tstatus\t26", "admin\u2009status\u200926", "\uff41\uff44\uff4d\uff49\uff4e status \uff12\uff16", "admin\u00a0status\u00a026", "admin status\u200b 26", "Admin status 26"):
            self.assertFalse(S(r, self.key), repr(r))
        for r in ("admin status 26", "  admin status 26  ", "\u00a0admin status 26\u00a0", '"admin status 26"', "\u2018admin status 26\u2019", "`admin status 26`", "admin status 26."):
            self.assertTrue(S(r, self.key), repr(r))
        alpha = {"score": "exact_case", "accept": ["CAovnsJWTx8ft24"], "kind": "encoding", "question": "Give the 15-character alphabet."}
        self.assertFalse(S("CAovnsJWT\\x8ft24", alpha))
        self.assertTrue(S("CAovnsJWTx8ft24", alpha))

    def test_a_noun_in_front_of_the_key_is_part_of_the_string(self):
        for r in ("secret admin status 26", "path admin status 26", "output admin status 26", "in admin status 26", "the admin status 26", "string admin status 26",
                  "key admin status 26", "result admin status 26", "returns admin status 26", "plain admin status 26", "text admin status 26"):
            self.assertFalse(S(r, self.key), r)
        for r in ("The string is admin status 26", "The string is: admin status 26", "It writes admin status 26.", "Plaintext: admin status 26", "String: admin status 26",
                  "It reads admin status 26", "It prints admin status 26", "It decodes to admin status 26", "result = admin status 26", "= admin status 26", "The plaintext is \"admin status 26\""):
            self.assertEqual(S(r, self.key), r != "= admin status 26", r)
        for r in ("writesadmin status 26", "isadmin status 26", "The string is-admin status 26"):
            self.assertFalse(S(r, self.key), r)

    def test_a_lead_in_line_is_function_words_and_the_questions_words(self):
        for r in ("Answer:\nadmin status 26", "Here is the string:\nadmin status 26", "Plaintext:\nadmin status 26", "The string:\nadmin status 26\nIt is XORed first."):
            self.assertTrue(S(r, self.key), r)
        for r in ("The size:\nadmin status 26", "Offset:\nadmin status 26", "Hex:\nadmin status 26"):
            self.assertFalse(S(r, self.key), r)
        nxt = variant("structs", "linked_next")
        k = nxt["accept"][0][0]
        self.assertTrue(S(f"Offset:\n{k}", nxt) and S(f"Answer:\n{k}", nxt))
        self.assertFalse(S(f"Size:\n{k}", nxt) or S(f"Key length:\n{k}", nxt))


class M1LegacySkip(unittest.TestCase):
    def test_the_value_is_hex_and_no_decimal_numeral_is_accepted(self):
        for b in (0x20, 0x0a, 0x00, 0x2c, 0x3b, 0x09, 0xff):
            t = {"id": "skip-7-0001", "kind": "skip", "score": "any", "accept": [f"0x{b:02x}", f"{b:02x}", str(b)],
                 "question": "The checksum skips bytes with one specific value. Which value (hex)?"}
            self.assertTrue(S(f"0x{b:02x}", t) and S(f"The value is 0x{b:02x}", t), b)
            if b >= 10:
                self.assertFalse(S(str(b), t), f"decimal {b} is {int(str(b), 16):#x} read as hex")
                x = f"{b:02x}"
                self.assertEqual(S(x, t), scoring._bare_hex_ok(x) or (x.isdigit() and int(x, 10) == int(x, 16)), f"bare {x}: a letter and a digit, or digits that read the same in both bases")
            if f"{b:02x}".isdigit() and b >= 10:
                self.assertFalse(S(f"0x{int(f'{b:02x}'):x}", t), "the digits of an all-digit hex byte read as decimal")
        t = {"kind": "skip", "score": "any", "accept": ["0x20", "20", "32"], "question": "The checksum skips bytes with one specific value. Which value (hex)?"}
        for r in ("0x14", "20", "32", "0x32"):
            self.assertFalse(S(r, t), r)
        self.assertTrue(S("0x20", t))

    def test_skip_carries_a_chance_so_agreement_applies(self):
        t = {"id": "skip-7-0001", "kind": "skip", "score": "any", "accept": ["0x20", "20", "32"], "question": "Which value (hex)?"}
        self.assertAlmostEqual(scoring.strict_task(t)["chance"], 0.14)
        self.assertTrue(sft.needs_agreement(t))
        self.assertEqual(sft.select([("0x20", True)], t)[1], "agreement", "a constant guess among 7 values is not one lucky sample")
        self.assertEqual(sft.select([("0x20", True), ("The value is 0x20.", True)], t)[1], "kept")
        other = legacy("lcg")
        self.assertFalse(sft.needs_agreement(other))


class M3M5Robustness(unittest.TestCase):
    def test_a_run_of_digits_longer_than_any_answer_cannot_crash_the_scorer(self):
        t = variant("bounds", "range_chain")
        for junk in ("1751" * 1200, "9" * 5000, "0x" + "f" * 5000, "7" * 5000 + "-bit", "-" + "1" * 4400, "1" * 4400 + "-bit key"):
            for reply in (junk, f"{t['accept'][0][0]}, {junk}", f"{junk}\nIt is read in the loop of the function."):
                self.assertFalse(S(reply, t))
                self.assertFalse(scoring.score(reply, t) or scoring.score_any_line(reply, t))
                self.assertIsNone(scoring.sft_target(reply, t))
        self.assertIsNone(grammar.num_value("1" * 5000))
        self.assertEqual(grammar.num_value("1" * 40), int("1" * 40))

    def test_digit_separators_only_in_the_groups_c23_writes(self):
        t = {"score": "tokens", "accept": [["40", "0x28"]], "kind": "structs", "question": "How many bytes?"}
        for r in ("4'0", "4_0", "0x2'8", "0x2_8", "4_0_0", "1'00", "4'000"):
            self.assertFalse(S(r, t), r)
        big = {"score": "tokens", "accept": [["1540000000", "0x5bca8900"]], "kind": "antidebug", "variant": "timing_ns", "question": "above how many nanoseconds"}
        for r in ("1'540'000'000", "1_540_000_000", "1,540,000,000", "0x5bca'8900", "0x5bca_8900"):
            self.assertTrue(S(r, big), r)
        for r in ("15'40'000'000", "1_54_0000000", "1540'000000"):
            self.assertFalse(S(r, big), r)


if __name__ == "__main__":
    unittest.main()

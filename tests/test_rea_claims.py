"""Round 5: the claims table. Every sentence of every truth template makes claims; every claim has a checker (tests/rea_claims.py) that verifies it against
the task's C source and its compiled behaviour; the checkers run over at least REA_CLAIMS_N (default 200) generated tasks of every variant at each of two seeds
(a training seed and the held-out seed), every code form of the variant included.

  * TABLE            template key -> for each sentence of the template, the names of the claims it makes. A template without an entry, a sentence without a claim, a
                     claim without a checker, a checker that no sentence uses: each fails `TableIsComplete`.
  * LINT             a sentence never uses a word of the code-structure list (loop, array, switch, table, comparison, stack, copied, shifts, ...): `NoStructureWords`.
  * CHECKERS         `ClaimsHold`: for every family, every task of every variant, every claim of every sentence that could be rendered for it (for the families whose
                     tasks ask one of several questions - encoding: the identity or the alphabet, pktsum: endianness, start or length byte - the claims of the other
                     questions are checked too, with the facts the template would state for that question).
  * TEETH            `Teeth`: a false fact (one more iteration, another initial CRC value, the wrong byte order, the name MurmurHash2 over the murmur-style loop, a window
                     one wider) makes the checker fail, so a checker that always says yes cannot pass.
  * REFERENCES       `References`: the Python references the checkers use are anchored (MurmurHash2 against Austin Appleby's C, FNV-1a against its published values, the delta
                     parameter of TEA/XTEA against f_cipher's fixed one).

    REA_CLAIMS_N=8 python3 -m unittest tests.test_rea_claims      a quick run (the full run compiles about 30,000 programs)
    REA_CLAIMS_WORKERS=1                                           one process instead of two
"""
from __future__ import annotations

import collections
import concurrent.futures
import multiprocessing
import os
import random
import re
import shutil
import sys
import unittest

from .helpers import ROOT
from . import rea_claims as RC

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, f_cipher, f_pktsum, families, targets  # noqa: E402

N = int(os.environ.get("REA_CLAIMS_N", "200"))
WORKERS = int(os.environ.get("REA_CLAIMS_WORKERS", "2"))
SEEDS = (4241, core.HELDOUT_SEED)
GCC = bool(shutil.which("gcc"))


def same(*names, n=3):
    return [tuple(names)] * n


# key -> one tuple of claim names per sentence of the template (the order of targets.SPECS[key].sentences)
TABLE: dict = {
    "structs/array_stride": same("stride_records"),
    "structs/field_read": same("field_read_offset"),
    "structs/init_magic": [("init_magic_offset",), ("init_magic_offset", "init_other_fields"), ("init_magic_offset", "init_other_fields")],
    "structs/alloc_size": same("alloc_structure_size"),
    "structs/nested_offset": same("nested_field_offset"),
    "structs/linked_next": same("linked_pointer_offset"),
    "fsm/table_input": [("fsm_input_final",), ("fsm_input_final",), ("fsm_input_final", "fsm_input_shortest")],
    "fsm/switch_input": [("fsm_input_final",), ("fsm_input_final",), ("fsm_input_final", "fsm_input_shortest")],
    "fsm/table_states": [("fsm_states_count",), ("fsm_states_count",), ("fsm_states_reachable", "fsm_states_count")],
    "fsm/switch_states": [("fsm_states_count",), ("fsm_states_count",), ("fsm_states_reachable", "fsm_states_count")],
    "fsm/opcode_value": [("opcode_returns_expr",), ("opcode_returns_expr", "opcode_unique"), ("opcode_returns_expr", "opcode_unique")],
    "fsm/opcode_count": same("opcode_count_handled", "opcode_default"),
    "crc/bitwise_poly": same("crc_poly_form"), "crc/table_poly": same("crc_poly_form"),
    "crc/bitwise_init": same("crc_init_value"), "crc/table_init": same("crc_init_value"),
    "crc/bitwise_xorout": same("crc_xorout_value"), "crc/table_xorout": same("crc_xorout_value"),
    "cipher/tea_enc": same("tea_reference"), "cipher/tea_dec": same("tea_reference"),
    "cipher/xtea_enc": same("tea_reference"), "cipher/xtea_dec": same("tea_reference"),
    "cipher/rc4": same("rc4_reference"),
    "cipher/chacha": same("stream_reference"), "cipher/salsa": same("stream_reference"),
    "strobf/stack_string": same("writes_exact_string"), "strobf/xor_const": same("writes_exact_string"), "strobf/xor_rolling": same("writes_exact_string"),
    "strobf/alt_xor": same("writes_exact_string"), "strobf/sub_const": same("writes_exact_string"), "strobf/reverse_xor": same("writes_exact_string"),
    "strobf/rot_n": same("writes_exact_string"),
    "bounds/range_chain": same("returns_one_exactly_in_range"),
    "bounds/window": same("returns_one_for_window"),
    "bounds/index_limit": same("index_limit_range"),
    "bounds/clamp": same("clamp_interval"),
    "bounds/loop_iterations": same("returns_loop_value"),
    "bounds/char_class": same("char_class_count"),
    "antidebug/ptrace": same("ptrace_request"), "antidebug/ptrace_raw": same("ptrace_request"),
    "antidebug/tracerpid": same("tracerpid_status"),
    "antidebug/env": same("env_variable_read"),
    "antidebug/timing_ns": same("timing_threshold"), "antidebug/timing_us": same("timing_threshold"),
    "multi/eval_arith": same("returns_for_argument"), "multi/chain": same("returns_for_argument"),
    "multi/key_helper": same("xors_key_byte"),
    "multi/decode_secret": same("accepts_only_the_secret"),
    "multi/limit_helper": same("limit_for_mode"),
    "xor-key": same("xor_key_repeats"),
    "password": same("password_equal_only"),
    "keycount": same("keycount_lines"),
    "alphabet": same("alphabet_position"),
    "magic": same("magic_header"),
    "lcg": same("lcg_reference"),
    "hash": [("hash_start_value", "hash_final_xor"), ("hash_name_is_the_algorithm", "hash_final_xor"), ("hash_final_xor",)],
    "skip": same("skip_one_value"),
    "rotate": same("rotate_reference"),
    "threshold": same("threshold_values"),
}
for _b in ("base64", "base32", "base58"):
    for _d in ("encode", "decode"):
        TABLE[f"encoding/{_b}_{_d}"] = same("encoding_reference")
        TABLE[f"encoding/{_b}_{_d}#alphabet"] = same("alphabet_semantics")
for _v in ("sum8", "xor8", "sum16_le", "sum16_be", "fletcher16", "crc16", "len_field"):
    TABLE[f"pktsum/{_v}"] = same("pkt_endian_behaviour") + same("pkt_range_start") + same("pkt_length_byte")      # ends + starts + lens


# Words that describe code structure or compiler output (owner decision, round 5). A sentence uses none of them.
STRUCTURE = re.compile(
    r"\b(loops?|looping|looped|iterates?|byte by byte|on the stack|stacks?|copied|copies|copy|arrays?|switch(es)?|branch(es)?|comparisons?|compar(e|es|ed|ing)|"
    r"tables?|look(s|ed|ing)? ?ups?|looks? up|shifts?|shifted|shifting|unsigned|signed|memcpy|strcmp|strncmp|strchr|helpers?|calls?|divides?|divided|"
    r"again and again|registers?|local|inlined?|immediates?|pseudocode|decompil\w+|opcodes? handler|handlers?|cases?)\b", re.I)


def applicable(key: str, f: dict) -> range:
    """The sentences of template `key` that can be rendered for a task with facts `f` (pktsum: the set of the question that was asked)."""
    if key.startswith("pktsum/"):
        lo, hi = targets.PKT_SETS[key.split("/")[1]][f["ask"]]
        return range(lo, hi)
    return range(len(targets.SPECS[key].sentences))


# ---- the other questions a task could have asked (the claims do not depend on which one it did) ---------------------------------------

def other_questions(task) -> list:
    """Records of the same task asking its other questions, for the families where one program has several: encoding (identity / alphabet), pktsum."""
    m, out = task.meta(), []
    if task.kind == "encoding":
        base, direction = task.variant.split("_")
        n = RC.f_encoding.RADIX[base]
        if m["score"] == "exact_case":
            out.append({**m, "score": "tokens", "question": "Does this function encode or decode, and which encoding does it implement (base32, base58 or base64)?",
                        "accept": [[direction], [base]]})
        else:
            out.append({**m, "score": "exact_case", "question": f"Give the {n}-character alphabet this function uses, in index order (the first character stands for the value 0).",
                        "accept": [RC._enc_alphabet(core.strip_decoys(task.body))]})
    if task.kind == "pktsum":
        p = task.params
        asks = (["len_offset", "len_start"] if task.variant == "len_field" else ["start"] + (["endian"] if p["W"] == 2 else []))
        for ask in asks:
            if ask == p["ask"]:
                continue
            if ask == "endian":
                acc = {"score": "tokens", "accept": [[p["endian"]]], "reject": ["big" if p["endian"] == "little" else "little"]}
            else:
                acc = {"score": "tokens", "accept": [core.dec({"start": p["S"], "len_offset": p["L"], "len_start": p["H"]}[ask])]}
            out.append({**m, "question": f_pktsum.QUESTIONS[ask], **acc})
    return out


def exercise(task):
    """[(key, claim, None | why)] for every claim of every renderable sentence of every question of this task; one compile per probe (shared cache)."""
    cache: dict = {}
    res = []
    for m in [task.meta()] + other_questions(task):
        r = targets.render(m)
        key, f = r["key"], r["facts"]
        names = sorted({n for i in applicable(key, f) for n in TABLE[key][i]})
        ctx = RC.Ctx(task, f, key)
        ctx.cache = cache
        for name in names:
            try:
                RC.CLAIMS[name](ctx)
                res.append((key, name, None))
            except RC.ClaimFailed as e:
                res.append((key, name, f"{task.id}: {e}"[:300]))
            except Exception as e:                       # a crash of a checker is a failure of the claim, with the reason
                res.append((key, name, f"{task.id}: checker crashed: {type(e).__name__}: {e}"[:300]))
    return res


def _worker(task):
    return exercise(task)


def run_family(fam: str, n_per_variant: int = N, seeds=SEEDS) -> tuple:
    """(counts {(key, seed): tasks}, failures {(key, claim): [why]}, forms)"""
    mod = families.modules()[fam]
    counts, failures = collections.Counter(), collections.defaultdict(list)
    forms = collections.Counter()
    ctx = multiprocessing.get_context("fork")
    for seed in seeds:
        tasks = families.generate(fam, n_per_variant * len(mod.VARIANTS), seed)
        for t in tasks:
            forms.update(form_keys(t))
        if WORKERS > 1:
            with concurrent.futures.ProcessPoolExecutor(WORKERS, mp_context=ctx) as ex:
                results = list(ex.map(_worker, tasks, chunksize=8))
        else:
            results = [exercise(t) for t in tasks]
        for t, res in zip(tasks, results):
            for key in {k for k, _c, _w in res}:
                counts[(key, seed)] += 1
            for key, name, why in res:
                if why:
                    failures[(key, name)].append(why)
    return counts, failures, forms


# ---- code forms ---------------------------------------------------------------------------------------------------------------------

FORM_PARAMS = ("form", "style", "shape", "pad", "table", "custom", "ask", "raw", "refl", "width", "kind", "asked", "direction", "unit")


def form_keys(t) -> list:
    """The code forms a task is one of: the generator's own form parameters, plus the forms only the C source shows."""
    out = []
    for name in FORM_PARAMS:
        v = t.params.get(name)
        if v is not None and not isinstance(v, (list, dict, bytes)):
            out.append((f"{t.kind}/{t.variant}", name, v))
    b = t.body
    if t.kind == "structs" and t.variant == "alloc_size":
        out.append(("structs/alloc_size", "allocator", "malloc" if "malloc(" in b else ("calloc1" if "calloc(1," in b else "calloc_n")))
    if t.kind == "encoding":
        out.append((f"encoding/{t.variant}", "lookup", "reverse table" if "rev[" in b else ("strchr" if "strchr" in b else "index")))
    if t.kind == "cipher" and t.variant in ("chacha", "salsa"):
        out.append((f"cipher/{t.variant}", "sigma", "inline" if "sigma[" not in b else "static"))
    if t.kind == "fsm" and t.variant.endswith("input"):
        out.append((f"fsm/{t.variant}", "machine", "table" if "static const uint8_t T[" in b else "switch"))
    return out


# domains that every value of must appear, over the two seeds together, at least MIN_FORM times at the default N (scaled down with N)
FORM_DOMAINS = {("antidebug/env", "form"): {0, 1, 2, 3}, ("antidebug/ptrace", "form"): {0, 1, 2, 3}, ("antidebug/ptrace_raw", "form"): {0, 1, 2, 3},
                ("antidebug/tracerpid", "form"): {0, 1, 2}, ("antidebug/timing_ns", "form"): {0, 1, 2}, ("antidebug/timing_us", "form"): {0, 1, 2},
                ("bounds/index_limit", "form"): set(range(6)), ("bounds/clamp", "form"): {0, 1, 2},
                ("bounds/loop_iterations", "form"): {"up_lt", "up_le", "down_gt", "down_ge"},
                ("cipher/chacha", "style"): {0, 1, 2}, ("cipher/salsa", "style"): {0, 1, 2},
                ("cipher/chacha", "sigma"): {"inline", "static"}, ("cipher/salsa", "sigma"): {"inline", "static"},
                ("structs/alloc_size", "allocator"): {"malloc", "calloc1", "calloc_n"},
                ("fsm/table_input", "machine"): {"table"}, ("fsm/switch_input", "machine"): {"switch"}}
for _b in ("base64", "base32", "base58"):
    for _d in ("encode", "decode"):
        FORM_DOMAINS[(f"encoding/{_b}_{_d}", "pad")] = {True, False}
        FORM_DOMAINS[(f"encoding/{_b}_{_d}", "lookup")] = {"reverse table", "strchr"} if _d == "decode" else {"index"}
for _v in ("bitwise_poly", "bitwise_init", "bitwise_xorout", "table_poly", "table_init", "table_xorout"):
    FORM_DOMAINS[(f"crc/{_v}", "width")] = {16, 32}
    FORM_DOMAINS[(f"crc/{_v}", "refl")] = {True, False}
for _v in ("sum16_le", "sum16_be"):
    FORM_DOMAINS[(f"pktsum/{_v}", "ask")] = {"start", "endian"}
FORM_DOMAINS[("pktsum/len_field", "kind")] = {"sum8", "xor8"}
FORM_DOMAINS[("pktsum/len_field", "ask")] = {"len_offset", "len_start"}


class TableIsComplete(unittest.TestCase):
    def test_every_sentence_of_every_template_has_claims_and_every_claim_a_checker(self):
        missing = [k for k in targets.SPECS if k not in TABLE]
        self.assertEqual(missing, [], "a template without an entry in the claims table")
        self.assertEqual([k for k in TABLE if k not in targets.SPECS], [], "an entry for a template that does not exist")
        for key, spec in targets.SPECS.items():
            self.assertEqual(len(TABLE[key]), len(spec.sentences), f"{key}: one tuple of claims per sentence")
            for i, names in enumerate(TABLE[key]):
                self.assertTrue(names, f"{key} sentence {i} makes no claim: {spec.sentences[i]}")
                for name in names:
                    self.assertIn(name, RC.CLAIMS, f"{key} sentence {i}: no checker for the claim {name}")
        used = {name for names in TABLE.values() for tup in names for name in tup}
        self.assertEqual(set(RC.CLAIMS) - used, set(), "a checker that no sentence uses")
        self.assertEqual(len(targets.SPECS), 78)

    def test_the_summary_numbers(self):
        sentences = sum(len(s.sentences) for s in targets.SPECS.values())
        claims = sum(len(t) for names in TABLE.values() for t in names)
        self.assertGreaterEqual(sentences, 234)
        self.assertGreaterEqual(claims, sentences)


class NoStructureWords(unittest.TestCase):
    def test_no_sentence_describes_code_structure(self):
        bad = [(k, i, m.group(0), s) for k, spec in targets.SPECS.items() for i, s in enumerate(spec.sentences) for m in [STRUCTURE.search(s)] if m]
        self.assertEqual(bad, [])

    def test_the_lint_has_teeth(self):
        for s in ("The string is built byte by byte on the stack.", "A loop shifts the register right.", "A single unsigned comparison accepts it.",
                  "The encoder divides the input again and again.", "It is copied into a local array.", "The table lookup works on a register.", "It calls a helper."):
            self.assertTrue(STRUCTURE.search(s), s)

    def test_a_checksum_that_is_an_xor_is_never_called_a_sum(self):
        """B2: `pktsum/len_field` said 'how many bytes are summed' over an XOR checksum. No pktsum sentence uses the word sum, summed or sums."""
        word = re.compile(r"\bsum(s|med|ming)?\b", re.I)
        for key, spec in targets.SPECS.items():
            if key.startswith("pktsum/"):
                self.assertEqual([s for s in spec.sentences if word.search(s)], [], key)
        tasks = [t for t in families.generate("pktsum", 7 * 60, 99) if t.variant in ("xor8", "len_field")]
        n = 0
        for t in tasks:
            if t.variant == "len_field" and t.params["kind"] != "xor8":
                continue
            self.assertIsNone(word.search(targets.render(t.meta())["text"]), t.id)
            n += 1
        self.assertGreater(n, 40)


@unittest.skipUnless(GCC, "gcc is needed")
class References(unittest.TestCase):
    def test_murmurhash2_port_equals_appleby_s_c_and_the_legacy_loop_is_not_it(self):
        c_src = """#include <stdio.h>
#include <stdint.h>
#include <string.h>
static uint32_t MurmurHash2(const void *key, int len, uint32_t seed) {
    const uint32_t m = 0x5bd1e995; const int r = 24;
    uint32_t h = seed ^ (uint32_t)len;
    const unsigned char *data = (const unsigned char *)key;
    while (len >= 4) { uint32_t k; memcpy(&k, data, 4); k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; data += 4; len -= 4; }
    switch (len) { case 3: h ^= data[2] << 16; /* fallthrough */ case 2: h ^= data[1] << 8; /* fallthrough */ case 1: h ^= data[0]; h *= m; }
    h ^= h >> 13; h *= m; h ^= h >> 15;
    return h;
}
int main(void) {
    const char *w[] = {"", "a", "ab", "abc", "abcd", "hello world", "The quick brown fox jumps"};
    for (int i = 0; i < 7; i++) printf("%u\\n", (unsigned)MurmurHash2(w[i], (int)strlen(w[i]), 0x9747b28cu));
    return 0;
}
"""
        got = RC.ints(RC.gcc_run(c_src))
        words = [b"", b"a", b"ab", b"abc", b"abcd", b"hello world", b"The quick brown fox jumps"]
        self.assertEqual(got, [RC.murmur2(w, 0x9747b28c) for w in words])
        self.assertNotEqual([RC.h_murmur_style(w) for w in words][1:], got[1:], "the legacy per-byte loop is not MurmurHash2")
        self.assertEqual(RC.h_murmur_style(b""), 0x9747b28c, "it starts from the seed, with no length mixed in")

    def test_published_values_of_the_other_hashes(self):
        self.assertEqual((RC.h_fnv(b""), RC.h_fnv(b"a"), RC.h_fnv(b"foobar")), (0x811c9dc5, 0xe40c292c, 0xbf9cf968))
        self.assertEqual(RC.h_djb2(b"a"), 5381 * 33 + 97)
        self.assertEqual(RC.h_sdbm(b"a"), 97)
        self.assertEqual(RC.h_sdbm(b"ab"), (98 + (97 << 6) + (97 << 16) - 97) & RC.M32)

    def test_the_tea_references_with_a_delta_are_f_cipher_s_for_the_standard_delta(self):
        rng = random.Random(5)
        for _ in range(20):
            k = [rng.getrandbits(32) for _ in range(4)]
            v = (rng.getrandbits(32), rng.getrandbits(32))
            r = rng.choice([8, 16, 32, 64])
            for dec in (False, True):
                self.assertEqual(RC.tea_ref("TEA", v, k, r, RC.targets.TEA_DELTA, dec), f_cipher.tea(v, k, r, dec))
                self.assertEqual(RC.tea_ref("XTEA", v, k, r, RC.targets.TEA_DELTA, dec), f_cipher.xtea(v, k, r, dec))
            self.assertNotEqual(RC.tea_ref("TEA", v, k, r, 0x9E3779B8, False), f_cipher.tea(v, k, r, False), "another delta is another cipher")


@unittest.skipUnless(GCC, "gcc is needed")
class Teeth(unittest.TestCase):
    """A false fact must fail its checker (and the true one must pass), or the checker proves nothing."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = {}
        for fam in ("cipher", "crc", "pktsum", "bounds", "legacy", "strobf", "structs", "encoding", "antidebug", "fsm", "multi"):
            cls.tasks[fam] = families.generate(fam, 12 * len(families.modules()[fam].VARIANTS), 77)

    def task(self, fam, variant, pred=lambda t: True):
        return next(t for t in self.tasks[fam] if (t.variant == variant or t.kind == variant) and pred(t))

    def facts(self, t, **over):
        r = targets.render(t.meta())
        return r["key"], {**r["facts"], **over}

    def holds(self, claim, t, **over):
        key, f = self.facts(t, **over)
        try:
            RC.CLAIMS[claim](RC.Ctx(t, f, key))
            return True
        except RC.ClaimFailed:
            return False

    def check(self, claim, t, **over):
        self.assertTrue(self.holds(claim, t), f"{claim}: the true fact fails")
        for k, v in over.items():
            self.assertFalse(self.holds(claim, t, **{k: v(self.facts(t)[1][k]) if callable(v) else v}), f"{claim}: a false {k} passes")

    def test_cipher_and_crc(self):
        self.check("tea_reference", self.task("cipher", "xtea_dec"), rounds=lambda n: n + 1, delta=0x9E3779B8, dir=lambda d: "encrypt")
        self.check("tea_reference", self.task("cipher", "tea_enc"), alg="XTEA")
        self.check("crc_init_value", self.task("crc", "bitwise_init"), x=lambda x: x ^ 1)
        self.check("crc_xorout_value", self.task("crc", "table_xorout"), x=lambda x: x ^ 0x10)
        self.check("crc_poly_form", self.task("crc", "bitwise_poly"), lit=lambda x: x ^ 2, refl=lambda r: not r)
        self.check("stream_reference", self.task("cipher", "chacha"), rounds=lambda n: n + 2)

    def test_pktsum_and_bounds(self):
        t = self.task("pktsum", "sum16_le", lambda t: t.params["ask"] == "endian")
        self.check("pkt_endian_behaviour", t, e="big", first="high")
        t = self.task("pktsum", "sum8", lambda t: t.params["S"] > 1)
        self.check("pkt_range_start", t, n=lambda n: n - 1)
        self.assertFalse(self.holds("pkt_range_start", t, n=t.params["S"] + 1))
        t = self.task("pktsum", "len_field", lambda t: t.params["ask"] == "len_offset")
        self.check("pkt_length_byte", t, n=lambda n: n + 1)
        t = self.task("bounds", "window")
        self.check("returns_one_for_window", t, hi=lambda h: h + 1, w=lambda w: w + 1)
        t = self.task("bounds", "range_chain")
        self.check("returns_one_exactly_in_range", t, lo=lambda x: x - 1)
        t = self.task("bounds", "clamp")
        self.check("clamp_interval", t, hi=lambda h: h + 1)

    def test_legacy_and_strobf(self):
        t = self.task("legacy", "hash", lambda t: t.truth["accept"][0] == ["murmur"])
        self.assertTrue(self.holds("hash_name_is_the_algorithm", t))
        self.assertFalse(self.holds("hash_name_is_the_algorithm", t, name="FNV-1a"), "the murmur loop is not FNV-1a")
        self.check("hash_final_xor", t, c=lambda c: c ^ 1)
        self.check("hash_start_value", t, init=0x9747b28d)
        self.check("magic_header", self.task("legacy", "magic"), v=lambda v: v + 1)
        self.check("skip_one_value", self.task("legacy", "skip"), b=lambda b: b ^ 1)
        self.check("lcg_reference", self.task("legacy", "lcg"), r=lambda r: r + 1)
        t = self.task("strobf", "xor_const")
        self.check("writes_exact_string", t, s=t.params["plain"][:-1] + "~", len=lambda n: n + 1)

    def test_structs_encoding_antidebug_fsm_multi(self):
        self.check("stride_records", self.task("structs", "array_stride"), n=lambda n: n + 8)
        self.check("field_read_offset", self.task("structs", "field_read"), off=lambda o: o + 1)
        self.check("linked_pointer_offset", self.task("structs", "linked_next"), off=lambda o: o + 8)
        self.check("alloc_structure_size", self.task("structs", "alloc_size"), n=lambda n: n + 8)
        t = self.task("encoding", "base58_encode")
        key, f = self.facts(t)
        self.assertTrue(self.holds("encoding_reference", t))
        self.assertFalse(self.holds("encoding_reference", t, dir="decode"))
        self.check("ptrace_request", self.task("antidebug", "ptrace"), rv=lambda r: r + 1)
        self.check("timing_threshold", self.task("antidebug", "timing_ns"), thr=lambda x: x + 1)
        self.check("env_variable_read", self.task("antidebug", "env"), NAME="NOT_THE_VARIABLE")
        t = self.task("fsm", "table_input")
        self.check("fsm_input_final", t, final=lambda s: s + 1)
        self.check("opcode_returns_expr", self.task("fsm", "opcode_value"), n=lambda n: n + 1)
        self.check("opcode_count_handled", self.task("fsm", "opcode_count"), n=lambda n: n + 1)
        self.check("returns_for_argument", self.task("multi", "eval_arith"), n=lambda n: n + 1)
        self.check("limit_for_mode", self.task("multi", "limit_helper"), n=lambda n: n + 1)


def _family_test(fam):
    def test(self):
        mod = families.modules()[fam]
        counts, failures, forms = run_family(fam)
        for key in targets.SPECS:
            if key.split("/")[0] != fam and not (fam == "legacy" and "/" not in key):
                continue
            for seed in SEEDS:
                self.assertGreaterEqual(counts[(key, seed)], N, f"{key}: fewer than {N} tasks checked at seed {seed}")
        bad = {k: v for k, v in failures.items()}
        self.assertEqual({k: (len(v), v[:2]) for k, v in bad.items()}, {}, f"{fam}: claims that do not hold")
        for (key, name), domain in FORM_DOMAINS.items():
            if not key.startswith(fam + "/") or N < 100:       # a quick run (REA_CLAIMS_N below 100) does not claim to cover every form
                continue
            seen = {v: forms[(key, name, v)] for v in domain}
            self.assertTrue(all(c >= 15 * N // 200 for c in seen.values()), f"{key} {name}: a code form is rarely or never exercised: {seen}")
    test.__name__ = f"test_every_claim_of_every_sentence_holds_{fam.replace('-', '_')}"
    return test


@unittest.skipUnless(GCC, "gcc is needed")
class ClaimsHold(unittest.TestCase):
    pass


for _fam in families.modules():
    setattr(ClaimsHold, f"test_{_fam}", _family_test(_fam))


if __name__ == "__main__":
    unittest.main()

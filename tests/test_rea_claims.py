"""Rounds 5 and 6: the claims table. Every sentence of every truth template has a PATTERN and makes claims (tests/rea_sentences.py); every claim has a checker
(tests/rea_claims.py) that reads the numbers, names and direction words FROM THE RENDERED SENTENCE and verifies them against the task's C source and its
compiled behaviour; the checkers run over at least REA_CLAIMS_N (default 200) generated tasks of every variant at each of two seeds (a training seed and the
held-out seed), every code form of the variant included, and EVERY sentence of the template is rendered for every task (not only the one the seeded RNG picks).

  * TABLE            template key -> for each sentence of the template, its pattern and the names of the claims it makes (tests/rea_sentences.py). A template without
                     an entry, a sentence without a pattern or a claim, a claim without a checker, a checker that no sentence uses, a pattern that does not match its
                     own template: each fails `TableIsComplete`.
  * HEADS            (round 7) the same for the headline: a pattern and the claims per headline phrasing of every key, parsed from the rendered headline (`exercise` renders
                     and checks every headline too); what a headline does not state is implied (the question, a constant). `HeadWording`: every literal word of a headline is
                     registered with the keys it is true for. `HeadTeeth`: a wrong value in any group of any headline is rejected.
  * WORDING          a rendered sentence that matches no pattern fails ("wording the table does not know"): a sentence contains only its pattern's literal text and its
                     captured groups. This replaces the structure-word blacklist of round 5. A group that no claim reads fails too.
  * GOLDEN           tests/golden/rea_sentences.txt: every headline and sentence of every template rendered with a named sentinel per placeholder, so that any edit of a
                     template shows in review (REA_GOLDEN_UPDATE=1 rewrites it).
  * CHECKERS         `ClaimsHold`: for every family, every task of every variant, every claim of every sentence that could be rendered for it (for the families whose
                     tasks ask one of several questions - encoding: the identity or the alphabet, pktsum: endianness, start or length byte - the sentences of the other
                     questions are checked too).
  * TEETH            `Teeth`: a false value in a group of the sentence (one more iteration, another initial CRC value, the wrong byte order, the name MurmurHash2 over
                     the murmur-style loop, a window one wider, a width of 32 over a reflected 16-bit CRC) makes the checker fail.
  * REFERENCES       `References`: the Python references the checkers use are anchored to the standard library or to published vectors (MurmurHash2 against Austin
                     Appleby's C, FNV-1a, RC4, ChaCha against RFC 8439, Salsa20, the CRCs against zlib and binascii, base64/base32 against base64, base58 against the
                     Bitcoin vectors).
  * MUTATIONS        tests/test_rea_claims_mutations.py: 233 one-edit false sentences, headlines and facts (rounds 6 and 7) are applied to a copy of targets.py; the claims
                     machinery (not the golden file) must reject every one.

    REA_CLAIMS_N=8 python3 -m unittest tests.test_rea_claims      a quick run (the full run compiles about 30,000 programs)
    REA_CLAIMS_WORKERS=1                                           one process instead of two
"""
from __future__ import annotations

import base64
import binascii
import collections
import concurrent.futures
import multiprocessing
import os
import random
import re
import shutil
import string
import sys
import time
import unittest
import zlib
from unittest import mock

from .helpers import ROOT
from . import rea_claims as RC
from . import rea_perturb as RP
from . import rea_sentences as RS

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, f_cipher, f_crc, f_encoding, f_pktsum, families, scoring, targets  # noqa: E402

N = int(os.environ.get("REA_CLAIMS_N", "200"))
WORKERS = int(os.environ.get("REA_CLAIMS_WORKERS", "2"))
SEEDS = (4241, core.HELDOUT_SEED)
CLAIM_SECONDS = float(os.environ.get("REA_CLAIM_SECONDS", "120"))      # a claim that runs longer fails (round 7, m4)
GCC = bool(shutil.which("gcc"))
GOLDEN = ROOT / "tests" / "golden" / "rea_sentences.txt"
TABLE = RS.TABLE
HEADS = RS.HEADS


def claims_of(key: str, i: int) -> tuple:
    """The names of the claims that sentence `i` of template `key` makes."""
    return TABLE[key][i][1]


def applicable(key: str, f: dict, T=targets) -> range:
    """The sentences of template `key` that can be rendered for a task with facts `f` (pktsum: the set of the question that was asked)."""
    if key.startswith("pktsum/"):
        lo, hi = T.PKT_SETS[key.split("/")[1]][f["ask"]]
        return range(lo, hi)
    return range(len(T.SPECS[key].sentences))


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


ASK_ORDER = ("endian", "start", "len")          # the order of the three sets of headlines (and of sentences) of a pktsum variant (targets._pkt_spec)


def head_slots(key: str, f: dict, T=targets) -> list:
    """[(row of HEADS[key], index of the headline within the set that applies)] for a task with facts `f`: the headlines that can be rendered for it."""
    if key.startswith("pktsum/"):
        v = key.split("/")[1]
        base = sum(len(targets.PKT_HEADS[v][a]) for a in ASK_ORDER[:ASK_ORDER.index(f["ask"])])
        return [(base + k, k) for k in range(len(T.PKT_HEADS[v][f["ask"]]))]
    return [(k, k) for k in range(len(T.SPECS[key].heads))]


def exercise(task, T=targets, affected=None, stop=False):
    """[(key, claim, None | why)] for every claim of every renderable sentence AND headline of every question of this task; one compile per probe (shared cache).
    Each sentence and headline is rendered by template module `T` (the real one, or a mutated copy), parsed with its pattern (TABLE for sentences, HEADS for
    headlines), and the claims read the groups. `affected` ({key: {sentence indexes and "h<row>" for a headline row}}) limits the run to those, `stop` returns
    at the first failure (the mutation test)."""
    cache: dict = {}
    res = []

    def check(key, rec, label, text, caps, claims) -> bool:
        """Run the claims of one parsed sentence or headline; True when the run must end (`stop` and a failure)."""
        ctx = RC.Ctx(task, rec, caps, key)
        ctx.cache = cache
        ok = True
        for name in claims:
            try:
                with RC.time_limit(CLAIM_SECONDS):
                    RC.CLAIMS[name](ctx)
                res.append((key, name, None))
            except RC.ClaimFailed as e:
                ok = False
                res.append((key, name, f"{task.id}[{label}]: {e}"[:300]))
            except Exception as e:                       # a crash of a checker is a failure of the claim, with the reason
                ok = False
                res.append((key, name, f"{task.id}[{label}]: checker crashed: {type(e).__name__}: {e}"[:300]))
            if not ok and stop:
                return True
        if ok and caps.unread():
            res.append((key, "unread", f"{task.id}[{label}]: the groups {sorted(caps.unread())} of {text!r} are read by no claim"[:300]))
            return stop
        return False

    for rec in [task.meta()] + other_questions(task):
        key = T.key_of(rec)
        facts = T.SPECS[key].facts(rec)
        for j, i in enumerate(applicable(key, facts, T)):
            if affected is not None and i not in affected.get(key, ()):
                continue
            pattern, claims = TABLE[key][i]
            try:
                sentence = T.render(rec, j)["sentence"]
            except Exception as e:                       # a template that names a fact the task does not have
                res.append((key, "wording", f"{task.id}[{i}]: the sentence does not render: {type(e).__name__}: {e}"[:300]))
                if stop:
                    return res
                continue
            caps = RS.parse(pattern, sentence)
            if caps is None:
                res.append((key, "wording", f"{task.id}[{i}]: wording the table does not know: {sentence!r}"[:300]))
                if stop:
                    return res
                continue
            if check(key, rec, i, sentence, caps, claims):
                return res
        for row, k in head_slots(key, facts, T):
            if affected is not None and f"h{row}" not in affected.get(key, ()):
                continue
            label = f"head {row}"
            try:
                head = T.render(rec, 0, k)["headline"]
            except Exception as e:
                res.append((key, "wording", f"{task.id}[{label}]: the headline does not render: {type(e).__name__}: {e}"[:300]))
                if stop:
                    return res
                continue
            hrow = HEADS[key][row]
            try:
                caps = RS.parse_head(hrow, head, rec["question"])
            except Exception as e:                       # an implied value that the question does not state
                res.append((key, "wording", f"{task.id}[{label}]: {type(e).__name__}: {e}"[:300]))
                if stop:
                    return res
                continue
            if caps is None:
                res.append((key, "wording", f"{task.id}[{label}]: headline wording the table does not know: {head!r}"[:300]))
                if stop:
                    return res
                continue
            if check(key, rec, label, head, caps, hrow[1]):
                return res
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


# ---- the sentinel rendering (golden file) ------------------------------------------------------------------------------------------

class Sentinel:
    """A placeholder value that renders as its own name and format spec: `{off:#x}` -> `<off:#x>`, `{rot[0]}` -> `<rot[0]>`."""

    def __init__(self, name):
        self.name = name

    def __format__(self, spec):
        return f"<{self.name}{':' + spec if spec else ''}>"

    def __getitem__(self, i):
        return Sentinel(f"{self.name}[{i}]")


def template_fields(template: str) -> list:
    """[(group name in the pattern, field name in the template, base name, format spec)] of a template: `rot[0]` is the group `rot0`."""
    out = []
    for _lit, field, spec, _conv in string.Formatter().parse(template):
        if field is None:
            continue
        base = re.match(r"\w+", field).group(0)
        idx = re.findall(r"\[(\d+)\]", field)
        out.append((base + "".join(idx), field, base, spec))
    return out


def sample_text(template: str, pattern: str) -> str:
    """The template rendered with a sample value of the kind of each group of its pattern (an integer, a hex number, a word of the list, ...)."""
    _rx, kinds = RS.compile_pattern(pattern)
    kw, idx = {}, 7000
    for group, field, base, fspec in template_fields(template):
        idx += 1
        kind = kinds[group]
        v = idx if kind == "i" else (0x1000 + idx if kind == "h" else None)
        if v is None:
            v = {"p": "/proc/self/status", "q": "PTRACE_TRACEME", "e": "a ^ b", "w": "Name", "l": "abc", "t": "sample string"}.get(kind) or kind.split("|")[0]
        if "[" in field:
            kw.setdefault(base, {})[int(re.search(r"\[(\d+)\]", field).group(1))] = v
        else:
            kw[base] = v
    kw = {k: ([v[j] for j in sorted(v)] if isinstance(v, dict) else v) for k, v in kw.items()}
    return template.format(**kw)


def sentinel_text(template: str) -> str:
    names = {base for _g, _f, base, _s in template_fields(template)}
    return template.format(**{n: Sentinel(n) for n in names})


def golden_lines() -> list:
    out = []
    for key, spec in targets.SPECS.items():
        for i, h in enumerate(spec.heads):
            out.append(f"{key} head[{i}] {sentinel_text(h)}")
        for i, s in enumerate(spec.sentences):
            out.append(f"{key} sentence[{i}] {sentinel_text(s)}")
    return out


class TableIsComplete(unittest.TestCase):
    def test_every_sentence_of_every_template_has_a_pattern_and_claims_and_every_claim_a_checker(self):
        missing = [k for k in targets.SPECS if k not in TABLE]
        self.assertEqual(missing, [], "a template without an entry in the claims table")
        self.assertEqual([k for k in TABLE if k not in targets.SPECS], [], "an entry for a template that does not exist")
        for key, spec in targets.SPECS.items():
            self.assertEqual(len(TABLE[key]), len(spec.sentences), f"{key}: one pattern and one tuple of claims per sentence")
            for i, (pattern, names) in enumerate(TABLE[key]):
                RS.compile_pattern(pattern)
                self.assertTrue(names, f"{key} sentence {i} makes no claim: {spec.sentences[i]}")
                for name in names:
                    self.assertIn(name, RC.CLAIMS, f"{key} sentence {i}: no checker for the claim {name}")
        used = {name for rows in TABLE.values() for _p, names in rows for name in names} | {name for rows in HEADS.values() for _p, names, _i in rows for name in names}
        self.assertEqual(set(RC.CLAIMS) - used, set(), "a checker that no sentence and no headline uses")
        self.assertEqual(len(targets.SPECS), 78)

    def test_every_headline_of_every_template_has_a_pattern_and_claims_and_every_claim_a_checker(self):
        """Round 7, M1: the headline is read like the sentence. One pattern and one tuple of claims per headline phrasing of every key (pktsum: the three questions in a row)."""
        self.assertEqual([k for k in targets.SPECS if k not in HEADS], [], "a template without headline patterns")
        self.assertEqual([k for k in HEADS if k not in targets.SPECS], [], "headline patterns for a template that does not exist")
        for key, spec in targets.SPECS.items():
            self.assertEqual(len(HEADS[key]), len(spec.heads), f"{key}: one pattern and one tuple of claims per headline")
            for i, (pattern, names, implied) in enumerate(HEADS[key]):
                RS.compile_pattern(pattern)
                self.assertTrue(names, f"{key} head {i} makes no claim: {spec.heads[i]}")
                for name in names:
                    self.assertIn(name, RC.CLAIMS, f"{key} head {i}: no checker for the claim {name}")
                self.assertFalse(set(implied) & RS.groups(pattern), f"{key} head {i}: a group is both captured and implied")

    def test_every_placeholder_of_every_headline_is_a_group_of_its_pattern(self):
        for key, spec in targets.SPECS.items():
            for i, h in enumerate(spec.heads):
                have = RS.groups(HEADS[key][i][0])
                want = {g for g, _f, _b, _s in template_fields(h)}
                self.assertEqual(sorted(want - have), [], f"{key} head {i}: a placeholder of the headline that the pattern does not capture: {h}")

    def test_every_headline_pattern_matches_its_template_rendered_with_sample_values(self):
        for key, spec in targets.SPECS.items():
            for i, h in enumerate(spec.heads):
                pattern = HEADS[key][i][0]
                text = sample_text(h, pattern)
                self.assertIsNotNone(RS.parse(pattern, text), f"{key} head {i}: the pattern does not match its template\n  template: {h}\n  rendered: {text}\n  pattern:  {pattern}")

    def test_every_placeholder_of_every_template_is_a_group_of_its_pattern(self):
        """A number the sentence states is read from the sentence: every placeholder of a template is a captured group of the pattern."""
        for key, spec in targets.SPECS.items():
            for i, s in enumerate(spec.sentences):
                have = RS.groups(TABLE[key][i][0])
                want = {g for g, _f, _b, _s in template_fields(s)}
                self.assertEqual(sorted(want - have), [], f"{key}[{i}]: a placeholder of the template that the pattern does not capture: {s}")

    def test_every_pattern_matches_its_template_rendered_with_sample_values(self):
        """The literal text of a pattern is the literal text of its template: render the template with a sample value of the kind of each group, and parse."""
        for key, spec in targets.SPECS.items():
            for i, s in enumerate(spec.sentences):
                pattern = TABLE[key][i][0]
                text = sample_text(s, pattern)
                self.assertIsNotNone(RS.parse(pattern, text), f"{key}[{i}]: the pattern does not match its template\n  template: {s}\n  rendered: {text}\n  pattern:  {pattern}")

    def test_the_summary_numbers(self):
        sentences = sum(len(s.sentences) for s in targets.SPECS.values())
        claims = sum(len(names) for rows in TABLE.values() for _p, names in rows)
        self.assertGreaterEqual(sentences, 234)
        self.assertGreaterEqual(claims, sentences)
        heads = sum(len(h) for h in HEADS.values())
        self.assertEqual(heads, sum(len(s.heads) for s in targets.SPECS.values()))
        self.assertGreaterEqual(heads, 368)


class Wording(unittest.TestCase):
    def test_the_parser_rejects_a_sentence_the_table_does_not_know(self):
        pat = TABLE["bounds/range_chain"][0][0]
        good = "proc returns 1 for every integer from 3 to 9, both included, and for no other integer."
        caps = RS.parse(pat, good)
        self.assertEqual((caps["ret"], caps["lo"], caps["hi"]), (1, 3, 9))
        for bad in (good.replace("every", "each"), good.replace("both included", "both included, by a loop"), good + " It is fast.", "It " + good,
                    good.replace("integer from", "union from"), good.replace("no other", "any other")):
            self.assertIsNone(RS.parse(pat, bad), bad)

    def test_a_group_that_no_claim_reads_is_a_failure(self):
        caps = RS.parse(TABLE["bounds/range_chain"][0][0], "proc returns 1 for every integer from 3 to 9, both included, and for no other integer.")
        caps["lo"], caps["hi"]
        self.assertEqual(caps.unread(), {"ret"})

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


class HeadWording(unittest.TestCase):
    """Round 7, B1: a verb or label of a headline is true for the keys that use it. `multi/decode_secret` shared the strobf tuple and said `It writes: <secret>`
    of a function that only compares its input. Every literal word of every headline is registered (tests/rea_sentences.py HEAD_WORDS) with the keys it is
    true for; a headline tuple shared by two keys is therefore true for both or the lint fails."""

    def heads(self):
        return {key: spec.heads for key, spec in targets.SPECS.items()}

    def test_every_word_of_every_headline_is_registered_for_the_key_that_uses_it(self):
        self.assertEqual(RS.head_word_problems(self.heads()), [])

    def test_the_registry_has_no_word_and_no_key_prefix_that_no_headline_uses(self):
        used = collections.defaultdict(set)
        for key, templates in self.heads().items():
            for t in templates:
                for w in RS.head_words(t):
                    used[w].add(key)
        self.assertEqual(sorted(set(RS.HEAD_WORDS) - set(used)), [], "a registered word that no headline uses")
        for w, prefixes in RS.HEAD_WORDS.items():
            for pre in prefixes:
                self.assertTrue(any(k.startswith(pre) for k in used[w]), f"{w!r} is registered for {pre!r}, no key under it uses the word")

    def test_the_lint_rejects_the_headlines_of_round_7(self):
        """The edit that was in the tree: the strobf tuple (`It writes:`) on decode_secret, and on the legacy xor-key and keycount."""
        bad = RS.head_word_problems({"multi/decode_secret": targets.EXACT})
        self.assertTrue(any("'writes'" in b and "not for this key" in b for b in bad), bad)
        self.assertTrue(RS.head_word_problems({"xor-key": ("{s}", "The key is: {s}", "Key: {s}", "It writes: {s}")}))
        self.assertTrue(RS.head_word_problems({"keycount": ("{s}", "The key is: {s}", "Key: {s}", "It counts: {s}")}))
        self.assertTrue(RS.head_word_problems({"strobf/xor_const": ("{s}", "The string is: {s}", "It reads: {s}", "Plaintext: {s}")}), "an unregistered verb")
        self.assertTrue(RS.head_word_problems({"antidebug/timing_us": ("{tech}, {thr}", "{tech}, {thr} ns")}), "a unit of the other variant")
        self.assertTrue(RS.head_word_problems({"antidebug/timing_us": ("{tech}, {thr}", "{tech}, {thr} us")}), "round 8, m2: no timing headline carries a unit")
        self.assertFalse(RS.head_word_problems({"strobf/xor_const": targets.EXACT, "multi/decode_secret": targets.ACCEPTED}))

    def test_no_decode_secret_or_xor_key_record_says_it_writes(self):
        for key in ("multi/decode_secret", "xor-key", "password", "keycount"):
            for i, h in enumerate(targets.SPECS[key].heads):
                self.assertNotIn("writes", h.lower(), f"{key} head[{i}]")
        tasks = [t for t in families.generate("multi", 5 * 200, 99) if t.variant == "decode_secret"] + [t for t in families.generate("legacy", 12 * 200, 99) if t.kind == "xor-key"]
        self.assertGreater(len(tasks), 150)
        for t in tasks:
            self.assertNotIn("writes", targets.render(t.meta())["headline"].lower(), t.id)


class Phrasings(unittest.TestCase):
    """Round 7, m2: the headline is chosen independently of the sentence, so every headline and every sentence of every key is rendered for some task (head[3] of all
    78 keys never was), a task is rendered the same way every time, and the sentence of a task is the one that round 6 rendered."""

    @classmethod
    def setUpClass(cls):
        cls.seen = collections.defaultdict(lambda: {"heads": set(), "sentences": set(), "pairs": set(), "n": 0, "of": None})
        cls.tasks = {}
        for fam, mod in families.modules().items():
            for t in families.generate(fam, 150 * len(mod.VARIANTS), 7):
                r = targets.render(t.meta())
                cls.tasks[t.id] = (t, r)
                d = cls.seen[r["key"]]
                d["heads"].add(r["head_template"])
                d["sentences"].add(r["template"])
                d["pairs"].add((r["head_template"], r["template"]))
                d["n"] += 1
                d["of"] = (r["head_templates"], r["templates"])

    def test_every_headline_and_every_sentence_of_every_key_is_rendered(self):
        self.assertEqual(sorted(self.seen), sorted(targets.SPECS), "every key is rendered for some task")
        for key, d in self.seen.items():
            heads, sentences = d["of"]
            self.assertEqual(d["heads"], set(range(heads)), f"{key}: a headline that no task is rendered with ({d['n']} tasks)")
            self.assertEqual(d["sentences"], set(range(sentences)), f"{key}: a sentence that no task is rendered with")

    def test_the_headline_does_not_follow_from_the_sentence(self):
        for key, d in self.seen.items():
            heads, sentences = d["of"]
            if d["n"] >= 60:
                self.assertGreaterEqual(len(d["pairs"]), min(heads * sentences, 9), f"{key}: the headline is a function of the sentence: {sorted(d['pairs'])}")

    def test_every_headline_of_every_key_is_an_answer_under_the_strict_grammar(self):
        """Round 7: making head[3] reachable showed seven headlines that `score_strict` rejects (`{thr} ns, {tech}`: a number before the name, `It has {n} states.`,
        `byte 0x3b`, `XOR key: <key>`: not a lead-in of the grammar) and `sft_target` raises on such a text. All 4 headlines of every key, with every sentence, are
        an answer for 15 tasks of the key: whatever the RNG draws, the stored text scores."""
        per = collections.defaultdict(list)
        for t, r in self.tasks.values():
            if len(per[r["key"]]) < 15:
                per[r["key"]].append(t)
        self.assertEqual(sorted(per), sorted(targets.SPECS))
        n = 0
        for key, tasks in per.items():
            for t in tasks:
                m = t.meta()
                heads = targets.render(m)["head_templates"]
                for k in range(heads):
                    for j in range(targets.render(m)["templates"]):
                        text = targets.render(m, j, k)["text"]
                        n += 1
                        self.assertTrue(scoring.score_strict(text, m), f"{key} head {k} sentence {j}: score_strict rejects {text!r}")
        self.assertGreater(n, 12000)

    def test_a_task_is_always_rendered_the_same_way_and_the_sentence_is_the_round_6_one(self):
        for tid, (t, r) in list(self.tasks.items())[::37]:
            again = targets.render(t.meta())
            self.assertEqual(r, again, tid)
            key = r["key"]
            self.assertEqual(r["template"], random.Random(f"truth-template/{tid}/{key}").randrange(r["templates"]), "the sentence RNG is the one of round 6")
            self.assertEqual(r["head_template"], random.Random(f"truth-template/{tid}/{key}/head").randrange(r["head_templates"]))
            self.assertEqual(targets.render(t.meta(), r["template"], r["head_template"]), r)


class Golden(unittest.TestCase):
    def test_every_headline_and_sentence_with_a_sentinel_per_placeholder(self):
        """Any edit of a template shows here: the diff of tests/golden/rea_sentences.txt is the review of the text."""
        lines = golden_lines()
        if os.environ.get("REA_GOLDEN_UPDATE"):
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text("\n".join(lines) + "\n")
        self.assertTrue(GOLDEN.exists(), "run once with REA_GOLDEN_UPDATE=1")
        self.assertEqual(GOLDEN.read_text().splitlines(), lines, "a template changed: review the diff, then REA_GOLDEN_UPDATE=1 python3 -m unittest tests.test_rea_claims.Golden")


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
                self.assertEqual(RC.tea_ref("TEA", v, k, r, targets.TEA_DELTA, dec), f_cipher.tea(v, k, r, dec))
                self.assertEqual(RC.tea_ref("XTEA", v, k, r, targets.TEA_DELTA, dec), f_cipher.xtea(v, k, r, dec))
            self.assertNotEqual(RC.tea_ref("TEA", v, k, r, 0x9E3779B8, False), f_cipher.tea(v, k, r, False), "another delta is another cipher")

    # ---- m5: the references that the "name" claims lean on, anchored to the standard library or to published vectors -------------------

    def test_rc4_equals_the_published_vectors(self):
        self.assertEqual(f_cipher.rc4(b"Key", b"Plaintext").hex(), "bbf316e8d940af0ad3")
        self.assertEqual(f_cipher.rc4(b"Wiki", b"pedia").hex(), "1021bf0420")
        self.assertEqual(f_cipher.rc4(b"Secret", b"Attack at dawn").hex(), "45a01f645fc35b383552544b9bf5")

    def test_chacha20_block_equals_rfc_8439_section_2_3_2(self):
        key = [int.from_bytes(bytes(range(i, i + 4)), "little") for i in range(0, 32, 4)]
        nonce = [0x09000000, 0x4a000000, 0x00000000]
        want = [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
                0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2]
        self.assertEqual(f_cipher.chacha_block(key, 1, nonce, 20), want)

    def test_salsa20_core_equals_the_specification_s_example(self):
        """Bernstein's Salsa20 specification, section 8: Salsa20(x) for a 64-byte x, and the identity that the core of 64 zero bytes is zero."""
        x = bytes([211, 159, 13, 115, 76, 55, 82, 183, 3, 117, 222, 37, 191, 187, 234, 136, 49, 237, 179, 48, 1, 106, 178, 219, 175, 199, 166, 48, 86, 16, 179, 207,
                   31, 240, 32, 63, 15, 83, 93, 161, 116, 147, 48, 113, 238, 55, 204, 36, 79, 201, 235, 79, 3, 81, 156, 47, 203, 26, 244, 243, 88, 118, 104, 54])
        want = bytes([109, 42, 178, 168, 156, 240, 248, 238, 168, 196, 190, 203, 26, 110, 170, 154, 29, 29, 150, 26, 150, 30, 235, 249, 190, 163, 251, 48, 69, 144, 51, 57,
                      118, 40, 152, 157, 180, 57, 27, 94, 107, 42, 236, 35, 27, 111, 114, 114, 219, 236, 232, 135, 111, 155, 110, 18, 24, 232, 95, 158, 179, 19, 48, 202])
        inp = [int.from_bytes(x[i:i + 4], "little") for i in range(0, 64, 4)]
        out = f_cipher.salsa_core(inp, 20)
        self.assertEqual(b"".join(w.to_bytes(4, "little") for w in out), want)
        self.assertEqual(f_cipher.salsa_core([0] * 16, 20), [0] * 16)

    def test_the_crc_model_equals_zlib_binascii_and_the_catalogue_check_values(self):
        rng = random.Random(8)
        for _ in range(10):
            data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 50)))
            self.assertEqual(f_crc.crc_ref(data, 32, 0x04C11DB7, 0xFFFFFFFF, True, 0xFFFFFFFF), zlib.crc32(data))
            self.assertEqual(f_crc.crc_ref(data, 16, 0x1021, 0xFFFF, False, 0), binascii.crc_hqx(data, 0xFFFF), "CRC-16/CCITT-FALSE")
        self.assertEqual(f_crc.crc_ref(b"123456789", 16, 0x8005, 0, True, 0), 0xBB3D, "CRC-16/ARC")
        self.assertEqual(f_crc.crc_ref(b"123456789", 16, 0x1021, 0xFFFF, False, 0), 0x29B1, "CRC-16/CCITT-FALSE")
        self.assertEqual(f_crc.crc_ref(b"123456789", 32, 0x04C11DB7, 0xFFFFFFFF, True, 0xFFFFFFFF), 0xCBF43926, "CRC-32")
        self.assertEqual(f_pktsum.csum("crc16", b"123456789"), 0xBB3D)
        self.assertEqual(f_pktsum.csum("fletcher16", b"abcde"), 0xC8F0, "Wikipedia's Fletcher-16 example")
        self.assertEqual(f_pktsum.csum("fletcher16", b"abcdef"), 0x2057)

    def test_base64_and_base32_equal_the_standard_library_and_base58_the_bitcoin_vectors(self):
        a64 = string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"
        a32 = string.ascii_uppercase + "234567"
        a58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        rng = random.Random(9)
        for n in list(range(0, 12)) + [30, 31]:
            v = bytes(rng.randrange(256) for _ in range(n))
            self.assertEqual(f_encoding.encode_ref("base64", v, a64, True).encode(), base64.b64encode(v))
            self.assertEqual(f_encoding.encode_ref("base32", v, a32, True).encode(), base64.b32encode(v))
            self.assertEqual(f_encoding.encode_ref("base64", v, a64, False).encode(), base64.b64encode(v).rstrip(b"="))
        self.assertEqual(f_encoding.encode_ref("base58", b"hello world", a58, False), "StV1DL6CwTryKyV")
        self.assertEqual(f_encoding.encode_ref("base58", b"\x00\x00\x28\x7f\xb4\xcd", a58, False), "11233QC4", "leading zero bytes are leading '1's")
        self.assertEqual(f_encoding.encode_ref("base58", b"", a58, False), "")


@unittest.skipUnless(GCC, "gcc is needed")
class Teeth(unittest.TestCase):
    """A false value in a group of the sentence must fail its checker (and the true one must pass), or the checker proves nothing."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = {}
        for fam in ("cipher", "crc", "pktsum", "bounds", "legacy", "strobf", "structs", "encoding", "antidebug", "fsm", "multi"):
            cls.tasks[fam] = families.generate(fam, 12 * len(families.modules()[fam].VARIANTS), 77)

    def task(self, fam, variant, pred=lambda t: True):
        return next(t for t in self.tasks[fam] if (t.variant == variant or t.kind == variant) and pred(t))

    def holds(self, claim, t, group, over=None):
        """Parse the first sentence of the task's template that makes `claim` and has the group `group`, put `over` in the group, run the claim."""
        rec = t.meta()
        key = targets.key_of(rec)
        f = targets.SPECS[key].facts(rec)
        for j, i in enumerate(applicable(key, f)):
            pattern, names = TABLE[key][i]
            if claim in names and group in RS.groups(pattern):
                caps = RS.parse(pattern, targets.render(rec, j)["sentence"])
                self.assertIsNotNone(caps, f"{key}[{i}]")
                if over is not None:
                    caps[group] = over(caps[group]) if callable(over) else over
                try:
                    RC.CLAIMS[claim](RC.Ctx(t, rec, caps, key))
                    return True
                except RC.ClaimFailed:
                    return False
        self.fail(f"no sentence of {key} makes {claim} with a group {group}")

    def check(self, claim, t, **over):
        for group, v in over.items():
            self.assertTrue(self.holds(claim, t, group), f"{claim}: the true {group} fails")
            self.assertFalse(self.holds(claim, t, group, v), f"{claim}: a false {group} passes")

    def test_cipher_and_crc(self):
        self.check("tea_reference", self.task("cipher", "xtea_dec"), rounds=lambda n: n + 1, delta=0x9E3779B8, done="encrypted", verb="encrypts", key_bits=lambda n: n * 2,
                   block=16)
        self.check("tea_reference", self.task("cipher", "tea_enc"), alg="XTEA")
        self.check("crc_init_value", self.task("crc", "bitwise_init"), x=lambda x: x ^ 1)
        self.check("crc_xorout_value", self.task("crc", "table_xorout"), x=lambda x: x ^ 0x10)
        self.check("crc_poly_form", self.task("crc", "bitwise_poly"), lit=lambda x: x ^ 2, dir=lambda d: "normal" if d == "reflected" else "reflected")
        for refl in (True, False):                       # m3(a): the width is the register's, a 16-bit CRC is not a 32-bit one whatever the values say
            t = self.task("crc", "bitwise_poly", lambda t: t.params["width"] == 16 and t.params["refl"] == refl)
            self.check("crc_poly_form", t, w=32)
        self.check("stream_reference", self.task("cipher", "chacha"), rounds=lambda n: n + 2, rot0=lambda r: r + 1, alg="Salsa", half=lambda h: h + 1, block=32)
        self.check("rc4_reference", self.task("cipher", "rc4"), n=lambda n: n + 1, state=255, alg="ChaCha")

    def test_pktsum_and_bounds(self):
        t = self.task("pktsum", "sum16_le", lambda t: t.params["ask"] == "endian")
        self.check("pkt_endian_behaviour", t, e="big", first="high")
        t = self.task("pktsum", "fletcher16", lambda t: t.params["S"] > 1 and t.params["ask"] == "start")
        self.check("pkt_range_start", t, n=lambda n: n - 1)
        self.assertFalse(self.holds("pkt_range_start", t, "n", t.params["S"] + 1))
        t = self.task("pktsum", "len_field", lambda t: t.params["ask"] == "len_offset")
        self.check("pkt_length_byte", t, n=lambda n: n + 1)
        t = self.task("bounds", "window")
        self.check("returns_one_for_window", t, hi=lambda h: h + 1, w=lambda w: w + 1, ret=0)
        t = self.task("bounds", "range_chain")
        self.check("returns_one_exactly_in_range", t, lo=lambda x: x - 1, lw="highest")
        t = self.task("bounds", "clamp")
        self.check("clamp_interval", t, hi=lambda h: h + 1, d1="above", lb="upper")
        self.check("index_limit_range", self.task("bounds", "index_limit"), neg=0, o1="inside", lw="largest")
        self.check("char_class_count", self.task("bounds", "char_class"), n=lambda n: n + 1, total=255, top=254)

    def test_legacy_and_strobf(self):
        t = self.task("legacy", "hash", lambda t: t.truth["accept"][0] == ["murmur"])
        self.assertTrue(self.holds("hash_name_is_the_algorithm", t, "name"))
        for wrong in ("FNV-1a", "MurmurHash2", "Murmur"):
            self.assertFalse(self.holds("hash_name_is_the_algorithm", t, "name", wrong), f"the murmur loop is not {wrong}")
        self.check("hash_final_xor", t, c=lambda c: c ^ 1)
        self.check("hash_start_value", t, init=0x9747b28d)
        self.check("magic_header", self.task("legacy", "magic"), v=lambda v: v + 1, e=lambda e: "big" if e == "little" else "little")
        self.check("skip_one_value", self.task("legacy", "skip"), b=lambda b: b ^ 1)
        self.check("lcg_reference", self.task("legacy", "lcg"), r=lambda r: r + 1, a=lambda a: a + 2)
        self.check("rotate_reference", self.task("legacy", "rotate"), dir=lambda d: "right" if d == "left" else "left", r=lambda r: r + 1)
        self.check("threshold_values", self.task("legacy", "threshold"), below=lambda b: b + 1, d1="Above", lb="upper")
        self.check("writes_exact_string", self.task("strobf", "xor_const"), len=lambda n: n + 1)

    def test_a_number_that_would_make_a_claim_loop_fails_fast(self):
        """Round 7, m4: the numbers of a swapped sentence (rounds <-> multiplier, a delta as the rounds, a huge group) are out of range: the claim fails in a moment."""
        cases = [("lcg_reference", self.task("legacy", "lcg"), "r", 1103515245), ("tea_reference", self.task("cipher", "xtea_dec"), "rounds", 2 ** 31),
                 ("stream_reference", self.task("cipher", "chacha"), "rounds", 2 ** 31), ("encoding_reference", self.task("encoding", "base64_encode"), "bits", 10 ** 9),
                 ("encoding_reference", self.task("encoding", "base64_encode"), "gin", 10 ** 9), ("magic_header", self.task("legacy", "magic"), "v", 2 ** 31)]
        for claim, t, group, v in cases:
            t0 = time.time()
            self.assertFalse(self.holds(claim, t, group, v), f"{claim} {group}={v} passes")
            self.assertLess(time.time() - t0, 10, f"{claim} {group}={v} took {time.time() - t0:.1f} s to fail")

    def test_structs_encoding_antidebug_fsm_multi(self):
        self.check("stride_records", self.task("structs", "array_stride"), n=lambda n: n + 8)
        self.check("field_read_offset", self.task("structs", "field_read"), off=lambda o: o + 1, flag="zero")
        self.check("linked_pointer_offset", self.task("structs", "linked_next"), off=lambda o: o + 8)
        self.check("alloc_structure_size", self.task("structs", "alloc_size"), n=lambda n: n + 8)
        self.check("init_magic_offset", self.task("structs", "init_magic"), off=lambda o: o + 8)
        self.check("encoding_reference", self.task("encoding", "base64_encode"), gin=4, radix=63)
        self.check("ptrace_request", self.task("antidebug", "ptrace"), rv=lambda r: r + 1, outcome="granted", req="PTRACE_ATTACH")
        self.check("tracerpid_status", self.task("antidebug", "tracerpid"), field="PPid", file="/proc/self/maps")
        self.check("timing_threshold", self.task("antidebug", "timing_ns"), thr=lambda x: x + 1, unit="microseconds", cmp="below")
        self.check("timing_threshold", self.task("antidebug", "timing_us"), unit="milliseconds")
        self.check("env_variable_read", self.task("antidebug", "env"), NAME="NOT_THE_VARIABLE")
        self.check("fsm_input_final", self.task("fsm", "table_input"), final=lambda s: s + 1, z=1)
        self.check("fsm_states_count", self.task("fsm", "table_states"), n=lambda n: n + 1)
        self.check("opcode_returns_expr", self.task("fsm", "opcode_value"), n=lambda n: n + 1)
        self.check("opcode_count_handled", self.task("fsm", "opcode_count"), n=lambda n: n + 1)
        self.check("opcode_default", self.task("fsm", "opcode_count"), dflt=1)
        self.check("returns_for_argument", self.task("multi", "eval_arith"), n=lambda n: n + 1, n0=lambda n: n + 1)
        self.check("xors_key_byte", self.task("multi", "key_helper"), k=lambda k: k ^ 1)
        self.check("limit_for_mode", self.task("multi", "limit_helper"), n=lambda n: n + 1)


@unittest.skipUnless(GCC, "gcc is needed")
class HeadTeeth(unittest.TestCase):
    """Round 7, M1: every group of every headline phrasing of every key, over one task of the key, is replaced by a wrong value (+1, -1, a flipped bit, every other word of a
    list, a mangled string, two numbers swapped) and a claim of its row must fail: a group that the table does not constrain would let a headline say something false."""

    def test_a_wrong_value_in_any_group_of_any_headline_is_rejected(self):
        survivors, n, slow = [], 0, []
        for fam, mod in families.modules().items():
            seen = set()
            for task in families.generate(fam, 3 * len(mod.VARIANTS) + 12, 4242):
                base = targets.key_of(task.meta()).split("#")[0]
                if base in seen:
                    continue
                seen.add(base)
                cache: dict = {}
                for rec in [task.meta()] + other_questions(task):
                    key = targets.key_of(rec)
                    for row, k in head_slots(key, targets.SPECS[key].facts(rec)):
                        hrow = HEADS[key][row]
                        caps = RS.parse_head(hrow, targets.render(rec, 0, k)["headline"], rec["question"])
                        self.assertIsNotNone(caps, f"{key} head {row}")
                        _rx, kinds = RS.compile_pattern(hrow[0])
                        for label, new in RP.wrong_values(kinds, caps):
                            n += 1
                            why = RP.rejects(task, rec, key, hrow[1], caps, new, cache, hrow[2], seconds=20)
                            if not why:
                                survivors.append(f"{key} head {row} {label} on {task.id}: {targets.render(rec, 0, k)['headline']!r}")
                            elif why.startswith("hang"):
                                slow.append(f"{key} head {row} {label}")
        self.assertGreater(n, 1000)
        self.assertEqual(survivors, [], "a wrong value in a headline that no claim rejects")
        self.assertEqual(slow, [], "a claim that hangs on a wrong value instead of failing")


@unittest.skipUnless(GCC, "gcc is needed")
class Guards(unittest.TestCase):
    """Round 7, m4: a claim FAILS on a false sentence, it does not hang. A number of a swapped sentence (the rounds of an LCG that became its multiplier: 1,103,515,245
    iterations of a Python loop) made `ClaimsHold` and the mutation test run for minutes; a claim now rejects a number outside the range of its kind of task, and a
    claim that still does not return within CLAIM_SECONDS is a failed claim with that reason."""

    def test_a_claim_that_does_not_return_fails_with_the_reason(self):
        t0 = time.time()
        with self.assertRaises(RC.ClaimFailed) as cm:
            with RC.time_limit(0.3):
                while True:
                    pass
        self.assertIn("did not return within 0.3 s", str(cm.exception))
        self.assertLess(time.time() - t0, 3)

    def test_the_limit_is_lifted_when_the_claim_returns(self):
        with RC.time_limit(0.3):
            pass
        time.sleep(0.6)                                    # an alarm that was left armed would raise here

    def test_exercise_turns_a_hang_into_a_failed_claim(self):
        task = next(t for t in families.generate("legacy", 12 * 5, 99) if t.kind == "lcg")

        def slow(_ctx):
            time.sleep(30)
        with mock.patch.dict(RC.CLAIMS, {"lcg_reference": slow}), mock.patch.object(sys.modules[__name__], "CLAIM_SECONDS", 0.5):
            t0 = time.time()
            res = exercise(task, stop=True)
        self.assertLess(time.time() - t0, 10)
        self.assertEqual([r[1] for r in res if r[2]], ["lcg_reference"])
        self.assertIn("did not return within 0.5 s", [r[2] for r in res if r[2]][0])

    def test_an_expression_with_an_exponent_or_a_huge_shift_is_not_evaluated(self):
        self.assertEqual(RC._eval_expr("a ^ b", 6, 3), 5)
        for bad in ("a ** b", "a ** 99999999", "1 << 99999999999", "a<<4000000000"):
            with self.assertRaises(RC.ClaimFailed, msg=bad):
                RC._eval_expr(bad, 2, 3)


class _Stub:
    """A task that is only a program: the claims that read nothing but `source` and the groups run on it."""
    id = "stub"

    def __init__(self, proc: str):
        self.proc = proc

    def source(self, main: str, pre: str = "") -> str:
        return "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <strings.h>\n" + pre + self.proc + "\n" + main


@unittest.skipUnless(GCC, "gcc is needed")
class KeycountDomain(unittest.TestCase):
    """Round 7, m3: `keycount_lines` covers every shape of one line (800 lines: 5 prefixes x 5 spellings of the key x 4 separators x 8 values), not 6 probe lines. The
    legacy proc passes; seven wrong counters fail: five that the 6 probe lines of round 6 accepted (they skip a leading space or tab, accept a colon, ignore the case of
    the key, do not count a value that is one space) and two that those probes caught as well (a longer key, a comment line)."""

    KEY = "port"

    def proc(self, cond: str) -> str:
        return ('int proc(const char *text) { int k = 0; const char *line = text; while (*line) { const char *nl = strchr(line, 10); '
                'size_t len = nl ? (size_t)(nl - line) : strlen(line); if (' + cond + ') k++; line += len + (nl ? 1 : 0); } return k; }')

    def holds(self, cond: str, key: str = None) -> bool:
        key = key or self.KEY
        c = RC.Ctx(_Stub(self.proc(cond)), {}, RS.Caps({"s": key}), "keycount")
        try:
            RC.CLAIMS["keycount_lines"](c)
            return True
        except RC.ClaimFailed:
            return False

    def test_the_domain_is_800_lines_of_every_shape(self):
        lines = RC.key_lines(self.KEY)
        self.assertEqual(len(lines), 800)
        self.assertGreater(len(set(lines)), 700, "the shapes are distinct lines")
        self.assertEqual(sum(ln.startswith("port=") and len(ln) > len("port=") for ln in lines), 7,
                         "no prefix, the key itself, an equals sign, and any of the 7 values that are not empty")

    def test_the_proc_of_the_generator_passes(self):
        n = len(self.KEY) + 1
        self.assertTrue(self.holds(f'len > {n} && line[0] != 35 && strncmp(line, "{self.KEY}=", {n}) == 0'))
        self.assertTrue(self.holds('len > 10 && line[0] != 35 && strncmp(line, "host_name=", 10) == 0', "host_name"))

    def test_every_wrong_counter_fails_including_five_that_the_probe_lines_of_round_6_accepted(self):
        n = len(self.KEY) + 1
        base = f'strncmp(line + OFF, "{self.KEY}=", {n}) == 0 && len > {n}'
        wrong = {
            "skips one leading space": base.replace("OFF", "(line[0] == 32)"),
            "skips a tab or a space": base.replace("OFF", "(line[0] == 32 || line[0] == 9)"),
            "accepts a colon": f'len > {n} && strncmp(line, "{self.KEY}", {n - 1}) == 0 && (line[{n - 1}] == 61 || line[{n - 1}] == 58)',
            "ignores the case of the key": f'len > {n} && strncasecmp(line, "{self.KEY}=", {n}) == 0',
            "does not count a value that is a single space": f'len > {n} && strncmp(line, "{self.KEY}=", {n}) == 0 && !(len == {n + 1} && line[{n}] == 32)',
            "counts a longer key": f'len > {n} && strncmp(line, "{self.KEY}", {n - 1}) == 0 && strchr(line, 61) != NULL',
            "counts a comment line": f'len > {n} && strstr(line, "{self.KEY}=") != NULL',
        }
        for why, cond in wrong.items():
            self.assertFalse(self.holds(cond), f"a counter that {why} is accepted")


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

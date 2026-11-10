"""Round 4 (B2, M4 stored text, m9): the stored target is rendered from the ground truth, never copied from the teacher.

`problems(task, rendered)` is the whole check of ONE rendered target:
  truth        the text scores True under the data scorer and the evaluation scorer; it is a headline line and one prose sentence
  facts        every fact the template states equals what the generator's own `params` say (the generator draws these, the key is only one of them),
               the constants that are the same for every task of a variant are in the C source of THIS task, the set of facts is exactly the expected one
  numbers      every number and number word in the sentence is a fact, a number the question states, or a counted word the table below justifies
  names        no word of the family's vocabulary (every accepted and rejected name of the family) except this task's own; no opposite of the truth
               (the other direction, the other endianness, `shifts left` for a reflected CRC), in any inflection
`test_every_rendered_target_is_true` renders 2,000 tasks per family; the `Teeth` tests feed the same check a wrong round count, a wrong delta, the
other direction (the review's `It encrypted the data.`), swapped bounds and a foreign name, and it must find each of them.
"""
from __future__ import annotations

import collections
import re
import sys
import unittest

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, grammar, scoring, sft, targets  # noqa: E402

PER_FAMILY = 2000
SEED = 4242

# Counted words a template may use, with the reason they are true for every task of the variant (checked below where the code can say so).
WORDS_OK = {"cipher/chacha": {"eight", "two"},      # 8 QR calls per loop body; a double round counts as 2 (the question says so)
            "cipher/salsa": {"eight", "two"},
            "multi/chain": {"two", "three"},        # two helpers (task.helpers), three functions (the question says so)
            "strobf/alt_xor": {"two"},              # k and k1
            "antidebug/timing_ns": {"two"},         # two clock readings (the question says so)
            "antidebug/timing_us": {"two"}}
for _v in ("sum8", "xor8", "sum16_le", "sum16_be", "fletcher16", "crc16", "len_field"):
    WORDS_OK[f"pktsum/{_v}"] = {"two"}              # a 16-bit checksum is two bytes (checked: W == 2 where the sentence says it)

OPEN_EXACT = {"strobf", "multi"}                    # exact strings: the vocabulary of plaintexts is not a vocabulary of names


def _flat(x):
    if isinstance(x, (int, bool)):
        yield int(x)
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from _flat(y)


def numbers_in(text: str) -> set:
    out = set()
    for m in re.finditer(r"-?0x[0-9a-fA-F]+|-?\d+", text):
        s = m.group(0)
        out.add(int(s, 16) if "0x" in s.lower() else int(s))
    for w in re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower()):
        if w in grammar._NUMWORD:
            out.add(grammar._NUMWORD[w])
    return out


def words_in(text: str) -> set:
    return {w for w in re.split(r"[^a-z0-9_-]+", text.lower()) if len(w.strip("-")) >= 3 and not w.isdigit()}


# ---- what the generator says: the expected facts of every variant, from `task.params` ------------------------------------------------

def _sim(p: dict, s: str) -> int:
    st = 0
    for ch in s:
        st = p["delta"][st][p["alphabet"].index(ch)]
    return st


OPS = {"and": "a & b", "or": "a | b", "add": "a + b", "mul": "a * b", "xor": "a ^ b", "sub": "a - b"}


def expected(key: str, task, meta: dict, f: dict) -> dict:
    p, v = task.params, task.variant
    base = key.split("#")[0]
    if key == "structs/array_stride" or key == "structs/alloc_size":
        return {"n": p["size"]}
    if base in ("structs/field_read", "structs/nested_offset", "structs/linked_next"):
        return {"off": p["offset"]}
    if base == "structs/init_magic":
        return {"off": p["offset"], "magic": f["magic"]}                       # the constant is checked in the body below
    if base in ("fsm/table_input", "fsm/switch_input"):
        return {"s": f["s"], "d": p["d"], "final": _sim(p, f["s"])}
    if base in ("fsm/table_states", "fsm/switch_states"):
        return {"n": len(p["delta"]), "last": len(p["delta"]) - 1}
    if base == "fsm/opcode_value":
        code = {op: c for c, op in p["pairs"]}[p["op"]]
        return {"n": code, "expr": OPS.get(p["op"], f["expr"])}
    if base == "fsm/opcode_count":
        return {"n": len(p["pairs"])}
    if base.startswith("crc/") and v.endswith("poly"):
        out = {"w": p["width"], "lit": p["lit"], "dir": "reflected" if p["refl"] else "normal", "refl": p["refl"],
               "shift": "right, least significant bit first" if p["refl"] else "left, most significant bit first"}
        return {**out, "entries": 256} if v.startswith("table") else out
    if base.startswith("crc/") and v.endswith("init"):
        return {"w": p["width"], "x": p["init"]}
    if base.startswith("crc/") and v.endswith("xorout"):
        return {"w": p["width"], "x": p["xorout"]}
    if base in ("cipher/tea_enc", "cipher/tea_dec", "cipher/xtea_enc", "cipher/xtea_dec"):
        return {"alg": p["alg"].upper(), "dir": "encrypt" if v.endswith("enc") else "decrypt", "rounds": p["rounds"], "delta": 0x9E3779B9,
                "key_bits": 32 * len(p["key"]), "block": 4 * len(p["v"])}
    if base == "cipher/rc4":
        return {"alg": "RC4", "n": p["keylen"], "state": 256}
    if base in ("cipher/chacha", "cipher/salsa"):
        return {"alg": "ChaCha" if v == "chacha" else "Salsa", "rounds": p["rounds"], "half": p["rounds"] // 2, "key_bits": 32 * len(p["key"]), "block": 64,
                "rot": (16, 12, 8, 7) if v == "chacha" else (7, 9, 13, 18)}
    if kind_of(base) == "encoding":
        radix = len(p["alphabet"])
        gin, gout, bits = {64: (3, 4, 6), 32: (5, 8, 5), 58: (0, 0, 0)}[radix]
        out = {"base": p["base"], "dir": p["direction"], "radix": radix, "gin": gin, "gout": gout, "bits": bits}
        if key.endswith("#alphabet"):
            out.update({"s": p["alphabet"], "alen": radix, "agent": "encoder" if p["direction"] == "encode" else "decoder",
                        "use": "looks up each output character in" if p["direction"] == "encode" else "looks up each input character in"})
        else:
            out["Dir"] = p["direction"].capitalize()
        return out
    if kind_of(base) == "strobf":
        return {"s": p["plain"], "len": len(p["plain"]), **({"modulus": 26} if v == "rot_n" else {})}
    if base == "multi/decode_secret":
        return {"s": p["secret"], "len": len(p["secret"])}
    if kind_of(base) == "pktsum":
        if p["ask"] == "endian":
            return {"ask": "endian", "e": p["endian"], "first": "low" if p["endian"] == "little" else "high", "kind": v}
        return {"ask": "len" if p["ask"] == "len_offset" else "start", "n": p["L"] if p["ask"] == "len_offset" else p["S"], "kind": v}
    if base in ("bounds/range_chain", "bounds/index_limit", "bounds/clamp"):
        return {"lo": p["lo"], "hi": p["hi"]}
    if base == "bounds/window":
        return {"lo": p["lo"], "hi": p["hi"], "w": p["w"]}
    if base == "bounds/loop_iterations":
        return {"n": f["n"], "n0": p["n0"]}
    if base == "bounds/char_class":
        cover = set()
        for a, b in p["ranges"]:
            cover |= set(range(a, b + 1))
        cover |= {p["single"]} if p.get("single") is not None else set()
        return {"n": len(cover), "total": 256, "top": 255}
    if base in ("antidebug/ptrace", "antidebug/ptrace_raw"):
        return {"tech": "ptrace", "rv": p["rv"]}
    if base == "antidebug/tracerpid":
        return {"tech": "tracerpid", "rv": p["rv"]}
    if base == "antidebug/env":
        return {"tech": "env", "NAME": p["name"]}
    if base in ("antidebug/timing_ns", "antidebug/timing_us"):
        ns = base.endswith("ns")
        return {"tech": "timing", "thr": p["thr"], "unit": "nanoseconds" if ns else "microseconds", "call": "clock_gettime" if ns else "gettimeofday",
                "scale": 10 ** 9 if ns else 10 ** 6}
    if base == "multi/eval_arith" or base == "multi/chain":
        return {"n": f["n"], "n0": p["n0"]}
    if base == "multi/key_helper":
        return {"k": p["key"], "seed": p["seed"]}
    if base == "multi/limit_helper":
        return {"n": p["limit"], "mode": p["mode"]}
    return None                                                                  # legacy: checked against the C source below


def kind_of(key: str) -> str:
    return key.split("/")[0]


def legacy_expected(key: str, meta: dict, body: str) -> dict:
    """The legacy templates keep no params: the facts are read back from the C source of the task."""
    a = meta["accept"]
    if key in ("xor-key", "password", "keycount", "alphabet"):
        s = a[0]
        assert f'"{s}' in body, (key, s)
        out = {"s": s, "len": len(s)}
        if key == "xor-key":
            assert f"% {len(s)}" in body
        if key == "alphabet":
            assert ": -1;" in body
            out["miss"] = -1
        return out
    if key == "magic":
        m, v = int(a[0], 16), int(a[1])
        assert f"0x{m:08X}u" in body and f"ver > {v} " in body
        return {"m": m, "v": v}
    if key == "lcg":
        x, r = int(a[0]), int(a[1])
        assert f"seed * {x}u" in body and f"i < {r};" in body and "0x7fffffffu" in body
        return {"a": x, "r": r, "mask": 0x7fffffff}
    if key == "hash":
        stem, c = a[0][0], int(a[1][0], 16)
        assert f"h ^ 0x{c:08x}u" in body
        init = {"fnv": 2166136261, "djb2": 5381, "sdbm": 0, "murmur": 0x9747b28c}[stem]
        assert {"fnv": "2166136261u", "djb2": "5381u", "sdbm": "h = 0;", "murmur": "0x9747b28cu"}[stem] in body
        return {"name": {"fnv": "FNV-1a", "djb2": "djb2", "sdbm": "sdbm", "murmur": "MurmurHash2"}[stem], "c": c, "init": init}
    if key == "skip":
        b = int(a[0], 16)
        assert f"p[i] != 0x{b:02x}" in body
        return {"b": b}
    if key == "rotate":
        k, r = int(a[0], 16), int(a[1])
        assert f"x ^= 0x{k:08x}u" in body and f"x << {r})" in body
        return {"k": k, "r": r}
    if key == "threshold":
        lo, hi = int(a[0]), int(a[1])
        assert f"v < {lo})" in body and f"v > {hi})" in body and "return -1" in body and "return 1" in body
        return {"lo": lo, "hi": hi, "below": -1, "inside": 0, "above": 1}
    raise KeyError(key)


# What the C source of every task of a variant must hold for the constants and structural words its sentences state.
def body_checks(key: str, f: dict, task) -> list:
    v = task.variant
    out = []
    if key.startswith("cipher/") and v in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec"):
        out += [rf"sum {'\\+=' if v.endswith('enc') else '-='} 0x{f['delta']:08x}u", r"for \(i = 0; i < %d; i\+\+\)" % f["rounds"]]
        if v.startswith("xtea"):
            out += [r"k\[sum & 3\]", r"k\[\(sum >> 11\) & 3\]"]
    if key == "cipher/rc4":
        out += [r"S\[256\]"]
    if key in ("cipher/chacha", "cipher/salsa"):
        out += [r"0x61707865u"] + [rf"<< {r}\b|, {r}\)|\b{r}\b" for r in f["rot"]]
    if key.startswith("crc/"):
        w = f["w"]
        out += [r"table\[256\]"] if v.startswith("table") else [r"for \(int k = 0; k < 8; k\+\+\)"]
        if v.endswith("poly"):
            out += [rf"0x{f['lit']:0{w // 4}x}u", r">> 1" if f["refl"] else r"<< 1"]
        elif v.endswith("init"):
            out += [rf"0x{f['x']:0{w // 4}x}u"]
        else:
            out += [rf"\^ 0x{f['x']:0{w // 4}x}u"]
    if key.startswith("encoding/") and not key.endswith("#alphabet"):
        radix = f["radix"]
        out += {("encode", 64): [r"& 63"], ("encode", 32): [r"& 31"], ("encode", 58): [r"% 58u", r"/=? 58u"],
                ("decode", 64): [r"<< 6", r"'='"], ("decode", 32): [r"<< 5", r"'='"], ("decode", 58): [r"\* 58u"]}[(f["dir"], radix)]
    if key == "structs/init_magic":
        out += [rf"0x{f['magic']:x}"]
    if key == "multi/decode_secret":
        out += [r"strcmp"]
    if key.startswith("antidebug/"):
        out += {"ptrace": [r"ptrace\(PTRACE_TRACEME"], "ptrace_raw": [r"syscall\(SYS_ptrace, PTRACE_TRACEME"], "tracerpid": [r"TracerPid:", r"/proc/self/status"],
                "env": [rf'getenv\("{f["NAME"]}"\)'] if "NAME" in f else [], "timing_ns": [r"clock_gettime", rf"d > {f.get('thr', 0)}LL"],
                "timing_us": [r"gettimeofday", rf"d > {f.get('thr', 0)}LL"]}[v]
    if key == "bounds/char_class":
        out += [r"unsigned char c"]
    if key.startswith("pktsum/"):
        if f["ask"] == "endian":
            out += [r"p\[n - 2\] << 8\) \| p\[n - 1\]|\(p\[n - 2\] \| p\[n - 1\] << 8\)|p\[n - 2\] \| \(?p\[n - 1\]"]
        elif f["ask"] == "start":
            out += [rf"i = {f['n']};"]
        else:
            out += [rf"p\[{f['n']}\]"]
    if key.startswith("strobf/"):
        out += {"stack_string": [r"memcpy"], "xor_const": [r"\^ 0x"], "xor_rolling": [r"i \* \d"], "alt_xor": [r"\(i & 1\)"], "sub_const": [r"enc\[i\] - "],
                "reverse_xor": [r"enc\[n ?- ?1 ?- ?i\]|enc\[\w+ - 1 - i\]"], "rot_n": [rf"% {f.get('modulus', 26)}\b"]}[v]
    return out


def opposite(key: str, f: dict):
    """A regex of what the text must NOT say, given the truth: the other direction, the other endianness, the other shift, in any inflection."""
    if key.startswith("cipher/") and key.split("/")[1] in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec"):
        return r"decrypt|decipher|subtract" if f["dir"] == "encrypt" else r"encrypt|encipher|\badds?\b"
    if key.startswith("encoding/"):
        return r"decod" if f["dir"] == "encode" else r"encod"
    if key.startswith("pktsum/") and f.get("ask") == "endian":
        return r"\bbig\b|\bhigh\b" if f["e"] == "little" else r"\blittle\b|\blow\b"
    if key.startswith("crc/") and "dir" in f:
        return r"\bleft\b|most significant|\bnormal\b|\bmsb\b" if f["refl"] else r"\bright\b|least significant|reflected|\blsb\b"
    return None


class Corpus:
    """Everything the checks need that comes from the family as a whole: its vocabulary of names, per kind and per variant."""

    def __init__(self):
        self.vocab = collections.defaultdict(set)         # kind -> words of every accepted and rejected name of its token tasks
        self.own = collections.defaultdict(set)           # (kind, variant) -> words of every accepted name of that variant


def collect(tasks) -> Corpus:
    c = Corpus()
    for t in tasks:
        m = t.meta()
        if m["score"] != "tokens":
            continue
        for g in m["accept"]:
            for lit in g:
                if grammar.num_value(lit.lower()) is None:
                    c.vocab[t.kind] |= words_in(lit)
                    c.own[(t.kind, t.variant)] |= words_in(lit)
        for lit in m.get("reject", []):
            c.vocab[t.kind] |= words_in(lit)
    return c


def problems(task, r: dict, corpus: Corpus) -> list:
    """Everything wrong with one rendered target `r` of `task`; [] when it is true."""
    meta, key, f, text = task.meta(), r["key"], r["facts"], r["text"]
    out = []
    # 1. truth: the stored text is an answer to the task under both scorers, a headline line and one prose sentence
    if not scoring.score_strict(text, meta):
        out.append("score_strict rejects the text")
    if not scoring.score(text, meta):
        out.append("score rejects the text")
    lines = text.split("\n")
    if len(lines) != 2 or lines[0] != r["headline"] or lines[1] != r["sentence"]:
        out.append("not headline + one sentence")
    # 2. facts against the generator's own parameters
    base = key.split("#")[0]
    if kind_of(base) == "legacy" or base in families.modules()["legacy"].VARIANTS or task.kind in ("xor-key", "password", "magic", "lcg", "hash", "skip", "keycount", "alphabet", "rotate", "threshold"):
        try:
            want = legacy_expected(key, meta, task.body)
        except AssertionError as e:
            out.append(f"legacy fact not in the C source: {e}")
            want = None
    else:
        want = expected(key, task, meta, f)
    if want is None:
        out.append("no expected facts for this variant")
    elif want != f:
        out.append(f"facts differ from the parameters: {sorted(set(f.items()) ^ set(want.items()), key=str)[:6]}")
    # 3. the constants and structural words in the C source of this very task
    for rx in body_checks(key, f, task):
        if not re.search(rx, task.body):
            out.append(f"not in the C source: {rx}")
    if key.startswith("cipher/") and key.split("/")[1] in ("chacha", "salsa"):
        if len(re.findall(r"QR\(", task.body)) - 1 != 8 and len(re.findall(r"QR\(", task.body)) != 8:
            out.append("the loop body does not hold eight quarter-rounds")
    if key == "multi/chain" and len(task.helpers) != 2:
        out.append("chain: not two helpers")
    if key.startswith("pktsum/") and f["ask"] == "endian" and task.params["W"] != 2:
        out.append("pktsum: the checksum is not two bytes")
    # 4. numbers: facts, numbers of the question, counted words that this variant justifies
    sentence = r["sentence"]
    for val in f.values():
        if isinstance(val, str) and val:
            sentence = re.sub(re.escape(val), " ", sentence, flags=re.I)
    allowed = set(_flat(list(f.values()))) | numbers_in(meta["question"]) | {grammar._NUMWORD[w] for w in WORDS_OK.get(base, ())}
    stray = numbers_in(sentence) - allowed
    if stray:
        out.append(f"a number that is no fact of the task: {sorted(stray)}")
    # 5. names and opposites
    if task.kind not in OPEN_EXACT and task.kind in corpus.vocab:
        own = corpus.own[(task.kind, task.variant)] | set().union(*(words_in(a) for g in meta["accept"] for a in g if isinstance(a, str)))
        foreign = (words_in(r["sentence"]) & corpus.vocab[task.kind]) - own
        if foreign:
            out.append(f"names of other answers: {sorted(foreign)}")
    rx = opposite(key, f)
    if rx and re.search(rx, text, re.I):
        out.append(f"says the opposite of the truth: {re.search(rx, text, re.I).group(0)}")
    return out


def render_all(fam: str, n: int = PER_FAMILY, seed: int = SEED) -> tuple:
    tasks = families.generate(fam, n, seed)
    return tasks, collect(tasks)


class RenderedTargets(unittest.TestCase):
    def test_every_rendered_target_is_true(self):
        """2,000 tasks per family, all eleven families: every rendered text is an answer under both scorers, every fact equals the generator's
        parameter, every constant is in the task's C source, every number and name in the sentence is the task's own."""
        total = collections.Counter()
        for fam in families.modules():
            tasks, corpus = render_all(fam)
            bad, used = [], collections.defaultdict(set)
            for t in tasks:
                r = targets.render(t.meta())
                p = problems(t, r, corpus)
                if p:
                    bad.append((t.id, r["key"], p[:2], r["text"][:160]))
                used[(r["key"], r["facts"].get("ask"))].add(r["template"])
                self.assertEqual(r["templates"], len(set(range(r["templates"]))))
                total[fam] += 1
            self.assertEqual(bad[:3], [], f"{fam}: {len(bad)} of {len(tasks)} rendered targets are not true")
            for (key, ask), idx in used.items():
                n = max(idx) + 1
                self.assertGreaterEqual(n, 3, f"{key}: fewer than 3 phrasings in use")
                self.assertLessEqual(n, 6, key)
                self.assertEqual(idx, set(range(n)), f"{key} {ask}: a phrasing is never rendered")
            self.assertEqual(total[fam], PER_FAMILY if fam != "legacy" else PER_FAMILY)
        self.assertEqual(set(total), set(families.modules()))

    def test_every_variant_has_a_template_and_the_set_is_3_to_6(self):
        seen = set()
        for fam, mod in families.modules().items():
            for t in families.generate(fam, 12 * max(len(mod.VARIANTS), 10), 5):
                m = t.meta()
                self.assertTrue(targets.known(m), (t.id, targets.key_of(m)))
                seen.add(targets.key_of(m))
        for key in seen:
            s = targets.SPECS[key]
            self.assertIn(len(s.sentences), range(3, 7 if not key.startswith("pktsum/") else 10), key)
            self.assertGreaterEqual(len(s.heads), 3, key)
        self.assertGreaterEqual(len(seen), 70)

    def test_the_same_task_always_renders_the_same_text_and_the_teacher_plays_no_part(self):
        t = next(t for t in families.generate("cipher", 60, 7) if t.variant == "xtea_dec").meta()
        first = targets.render(t)
        for _ in range(3):
            self.assertEqual(targets.render(t), first)
        self.assertEqual(targets.render({**t, "pseudocode": "int x;", "refs": "abc", "chance": 0.9}), first, "fields of the sample stage do not change the rendering")
        ids = {targets.render({**t, "id": f"cipher-7-{i:04d}"})["template"] for i in range(60)}
        self.assertEqual(ids, set(range(first["templates"])), "the seed of the task id spreads the phrasings of a variant")

    def test_an_unknown_variant_is_an_error_not_a_free_text(self):
        with self.assertRaises(ValueError):
            targets.render({"id": "x-1", "kind": "cipher", "variant": "aes", "accept": [["aes"]], "question": "?"})
        with self.assertRaises(ValueError):
            targets.render({"id": "x-1", "kind": "nope", "accept": [["1"]], "question": "?"})


class Teeth(unittest.TestCase):
    """The same `problems` check must fail on a rendering that is wrong in the ways the reviews found."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = {}
        for fam in ("cipher", "bounds", "encoding", "crc", "pktsum", "strobf", "legacy", "structs"):
            ts, corpus = render_all(fam, 400, 99)
            cls.tasks[fam] = (ts, corpus)

    def pick(self, fam, variant):
        ts, corpus = self.tasks[fam]
        t = next(t for t in ts if t.variant == variant or t.kind == variant)
        return t, corpus, targets.render(t.meta())

    def test_the_honest_rendering_is_clean(self):
        for fam, v in (("cipher", "xtea_dec"), ("bounds", "range_chain"), ("encoding", "base64_encode"), ("crc", "bitwise_poly"), ("pktsum", "sum16_le"),
                       ("strobf", "xor_const"), ("legacy", "lcg"), ("structs", "field_read")):
            t, corpus, r = self.pick(fam, v)
            self.assertEqual(problems(t, r, corpus), [], (v, r["text"]))

    def test_a_wrong_round_count_a_wrong_delta_and_a_wrong_number_are_found(self):
        t, corpus, r = self.pick("cipher", "xtea_dec")
        n = r["facts"]["rounds"]
        wrong = {**r, "facts": {**r["facts"], "rounds": n + 1}}
        self.assertTrue(any("facts differ" in p for p in problems(t, wrong, corpus)))
        wrong = {**r, "facts": {**r["facts"], "delta": 0x9E3779B8}}
        self.assertTrue(any("facts differ" in p or "C source" in p for p in problems(t, wrong, corpus)))
        wrong = {**r, "sentence": f"The main loop runs {n + 1} iterations.", "text": f"{r['headline']}\nThe main loop runs {n + 1} iterations."}
        self.assertTrue(any("number" in p for p in problems(t, wrong, corpus)))
        wrong = {**r, "sentence": f"The main loop runs {n} iterations and 7 key words.", "text": f"{r['headline']}\nThe main loop runs {n} iterations and 7 key words."}
        self.assertTrue(any("number" in p for p in problems(t, wrong, corpus)))

    def test_the_other_direction_is_found_in_any_inflection(self):
        """The review's B2 sentences, after a right headline: each one is a problem of the rendering check."""
        t, corpus, r = self.pick("cipher", "xtea_dec")
        for s in ("It encrypted the data.", "It is the encryption routine.", "It runs the forward cipher and encrypts.", "The loop adds the delta to the sum in every iteration."):
            wrong = {**r, "sentence": s, "text": f"{r['headline']}\n{s}"}
            self.assertTrue(any("opposite" in p for p in problems(t, wrong, corpus)) or s == "It runs the forward cipher and encrypts.", s)
        t, corpus, r = self.pick("encoding", "base64_encode")
        t = next(t for t in self.tasks["encoding"][0] if t.variant == "base64_encode" and t.truth["score"] == "tokens")
        r = targets.render(t.meta())
        for s in ("It decoded the input.", "The output is the decoded buffer.", "It is the decoding step."):
            wrong = {**r, "sentence": s, "text": f"{r['headline']}\n{s}"}
            self.assertTrue(any("opposite" in p for p in problems(t, wrong, corpus)), s)
        t = next(t for t in self.tasks["pktsum"][0] if t.variant == "sum16_le" and t.params["ask"] == "endian")
        r = targets.render(t.meta())
        for s in ("The checksum is stored big-endian.", "The bytes are stored in big-endian order with the high byte first."):
            wrong = {**r, "sentence": s, "text": f"{r['headline']}\n{s}"}
            self.assertTrue(any("opposite" in p for p in problems(t, wrong, corpus)), s)
        t = next(t for t in self.tasks["crc"][0] if t.variant == "bitwise_poly" and t.params["refl"])
        r = targets.render(t.meta())
        for s in ("The algorithm shifts left, MSB first.", "It processes the most significant bit first."):
            wrong = {**r, "sentence": s, "text": f"{r['headline']}\n{s}"}
            self.assertTrue(any("opposite" in p for p in problems(t, wrong, corpus)), s)

    def test_swapped_bounds_and_a_foreign_name_are_found(self):
        t, corpus, r = self.pick("bounds", "range_chain")
        f = r["facts"]
        wrong = {**r, "facts": {"lo": f["hi"], "hi": f["lo"]}}
        self.assertTrue(any("facts differ" in p for p in problems(t, wrong, corpus)))
        t, corpus, r = self.pick("cipher", "xtea_enc")
        s = "It is the TEA cipher with a key schedule."
        wrong = {**r, "sentence": s, "text": f"{r['headline']}\n{s}"}
        self.assertTrue(any("names of other answers" in p for p in problems(t, wrong, corpus)))

    def test_a_template_that_lies_about_the_task_is_found(self):
        """A fake spec that states the CRC width of another task, or the other shift direction, is caught by the facts and the opposite rule."""
        t, corpus, r = self.pick("crc", "bitwise_poly")
        w = r["facts"]["w"]
        wrong = {**r, "facts": {**r["facts"], "w": 48 - w}}
        self.assertTrue(any("facts differ" in p for p in problems(t, wrong, corpus)))
        flipped = "left, most significant bit first" if r["facts"]["refl"] else "right, least significant bit first"
        wrong = {**r, "facts": {**r["facts"], "shift": flipped}, "sentence": f"The loop shifts the register {flipped}.", "text": f"{r['headline']}\nThe loop shifts the register {flipped}."}
        self.assertTrue(len(problems(t, wrong, corpus)) >= 2)


class SelectionStoresTheTemplate(unittest.TestCase):
    """What sft stores: the rendered text, the raw teacher headline for analysis, `truth-template`; no sentence of the teacher is ever stored."""

    def test_the_review_sentences_are_never_stored_and_the_stored_text_is_the_template(self):
        t = next(t for t in families.generate("cipher", 60, 7) if t.variant == "xtea_dec").meta()
        n = t["accept"][2][0]
        want = targets.render(t)["text"]
        poison = ("It encrypted the data.", "It is the encryption routine.", "It runs the forward cipher.", "It decrypts the block with a 128-bit key and runs 128 rounds.",
                  "The block is 128 bits wide.", "Wait, actually it is TEA.")
        t = {**t, "chance": 0.01}                              # below the agreement threshold: one accepted sample is enough
        for s in poison:
            recs, why = sft.select([(f"XTEA, decrypt, {n}\n{s}", True)], t)
            self.assertEqual(why, "kept", s)
            self.assertEqual([r["text"] for r in recs], [want], s)
            self.assertEqual((recs[0]["target_source"], recs[0]["teacher_headline"]), ("truth-template", f"XTEA, decrypt, {n}"))
            self.assertNotIn("sentence_dropped", recs[0])
            self.assertNotIn(s, recs[0]["text"])
        self.assertEqual(sft.select([("TEA, decrypt, 99\nIt decrypts.", True)], t), ([], "no_correct"))

    def test_every_accepted_reply_of_a_task_gives_one_record(self):
        t = next(t for t in families.generate("cipher", 60, 7) if t.variant == "xtea_dec").meta()
        n = t["accept"][2][0]
        replies = [(f"XTEA, decrypt, {n}", True), (f"decrypt, XTEA, {n} rounds\nIt runs the loop.", True), (f"The answer is XTEA decrypt {n}.", True), ("TEA, encrypt, 3", True)]
        recs, why = sft.select(replies, t)
        self.assertEqual((why, len(recs)), ("kept", 1), "three accepted samples agree; one record per task")
        self.assertEqual(recs[0]["teacher_headline"], f"XTEA, decrypt, {n}", "the first accepted reply's headline, raw")

    def test_assemble_stores_the_rendered_text_only(self):
        t = next(t for t in families.generate("structs", 60, 7) if t.variant == "linked_next")
        meta = {**t.meta(), "chance": 0.01}
        off = meta["accept"][0][0]
        results = {meta["id"]: {"outs": [[f"{off}\nThe pointer is at 0x70, not at 0x8 in the struct.", True], ["[error TimeoutError('x')]", False]]}}
        kept, st = sft.assemble([meta], results, "m", lambda task: [{"role": "user", "content": "q"}])
        self.assertEqual(len(kept), 1)
        rec = kept[0]
        self.assertEqual(rec["messages"][-1]["content"], targets.render(meta)["text"])
        self.assertEqual((rec["target_source"], rec["teacher_headline"]), ("truth-template", off))
        self.assertTrue(scoring.score_strict(rec["messages"][-1]["content"], meta))
        self.assertNotIn("sentence_dropped", rec)
        self.assertNotIn("sentence_dropped", st)
        self.assertEqual((st["errors"], st["kept_records"]), (1, 1))


if __name__ == "__main__":
    unittest.main()

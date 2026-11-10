"""The reviewer's scorer attacks (round 2: repro_scorer.py, s_attack.py, s_attack2.py), ported to run on generated tasks.

`matrix(task_dicts)` returns {(name, expect): [passed, total]} plus one example per row that went the wrong way:
  expect False    a WRONG reply: must be rejected (candidate lists, `X or Y`, negations, supersets, a lead-in over a list, ...)
  expect True     a RIGHT reply in a realistic format: must be accepted (the 28 formats of the review, see RIGHT_FORMATS)
  expect "yield"  a RIGHT reply in a format the whitelist grammar of round 3 does not parse: it must be rejected. These rows are the yield
                  cost of the design ("yield cost is acceptable, poison is not"), pinned so the list in the README stays exact.
Rows that the review marked "info" (policy questions) are not counted either way. Where a round changed the policy the old expectation
is flipped on purpose: a key followed by a parenthesis, spaced-out alphabets, upper-cased plaintexts and `X or Y` over two accepted answers
(round 2); a number followed by `approximately` (a hedge), a unit the question did not use, a name group written in another order than the
question asked, a second line that is not a sentence (round 3).

    python3 -m tests.rea_attack_matrix [seed] [per-variant]        the numbers of the final report, with the right-format pass rate per family
"""
from __future__ import annotations

import collections
import itertools
import re
import sys

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families  # noqa: E402
from rea_bench.scoring import num_value, score, strict_task  # noqa: E402

NEW_FAMILIES = [f for f in families.modules() if f != "legacy"]
HEADLINE_MODES = ("tokens", "exact_case", "exact_head")

CHANGED_BY_DESIGN = ["eh key + more (a parenthesis after the key)", "alpha spaced chars", "eh uppercase", "fsm two accepted (X or Y)"]

# The 28 right-answer formats of the review (s_attack "R ..." rows: 12 generic + 11 single-number + 5 two-bound), 27 of them here: "approximately N"
# is a hedge, not an answer, and is a wrong row since round 3 ("W approx"). Five more are yield-class (a non-sentence second line, a bullet before a digit).
RIGHT_FORMATS = ["R bare", "R md-bold", "R backtick", "R answer-colon", "R answer-colon-nl", "R sentence", "R +just-line2", "R heading",
                 "R bullet", "R trailing-dot", "R blank-first", "R parens",
                 "R hex-dec-both", "R dec-in-hex-paren", "R zero-padded", "R u-suffix", "R ull-suffix hex", "R bytes", "R offset-word",
                 "R it-is", "R commas", "R trailing-colon",
                 "R lo-to-hi", "R lo-and-hi", "R lo hi interval", "R lo..hi", "R lo-hi hyphen"]


def tasks_for(seed: int, per_variant: int, fams=None, gate=None) -> list:
    out = []
    for fam in fams or NEW_FAMILIES:
        m = families.modules()[fam]
        out += [t.meta() | {"_truth": t.truth} for t in families.generate(fam, per_variant * len(m.VARIANTS), seed, "mix", gate)]
    return out


def legacy_tasks(seed: int, n: int) -> list:
    """The ten legacy templates as the data path scores them (`strict_task`): grammar groups and exact strings, kind kept."""
    return [strict_task(t.meta()) for t in families.generate("legacy", n, seed)]


def _nums(t):
    out = []
    for g in t["accept"]:
        v = [num_value(a.lower()) for a in g]
        v = [x for x in v if x is not None]
        out.append(v[0] if v else None)
    return out


def _bare(t):
    return ", ".join(g[0] for g in t["accept"]) if t["score"] == "tokens" else t["accept"][0]


def _other(v):
    return v + 1 if v < 10 ** 9 else v - 1


def generic_attacks(t) -> list:
    """s_attack.py: (name, reply, expect) for every headline-mode task."""
    A, mode, k = _bare(t), t["score"], _nums(t)
    numeric = [x for x in k if x is not None]
    interval = t.get("kind") in ("bounds", "threshold") and len(k) == 2 and None not in k
    out = [("R bare", A, True), ("R md-bold", f"**{A}**", True), ("R backtick", f"`{A}`", True), ("R answer-colon", f"Answer: {A}", True),
           ("R answer-colon-nl", f"Answer:\n{A}\nbecause of the loop", True),
           ("R sentence", f"The answer is {A}.", True),
           ("R +just-line2", f"{A}\nThe constant 1 and 2 and 3 appear in the loop at offset 4.", True), ("R heading", f"## {A}\njustification 7", "yield"),
           ("R bullet", f"- {A}", "yield" if re.match(r"[\d-]", A) else True), ("R trailing-dot", f"{A}.", True),
           ("R blank-first", f"\n\n{A}\nbecause", "yield"),
           (("W interval-parens" if interval else "R parens"), f"({A})", False if interval else True)]
    if mode == "tokens" and numeric and len(k) == 1:
        v = numeric[0]
        out += [("R hex-dec-both", f"0x{v:x} ({v})" if v >= 0 else f"{v}", True), ("R dec-in-hex-paren", f"{v} (0x{v:x})" if v >= 0 else f"{v}", True),
                ("R zero-padded", f"0x{v:08x}" if v >= 0 else f"{v}", True), ("R u-suffix", f"{v}u" if v >= 0 else f"{v}", True),
                ("R ull-suffix hex", f"0x{v:x}ULL" if v >= 0 else f"{v}", True), ("R offset-word", f"Offset {v}", True),
                ("R it-is", f"It is {v}", True), ("W approx", f"approximately {v}", False), ("R commas", f"{v:,}" if v >= 1000 else f"{v}", True),
                ("R trailing-colon", f"{v}:", True)]
        if t.get("kind") in ("structs", "pktsum") or t.get("variant") in ("rc4", "limit_helper"):
            out.append(("R bytes", f"{v} bytes", True))               # a unit the question uses; `5 bytes` for a state count is a wrong unit
    if mode == "tokens" and numeric:
        v = numeric[0]

        def with_wrong(idx):
            return ", ".join(str(_other(k[gi])) if gi == idx and k[gi] is not None else g[0] for gi, g in enumerate(t["accept"]))
        out += [(f"W off-by-one g{gi}", with_wrong(gi), False) for gi in range(len(k)) if k[gi] is not None]
        cl = ", ".join(str(x) for x in sorted({v - 1, v, v + 1, v * 2, v + 8, 0, 1, 2}))
        out += [("W cand-list-numeric", cl, False), ("W cand-list-line1+line2", f"{cl}\n{A}", False), ("W X-or-Y", f"{A} or {v + 1}", False),
                ("W X-or-Y rev", f"{v + 1} or {A}", False), ("W either", f"either {v} or {v + 2}", False),
                ("W negation not-K-it-is-W", f"not {v}, it is {v + 1}", False), ("W negation it-is-K-not-W", f"it is {v}, not {v + 1}", False),
                ("W junk-line1 wrong, right line2", f"{v + 1}\n{A}", False), ("W wrong, right in justification", f"{v + 1}\nActually {A}", False),
                ("W range", f"{v - 2}-{v + 2}", False), ("W range-to", f"between {v - 1} and {v + 1}", False), ("W not-K only", f"not {v}", False),
                ("W K-in-expr", f"{v + 3} - {v + 1} = 2", False), ("W sum", f"{v} + {v}", False)]
        if len(numeric) == 2:
            lo, hi = numeric
            out += [("W swapped-order", f"{hi}, {lo}", False), ("W only-lo", f"{lo}", False), ("W only-hi", f"{hi}", False)]
            if interval:
                out += [("R lo-to-hi", f"{lo} to {hi}", True), ("R lo-and-hi", f"lowest {lo}, highest {hi}", True), ("R lo hi interval", f"[{lo}, {hi}]", True),
                        ("R lo..hi", f"{lo}..{hi}", True)]
                if lo >= 0:
                    out.append(("R lo-hi hyphen", f"{lo}-{hi}", True))
            else:
                out.append(("R two numbers", f"{lo}, {hi}", True))
    return out


def repro_attacks(t) -> list:
    """repro_scorer.py: the round-2 blockers, per task shape."""
    out, mode = [], t["score"]
    if mode == "exact_head":
        k = t["accept"][0]
        out += [("B1 key or key-with-x", f"{k} or {k[:-1]}x", False), ("B1 key-with-x or key", f"{k[:-1]}x or {k}", False), ("B1 not key", f"not {k}", False),
                ("B1 key+7", f"{k}7", False), ("B1 x+key", f"x{k}", False), ("B1 path", f"/etc/{k}", False), ("B1 UPPER", k.upper(), False),
                ("B1 wrong then right", f"{k[:-1]}x\n{k}", False), ("B1 spaces removed", k.replace(" ", ""), False if " " in k else None)]
    if mode == "exact_case":
        a = t["accept"][0]
        sw = a[1] + a[0] + a[2:]
        out += [("B1 two alphabets", f"{a} or {sw}", False), ("B1 superstring", f"x{a}y", False), ("B1 swapped", sw, False),
                ("B1 wrong then right", f"{sw}\n{a}", False), ("B1 swapcase", a.swapcase(), False), ("B1 rotated", a[1:] + a[0], False),
                ("B1 truncated", a[:-1], False), ("B1 reversed", a[::-1], False), ("B1 sorted", "".join(sorted(a)), False),
                ("R alphabet quoted", f'"{a}"', True), ("R Alphabet:", f"Alphabet: {a}", True), ("R alphabet md code", f"`{a}`", True),
                ("R alphabet prefix text", f"The alphabet is {a}", True), ("R alphabet trailing dot", a + ".", True)]
    return out


def family_attacks(t) -> list:
    """s_attack2.py: per-family shapes (names, directions, rounds, techniques, env variables, fsm strings, plaintexts)."""
    out, kind, mode, acc = [], t["kind"], t["score"], t["accept"]
    v = t.get("variant", "")
    if kind == "cipher":
        alg = {"rc4": "RC4", "tea_enc": "TEA", "tea_dec": "TEA", "xtea_enc": "XTEA", "xtea_dec": "XTEA", "chacha": "ChaCha20", "salsa": "Salsa20"}[v]
        num = [g for g in acc if g and g[0].isdigit()]
        rounds = num[0][0]
        d = "decrypt" if v.endswith("dec") else "encrypt"
        dd = "decryption" if d == "decrypt" else "encryption"
        tea = v in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec")
        if tea:
            right = [f"{alg}, {d}, {rounds}", f"{alg} {dd}, {rounds} iterations", f"It's {alg} ({d}), {rounds} rounds.", f"**{alg}**, {d}, {rounds} rounds",
                     f"{alg}: {d}, {rounds}", f"The function is {alg} {dd} running {rounds} rounds.", f"{alg} {d}s with {rounds} iterations of the main loop",
                     f"{alg}, {d}, {rounds} rounds (128-bit key)"]
            yields = [f"{alg} {d}, {rounds} iterations (a 128-bit key, 64-bit blocks)"]          # 64-bit blocks: the question says 8-byte
        elif v == "rc4":
            right = [f"{alg}, {rounds}", f"{alg}: {rounds} bytes", f"It's {alg}, key length {rounds}.", f"**{alg}**, {rounds}",
                     f"{alg}, {rounds} bytes ({int(rounds) * 8} bits)"]
            yields = [f"{alg} with a {rounds}-byte key"]                                          # `N-byte` is a context form, not a hit
        else:
            right = [f"{alg}, {rounds}", f"{alg}: {rounds} rounds", f"It's {alg}, {rounds} rounds.", f"**{alg}**, {rounds}", f"{alg}, {rounds} rounds (256-bit key)"]
            yields = []
        out += [("cipher right", r, True) for r in right] + [("cipher right (yield)", r, "yield") for r in yields]
        if v in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec"):
            other = "decrypt" if d == "encrypt" else "encrypt"
            twin = "TEA" if alg == "XTEA" else "XTEA"
            out += [("tea wrong-direction", f"{alg}, {other}, {rounds}", False), ("tea wrong-direction+ok in line2", f"{alg}, {other}, {rounds}\nit {d}s", False),
                    ("tea both dirs", f"{alg}, encrypt/decrypt, {rounds}", False), ("tea both dirs2", f"{alg}, encrypts and decrypts, {rounds}", False),
                    ("tea 'not <dir>'", f"{alg}, not {other}, {rounds}", False), ("tea not-right-dir", f"{alg}, not {d}, {rounds}", False),
                    ("tea direction word absent", f"{alg}, {rounds}", False), ("tea wrong alg other (no neg)", f"{twin}, {d}, {rounds}", False),
                    ("tea alg negation (wrong alg with 'not key')", f"{twin} (not {alg}), {d}, {rounds}", False),
                    ("tea 'X, not Y' wrong", f"{twin}, not {alg}, {d}, {rounds}", False), ("tea XTEA or TEA", f"{alg} or {twin}, {d}, {rounds}", False),
                    ("tea XTEA, TEA (list)", f"{alg}, {twin}, {d}, {rounds}", False), ("tea TEA, XTEA (list, key last)", f"{twin}, {alg}, {d}, {rounds}", False),
                    ("tea (eXtended TEA)", f"{alg}{' (eXtended TEA)' if alg == 'XTEA' else ''}, {d}, {rounds}", True),
                    ("tea 'Tiny Encryption Algorithm'", f"TEA (Tiny Encryption Algorithm), {d}, {rounds}" if alg == "TEA" else f"XTEA, {d}, {rounds}", True),
                    ("tea plain 'TEA' for XTEA key", f"TEA, {d}, {rounds}" if alg == "XTEA" else f"XTEA, {d}, {rounds}", False),
                    ("tea words: Block TEA", f"{alg} {d}, {rounds}, a block cipher", True),
                    ("tea (TEA not XTEA) mis-identification", f"TEA (not XTEA), {d}, {rounds}", False if alg == "XTEA" else None),
                    ("tea key negated, number negated", f"{alg}, {d}, not {rounds}", False)]
        if v in ("chacha", "salsa"):
            other_alg = "Salsa20" if alg == "ChaCha20" else "ChaCha20"
            x_alg = "XSalsa20" if alg == "Salsa20" else "XChaCha20"
            out += [("stream wrong alg", f"{other_alg}, {rounds}", False), ("stream both algs", f"{alg} or {other_alg}, {rounds}", False),
                    ("stream X-variant or", f"{alg} or {x_alg}, {rounds}", False), ("stream X-variant list", f"{alg}, {x_alg}, {rounds}", False),
                    ("stream alg w/ rounds off-by-one", f"{alg}, {int(rounds) + 2}", False), ("stream name+digits no rounds", f"{alg}", False),
                    ("stream 'ChaCha-12 style name digits only'", f"{alg.rstrip('0123456789')}, {rounds}", True)]
        if v == "rc4":
            out += [("rc4 klen+-1", f"RC4, {int(rounds) + 1}", False), ("rc4 'arcfour'", f"ARCFOUR, {rounds} bytes", True),
                    ("rc4 bits not bytes", f"RC4, {int(rounds) * 8} bits", False), ("rc4 'RC4/ARC4' (same)", f"RC4/ARC4, {rounds}", True),
                    ("rc4 key length text", f"RC4, key length {rounds} bytes", True), ("rc4 RC4 vs RC5", f"RC5, {rounds}", False),
                    ("rc4 RC4 or AES", f"RC4 or AES, {rounds}", False), ("rc4 RC4 or Camellia", f"RC4 or Camellia, {rounds}", False),
                    ("rc4 bytes and bits", f"RC4, {rounds} bytes ({int(rounds) * 8} bits)", True)]
    elif kind == "encoding" and mode == "tokens":
        base, direction = v.split("_")
        opp = "decode" if direction == "encode" else "encode"
        other_base = "base32" if base != "base32" else "base64"
        right = [f"It {direction}s {base}", f"{direction.capitalize()}s {base}", f"{direction}, {base}"]
        yields = [f"{base.capitalize()} {direction}", f"{base} {direction}r", f"{base}-{direction}", f"It {direction}s data in {base}.",
                  f"Base-{re.sub('[^0-9]', '', base)} {direction}"]                              # base first, or a hyphen word, or a word the grammar does not know
        out += [("enc right", r, True) for r in right] + [("enc right (reversed order, yield)", r, "yield") for r in yields if not r.endswith("decoder") or direction == "decode"]
        out += [("enc wrong dir", f"{base} {opp}", False), ("enc wrong dir sentence", f"It {opp}s {base}", False), ("enc wrong base", f"{other_base} {direction}", False),
                ("enc both dirs", f"{base} encode/decode", False), ("enc 'not <dir>'", f"{base}, not {direction}", False), ("enc dir absent", f"{base}", False),
                ("enc base absent", f"{direction}", False), ("enc ascii85", f"ascii85 {direction}", False), ("enc base16", f"base16 {direction}", False),
                ("enc or other base", f"{base} {direction} or base36 {direction}", False), ("enc negated other base", f"{base} {direction} (not base45)", False),
                ("enc list of bases", f"{base}, {other_base} {direction}", False), ("enc base36 list", f"{base}, base36 {direction}", False),
                ("enc 'base64, not encode'", f"{base}, not {direction}", False)]
    elif kind == "antidebug":
        tech, det = acc[0][0], acc[1][0]
        out += [("ad right", r, True) for r in (f"{tech}, {det}", f"{tech} ({det})", f"**{tech}**: {det}", f"It's a {tech} check; {det}",
                                                f"{tech.upper()}, {det.upper()}" if not det.isdigit() else f"{tech.upper()}, {det}")]
        oth = {"ptrace": "timing", "timing": "ptrace", "env": "ptrace", "tracerpid": "env"}.get(tech, "timing")
        out += [("ad wrong tech", f"{oth}, {det}", False), ("ad both techs", f"{tech}/{oth}, {det}", False),
                ("ad tech or tech", f"{tech} or tracerpid, {det}" if tech != "tracerpid" else f"{tech} or ptrace, {det}", False)]
        if det.isdigit():
            n = int(det)
            out += [("ad wrong rv", f"{tech}, {n + 1}", False), ("ad rv list", f"{tech}, {n - 1} {n} {n + 1}", False), ("ad rv or", f"{tech}, {det} or {n + 1}", False),
                    ("ad hex rv", f"{tech}, 0x{n:x}", True), ("ad not rv", f"{tech}, not {det}", False)]
            if v.startswith("timing"):
                out += [("ad sep commas", f"{tech}, {n:,}", True), ("ad underscores", f"{tech}, {n:_}", True),
                        ("ad with unit text", f"{tech}, {det} {'ns' if v.endswith('ns') else 'us'}", True)]
        else:
            out += [("ad env other var", "env, LD_PRELOAD" if det != "ld_preload" else "env, LD_AUDIT", False),
                    ("ad env two vars", f"env, {det.upper()} or LD_PRELOAD" if det != "ld_preload" else f"env, {det.upper()} or LD_AUDIT", False),
                    ("ad env getenv", f'env, getenv("{det.upper()}")', True), ("ad env quoted", f'env, "{det.upper()}"', True), ("ad env lower", f"env, {det}", True),
                    ("ad env two vars list", f"env, {det.upper()}, LD_PRELOAD" if det != "ld_preload" else f"env, {det.upper()}, LD_AUDIT", False),
                    ("ad env only tech", "env", False), ("ad env name+suffix", f"env, {det.upper()}_X", False)]
    if mode == "exact_head":
        p = acc[0]
        out += [("eh right", p, True), ("eh quoted", f'"{p}"', True), ("eh code", f"`{p}`", True), ("eh 'It writes X'", f"It writes {p}", True),
                ("eh 'The string is X.'", f"The string is {p}.", True), ("eh 'Plaintext: X'", f"Plaintext: {p}", True),
                ("eh uppercase", p.upper(), False), ("eh key+suffix", p + "x", False), ("eh key+digit", p + "7", False), ("eh prefix", "x" + p, False),
                ("eh wrong last char", p[:-1] + ("x" if p[-1] != "x" else "y"), False), ("eh negated", f"not {p}", False), ("eh wrong then right", f"{p[:-1]}x\n{p}", False),
                ("eh key + more (a parenthesis after the key)", f"{p} (and more text)", False), ("eh path-embed", f"/etc/{p}", False),
                ("eh lowercase-if-case", p.lower(), True if p == p.lower() else False), ("eh two keys", f"{p} and {p}", False)]
    if kind == "fsm" and "cand" in t and mode == "tokens":
        good = acc[0]
        g = good[0]
        al = sorted(set("".join(good)))
        out += [("fsm right", g, True), ("fsm right spaced", " ".join(g), True), ("fsm right comma", ", ".join(g), True), ("fsm right quoted", f'"{g}"', True),
                ("fsm right sentence", f"The input is {g}.", True), ("fsm 'Input: bcd'", f"Input: {g}", True)]
        if len(good) > 1:
            out.append(("fsm two accepted (X or Y)", f"{good[0]} or {good[1]}", False))
        bad = [w for w in ("".join(p) for p in itertools.product(al, repeat=len(g))) if w not in good]
        if bad:
            out += [("fsm wrong string", bad[0], False), ("fsm right+wrong", f"{g} or {bad[0]}", False), ("fsm wrong, right in line2", f"{bad[0]}\n{g}", False),
                    ("fsm right, wrong (list)", f"{g}, {bad[0]}", False)]
        out += [("fsm shorter string", g[:-1], False), ("fsm longer string containing key", g + al[0], False), ("fsm prefix + key", "x" + g, False),
                ("fsm key letters inside a word", f"{g}ing", False)]
    return out


LEGACY_KINDS = ("xor-key", "password", "magic", "lcg", "hash", "skip", "keycount", "alphabet", "rotate", "threshold")


def legacy_attacks(t) -> list:
    """Right and wrong replies for the ten legacy templates as the data path scores them (the task is a `strict_task`)."""
    kind, acc = t["kind"], t["accept"]
    if kind in ("xor-key", "password", "keycount", "alphabet"):
        k = acc[0]
        return [("legacy right", r, True) for r in (k, f"The key is {k}", f"`{k}`", f'"{k}"', f"{k}.")] + [
            ("legacy wrong exact", r, False) for r in (k[:-1], k + "x", f"x{k}", f"{k} or {k[:-1]}x", f"not {k}", k.upper() if k.upper() != k else k + "9")]
    nums = [[num_value(a.lower()) for a in g if num_value(a.lower()) is not None] for g in acc]
    if kind == "skip":
        v, first, last = nums[0][0], acc[0][0], acc[0][-1]
        return [("legacy right", r, True) for r in (first, last, f"0x{v:02x}", f"The value is {first}")] + [
            ("legacy wrong", r, False) for r in ("a", "the letter a", f"{v + 1}", "0x%02x" % ((v + 1) % 256), f"not {first}", f"{first} or {v + 1}")]
    if kind == "hash":
        name, h = acc[0][0], acc[1][0]
        wrong_name = "sdbm" if name != "sdbm" else "djb2"
        return [("legacy right", r, True) for r in (f"{name}, 0x{h}", f"{acc[0][-1]}, {h}", f"{name} and {h}", f"{name.upper()}, 0x{h.upper()}")] + [
            ("legacy wrong", r, False) for r in (f"{wrong_name}, 0x{h}", f"{name}, 0x{h[:-1]}0" if h[-1] != "0" else f"{name}, 0x{h[:-1]}1", f"{name}", f"0x{h}",
                                                 f"{name} or {wrong_name}, 0x{h}")]
    if kind == "threshold":
        lo, hi = nums[0][0], nums[1][0]
        return [("legacy right", r, True) for r in (f"{lo} to {hi}", f"[{lo}, {hi}]", f"lowest {lo}, highest {hi}", f"{lo}, {hi}")] + [
            ("legacy wrong", r, False) for r in (f"{hi}, {lo}", f"({lo}, {hi})", f"x > {lo} and x < {hi}", f"{lo + 1} to {hi - 1}", f"{lo}")]
    a, b = nums[0][0], nums[1][0]                                        # magic, lcg, rotate: two values in the order of the question
    unit = " rounds" if kind == "lcg" else " bits" if kind == "rotate" else ""
    return [("legacy right", r, True) for r in (f"{acc[0][0]}, {b}", f"0x{a:x}, {b}", f"{a} and {b}{unit if unit != ' bits' else ''}", f"{a}; {b}")] + [
        ("legacy wrong", r, False) for r in (f"{b}, {acc[0][0]}", f"{acc[0][0]}, {b + 1}", f"{acc[0][0]}", f"{b}", f"{acc[0][0]} or {b}")]


def lead_in_attacks(t) -> list:
    """The M3 shapes: a lead-in line over a list of candidates (the key among them, first or last)."""
    if t["score"] != "tokens":
        return []
    k = [x for x in _nums(t) if x is not None]
    if not k:
        return []
    a = _bare(t)
    v = k[0]
    return [("M lead-in + list (key first)", f"Possible offsets:\n{a}\n0x{v + 8:x}\n0x{v + 16:x}", False),
            ("M lead-in + bullet list", f"Candidates:\n- {a}\n- {v + 8}", False),
            ("M lead-in + list (key last)", f"Candidates:\n{v + 8}\n{v + 16}\n{a}", False),
            ("M lead-in + one answer + prose", f"Answer:\n{a}\nThe value is read from the third field.", True)]


# kind -> [right replies accepted, right replies tried, yield-class replies tried]: the right-format pass rate per family is
# accepted / (tried + yield tried), so the formats the grammar gives up are counted against it.
FAMILY_RATE = collections.defaultdict(lambda: [0, 0, 0])
YIELD_EXAMPLES = collections.defaultdict(set)              # yield row name -> kinds it applies to


def matrix(tasks) -> tuple:
    FAMILY_RATE.clear()
    YIELD_EXAMPLES.clear()
    res = collections.defaultdict(lambda: [0, 0])
    ex = {}
    for t in tasks:
        if t["score"] not in HEADLINE_MODES:
            continue
        truth = {k: v for k, v in t.items() if k != "_truth"}          # the full task record, as the sampler and modal_rea.py score it
        fam = FAMILY_RATE[t["kind"] if t["kind"] in NEW_FAMILIES else "legacy"]
        for name, reply, expect in generic_attacks(t) + repro_attacks(t) + family_attacks(t) + lead_in_attacks(t) + (legacy_attacks(t) if t["kind"] in LEGACY_KINDS else []):
            if expect is None:
                continue
            got = score(reply, truth)
            row = res[(name, expect)]
            row[0] += got
            row[1] += 1
            if expect is True:
                fam[0] += got
                fam[1] += 1
            elif expect == "yield":
                fam[2] += 1
                YIELD_EXAMPLES[name].add(t["kind"] if t["kind"] in NEW_FAMILIES else "legacy")
            if got != (expect is True):
                ex.setdefault((name, expect), (t["id"], reply))
    return dict(res), ex


def summary(res: dict) -> dict:
    wrong = [(n, a, b) for (n, e), (a, b) in res.items() if e is False]
    right = [(n, a, b) for (n, e), (a, b) in res.items() if e is True]
    yields = [(n, a, b) for (n, e), (a, b) in res.items() if e == "yield"]
    fmt = [(n, a, b) for n, a, b in right if n in RIGHT_FORMATS]
    return {"yield_total": sum(b for _, _, b in yields), "yield_accepted": sum(a for _, a, _ in yields), "yield_rows": sorted(n for n, _a, _b in yields),
            "wrong_total": sum(b for _, _, b in wrong), "wrong_accepted": sum(a for _, a, _ in wrong), "wrong_rows": len(wrong),
            "right_total": sum(b for _, _, b in right), "right_passed": sum(a for _, a, _ in right),
            "formats_total": sum(b for _, _, b in fmt), "formats_passed": sum(a for _, a, _ in fmt), "formats_rows": len(fmt)}


def family_table() -> list:
    """[(family, accepted, right replies, yield replies, pass rate over right + yield)] after a matrix() run."""
    return [(k, a, n, y, a / max(n + y, 1)) for k, (a, n, y) in sorted(FAMILY_RATE.items())]


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    res, ex = matrix(tasks_for(seed, per) + legacy_tasks(seed, per * 10))
    s = summary(res)
    print(f"seed {seed}, {per} tasks per variant: attacks (wrong replies) accepted {s['wrong_accepted']}/{s['wrong_total']} in {s['wrong_rows']} shapes; "
          f"right replies accepted {s['right_passed']}/{s['right_total']}; the review's {len(RIGHT_FORMATS)} formats {s['formats_passed']}/{s['formats_total']} "
          f"({100 * s['formats_passed'] / max(s['formats_total'], 1):.2f}%); formats the grammar gives up by design (yield): "
          f"{s['yield_total'] - s['yield_accepted']}/{s['yield_total']} rejected")
    print("right-format pass rate per family (accepted / (right-format replies + yield-class replies)):")
    for fam, a, n, y, rate in family_table():
        print(f"  {fam:10s} {a:5d}/{n + y:<5d} {100 * rate:6.1f}%   ({n} right-format replies, all {'accepted' if a == n else 'NOT all accepted'}; {y} in yield-class formats, rejected)")
    print("yield-class formats (right answers the grammar does not parse): " + ", ".join(f"{n} [{'/'.join(sorted(k))}]" for n, k in sorted(YIELD_EXAMPLES.items())))
    for (n, e), (a, b) in sorted(res.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
        bad = (e is False and a) or (e is True and a < b) or (e == "yield" and a)
        if bad:
            print(f"  {'WRONG ACCEPTED' if e is False else 'RIGHT REJECTED' if e is True else 'YIELD ACCEPTED':15s} {n:48s} {a}/{b}   e.g. {ex[(n, e)]}")

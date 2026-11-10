"""The round-3 reviewer's attack rows (attack_scorer.py, attack_exact_sft.py, sft_trunc.py, error_reply.py), ported to run on generated tasks.

Every reply returned by `attacks(t)` is WRONG or not an answer for the task it is built for: the scorer must reject all of them. A row
is built from the key but never from the key's own values (`fresh` picks a number that is not a value of any group and not a number the
question states), so a row that happens to equal the answer cannot count as an attack (the reviewer's script had a few such coincidences:
`2 to 5` for a task whose bounds are 2 and 5).

`stored_attacks(t)` are replies with a RIGHT headline and a bad second line: since round 4 the stored text is rendered from the truth, so whatever
the second line says, `sft_target` must return exactly the template's text.

    python3 -m tests.rea_attacks_r3 [seed] [n per family]       how many rows the scorer accepts (0 is the goal)
"""
from __future__ import annotations

import collections
import re
import sys

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, grammar  # noqa: E402
from rea_bench import targets  # noqa: E402
from rea_bench.scoring import canonical, score, score_strict, sft_target, strict_task  # noqa: E402

NUM_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
             12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
             20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def words(n):
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if 20 < n < 100:
        return NUM_WORDS[n // 10 * 10] + "-" + NUM_WORDS[n % 10]
    return None


def used_numbers(t) -> set:
    """Every number a reply may legitimately contain for this task: the key's values, what the question states, bit counts."""
    sp = grammar.Spec(t)
    return set(sp.values) | set(sp.q_numbers) | set(sp.bits_ok) | {n for n, _u in sp.ctx}


def fresh(t, start: int, low: int = 0) -> int:
    bad = used_numbers(t)
    n = max(start, low)
    while n in bad or n + 1 in bad:
        n += 1
    return n


def num_groups(t):
    """[(decimal spelling, hex spelling)] of the numeric groups of the key."""
    out = []
    for g in t["accept"]:
        d = [a for a in g if re.fullmatch(r"-?[0-9]+", a)]
        h = [a for a in g if re.fullmatch(r"0x[0-9a-f]+", a)]
        if d or h:
            out.append((d[0] if d else str(int(h[0], 16)), h[0] if h else hex(int(d[0]))))
    return out


NEGATIONS = [("neq-sym", "≠ {k}"), ("neq-ascii", "!= {k}"), ("neq-angle", "<> {k}"), ("no", "no {k}"), ("never", "never {k}"),
             ("other-than", "other than {k}"), ("but", "anything but {k}"), ("except", "except {k}"), ("all-but", "all but {k}"),
             ("without", "without {k}"), ("besides", "besides {k}"), ("excluding", "excluding {k}"), ("non", "non-{k}"),
             ("less-than", "less than {k}"), ("more-than", "more than {k}"), ("at-least", "at least {k}"), ("at-most", "at most {k}"),
             ("up-to", "up to {k}"), ("under", "under {k}"), ("over", "over {k}"), ("below", "below {k}"), ("above", "above {k}"),
             ("neither", "neither {k}"), ("lt", "< {k}"), ("gt", "> {k}"), ("tilde-not", "!{k}"), ("negative-uni-minus", "−{k}"),
             ("negative-endash", "–{k}"), ("not-curly", "isn’t {k}"), ("not-acute", "isn´t {k}"), ("not-modifier", "isnʼt {k}"),
             ("not-prime", "isn′t {k}"), ("not-fullwidth", "isn＇t {k}"), ("not-nospace", "isnt {k}"), ("dont", "don't think it is {k}"),
             ("na", "N/A {k}"), ("cant", "can't be {k}"), ("wouldnt", "wouldn't be {k}"), ("is-it", "{k}, isn't it"),
             ("unknown", "unknown, {k}"), ("none", "none of {k}"), ("avoid", "avoid {k}"), ("wrong", "wrong: {k}"), ("incorrect", "incorrect, {k}"),
             ("false", "false: {k}"), ("no-comma", "No, {k}"), ("nope", "Nope {k}"), ("not", "not {k}"), ("or-maybe", "{k} or maybe"),
             ("maybe", "maybe {k}"), ("perhaps", "perhaps {k}"), ("probably", "probably {k}"), ("either", "either {k}"), ("question", "{k}?"),
             ("dots", "{k}..."), ("i-think", "{k} I think"), ("might", "might be {k}"), ("could", "could be {k}"), ("cannot", "cannot be {k}"),
             ("instead", "instead of {k}"), ("rather", "rather than {k}"), ("vs", "vs {k}"), ("unsure", "unsure, {k}"), ("not-sign", "¬{k}"),
             ("le", "≤ {k}"), ("ge", "≥ {k}"), ("le-ascii", "<= {k}"), ("ge-ascii", ">= {k}"), ("eq", "== {k}"), ("eq1", "= {k}"), ("tilde", "~{k}"),
             ("pm", "±{k}"), ("approx", "approximately {k}"), ("about", "about {k}"), ("around", "around {k}"), ("roughly", "roughly {k}"),
             ("almost", "almost {k}"), ("nearly", "nearly {k}"), ("exceeds", "exceeds {k}"), ("plus", "+ {k}"), ("star", "*{k}"), ("amp", "&{k}"),
             ("pipe-lead", "| {k}"), ("caret", "^{k}"), ("pct", "%{k}"), ("shift", "<< {k}"), ("hash", "#{k}"), ("at", "@{k}"), ("dollar", "${k}")]

EXEMPT = [("bit", "{k}-bit"), ("byte-aligned", "{k}-byte aligned"), ("paren-bits", "({k} bits)"), ("paren-bit", "({k} bit)"), ("crc", "CRC-{k}"),
          ("x86", "x86-{k}"), ("x64", "x64-{k}"), ("aes", "AES-{k}"), ("sha", "SHA-{k}"), ("md", "MD-{k}"), ("rc", "RC-{k}"), ("des", "DES-{k}"),
          ("base", "base-{k}"), ("bits-plain", "{k}-bits"), ("bytes-plain", "{k}-bytes"), ("bit-sentence", "It is a {k}-bit value, not what you asked")]

SHAPED = [("sep-comma", "{a}, {b}"), ("sep-and", "{a} and {b}"), ("sep-slash", "{a} / {b}"), ("sep-pipe", "{a} | {b}"), ("sep-semi", "{a}; {b}"),
          ("sep-first", "{b}, {a}"), ("sep-dash", "{a} - {b}"), ("paren-other", "{a} ({b})"), ("paren-was", "{a} (was {b})"),
          ("paren-other-first", "{b} ({a})"), ("pipe", "{a} | {b}"), ("slash", "{a}/{b}"), ("codeblock-two", "```\n{a}\n{b}\n```"),
          ("two-lines", "{a}\n{b}"), ("lead-two", "Answer:\n{a}\n{b}"), ("dot-sep", "{a}.{b}"), ("comma-nospace", "{a},{b}"),
          ("range", "{a}-{b}"), ("range-dots", "{a}..{b}"), ("range-endash", "{a}–{b}"), ("to", "{a} to {b}"), ("approx-or", "~{a} or {b}"),
          ("plusminus", "{a}±3"), ("fullwidth", "{a}　{b}"), ("roman", "{a} (IV)")]


def _token_rows(t) -> list:
    c = canonical(t)
    out = []
    ng = num_groups(t)
    for d, h in ng:
        k = int(d)
        for lab, tmpl in NEGATIONS:
            if k == 0 and lab in ("negative-uni-minus", "negative-endash"):
                continue                                        # minus zero is zero: the key itself
            out.append((f"neg:{lab}", tmpl.format(k=k)))
        out.append(("negation-hex", f"not {h}"))
        other = fresh(t, k + 1, 0)
        w = words(other)
        if w:
            for lab, tmpl in [("cand-words-comma", "{k}, {w}"), ("cand-words-and", "{k} and {w}"), ("cand-words-slash", "{k} / {w}"),
                              ("cand-words-pipe", "{k} | {w}"), ("cand-words-semi", "{k}; {w}"), ("cand-words-first", "{w}, {k}"), ("cand-words-dash", "{k} - {w}")]:
                out.append((lab, tmpl.format(k=k, w=w)))
        for lab, tmpl in SHAPED:
            out.append((lab, tmpl.format(a=k, b=other)))
        out.append(("k-then-other-hex", f"{k} 0x{other:x}"))
        bits = fresh(t, k + 3, 0)
        out += [("paren-bits-other", f"{k} ({bits} bits)"), ("paren-bits-key", f"{bits} ({k} bits)"), ("paren-bit-key", f"{bits} ({k}-bit)")]
        for lab, tmpl in EXEMPT:
            out.append((f"exempt-{lab}", tmpl.format(k=k)))
        out.append(("exempt-byte-offset", f"{k}-byte aligned 0x{other:x}"))
    if len(ng) == 2 and t.get("kind") in ("bounds", "threshold"):       # a closed interval: the exclusive notations are the attack ((a, b) is a plain pair elsewhere)
        lo, hi = int(ng[0][0]), int(ng[1][0])
        out += [("excl-gt-lt", f"x > {lo} and x < {hi}"), ("excl-gt-lt-b", f"> {lo} and < {hi}"), ("excl-between", f"strictly between {lo} and {hi}"),
                ("between", f"between {lo} and {hi}"), ("excl-open", f"({lo}, {hi})"), ("half-open", f"[{lo}, {hi})"), ("half-open-b", f"({lo}, {hi}]"),
                ("half-open2", f"{lo} <= x < {hi}"), ("half-open3", f"{lo} inclusive, {hi} exclusive"), ("not-inclusive", f"{lo} to {hi} exclusive"),
                ("excl-lt-le", f"{lo} < x <= {hi}"), ("swapped", f"{hi}, {lo}"), ("greater-less", f"greater than {lo}, less than {hi}"),
                ("excl-code", f"v > {lo} && v < {hi}"), ("excl-words", f"above {lo}, below {hi}"), ("excl-after-before", f"after {lo}, before {hi}"),
                ("swapped-labels", f"lowest {hi}, highest {lo}"), ("labels-crossed", f"highest {lo}, lowest {hi}"), ("only-lo-label", f"lowest {lo}")]
    names = [g[0] for g in t["accept"] if not num_groups({"accept": [g]}) and g[0].isalpha()]
    for _nm in names[:1]:
        out += [("name-slash", f"{c} / Rabbit"), ("name-comma", f"{c}, Rabbit"), ("name-and", f"{c} and Rabbit"), ("name-pipe", f"{c} | Rabbit"),
                ("name-semi", f"{c}; Rabbit"), ("name-plus", f"{c} + Rabbit"), ("name-amp", f"{c} & Rabbit"), ("name-hyph", f"Rabbit-{c}"),
                ("name-aka", f"{c} aka Rabbit"), ("name-also", f"{c}, also Rabbit"), ("name-second-first", f"Rabbit, {c}"), ("name-lower", f"{c}, rabbit")]
    return out


def _exact_rows(t) -> list:
    k = t["accept"][0]
    alt = k[:-1] + ("x" if k[-1] != "x" else "y")
    url = "http://login.local/drivex"
    rows = [("or", f"{k} or {alt}"), ("not", f"not {k}"), ("neq", f"{k} ≠ {alt}"), ("kk", f"{k}7"), ("xk", f"x{k}"), ("path", f"/etc/{k}"),
            ("two-lines", f"{k}\n{alt}"), ("lead-two", f"Answer:\n{k}\n{alt}"), ("lead-two-url", f"Answer:\n{k}\n{url}"), ("isnt", f"It isn’t {alt}, it's {k}"),
            ("paren-other", f"{k} ({alt})"), ("pipe", f"{k} | {alt}"), ("semi", f"{k}; {alt}"), ("comma", f"{k}, {alt}"), ("slash", f"{k} / {alt}"),
            ("and", f"{k} and {alt}"), ("neither", f"neither {alt} nor {k}"), ("lead-is-not", f"It is not {k}"), ("lead-isnt-key", f"The answer isn't {k}"),
            ("lead-no", f"no {k}"), ("lead-never", f"never {k}"), ("prefix-minus", f"-{k}"), ("prefix-eq", f"x == {k}"), ("prefix-neq", f"x != {k}"),
            ("prefix-colon", f":{k}"), ("lead-instead", f"instead of {alt}: {k}"), ("lead-unlike", f"unlike {alt}, {k}"), ("lead-key-is", f"the key is {alt} {k}"),
            ("trail-dots", f"{k}..."), ("trail-q", f"{k}?"), ("trail-maybe", f"{k}, maybe"), ("trail-question", f"{k} (I think)"), ("trail-paren-alt", f"{k} (or {alt})"),
            ("embedded", f"The string x{k}y"), ("fullwidth-prefix", f"＄{k}"), ("zero-width", k[0] + "​" + k[1:])]
    if k.upper() != k:
        rows.append(("upper", k.upper()))
    if " " in k:
        rows.append(("spaces-removed", k.replace(" ", "")))
    else:
        rows.append(("spaced-out", " ".join(k)))
    return rows


def attacks(t) -> list:
    """[(label, reply)] for a task of any mode that the strict scorer knows (a legacy task is converted first)."""
    t = strict_task(t)
    return _token_rows(t) if t["score"] == "tokens" else _exact_rows(t)


def stored_attacks(t) -> list:
    """[(label, reply, forbidden)]: a right headline and a bad second line; `forbidden` must not appear in the stored text."""
    t = strict_task(t)
    c = canonical(t)
    if t["score"] == "tokens":
        ng = num_groups(t)
        if ng:
            other = fresh(t, int(ng[0][0]) + 1, 0)
            bad = [("2nd-number", f"The pointer is at {other} in the node.", str(other)), ("2nd-hedge", f"It could also be {other} for a packed layout.", str(other)),
                   ("2nd-alternatively", f"Alternatively the code reads {other} in the second loop.", str(other)),
                   ("2nd-wait", f"Wait, actually the value is {other} when the flag is set.", str(other)),
                   ("2nd-name", "Unlike TEA it indexes the key with the running sum.", "Unlike")]
        else:
            bad = [("2nd-hedge", "It could also be Rabbit for a packed layout of fields.", "Rabbit"), ("2nd-wait", "Wait, actually the value is Rabbit here.", "Wait")]
    else:
        k = t["accept"][0]
        alt = k[:-1] + ("x" if k[-1] != "x" else "y")
        bad = [("2nd-two-cands", f"The buffer then holds {alt}.", alt), ("2nd-url", "Next it opens http://login.local/drivex for reading.", "drivex"),
               ("2nd-could", f"It could also be {alt} in a variant.", alt), ("2nd-digit", f"Then each byte is XORed with 0x{37:x} once.", "0x37"),
               ("2nd-path", "It is also copied to /tmp/other/path on exit.", "/tmp/other")]
    return [(lab, f"{c}\n{body}", forbidden) for lab, body, forbidden in bad]


def run(seed: int, n: int, with_score: bool = True) -> dict:
    """{"rows": total, "accepted": [(task id, label, reply)], "stored_bad": [...], "by_family": {family: (rows, accepted)}} over every family."""
    rows, accepted, stored_bad = 0, [], []
    by_family = collections.defaultdict(lambda: [0, 0])
    for fam in families.modules():
        for task in families.generate(fam, n, seed):
            t = task.meta()
            for lab, reply in attacks(t):
                rows += 1
                by_family[fam][0] += 1
                if score_strict(reply, t) or (with_score and fam != "legacy" and score(reply, t)):
                    accepted.append((task.id, lab, reply))
                    by_family[fam][1] += 1
            for lab, reply, forbidden in stored_attacks(t):
                r = sft_target(reply, t)
                if r is not None and r["text"] != targets.render(t)["text"]:
                    stored_bad.append((task.id, lab, reply))              # round 4: whatever the second line says, the stored text is the template
    return {"rows": rows, "accepted": accepted, "stored_bad": stored_bad, "by_family": {k: tuple(v) for k, v in by_family.items()}}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    res = run(seed, n)
    print(f"seed {seed}, {n} tasks per family: {res['rows']} attack rows, {len(res['accepted'])} accepted, {len(res['stored_bad'])} stored with a bad body")
    print("  per family (rows, accepted): " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(res["by_family"].items())))
    for row in res["accepted"][:20] + res["stored_bad"][:20]:
        print("  ", row)

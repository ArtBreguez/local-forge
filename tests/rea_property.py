"""Property test of the scorer (round 3): mutate the canonical answer of random tasks of every family, accept nothing.

For each task the canonical answer (first spelling of every group) is changed in the ways a wrong or hedged reply is made:
  number      one number is moved by +-1 (decimal or hex, as the key was spelled)
  name        a name, direction or string is swapped for another one of the family's vocabulary (all names its tasks can produce plus the `reject`
              lists), for an exact string one character is changed
  second      a second candidate (another number, another name) is added before or after the answer, joined by `, or / ; and` or a space
  token       a negation, hedge, limit or operator template of the review is put around the whole answer or around one group (`NEGATIONS`);
              the exempt forms of m3 (`CRC-n`, `n-bit`, `(n bits)`, ...) stand in for a number (`EXEMPT`)
Every mutation is built from values that are NOT the key's (checked against every spelling of every group), so a mutated reply is wrong by
construction and the scorer must reject all of them. The separators the grammar allows (`, ; / and : whitespace`) are not "operators" here.

    python3 -m tests.rea_property [tasks per family] [seed] [templates per task (0 = all, at every position)]
"""
from __future__ import annotations

import collections
import random
import re
import sys

from .helpers import ROOT
from .rea_attacks_r3 import EXEMPT, NEGATIONS, fresh

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, grammar  # noqa: E402
from rea_bench.scoring import canonical, score, score_strict, strict_task  # noqa: E402

EXTRA = [("op-" + n, f"{{k}} {op} 1") for n, op in (("add", "+"), ("sub", "-"), ("mul", "*"), ("or", "|"), ("and", "&"), ("xor", "^"), ("shl", "<<"))]
# `+ x` and `* x` before a word are markdown bullets (the same list item as `- x`), harmless for a name; before a digit they are rejected (see
# scoring._MD_LEAD), so they only count as attacks on numbers.
BULLETS = ("{k}" if False else "+ {k}", "*{k}")
TEMPLATES = [(lab, tmpl) for lab, tmpl in NEGATIONS if tmpl not in BULLETS] + EXEMPT + EXTRA
SEPARATORS = [", ", " or ", " / ", "; ", " and ", " ", " | ", " + ", " aka ", " vs "]


def vocabulary(tasks) -> list:
    """The family's names: every non-numeric accepted spelling of its tasks and every `reject` word."""
    out = set()
    for t in tasks:
        t = strict_task(t)
        if t["score"] != "tokens":
            continue
        sp = grammar.Spec(t)
        for i, g in enumerate(t["accept"]):
            if i not in sp.numeric:
                out |= {a.lower() for a in g}
        out |= {r.lower() for r in t.get("reject", [])}
    return sorted(x for x in out if re.fullmatch(r"[a-z][a-z0-9_-]+", x))


def _spell(v: int, like: str) -> str:
    return f"0x{v:x}" if like.startswith("0x") and v >= 0 else str(v)


def _apply(tmpl: str, text: str) -> str:
    return tmpl.format(k=text)


def mutations(t: dict, vocab: list, rng: random.Random, templates: int = 12, separators: int = 0) -> list:
    """[(kind, reply)] of wrong replies for one task. templates=0: every template at every position; separators=0: every separator."""
    t = strict_task(t)
    out = []
    if t["score"] != "tokens":
        key = t["accept"][0]
        i = rng.randrange(len(key))
        c = key[i]
        bad = key[:i] + (str((int(c) + 1) % 10) if c.isdigit() else "x" if c != "x" else "y") + key[i + 1:]
        others = [v for v in vocab if v not in key.lower()] or ["zzzz"]
        out.append(("name", bad))
        for sep in (SEPARATORS if not separators else rng.sample(SEPARATORS, separators)):
            out += [("second", f"{key}{sep}{bad}"), ("second", f"{bad}{sep}{key}")]
        for lab, tmpl in (TEMPLATES if templates == 0 else rng.sample(TEMPLATES, min(templates, len(TEMPLATES)))):
            out.append(("token", _apply(tmpl, key)))
        return out
    sp = grammar.Spec(t)
    groups = t["accept"]
    parts = [g[0] for g in groups]
    spellings = [{a.lower() for a in g} for g in groups]
    allvals = {v for nums, _l in sp.groups for v in nums}
    # number: +-1 on one numeric group
    for i in sp.numeric if templates == 0 else [rng.choice(sp.numeric)] if sp.numeric else []:
        v = sorted(sp.groups[i][0])[0]
        for new in (v + 1, v - 1):
            if new in allvals or (new < 0 and parts[i].startswith("0x")):
                continue
            out.append(("number", ", ".join(_spell(new, parts[i]) if j == i else p for j, p in enumerate(parts))))
    # name: swap a non-numeric group for another name of the family
    for i in [j for j in range(len(groups)) if j not in sp.numeric]:
        alts = [v for v in vocab if v not in spellings[i] and not any(v in s for s in spellings)]
        if alts:
            out.append(("name", ", ".join(rng.choice(alts) if j == i else p for j, p in enumerate(parts))))
    # second candidate: another number or another name
    base = ", ".join(parts)
    if sp.numeric:
        i = sp.numeric[0]
        v = sorted(sp.groups[i][0])[0]
        extra = [_spell(fresh(t, abs(v) + d + 1), parts[i]) for d in (0, 5)]
    else:
        extra = []
    extra += [rng.choice([x for x in vocab if not any(x in s for s in spellings)] or ["rabbit"])]
    for x in extra:
        for sep in (SEPARATORS if not separators else rng.sample(SEPARATORS, separators)):
            out += [("second", f"{base}{sep}{x}"), ("second", f"{x}{sep}{base}")]
    # token: templates around the whole answer or one group; exempt forms in place of one numeric group
    chosen = TEMPLATES if templates == 0 else rng.sample(TEMPLATES, min(templates, len(TEMPLATES)))
    for lab, tmpl in chosen:
        if "bit" in tmpl and ("bit" in sp.units or "bits" in sp.units):
            continue                                               # `8 bits` is the key with its own unit where the question asks for bits
        if tmpl.startswith(("{k}-", "{k} (")) or any(tmpl.startswith(x) for x in ("CRC", "x8", "x6", "AES", "SHA", "MD", "RC", "DES", "base", "({k}", "It is")):
            if sp.numeric:                                         # the exempt forms stand in for the number
                i = rng.choice(sp.numeric) if templates else sp.numeric[0]
                v = sorted(sp.groups[i][0])[0]
                if v >= 0:
                    out.append(("exempt", ", ".join(_apply(tmpl, str(v)) if j == i else p for j, p in enumerate(parts))))
            continue
        if lab in ("negative-uni-minus", "negative-endash") and any(v == 0 for v in allvals):
            continue                                               # minus zero is zero
        out.append(("token", _apply(tmpl, base)))
        for j in (range(len(parts)) if templates == 0 else [rng.randrange(len(parts))]):
            out.append(("token", ", ".join(_apply(tmpl, p) if k == j else p for k, p in enumerate(parts))))
    return out


def run(n: int = 2000, seed: int = 31337, templates: int = 12, separators: int = 0, evaluation_too: bool = True) -> dict:
    """{family: {"tasks", "rows", "by_kind": {kind: rows}, "accepted": [(task id, kind, reply)]}}"""
    rng = random.Random(seed)
    res = {}
    for fam in families.modules():
        tasks = [t.meta() for t in families.generate(fam, n, seed)]
        vocab = vocabulary(tasks)
        row = {"tasks": len(tasks), "rows": 0, "by_kind": collections.Counter(), "accepted": []}
        for t in tasks:
            for kind, reply in mutations(t, vocab, rng, templates, separators):
                row["rows"] += 1
                row["by_kind"][kind] += 1
                if score_strict(reply, t) or (evaluation_too and fam != "legacy" and score(reply, t)):
                    row["accepted"].append((t["id"], kind, reply))
        res[fam] = row
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 31337
    tpl = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    res = run(n, seed, tpl)                                    # every separator; the unit test samples 3 per task
    total = sum(r["rows"] for r in res.values())
    acc = sum(len(r["accepted"]) for r in res.values())
    print(f"property test: {n} tasks per family, seed {seed}, {'all templates at every position' if tpl == 0 else f'{tpl} templates per task'}: "
          f"{total} mutated replies, {acc} accepted")
    for fam, r in res.items():
        print(f"  {fam:10s} {r['tasks']} tasks, {r['rows']:7d} replies ({', '.join(f'{k} {v}' for k, v in sorted(r['by_kind'].items()))}), {len(r['accepted'])} accepted")
        for a in r["accepted"][:8]:
            print("     ", a)

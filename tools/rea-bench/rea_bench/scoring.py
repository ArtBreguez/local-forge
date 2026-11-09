"""Answer scoring against a task's truth (`score` + `accept` [+ `reject`]).

Modes:
  exact / any       one accepted spelling must appear in the answer (whitespace and case ignored)
  all               every accepted value must appear (each with its numeric spellings, see `variants`)
  all_groups        every group must have one spelling in the answer
  tokens            every group must match a whole token of the answer; any `reject` token present fails the answer

The first four are the original modes and substring-based, which is fine for long values (a key, a 32-bit constant) but
vacuous for small ones: "0x18" contains "8", and "XTEA" contains "tea". `tokens` compares whole tokens, so small
numbers and algorithm names are checked properly. Used by the families that ask for offsets, rounds, key lengths and names.
"""
from __future__ import annotations

import re

# A token is a run of [0-9a-z_] with an optional leading minus that is not glued to a preceding word ("3-5" is 3 and 5,
# "x is -37" is -37). Hyphenated words ("x-tea", "base-64") are also offered as one token.
_TOK = re.compile(r"(?<![0-9a-z_])-?[0-9a-z_]+")
_HYPHEN_WORD = re.compile(r"[a-z_]+(?:-[0-9a-z_]+)+")


def variants(x: str) -> list:
    """An integer answer may be given in decimal or hex, signed or as the 32-bit pattern (0x97c59b7a == -0x683a6486); accept all."""
    out = [x]
    m = re.fullmatch(r"-?\d+", x)
    if m:
        v = int(x)
        if abs(v) > 9:
            out += [format(abs(v), "x") if v > 0 else "-0x" + format(-v, "x"), "0x" + format(v & 0xffffffff, "x")]
            if v > 0:
                out.append("0x" + format(v, "x"))
    elif re.fullmatch(r"[0-9a-f]{8}", x):
        v = int(x, 16)
        if v >= 0x80000000:
            out.append("-0x" + format(0x100000000 - v, "x"))
    return out


def tokens(answer: str) -> set:
    a = answer.lower()
    return set(_TOK.findall(a)) | set(_HYPHEN_WORD.findall(a))


def score(answer: str, t: dict) -> bool:
    if t["score"] == "tokens":
        toks = tokens(answer)
        if any(r.lower() in toks for r in t.get("reject", [])):
            return False
        return all(any(alt.lower() in toks for alt in g) for g in t["accept"])
    a = re.sub(r"\s+", "", answer.lower())
    if t["score"] == "all_groups":
        return all(any(re.sub(r"\s+", "", x.lower()) in a for x in g) for g in t["accept"])
    acc = [[re.sub(r"\s+", "", v.lower()) for v in variants(x)] for x in t["accept"]]
    return all(any(v in a for v in g) for g in acc) if t["score"] == "all" else any(v in a for g in acc for v in g)


def canonical(t: dict) -> str:
    """A correct answer written from the truth alone (first spelling of each group), for tests."""
    mode, acc = t["score"], t["accept"]
    if mode in ("tokens", "all_groups"):
        return ", ".join(g[0] for g in acc)
    if mode == "all":
        return ", ".join(acc)
    return acc[0]


_score, _variants = score, variants   # names modal_rea_data.py used before the move

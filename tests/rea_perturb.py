"""Round 7: group perturbation. Replace the value of a group parsed from a RENDERED sentence or headline by a wrong value of the same kind and run the claims of its row on
it: a perturbation that no claim rejects is a group the table does not constrain (the reviewer's test of round 7: 14,629 wrong group values of the sentences, 0 accepted).

Perturbations: +1 and -1 (an integer), a flipped bit and +4 (a hex number), every other alternative of a word group, a mangled string (reversed, one more character, one
less, another first character), another path, request or identifier, another expression, and every swap of two groups of the same kind (two numbers in swapped roles).
"""
from __future__ import annotations

import itertools
import signal

from . import rea_claims as RC
from . import rea_sentences as RS


def wrong_values(kinds: dict, caps) -> list:
    """[(label, {group: wrong value})], one per perturbation of the groups `kinds` ({group: kind}) that were parsed into `caps`."""
    out = []
    for g, kind in kinds.items():
        v = caps[g]
        if kind == "i":
            out += [(f"{g}+1", {g: v + 1}), (f"{g}-1", {g: v - 1})]
        elif kind == "h":
            out += [(f"{g}^1", {g: v ^ 1}), (f"{g}+4", {g: v + 4})]
        elif kind in ("t", "l"):
            out += [(f"{g}:reversed", {g: v[::-1] if v[::-1] != v else v + "z"}), (f"{g}:+char", {g: v + "z"}), (f"{g}:-char", {g: v[:-1]}),
                    (f"{g}:first", {g: ("a" if v[0] != "a" else "b") + v[1:]})]
        elif kind == "p":
            out += [(f"{g}=/proc/self/maps", {g: "/proc/self/maps"})]
        elif kind == "q":
            out += [(f"{g}=PTRACE_ATTACH", {g: "PTRACE_ATTACH"})]
        elif kind == "w":
            out += [(f"{g}=Uid", {g: "Uid"}), (f"{g}+x", {g: v + "x"})]
        elif kind == "e":
            out += [(f"{g}=a & b", {g: "a & b" if v != "a & b" else "a | b"})]
        else:
            out += [(f"{g}={alt}", {g: alt}) for alt in kind.split("|") if alt != v]
    for (g1, k1), (g2, k2) in itertools.combinations(kinds.items(), 2):
        if k1 == k2 and caps[g1] != caps[g2] and k1 in ("i", "h"):
            out.append((f"swap {g1}<->{g2}", {g1: caps[g2], g2: caps[g1]}))
    return out


class Hang(BaseException):
    pass


def _alarm(*_a):
    raise Hang()


def rejects(task, rec, key, claims, caps, new: dict, cache: dict, implied=None, seconds: int = 30) -> str:
    """Why the claims reject the groups `new` put over `caps` ("" when they all hold: a survivor). A claim that does not return within `seconds` is a "hang" (a claim must
    fail fast: it may loop on a number of the sentence). `implied` ({group: function of the caps}) is recomputed from the perturbed groups."""
    c2 = RS.Caps(caps)
    c2.update(new)
    for name, how in (implied or {}).items():
        if callable(how) and not isinstance(how, RS.Q):
            c2[name] = how(c2)
    ctx = RC.Ctx(task, rec, c2, key)
    ctx.cache = cache
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for name in claims:
            try:
                signal.setitimer(signal.ITIMER_REAL, seconds)
                RC.CLAIMS[name](ctx)
            except RC.ClaimFailed as e:
                return f"{name}: {e}"[:160]
            except Hang:
                return f"hang: {name} did not return in {seconds} s"
            except Exception as e:                                   # a crash of a checker is a failure of the claim, with the reason
                return f"{name} crashed: {type(e).__name__}: {e}"[:160]
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        return ""
    finally:
        signal.signal(signal.SIGALRM, old)

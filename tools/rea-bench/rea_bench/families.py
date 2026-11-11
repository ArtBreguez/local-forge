"""Registry of task families. A family module has FAMILY (str) and VARIANTS (name -> function(rng) -> Task);
`legacy` is special (frozen, one shared rng, no `cc` field) and has its own generate()."""
from __future__ import annotations

import re

from . import f_antidebug, f_bounds, f_cipher, f_crc, f_encoding, f_fsm, f_multifunc, f_pktsum, f_strobf, f_structs, legacy
from .core import HELDOUT_SEED, build, draw

NEW_MODULES: list = [f_structs, f_fsm, f_crc, f_cipher, f_encoding, f_strobf, f_pktsum, f_bounds, f_antidebug, f_multifunc]


def modules() -> dict:
    return {m.FAMILY: m for m in [legacy, *NEW_MODULES]}


def generate(family: str, n: int, seed: int, cc: str = "mix", gate=None) -> list:
    """`gate` (survive.Gate) resamples tasks whose asked fact does not survive compilation; legacy is frozen and never gated."""
    m = modules()[family]
    if hasattr(m, "generate"):
        return m.generate(n, seed, cc)
    return build(m.FAMILY, m.VARIANTS, n, seed, cc, gate)


def draws(family: str, n: int, seed: int, attempts: int, cc: str = "mix") -> list:
    """Every draw the fact gate could keep for the first n tasks: attempts 0..attempts-1 of each (legacy has one: it is never gated)."""
    m = modules()[family]
    if hasattr(m, "generate"):
        return m.generate(n, seed, cc)
    return [draw(m.FAMILY, m.VARIANTS, i, seed, cc, a) for i in range(n) for a in range(attempts)]


LCG_FOLDED_ROUNDS = 8     # a legacy lcg loop of this many rounds or fewer with an increment of 0 is multiplied out by gcc into one `imul` (-O1/-O2/-Os)
_LCG_STEP = re.compile(r"seed \* \d+u \+ (\d+)u\)")


def untrainable(t, seed: int, gate=None) -> str | None:
    """Why a task of a TRAINING seed is not trained on although the generator makes it, or None (round 5, M2). The frozen legacy generator
    cannot change, so the filter lives here: `skip` with b = 0x00 asks which byte value the checksum skips, and the test `p[i] != 0x00` is
    dead code that every optimiser removes (gcc -O1/-O2/-Os: `proc` identical with and without it, 60/60): there is nothing to see, the
    only answer is a guess. The held-out seed is not filtered: its bytes stay what they were."""
    if seed != HELDOUT_SEED and t.kind == "skip" and t.truth.get("accept", [None])[0] == "0x00":
        return "legacy skip with b = 0x00: the compare is dead code, the answer cannot be read"
    if seed != HELDOUT_SEED and t.kind == "lcg":
        # round 8, m1: with an increment of 0 the rounds are one product, and the compiler computes it: `imul eax, edi, 0x979e791` is 1664525^4, neither the multiplier
        # nor the round count is in the code (15 of 900 tasks at three seeds, all with 3-5 rounds). The rule is a margin; the gate compiles the task and asks for the multiplier.
        step = _LCG_STEP.search(t.body)
        if step and int(step.group(1)) == 0 and int(t.truth["accept"][1]) <= LCG_FOLDED_ROUNDS:
            return "legacy lcg with an increment of 0 and few rounds: the compiler multiplies the rounds out, the multiplier is not in the code"
        if gate is not None:
            ok, _why = gate.legacy_visible(t)
            if not ok:
                return "legacy lcg whose multiplier is not in the compiled code"
    if seed != HELDOUT_SEED and not t.cc and gate is not None:
        # round 9: the frozen strings (xor key, password, alphabet) must be inside what refdata prints for the code compiled with gcc at the task's -O level
        ok, _why = gate.legacy_shown(t)
        if not ok:
            return "legacy task whose static data is not shown to the model (a byte outside every window refdata prints)"
    return None


def generate_many(names: list, n: int, seed: int, cc: str = "mix", gate=None, dropped: list | None = None) -> list:
    """n tasks over the named families (split_counts), in family order, without the tasks `untrainable` names (appended to `dropped`
    as (task, reason) when a list is given): a training run asks for n tasks and may get a few fewer."""
    out = []
    for nm, c in zip(names, split_counts(n, len(names))):
        for t in generate(nm, c, seed, cc, gate):
            why = untrainable(t, seed, gate)
            if why and dropped is not None:
                dropped.append((t, why))
            elif not why:
                out.append(t)
    return out


def split_counts(n: int, k: int) -> list:
    """n tasks over k families: n//k each, the remainder to the first ones."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]

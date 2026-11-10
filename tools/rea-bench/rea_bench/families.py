"""Registry of task families. A family module has FAMILY (str) and VARIANTS (name -> function(rng) -> Task);
`legacy` is special (frozen, one shared rng, no `cc` field) and has its own generate()."""
from __future__ import annotations

from . import f_antidebug, f_bounds, f_cipher, f_crc, f_encoding, f_fsm, f_multifunc, f_pktsum, f_strobf, f_structs, legacy
from .core import build, draw

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


def generate_many(names: list, n: int, seed: int, cc: str = "mix", gate=None) -> list:
    """n tasks over the named families (split_counts), in family order."""
    return [t for nm, c in zip(names, split_counts(n, len(names))) for t in generate(nm, c, seed, cc, gate)]


def split_counts(n: int, k: int) -> list:
    """n tasks over k families: n//k each, the remainder to the first ones."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]

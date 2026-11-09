"""Registry of task families. A family module has FAMILY (str) and VARIANTS (name -> function(rng) -> Task);
`legacy` is special (frozen, one shared rng, no `cc` field) and has its own generate()."""
from __future__ import annotations

from . import f_cipher, f_crc, f_encoding, f_fsm, f_strobf, f_structs, legacy
from .core import build

NEW_MODULES: list = [f_structs, f_fsm, f_crc, f_cipher, f_encoding, f_strobf]


def modules() -> dict:
    return {m.FAMILY: m for m in [legacy, *NEW_MODULES]}


def generate(family: str, n: int, seed: int, cc: str = "mix") -> list:
    m = modules()[family]
    if hasattr(m, "generate"):
        return m.generate(n, seed, cc)
    return build(m.FAMILY, m.VARIANTS, n, seed, cc)


def split_counts(n: int, k: int) -> list:
    """n tasks over k families: n//k each, the remainder to the first ones."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]

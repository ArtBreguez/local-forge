"""Real task records for the tests of the sample stage (round 4: a stored target is rendered from the task's truth, so a synthetic dict without a
`kind` and `variant` has nothing to render)."""
from __future__ import annotations

import sys

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, scoring  # noqa: E402


def meta(fam: str, variant: str = "", seed: int = 7, n: int = 40, **extra) -> dict:
    """The task record (`task.meta()`) of the first task of `fam` whose variant (legacy: kind) is `variant`, with `extra` fields (e.g. chance=0.01)."""
    m = families.modules()[fam]
    count = n * (len(getattr(m, "VARIANTS", ())) or 1)
    for t in families.generate(fam, count, seed):
        if not variant or t.variant == variant or t.kind == variant:
            return {**t.meta(), **extra}
    raise LookupError((fam, variant))


def right(t: dict) -> str:
    """A reply that is right for the task: its canonical answer."""
    return scoring.canonical(scoring.strict_task(t))

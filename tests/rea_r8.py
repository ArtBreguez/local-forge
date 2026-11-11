"""Shared by the round-8 test modules (tests/test_rea_round8_text.py, _data.py, _modal.py)."""
from __future__ import annotations

import importlib.machinery
import shutil
import sys
import types

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families  # noqa: E402

GCC_OBJDUMP = bool(shutil.which("gcc") and shutil.which("objdump"))


def tasks_of(fam: str, variant: str, seed: int = 4241, per: int = 40) -> list:
    m = families.modules()[fam]
    n = per * (len(getattr(m, "VARIANTS", ())) or 1)
    return [t for t in families.generate(fam, n, seed) if t.variant == variant or (not variant and t.kind == fam)]


def refdata_module():
    """tools/modal/rea-bench/refdata (a script without .py) as a module."""
    path = ROOT / "tools" / "modal" / "rea-bench" / "refdata"
    loader = importlib.machinery.SourceFileLoader("refdata_under_test", str(path))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod

"""The subjects. A subject is a package here, `distill/domains/<name>/`, whose `__init__` defines `DOMAIN`, an instance of `distill.domain.Domain`; `load(name)` returns it.

    rea            reverse engineering of decompiled C (the first subject; wraps tools/rea-bench)
    example_arith  arithmetic word problems: a test fixture that proves the core does not depend on REA, not a subject to train on
"""
from __future__ import annotations

import importlib
import re


def load(name: str):
    """The `DOMAIN` of `distill/domains/<name>`; SystemExit for a name that is not a package of this directory."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        raise SystemExit(f"domain {name!r}: a lower-case package name of distill/domains")
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{name}":
            raise
        raise SystemExit(f"unknown domain {name!r}: there is no package distill/domains/{name}")
    return module.DOMAIN

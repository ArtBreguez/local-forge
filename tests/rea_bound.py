"""The API that `rea_bench.reasoning` and `rea_bench.sft` had before `distill/` was split out, bound to the REA domain: `rsn` and `sft` (test-only).

The generic half of those two modules moved to `distill/traces.py`, `distill/config.py`, `distill/records.py`, `distill/outputs.py` and `distill/backends/modal_sample.py`, where a function that needs the subject takes
it as `domain=`; what is REA's stayed in `rea_bench.reasoning` / `distill/domains/rea`. The tests of the REA runs were written against the old modules, and what they pin is the behaviour of the two halves together, so
they call it through these two namespaces: the same names, the same call shapes (`rsn.quality_drop(reply, task, thinking)`, `sft.assemble(tasks, results, model, messages_for, ...)`), the domain filled in. Every name here
is the function of the new module, not a copy of it.
"""
from __future__ import annotations

import functools
import sys
import types

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from distill import config, outputs, records, traces  # noqa: E402
from distill.backends import modal_sample  # noqa: E402
from distill.domains.rea import AGREE_CHANCE, DOMAIN  # noqa: E402
from rea_bench import reasoning as rea_reasoning  # noqa: E402


def _with_domain(f):
    """`f` with the REA domain filled in (a wrapper that `inspect.getsource` sees through, which `functools.partial` is not)."""
    @functools.wraps(f)
    def call(*args, **kw):
        return f(*args, domain=DOMAIN, **kw)
    return call


def _fill(ns, module, domain_functions=()):
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        if name in domain_functions:
            value = _with_domain(value)
        setattr(ns, name, value)


rsn = types.SimpleNamespace()
for _module, _bound in ((config, ()), (traces, ("verdict", "score_reasoning", "teacher_target", "quality_drop", "pick_traces", "mixed_language", "unreadable")), (modal_sample, ()), (rea_reasoning, ())):
    _fill(rsn, _module, _bound)
rsn.prompt_sha = outputs.prompt_sha

sft = types.SimpleNamespace()
_fill(sft, records, ("select_traces", "select", "assemble"))
sft.AGREE_CHANCE = AGREE_CHANCE
sft.needs_agreement = DOMAIN.needs_agreement

"""Which sampled replies become SFT pairs, and what exactly is stored (stdlib only; unit-tested without Modal).

`modal_rea_data.py --stage sample` asks a model K times per task. This module decides, per task, from the K replies and the STRICT scorer
(`scoring.score_strict`, the whitelist grammar, for every family including legacy; the frozen substring scorer is for `modal_rea.py`'s
evaluation only):

  filter        the teacher is a FILTER (round 4): a task enters the data only when at least one reply has a headline the strict scorer accepts
                (`scoring.accepted_headline`). That proves the task is answerable from the pseudocode. Every reply is scored again from its text,
                whatever flag the checkpoint holds: a `--resume` run or an older checkpoint never carries the verdict of an older scorer into the
                data (round 3, m3). The text of a failed request is never a reply.
  stored text   NEVER the teacher's text. The assistant message is rendered from the task's own truth (`targets.render`): the canonical headline and
                one factual sentence of a per-variant template set, chosen by a seeded RNG from the task id. The record keeps `teacher_headline` (the
                first accepted headline, raw, for analysis only) and `target_source: "truth-template"`. Free prose repeatedly carried wrong facts
                past every filter (a sentence that contradicted its headline, numbers in swapped roles), so none of it is stored any more.
  agreement     a task with `chance >= AGREE_CHANCE` (the best constant guess within its variant is right that often: CRC init and xorout
                0.4-0.6, pktsum 0.2-0.5, fsm 0.2-0.4, TEA/XTEA with the round count guessed 0.3, ...; legacy `skip` 0.14) is kept only when at
                least AGREE_MIN of the K samples have an accepted headline: one lucky guess is not evidence that the model read the code. Tasks
                below the threshold keep any right sample. Agreement filters luck, it does not make a 0.55 task hard: a model that always answers
                "all ones" still agrees with itself.
  one record    the rendered text of a task is the same for every accepted sample, so a task gives ONE record.
"""
from __future__ import annotations

import collections

from .scoring import is_error_reply, score_strict, sft_target, strict_task

AGREE_CHANCE = 0.05          # round 4 (m8): 0.06-0.063 variants (fsm/opcode_value, bounds/char_class, multi/limit_helper) kept one lucky sample of six in 31% of the tasks
AGREE_MIN = 2


def needs_agreement(t: dict) -> bool:
    c = strict_task(t).get("chance")                # a legacy task gets its `chance` from the conversion (skip: 0.14)
    return c is not None and c >= AGREE_CHANCE


def select(outs: list, t: dict) -> tuple:
    """([records], reason) for one task. `outs` = [(reply, flag)] for its K samples; the flag is not used (every reply is scored again). A record
    is {"text", "headline", "sentence", "template", "teacher_headline", "target_source"}: the text is rendered from the truth. reason is "kept",
    "no_correct" (no sample has an accepted headline) or "agreement" (too few accepted samples for the task's chance level)."""
    accepted = [r for r in (sft_target(o, t) for o, _flag in outs) if r is not None]
    if not accepted:
        return [], "no_correct"
    if needs_agreement(t) and len(accepted) < AGREE_MIN:
        return [], "agreement"
    return [accepted[0]], "kept"


class Stats:
    """The sample stage's counters, in total and per task family (`kind`)."""

    FIELDS = ("tasks", "answers", "errors", "correct", "solved_tasks", "kept_tasks", "kept_records", "no_correct", "agreement")

    def __init__(self):
        self.total = collections.Counter()
        self.by_kind = collections.defaultdict(collections.Counter)

    def add(self, t: dict, outs: list, kept: list, reason: str) -> None:
        for c in (self.total, self.by_kind[t.get("kind", "?")]):
            c["tasks"] += 1
            c["answers"] += len(outs)
            c["errors"] += sum(is_error_reply(o) for o, _ok in outs)          # failed requests: not answers, never right
            c["correct"] += sum(ok and not is_error_reply(o) for o, ok in outs)
            c["solved_tasks"] += any(ok and not is_error_reply(o) for o, ok in outs)
            c["kept_tasks"] += bool(kept)
            c["kept_records"] += len(kept)
            if reason in ("no_correct", "agreement"):
                c[reason] += 1

    def as_dict(self) -> dict:
        row = lambda c: {f: c[f] for f in self.FIELDS}
        return {**row(self.total), "by_kind": {k: row(c) for k, c in sorted(self.by_kind.items())}}


ERROR_RATE_WARN = 0.05


def error_warning(stats: dict):
    """A loud warning text when more than ERROR_RATE_WARN of the requests failed (in total, or in one family); None otherwise. A dead
    server, a timeout or an over-long prompt turns every following request into an error: the yield numbers are then not about the model."""
    rows = {"ALL": stats, **stats.get("by_kind", {})}
    bad = [(k, r["errors"], r["answers"]) for k, r in rows.items() if r.get("answers") and r.get("errors", 0) / r["answers"] > ERROR_RATE_WARN]
    if not bad:
        return None
    lines = "; ".join(f"{k}: {e}/{a} requests failed ({100 * e / a:.0f}%)" for k, e, a in bad)
    return (f"WARNING: MORE THAN {int(ERROR_RATE_WARN * 100)}% OF THE REQUESTS FAILED ({lines}). Failed requests are not scored and not stored, "
            f"but these counts say nothing about the model. Look at /tmp/server.log on the sampler (OOM, a prompt over the slot size, a timeout) "
            f"and re-run with --resume.")


def assemble(tasks: list, results: dict, model: str, messages_for) -> tuple:
    """(SFT records, stats dict) from the K replies of every task. `results` = {task id: {"outs": [[reply, flag], ...]}} (the sample stage's
    checkpoint format; the flag is ignored, see above); `messages_for(task)` = the system and user message. A task without a result is skipped."""
    st, kept = Stats(), []
    for t in tasks:
        rec = results.get(t["id"])
        if rec is None:
            continue
        outs = [(o, score_strict(o, t)) for o, _flag in rec["outs"]]
        records, reason = select(outs, t)
        st.add(t, outs, records, reason)
        for r in records:
            kept.append({"messages": [*messages_for(t), {"role": "assistant", "content": r["text"]}], "task": t["id"], "kind": t["kind"],
                         "model": model, "teacher_headline": r["teacher_headline"], "target_source": r["target_source"]})
    return kept, st.as_dict()

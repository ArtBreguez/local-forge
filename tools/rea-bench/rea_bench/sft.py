"""Which sampled replies become SFT pairs, and what exactly is stored (stdlib only; unit-tested without Modal).

`modal_rea_data.py --stage sample` asks a model K times per task. This module decides, per task, from the K replies and the scorer:

  stored text   the headline and at most ONE justification sentence (`scoring.sft_target`): the rest of the reply is cut off and the
                record says `truncated: true`. A justification that hedges, names another candidate value or changes its mind
                (wait / actually / correction / however) vetoes the reply, so a right first line cannot carry a wrong body into the data.
  agreement     a task with `chance >= AGREE_CHANCE` (a guess is right that often: CRC init and xorout 0.5, pktsum 0.2-0.36, fsm
                0.17-0.27) is kept only when at least AGREE_MIN of the K samples have a right headline: one lucky guess is not evidence
                that the model read the code. Tasks below the threshold keep any right sample, as before. Agreement filters luck, it
                does not make a 0.55 task hard: a model that always answers "all ones" still agrees with itself.
  distinct      identical stored texts of one task are kept once.
"""
from __future__ import annotations

import collections

from .scoring import sft_target

AGREE_CHANCE = 0.2
AGREE_MIN = 2


def needs_agreement(t: dict) -> bool:
    c = t.get("chance")
    return c is not None and c >= AGREE_CHANCE


def select(outs: list, t: dict) -> tuple:
    """([records], reason) for one task. `outs` = [(reply, scored right)] for its K samples. A record is
    {"text", "headline", "truncated"}; reason is "kept", "no_correct", "agreement" (too few agreeing right samples for the task's
    chance level) or "vetoed" (every right reply had a justification that is not storable)."""
    good = [o for o, ok in outs if ok]
    if not good:
        return [], "no_correct"
    if needs_agreement(t) and len(good) < AGREE_MIN:
        return [], "agreement"
    seen, kept = set(), []
    for reply in good:
        r = sft_target(reply, t)
        if r is None or r["text"] in seen:
            continue
        seen.add(r["text"])
        kept.append({"text": r["text"], "headline": r["headline"], "truncated": r["truncated"]})
    return kept, ("kept" if kept else "vetoed")


class Stats:
    """The sample stage's counters, in total and per task family (`kind`)."""

    FIELDS = ("tasks", "answers", "correct", "solved_tasks", "kept_tasks", "kept_records", "truncated", "no_correct", "agreement", "vetoed")

    def __init__(self):
        self.total = collections.Counter()
        self.by_kind = collections.defaultdict(collections.Counter)

    def add(self, t: dict, outs: list, kept: list, reason: str) -> None:
        for c in (self.total, self.by_kind[t.get("kind", "?")]):
            c["tasks"] += 1
            c["answers"] += len(outs)
            c["correct"] += sum(ok for _o, ok in outs)
            c["solved_tasks"] += any(ok for _o, ok in outs)
            c["kept_tasks"] += bool(kept)
            c["kept_records"] += len(kept)
            c["truncated"] += sum(r["truncated"] for r in kept)
            if reason in ("no_correct", "agreement", "vetoed"):
                c[reason] += 1

    def as_dict(self) -> dict:
        row = lambda c: {f: c[f] for f in self.FIELDS}
        return {**row(self.total), "by_kind": {k: row(c) for k, c in sorted(self.by_kind.items())}}


def assemble(tasks: list, results: dict, model: str, messages_for) -> tuple:
    """(SFT records, stats dict) from the K replies of every task. `results` = {task id: {"outs": [[reply, scored right], ...]}} (the
    sample stage's checkpoint format); `messages_for(task)` = the system and user message. A task without a result is skipped."""
    st, kept = Stats(), []
    for t in tasks:
        rec = results.get(t["id"])
        if rec is None:
            continue
        outs = [(o, bool(ok)) for o, ok in rec["outs"]]
        records, reason = select(outs, t)
        st.add(t, outs, records, reason)
        for r in records:
            kept.append({"messages": [*messages_for(t), {"role": "assistant", "content": r["text"]}], "task": t["id"], "kind": t["kind"],
                         "model": model, "truncated": r["truncated"]})
    return kept, st.as_dict()

"""Which sampled replies become SFT pairs, and what exactly is stored (stdlib only; unit-tested without Modal).

`modal_rea_data.py --stage sample` asks a model K times per task. This module decides, per task, from the K replies and the STRICT scorer
(`scoring.score_strict`, the whitelist grammar, for every family including legacy; the frozen substring scorer is for `modal_rea.py`'s
evaluation only):

  filter        the teacher is a FILTER (round 4): a task enters the data only when at least one reply has a headline the strict scorer accepts
                (`scoring.accepted_headline`). That proves the task is answerable from the pseudocode. Every reply is scored again from its text,
                whatever flag the checkpoint holds: a `--resume` run or an older checkpoint never carries the verdict of an older scorer into the
                data (round 3, m3). The text of a failed request is never a reply.
  stored text   NEVER the teacher's text. The assistant message is rendered from the task's own truth (`targets.render`): the canonical headline and
                one factual sentence of a per-variant template set, chosen by a seeded RNG from the task id. The record keeps `target_source: "truth-template"`
                and nothing else of the teacher: round 9 (m1) dropped `teacher_headline`, the raw first headline, from the record, so the dataset that is uploaded
                holds no teacher text at all (`RECORD_KEYS`, `check_records`). Free prose repeatedly carried wrong facts past every filter (a sentence that
                contradicted its headline, numbers in swapped roles), so none of it is stored any more.
  agreement     a task with `chance >= AGREE_CHANCE` (the best constant guess within its variant is right that often: CRC init and xorout
                0.4-0.6, pktsum 0.2-0.5, fsm 0.2-0.4, TEA/XTEA with the round count guessed 0.3, ...; legacy `skip` 0.14) is kept only when at
                least AGREE_MIN of the K samples have an accepted headline: one lucky guess is not evidence that the model read the code. Tasks
                below the threshold keep any right sample. Agreement filters luck, it does not make a 0.55 task hard: a model that always answers
                "all ones" still agrees with itself.
  one record    the rendered text of a task is the same for every accepted sample, so a task gives ONE record.

REASONING MODE (`reasoning=True`, `--stage sample --reasoning`, rea_bench/reasoning.py): the reply's final answer (the text after its last `Answer:` line) is the filter,
with the same agreement rule, counted on the answers. A kept task gives the truth-rendered record above and up to `max_traces` (3) records that hold an accepted reply VERBATIM
(`target_source: "teacher-reasoning"`; `<think>` blocks stripped, whitespace trimmed), distinct (round 12, RFT: several correct reasoning paths per task): each passes the trace quality
filters (mixed language, repetition loop, unreadable, self-contradiction) and is not a duplicate (identical normalized text, or token-set Jaccard >= 0.9) of one already kept; every other
accepted reply is counted per family under its reason (`stats["reasoning"]`). The steps in front of the answer are not checked: some may be wrong. A task whose traces were all filtered
keeps its truth record. `thinking=True` (a pilot): the trace is `<think>reasoning</think>` + the visible reply.
"""
from __future__ import annotations

import collections

from .reasoning import DEFAULT_MAX_TRACES, TARGET_SOURCE as TEACHER_SOURCE, TRACE_DROPS, pick_traces, score_reasoning, verdict
from .scoring import is_error_reply, score_strict, sft_target, strict_task, truth_target

AGREE_CHANCE = 0.05          # round 4 (m8): 0.06-0.063 variants (fsm/opcode_value, bounds/char_class, multi/limit_helper) kept one lucky sample of six in 31% of the tasks
AGREE_MIN = 2
LENGTH_OVERHEAD = 32     # tokens the chat template adds around the three messages (three role markers, the empty thinking block, the end-of-turn token): a margin, counted upward
NEAR_LIMIT = 0.95        # a record from this share of the window up is reported as near the limit: the sampler's tokenizer is not the training tokenizer, byte for byte
RECORD_KEYS = ("messages", "task", "kind", "model", "target_source")      # what the uploaded SFT file holds per record (round 9, m1: no `teacher_headline`)
TARGET_SOURCES = ("truth-template", TEACHER_SOURCE)      # where a record's assistant text comes from: rendered from the truth, or the teacher's reply (`--reasoning`, round 11)


def check_records(records: list) -> None:
    """Fail when an SFT record has a key outside RECORD_KEYS or a `target_source` outside TARGET_SOURCES: whatever else a record carried would be uploaded with the dataset. (A record's
    assistant message is the truth-rendered text, or, in a reasoning run, the teacher's reply as `assemble` got it from `reasoning.teacher_target`; nothing else is stored.)"""
    for r in records:
        if sorted(r) != sorted(RECORD_KEYS):
            raise ValueError(f"SFT record of {r.get('task', '?')} has keys {sorted(r)}, not {sorted(RECORD_KEYS)}: nothing but these may be stored")
        if r["target_source"] not in TARGET_SOURCES:
            raise ValueError(f"SFT record of {r.get('task', '?')} has target_source {r['target_source']!r}, not one of {list(TARGET_SOURCES)}")


def drop_over_length(records: list, stats: dict, count, max_len: int, overhead: int = LENGTH_OVERHEAD, report_windows: tuple = ()) -> tuple:
    """(records that fit, stats with an `over_length` block) (round 9, m2). The training run drops a pair whose prompt + answer + EOS is longer than `max_len` tokens
    (modal_lora.encode_ids: dropped, never cut); this measures the same thing at the sample stage, so the loss is a number before the GPU run and the file that is uploaded
    holds no record that training would throw away. `count(text)` is a token count of the sampler's tokenizer; a record's length is the sum over its messages plus `overhead`.
    The `over_length` block: max_len, dropped, near_limit (kept, within NEAR_LIMIT of the window), longest_tokens, by_kind {kind: dropped}; `kept_tasks`, `kept_records` and
    the same fields of each family are reduced by what was dropped (a task is lost when ALL of its records are: a reasoning run has two per task, and the long one goes first).
    `report_windows` (a reasoning run: REPORT_WINDOWS): the block also holds `fits`, per `target_source` the number of records and how many are within each window, and `by_source`
    (dropped per source), so the cost of a longer training window is known before it is chosen."""
    kept, over, longest, near = [], collections.Counter(), 0, 0
    fits = collections.defaultdict(collections.Counter)
    over_source = collections.Counter()
    for r in records:
        n = sum(count(m["content"]) for m in r["messages"]) + overhead
        longest = max(longest, n)
        if report_windows:
            fits[r["target_source"]]["records"] += 1
            for w in report_windows:
                fits[r["target_source"]][str(w)] += n <= w
        if n > max_len:
            over[r["kind"]] += 1
            if report_windows:
                over_source[r["target_source"]] += 1
        else:
            kept.append(r)
            near += n >= NEAR_LIMIT * max_len
    out = {**stats, "by_kind": {k: dict(row) for k, row in stats.get("by_kind", {}).items()}}
    dropped = sum(over.values())
    left, lost, seen = {r["task"] for r in kept}, collections.Counter(), set()
    for r in records:
        if r["task"] not in left and r["task"] not in seen:         # every record of this task was dropped
            seen.add(r["task"])
            lost[r["kind"]] += 1
    out["kept_tasks"] -= sum(lost.values())
    out["kept_records"] -= dropped
    for kind, c in over.items():
        out["by_kind"][kind]["kept_tasks"] -= lost[kind]
        out["by_kind"][kind]["kept_records"] -= c
    out["over_length"] = {"max_len": max_len, "dropped": dropped, "near_limit": near, "longest_tokens": longest, "by_kind": dict(sorted(over.items()))}
    if report_windows:
        out["over_length"]["fits"] = {src: dict(c) for src, c in sorted(fits.items())}
        out["over_length"]["by_source"] = dict(sorted(over_source.items()))
    return kept, out


def over_length_line(stats: dict):
    """The line the sample stage prints about the length rule (None when the stats carry no `over_length` block)."""
    o = stats.get("over_length")
    if o is None:
        return None
    if o.get("error"):
        return (f"WARNING: the token length of the SFT records was NOT measured ({o['error']}); the training run still drops every pair over its window, "
                f"but the loss is not known before it")
    by = ", ".join(f"{k} {c}" for k, c in o["by_kind"].items())
    return (f"over-length: {o['dropped']} of {o['dropped'] + stats.get('kept_records', 0)} records are longer than {o['max_len']} tokens (prompt + answer + EOS) and were dropped"
            + (f" ({by})" if by else "") + f"; {o['near_limit']} kept records are within {int(100 - NEAR_LIMIT * 100)}% of the window; the longest has {o['longest_tokens']} tokens")


def fits_line(stats: dict):
    """The line a reasoning run prints about the training window: how many teacher-reasoning records fit each window the run measured (a visible-reasoning run: 2048, 3072 and 4096 tokens;
    a thinking run: 4096 and 8192, `reasoning.THINKING_REPORT_WINDOWS`), as a share of the records: the projection of what `modal_lora.py --max-len` would keep. None without that measurement."""
    o = stats.get("over_length") or {}
    row = (o.get("fits") or {}).get(TEACHER_SOURCE)
    if not row or not row.get("records"):
        return None
    total = row["records"]
    parts = ", ".join(f"{w} tokens: {row.get(str(w), 0)} of {total} ({100 * row.get(str(w), 0) // total}%)" for w in sorted(int(k) for k in row if k.isdigit()))
    return f"teacher-reasoning records that fit the training window (prompt + answer + EOS): {parts}; this run dropped at --max-len {o['max_len']}"


def needs_agreement(t: dict) -> bool:
    c = strict_task(t).get("chance")                # a legacy task gets its `chance` from the conversion (skip: 0.14)
    return c is not None and c >= AGREE_CHANCE


def select_traces(outs: list, t: dict, thinking: bool = False, max_traces: int = DEFAULT_MAX_TRACES) -> tuple:
    """([records], reason, Counter of the accepted replies that were not stored, by reason) of one task of a reasoning run. A reply is ACCEPTED by `score_reasoning` (its text after the last
    `Answer:` line); the task is kept (the truth-rendered record) when it has an accepted reply, and for a task with `chance >= AGREE_CHANCE` when AGREE_MIN replies are accepted (the agreement rule
    counts answers: whether a trace is clean is not a statement about the answer). The teacher records are the first `max_traces` distinct clean accepted replies in sample order
    (`reasoning.pick_traces`); a kept task whose accepted replies are all filtered has the truth record alone. A task that is not kept has no traces and nothing is counted for its replies."""
    accepted = [o for o, _flag in outs if score_reasoning(o, t)]
    if not accepted:
        return [], "no_correct", collections.Counter()
    if needs_agreement(t) and len(accepted) < AGREE_MIN:
        return [], "agreement", collections.Counter()
    traces, dropped = pick_traces(accepted, t, max_traces, thinking)
    return [truth_target(t), *traces], "kept", dropped


def select(outs: list, t: dict, reasoning: bool = False, max_traces: int = DEFAULT_MAX_TRACES, thinking: bool = False) -> tuple:
    """([records], reason) for one task. `outs` = [(reply, flag)] for its K samples; the flag is not used (every reply is scored again). A record
    is {"text", "headline", "sentence", "template", "target_source"}: all of it rendered from the truth. reason is "kept",
    "no_correct" (no sample has an accepted headline) or "agreement" (too few accepted samples for the task's chance level).
    `reasoning`: a reply is accepted by `score_reasoning` (its text after the last `Answer:` line), and a kept task gives the truth-rendered record AND up to `max_traces` distinct clean
    accepted replies verbatim (`select_traces`)."""
    if reasoning:
        records, reason, _dropped = select_traces(outs, t, thinking, max_traces)
        return records, reason
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


def reasoning_report(stats: dict) -> list:
    """The lines a reasoning run prints about its replies (empty for any other run): how they were judged, what happened to the accepted ones, and what that run's own throughput was."""
    r = stats.get("reasoning")
    if not r:
        return []
    out = [f"{r.get('mode', 'reasoning')} replies: {r['replies']:,}: accepted {r['accepted']:,}, answer rejected {r['wrong']:,}, no final `Answer:` line (or text after it) {r['no_answer_line']:,}, "
           f"cut off at max_tokens {r['truncated']:,}, failed requests {r['error']:,}; {r['teacher_records']:,} teacher-reasoning records (up to {r.get('max_traces', 1)} per kept task)"]
    d = r.get("dropped")
    if d:
        out.append("accepted replies not stored: " + ", ".join(f"{k} {d.get(k, 0):,}" for k in TRACE_DROPS) + f"; in tasks that were not kept {r.get('in_lost_tasks', 0):,}; "
                   f"{r.get('tasks_without_trace', 0):,} kept tasks have no trace (their truth record stays)")
    if r["truncated"] and r["replies"] and r["truncated"] / r["replies"] > TRUNCATED_WARN:
        out.append(f"WARNING: {100 * r['truncated'] // r['replies']}% of the replies were cut off at max_tokens: raise --max-tokens (and look at the cost line again)")
    if r.get("mode") == "thinking" and d and r["accepted"] and d.get("no_think", 0) / r["accepted"] > NO_THINK_WARN:
        out.append(f"WARNING: {100 * d['no_think'] // r['accepted']}% of the accepted replies came back without reasoning: the server did not return `reasoning_content` or a <think> block "
                   f"(llama-server --reasoning-format), so they were dropped")
    s = stats.get("sampling")
    if s:
        out.append("sampling: " + ", ".join(f"{k} {v}" for k, v in s.items()))
    sl = stats.get("slot")
    if sl and sl.get("prompt_too_long"):
        out.append(f"WARNING: {sl['prompt_too_long']} task(s) were not asked: their prompt is longer than {sl['max_prompt_tokens']:,} tokens, which with max_tokens {s['max_tokens'] if s else '?'} does not fit one "
                   f"{sl['slot_context']:,}-token slot of the server ({', '.join(f'{k} {c}' for k, c in sl['by_kind'].items())})")
    p = stats.get("perf")
    if p and p.get("requests"):
        out.append(f"this call: {p['requests']:,} requests, {p['completion_tokens']:,} completion tokens in {p['wall_s']:,} s = {p['tok_per_s']} tokens/s over the four slots "
                   f"({p['s_per_request']} s per request): reasoning.PHASE_A_REPLY_S is 1.04 s per phase-A reply")
    return out


TRUNCATED_WARN = 0.10     # a reasoning run with more than this share of its replies cut off at max_tokens says so
NO_THINK_WARN = 0.10      # a thinking run in which more than this share of the accepted replies has no reasoning block says so


def _fate_row(judged: collections.Counter, fate: collections.Counter) -> dict:
    """The `reasoning` counters of one family (or of all): how the replies were judged and what became of the accepted ones."""
    row = {"replies": sum(judged.values()), **{v: judged[v] for v in ("accepted", "wrong", "no_answer_line", "truncated", "error")}, "teacher_records": fate["traces"],
           "dropped": {k: fate[k] for k in TRACE_DROPS}, "in_lost_tasks": fate["in_lost_tasks"], "tasks_without_trace": fate["tasks_without_trace"]}
    return row


def assemble(tasks: list, results: dict, model: str, messages_for, reasoning: bool = False, thinking: bool = False, max_traces: int = DEFAULT_MAX_TRACES) -> tuple:
    """(SFT records, stats dict) from the K replies of every task. `results` = {task id: {"outs": [[reply, flag], ...]}} (the sample stage's
    checkpoint format; the flag is ignored, see above); `messages_for(task)` = the system and user message. A task without a result is skipped.
    `reasoning`: see `select_traces`; the stats get a `reasoning` block (how the replies were judged, what became of the accepted ones, in total and per family), and no other key of a default
    run changes. `thinking`: the run kept the server's reasoning (`mode: "thinking"` in the block). `max_traces`: teacher records per kept task."""
    st, kept = Stats(), []
    judged, fate = collections.Counter(), collections.Counter()
    kind_judged, kind_fate = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    score = score_reasoning if reasoning else score_strict
    for t in tasks:
        rec = results.get(t["id"])
        if rec is None:
            continue
        outs = [(o, score(o, t)) for o, _flag in rec["outs"]]
        if reasoning:
            verdicts = collections.Counter(verdict(o, t) for o, _flag in outs)
            records, reason, dropped = select_traces(outs, t, thinking, max_traces)
            traces = sum(r["target_source"] == TEACHER_SOURCE for r in records)
            here = collections.Counter({"traces": traces, **dropped})
            if reason == "agreement":
                here["in_lost_tasks"] = verdicts["accepted"]
            if reason == "kept" and not traces:
                here["tasks_without_trace"] = 1
            for j, f in ((judged, fate), (kind_judged[t["kind"]], kind_fate[t["kind"]])):
                j.update(verdicts)
                f.update(here)
        else:
            records, reason = select(outs, t)
        st.add(t, outs, records, reason)
        for r in records:
            kept.append({"messages": [*messages_for(t), {"role": "assistant", "content": r["text"]}], "task": t["id"], "kind": t["kind"],
                         "model": model, "target_source": r["target_source"]})
    check_records(kept)
    stats = st.as_dict()
    if reasoning:
        stats["reasoning"] = {"mode": "thinking" if thinking else "reasoning", "max_traces": max_traces, **_fate_row(judged, fate),
                              "by_kind": {k: _fate_row(kind_judged[k], kind_fate[k]) for k in sorted(kind_judged)}}
    return kept, stats

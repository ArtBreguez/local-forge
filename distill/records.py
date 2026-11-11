"""Which sampled replies become SFT records, and what exactly is stored. Stdlib only, no subject in it (moved from `rea_bench/sft.py`; the subject is the `domain` argument: `distill.domain.Domain`).

The sample stage asks a model K times per task. This module decides, per task, from the K replies and the domain's `verify`:

  filter        the teacher is a FILTER: a task enters the data only when at least one reply is verified (plain mode: the whole reply; reasoning and thinking: the text after the last `Answer:` line,
                `traces.score_reasoning`). That proves the task is answerable. Every reply is scored again from its text, whatever flag the checkpoint holds: a `--resume` run or an older checkpoint never
                carries the verdict of an older scorer into the data. The text of a failed request is never a reply.
  stored text   plain mode: NEVER the teacher's text. The assistant message is the domain's `truth_target`, rendered from the task's own truth; the record keeps `target_source: "truth-template"` and nothing else
                of the teacher (`RECORD_KEYS`, `check_records`: whatever else a record carried would be uploaded with the dataset). Free prose repeatedly carried wrong facts past every filter in REA, so
                none of it is stored in this mode.
  agreement     a task for which `domain.needs_agreement` is true (a guess is right often) is kept only when at least AGREE_MIN of the K samples are verified: one lucky guess is not evidence that the model
                solved it. Agreement filters luck, it does not make a task hard: a model that always answers the same constant still agrees with itself.
  one record    the rendered text of a task is the same for every verified sample, so plain mode gives ONE record per task.

REASONING MODE (`reasoning=True`, `thinking=True` for the model's native reasoning): the final answer is the filter, with the same agreement rule, counted on the answers. A kept task gives the truth record (when the
domain has a `truth_target`) and up to `max_traces` (3) records that hold a verified reply VERBATIM (`target_source: "teacher-reasoning"`; `<think>` blocks stripped, whitespace trimmed), distinct (RFT: several
correct reasoning paths per task): each passes the trace quality filters (mixed language, repetition loop, unreadable, self-contradiction) and is not a duplicate (identical normalized text, or token-set Jaccard >= 0.9)
of one already kept; every other verified reply is counted per family under its reason (`stats["reasoning"]`). The steps in front of the answer are not checked: some may be wrong. A task whose traces were all
filtered keeps its truth record. `thinking=True`: the trace is `<think>reasoning</think>` + the visible reply.
"""
from __future__ import annotations

import collections

from .traces import DEFAULT_MAX_TRACES, TARGET_SOURCE as TEACHER_SOURCE, TRACE_DROPS, is_error_reply, pick_traces, score_reasoning, verdict

AGREE_MIN = 2                # samples that must be verified for a task `domain.needs_agreement` marks (REA: its chance level is 0.05 or more)
LENGTH_OVERHEAD = 32     # tokens the chat template adds around the three messages (three role markers, the empty thinking block, the end-of-turn token): a margin, counted upward
NEAR_LIMIT = 0.95        # a record from this share of the window up is reported as near the limit: the sampler's tokenizer is not the training tokenizer, byte for byte
RECORD_KEYS = ("messages", "task", "kind", "model", "target_source")      # what the uploaded SFT file holds per record (REA round 9, m1: no `teacher_headline`)
TRUTH_SOURCE = "truth-template"      # a record whose assistant text is the domain's `truth_target`
TARGET_SOURCES = (TRUTH_SOURCE, TEACHER_SOURCE)      # where a record's assistant text comes from: rendered from the truth, or the teacher's reply (`--reasoning`, round 11)


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
    (modal_train.encode_ids: dropped, never cut); this measures the same thing at the sample stage, so the loss is a number before the GPU run and the file that is uploaded
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
    a thinking run: 4096 and 8192, `config.THINKING_REPORT_WINDOWS`), as a share of the records: the projection of what `modal_train --max-len` would keep. None without that measurement."""
    o = stats.get("over_length") or {}
    row = (o.get("fits") or {}).get(TEACHER_SOURCE)
    if not row or not row.get("records"):
        return None
    total = row["records"]
    parts = ", ".join(f"{w} tokens: {row.get(str(w), 0)} of {total} ({100 * row.get(str(w), 0) // total}%)" for w in sorted(int(k) for k in row if k.isdigit()))
    return f"teacher-reasoning records that fit the training window (prompt + answer + EOS): {parts}; this run dropped at --max-len {o['max_len']}"


def truth_record(t: dict, domain):
    """The truth-rendered record of a task, {"text", "target_source"}, or None when the domain has no `truth_target`. The text must pass `domain.verify`: a truth that its own scorer rejects is a bug, found before
    anything is stored."""
    text = domain.truth_target(t)
    if text is None:
        return None
    if not domain.verify(text, t):
        raise ValueError(f"the truth target of task {t.get('id', '?')} renders a text that verify rejects: {text!r}")
    return {"text": text, "target_source": TRUTH_SOURCE}


def select_traces(outs: list, t: dict, thinking: bool = False, max_traces: int = DEFAULT_MAX_TRACES, *, domain) -> tuple:
    """([records], reason, Counter of the accepted replies that were not stored, by reason) of one task of a reasoning run. A reply is ACCEPTED by `score_reasoning` (its text after the last
    `Answer:` line); the task is kept (the truth record) when it has an accepted reply, and for a task with `domain.needs_agreement` when AGREE_MIN replies are accepted (the agreement rule
    counts answers: whether a trace is clean is not a statement about the answer). The teacher records are the first `max_traces` distinct clean accepted replies in sample order
    (`traces.pick_traces`); a kept task whose accepted replies are all filtered has the truth record alone. A task that is not kept has no traces and nothing is counted for its replies."""
    accepted = [o for o, _flag in outs if score_reasoning(o, t, domain=domain)]
    if not accepted:
        return [], "no_correct", collections.Counter()
    if domain.needs_agreement(t) and len(accepted) < AGREE_MIN:
        return [], "agreement", collections.Counter()
    traces, dropped = pick_traces(accepted, t, max_traces, thinking, domain=domain)
    truth = truth_record(t, domain)
    return [*([truth] if truth else []), *traces], "kept", dropped


def plain_ok(reply: str, t: dict, domain) -> bool:
    """A reply of a plain run is right when it is not the text of a failed request and `domain.verify` accepts the whole of it."""
    return not is_error_reply(reply) and domain.verify(reply, t)


def select(outs: list, t: dict, reasoning: bool = False, max_traces: int = DEFAULT_MAX_TRACES, thinking: bool = False, *, domain) -> tuple:
    """([records], reason) for one task. `outs` = [(reply, flag)] for its K samples; the flag is not used (every reply is scored again). A record
    is {"text", "target_source"}. reason is "kept", "no_correct" (no sample is verified) or "agreement" (too few verified samples for a task that needs it).
    `reasoning`: a reply is accepted by `score_reasoning` (its text after the last `Answer:` line), and a kept task gives the truth record AND up to `max_traces` distinct clean
    accepted replies verbatim (`select_traces`). A plain run stores the truth record only, so the domain must have a `truth_target`."""
    if reasoning:
        records, reason, _dropped = select_traces(outs, t, thinking, max_traces, domain=domain)
        return records, reason
    accepted = [o for o, _flag in outs if plain_ok(o, t, domain)]
    if not accepted:
        return [], "no_correct"
    if domain.needs_agreement(t) and len(accepted) < AGREE_MIN:
        return [], "agreement"
    truth = truth_record(t, domain)
    if truth is None:
        raise ValueError(f"domain {domain.name!r} has no truth_target: a plain run stores nothing of the teacher, so it has nothing to store; use mode reasoning")
    return [truth], "kept"


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
    budgets = (stats.get("sampling") or {}).get("max_tokens_by_family")
    if budgets:         # a family has a budget of its own: the share cut off is read against that budget
        cut = [f"{k} {b['truncated']}/{b['replies']} at {budgets.get(k, stats['sampling']['max_tokens'])}" for k, b in r.get("by_kind", {}).items() if b["replies"] and b["truncated"] / b["replies"] > TRUNCATED_WARN]
        if cut:
            out.append("WARNING: families with more than " + f"{int(TRUNCATED_WARN * 100)}% of their replies cut off at their own max_tokens: " + ", ".join(cut))
    if r.get("mode") == "thinking" and d and r["accepted"] and d.get("no_think", 0) / r["accepted"] > NO_THINK_WARN:
        out.append(f"WARNING: {100 * d['no_think'] // r['accepted']}% of the accepted replies came back without reasoning: the server did not return `reasoning_content` or a <think> block "
                   f"(llama-server --reasoning-format), so they were dropped")
    s = stats.get("sampling")
    if s:
        out.append("sampling: " + ", ".join(f"{k} {v}" for k, v in s.items()))
    sl = stats.get("slot")
    if sl and sl.get("prompt_too_long"):
        own = sl.get("max_prompt_tokens_by_kind") or {}          # a family with a budget of its own leaves its own room for the prompt
        what = ", ".join(f"{k} {c}" + (f" (over {own[k]:,} tokens)" if k in own else "") for k, c in sl["by_kind"].items())
        out.append(f"WARNING: {sl['prompt_too_long']} task(s) were not asked: their prompt is longer than {sl['max_prompt_tokens']:,} tokens, which with max_tokens {s['max_tokens'] if s else '?'} does not fit one "
                   f"{sl['slot_context']:,}-token slot of the server ({what})")
    p = stats.get("perf")
    if p and p.get("requests"):
        out.append(f"this call: {p['requests']:,} requests, {p['completion_tokens']:,} completion tokens in {p['wall_s']:,} s = {p['tok_per_s']} tokens/s over the four slots "
                   f"({p['s_per_request']} s per request): modal_sample.PHASE_A_REPLY_S is 1.04 s per phase-A reply")
    return out


TRUNCATED_WARN = 0.10     # a reasoning run with more than this share of its replies cut off at max_tokens says so
NO_THINK_WARN = 0.10      # a thinking run in which more than this share of the accepted replies has no reasoning block says so


def _fate_row(judged: collections.Counter, fate: collections.Counter) -> dict:
    """The `reasoning` counters of one family (or of all): how the replies were judged and what became of the accepted ones."""
    row = {"replies": sum(judged.values()), **{v: judged[v] for v in ("accepted", "wrong", "no_answer_line", "truncated", "error")}, "teacher_records": fate["traces"],
           "dropped": {k: fate[k] for k in TRACE_DROPS}, "in_lost_tasks": fate["in_lost_tasks"], "tasks_without_trace": fate["tasks_without_trace"]}
    return row


def assemble(tasks: list, results: dict, model: str, messages_for, reasoning: bool = False, thinking: bool = False, max_traces: int = DEFAULT_MAX_TRACES, *, domain) -> tuple:
    """(SFT records, stats dict) from the K replies of every task. `results` = {task id: {"outs": [[reply, flag], ...]}} (the sample stage's
    checkpoint format; the flag is ignored, see above); `messages_for(task)` = the system and user message. A task without a result is skipped.
    `reasoning`: see `select_traces`; the stats get a `reasoning` block (how the replies were judged, what became of the accepted ones, in total and per family), and no other key of a default
    run changes. `thinking`: the run kept the server's reasoning (`mode: "thinking"` in the block). `max_traces`: teacher records per kept task."""
    st, kept = Stats(), []
    judged, fate = collections.Counter(), collections.Counter()
    kind_judged, kind_fate = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    score = (lambda o, t: score_reasoning(o, t, domain=domain)) if reasoning else (lambda o, t: plain_ok(o, t, domain))
    for t in tasks:
        rec = results.get(t["id"])
        if rec is None:
            continue
        outs = [(o, score(o, t)) for o, _flag in rec["outs"]]
        if reasoning:
            verdicts = collections.Counter(verdict(o, t, domain=domain) for o, _flag in outs)
            records, reason, dropped = select_traces(outs, t, thinking, max_traces, domain=domain)
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
            records, reason = select(outs, t, domain=domain)
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

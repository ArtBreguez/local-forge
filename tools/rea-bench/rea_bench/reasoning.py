"""The teacher-reasoning mode of the sample stage (`modal_rea_data.py --stage sample --reasoning`), stdlib only, unit-tested without Modal.

Phase A used the teacher as a FILTER and stored a sentence rendered from the truth (sft.py). This mode keeps what the teacher wrote, with the rule of
rejection-sampling distillation (STaR, RFT, the R1 distills) and nothing more: **the reply's final answer is checked against the computed truth, the
reasoning in front of it is kept as it is and is not checked**. Some of its steps may be wrong while the answer is right; the dataset card says which
records those are (`target_source: "teacher-reasoning"`), next to the truth-rendered record of the same task (`"truth-template"`).

  asking      `SYSTEM_REASONING` replaces the system message (SYSTEM_RECIPE of modal_rea.py is untouched): a short step-by-step analysis, then a LAST
              line `Answer: <value>`. Thinking stays off (`enable_thinking: False`); the steps are the visible answer. The messages are the ones stored in the
              records, for both sources.
  scoring     `score_reasoning`: `scoring.score_strict` (the whitelist grammar, not loosened) applied to the text after the LAST `Answer:` line only. A reply
              with no such line, with anything after it, with a `<think>` tag left in it, cut off at `max_tokens` (`TRUNCATED`) or that is the text of a failed
              request is rejected; an `Answer:` earlier in the steps counts for nothing.
  stored      the first accepted reply, verbatim: `<think>` blocks stripped, whitespace trimmed, nothing else (`teacher_target`).
  sizing      `--quota kind=N,...` caps the tasks per family (`apply_quota`, round-robin so that a run stopped early is balanced) and `project` turns
              tasks x k x max_tokens into GPU hours and dollars before a GPU is rented (the assumptions are the constants below).
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re

from .scoring import is_error_reply, score_strict

TARGET_SOURCE = "teacher-reasoning"
DEFAULT_MAX_TOKENS = 768          # `--max-tokens` in reasoning mode (the phase-A sampler used 300 for a headline and one sentence)
REPORT_WINDOWS = (2048, 3072, 4096)     # the sample stage reports how many teacher records fit each of these training windows
TRUNCATED = "[truncated at max_tokens]"  # appended by the sampler to a reply the server cut off (finish_reason "length"); it is not an `Answer:` line, so the reply is rejected

SYSTEM_REASONING = ("You are a reverse engineer. You are given decompiled C pseudocode of one function from a stripped binary and a question. "
                    "First write a short step-by-step analysis, at most eight short lines: what the code reads, which constants it uses and why they give the answer. "
                    "Then write one last line of the form `Answer: <the concrete value or name asked for, in the form the question asks>` and nothing after it.")

# ---- the reply: the text after the last `Answer:` line -----------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S)
_ANSWER_LINE = re.compile(r"^[ \t]*(?:\*\*|__)?Answer:(?:\*\*|__)?(?P<rest>.*)$")      # `Answer:` at the start of a line; markdown bold around the label is not part of the answer


def clean_reply(text: str) -> str:
    """The reply as it is stored: `<think>` blocks removed, whitespace at both ends trimmed, nothing else touched."""
    return _THINK.sub("", text).strip()


def final_answer(reply: str):
    """The text after the label of the LAST `Answer:` line, when that line is the last line of the reply and has something after its label; else None.
    None for: no `Answer:` line, text after the last one, an empty one, a `<think>` tag still in the reply (an unterminated block is not stripped). An `Answer:` earlier
    in the steps is not looked at when a later one exists, and is a reply that ends in other text when none does."""
    text = clean_reply(reply)
    if "<think>" in text or "</think>" in text:
        return None
    lines = text.splitlines()
    last = [i for i, line in enumerate(lines) if _ANSWER_LINE.match(line)]
    if not last or last[-1] != len(lines) - 1:
        return None
    return _ANSWER_LINE.match(lines[-1]).group("rest").strip() or None


def verdict(reply: str, t: dict) -> str:
    """Why a reply is or is not an answer: "error" (the text of a failed request), "truncated", "no_answer_line" (no final `Answer:` line, text after it, or an empty
    one), "wrong" (the line is there and score_strict rejects its text) or "accepted"."""
    if is_error_reply(reply):
        return "error"
    if clean_reply(reply).endswith(TRUNCATED):
        return "truncated"
    answer = final_answer(reply)
    if answer is None:
        return "no_answer_line"
    return "accepted" if score_strict(answer, t) else "wrong"


def score_reasoning(reply: str, t: dict) -> bool:
    """DATA scorer of the reasoning mode: `score_strict` on the text after the last `Answer:` line; every other reply is rejected (see `verdict`)."""
    return verdict(reply, t) == "accepted"


def teacher_target(reply: str, t: dict):
    """The record text of an accepted reply, or None: the reply verbatim (`clean_reply`). {"text", "headline", "sentence", "template", "target_source"}; the headline is
    the text after `Answer:`, there is no sentence or template (nothing of it is rendered)."""
    if not score_reasoning(reply, t):
        return None
    return {"text": clean_reply(reply), "headline": final_answer(reply), "sentence": None, "template": None, "target_source": TARGET_SOURCE}


# ---- the tasks a run samples ---------------------------------------------------------------------------------------------------------

def known_kinds() -> list:
    """Every `kind` a task can have: the ten new family names and the legacy template names (`xor-key`, `lcg`, ...)."""
    from . import families
    return sorted({m for m in families.modules() if m != "legacy"} | {t.kind for t in families.generate("legacy", 10, 7)})


def parse_quota(text: str, known=None) -> dict:
    """`crc=250,structs=100` -> {"crc": 250, "structs": 100} (order kept). SystemExit on an empty name, a count that is not a positive integer, a repeated kind, or (when `known` is
    given) a kind no task has."""
    out: dict = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        kind, eq, count = part.partition("=")
        kind = kind.strip()
        if not eq or not kind or not count.strip().isdigit() or int(count) < 1:
            raise SystemExit(f"bad --quota entry {part!r}: write kind=N with N a positive integer, e.g. --quota crc=250,structs=100")
        if kind in out:
            raise SystemExit(f"--quota names {kind!r} twice")
        if known is not None and kind not in known:
            raise SystemExit(f"--quota names {kind!r}, which is no task kind; known: {', '.join(known)}")
        out[kind] = int(count)
    if not out:
        raise SystemExit("--quota is empty: write kind=N[,kind=N...], or leave it out to sample every decompiled task")
    return out


def apply_quota(tasks: list, quota: dict) -> list:
    """At most `quota[kind]` tasks of each listed kind (the first ones, in the order `tasks` has: sorted by id); tasks of other kinds are not sampled. The result goes round-robin over
    the kinds (every kind's first task, then every kind's second, ...), so a run that is stopped early, or times out and is not resumed, holds a balanced set. No quota: `tasks`."""
    if not quota:
        return tasks
    by = collections.defaultdict(list)
    for t in tasks:
        if t.get("kind") in quota:
            by[t["kind"]].append(t)
    chosen = {kind: by[kind][:n] for kind, n in quota.items()}
    out = []
    for rank in range(max((len(v) for v in chosen.values()), default=0)):
        out += [chosen[kind][rank] for kind in quota if rank < len(chosen[kind])]
    return out


def local_task_count(decompiled: pathlib.Path, quota: dict, n: int) -> tuple:
    """(tasks to sample, where the number comes from) for the cost line, which is printed before any GPU is rented. The decompile stage wrote `decompiled-<tag>.jsonl` here: its
    tasks with pseudocode, after the quota. Without that file: the sum of the quota (an upper bound: tasks the Modal compilers dropped are not known), else `--n`."""
    path = pathlib.Path(decompiled)
    if path.exists():
        tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        tasks = [t for t in tasks if "pseudocode" in t]
        return len(apply_quota(tasks, quota)), f"{len(tasks)} decompiled tasks in {path.name}" + (f", capped by --quota" if quota else "")
    if quota:
        return sum(quota.values()), f"the sum of --quota ({path.name} is not here: tasks the gate drops are not counted)"
    return n, f"--n {n} ({path.name} is not here)"


# ---- the cost, before the GPU is rented --------------------------------------------------------------------------------------------

# MEASURED (owner's figures for the phase-A CodeForge run; not recorded in the repo's logs): 3,187 tasks x 6 short replies in about 5.5 GPU hours on one L40S.
PHASE_A_REPLIES = 3187 * 6
PHASE_A_GPU_HOURS = 5.5
PHASE_A_REPLY_S = PHASE_A_GPU_HOURS * 3600 / PHASE_A_REPLIES        # 1.04 s of GPU per reply, prefill and decode, four requests in flight
# ASSUMED (not measured: the phase-A run did not record `usage`): a phase-A reply is a headline and one sentence, about 50 tokens; a reasoning reply is 5 to 8 times as long
# (the owner's estimate), so its time is the phase-A time scaled by its length. Prefill does not scale with the reply, so this OVERestimates the time: the line is a budget guard.
PHASE_A_REPLY_TOKENS = 50
LENGTH_FACTORS = {"low": 5.0, "expected": 6.5, "high": 8.0}
L40S_USD_PER_H = 1.95                # Modal list price, US$0.000542 per second (https://modal.com/pricing); CPU and memory of the container add a few cents per hour


def reply_seconds(max_tokens: int, factor: float) -> float:
    """GPU seconds per reply of `factor` times the phase-A length, cut off at `max_tokens`."""
    return PHASE_A_REPLY_S * min(max_tokens, factor * PHASE_A_REPLY_TOKENS) / PHASE_A_REPLY_TOKENS


def project(n_tasks: int, k: int, max_tokens: int, usd_per_h: float = L40S_USD_PER_H, call_hours: float = 5.0) -> dict:
    """GPU hours and dollars of a sample run: `n_tasks` x `k` replies of 5x, 6.5x and 8x the phase-A length (low, expected, high), and the worst case in which every reply runs to
    `max_tokens`. `calls`: how many sampler calls the expected time needs (a call times out after `call_hours`; the next one continues with --resume)."""
    replies = n_tasks * k
    out = {"tasks": n_tasks, "k": k, "replies": replies, "max_tokens": max_tokens, "usd_per_h": usd_per_h, "call_hours": call_hours}
    for name, factor in {**LENGTH_FACTORS, "worst": max_tokens / PHASE_A_REPLY_TOKENS}.items():
        s = reply_seconds(max_tokens, factor)
        hours = replies * s / 3600
        out[name] = {"s_per_reply": round(s, 2), "gpu_hours": round(hours, 2), "usd": round(hours * usd_per_h, 2)}
    out["calls"] = max(1, math.ceil(out["expected"]["gpu_hours"] / call_hours))
    return out


def cost_line(p: dict, source: str = "") -> str:
    lo, ex, hi, worst = p["low"], p["expected"], p["high"], p["worst"]
    return (f"projected cost: {p['tasks']:,} tasks x k={p['k']} = {p['replies']:,} replies, {ex['s_per_reply']} s of GPU each = {ex['gpu_hours']} GPU-h on one L40S = about "
            f"US${ex['usd']:.2f} at US${p['usd_per_h']:.2f}/h (replies 5x to 8x as long as phase A's: {lo['gpu_hours']}-{hi['gpu_hours']} GPU-h, US${lo['usd']:.2f}-US${hi['usd']:.2f}; "
            f"worst case, every reply runs to max_tokens={p['max_tokens']}: {worst['gpu_hours']} GPU-h, US${worst['usd']:.2f}). Assumes phase A's measured {PHASE_A_REPLY_S:.2f} s per reply "
            f"({PHASE_A_GPU_HOURS} GPU-h for {PHASE_A_REPLIES:,} replies) for a reply of ~{PHASE_A_REPLY_TOKENS} tokens, scaled by length (an overestimate: prefill does not scale); "
            f"GPU only, no decompile. A sampler call times out after {p['call_hours']:g} h: this takes ~{p['calls']} call(s), each after the first with --resume."
            + (f" Tasks: {source}." if source else ""))


def budget_warning(p: dict, budget_usd: float):
    """A warning text when even the high estimate is over budget (or the expected one, a stronger text), else None. The decompile stage costs about US$1 per 1,200 tasks on top."""
    if p["expected"]["usd"] > budget_usd:
        return f"WARNING: the expected cost US${p['expected']['usd']:.2f} is over the budget US${budget_usd:.2f}: lower --quota or --k, or raise --budget-usd on purpose"
    if p["high"]["usd"] > budget_usd:
        return f"WARNING: the expected cost fits the budget US${budget_usd:.2f}, the high estimate (replies 8x as long) US${p['high']['usd']:.2f} does not; --quota or --k lower"
    return None

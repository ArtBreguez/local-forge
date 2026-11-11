"""The teacher-reasoning mode of the sample stage (`modal_rea_data.py --stage sample --reasoning`), stdlib only, unit-tested without Modal.

Phase A used the teacher as a FILTER and stored a sentence rendered from the truth (sft.py). This mode keeps what the teacher wrote, with the rule of
rejection-sampling distillation (STaR, RFT, the R1 distills) and nothing more: **the reply's final answer is checked against the computed truth, the
reasoning in front of it is kept as it is and is not checked**. Some of its steps may be wrong while the answer is right; the dataset card says which
records those are (`target_source: "teacher-reasoning"`), next to the truth-rendered record of the same task (`"truth-template"`).

  asking      `SYSTEM_REASONING` replaces the system message (SYSTEM_RECIPE of modal_rea.py is untouched): a short step-by-step analysis, then a LAST
              line `Answer: <value>`. Thinking stays off (`enable_thinking: False`) unless `--thinking`; the steps are the visible answer. The messages are the
              ones stored in the records, for both sources.
  scoring     `score_reasoning`: `scoring.score_strict` (the whitelist grammar, not loosened) applied to the text after the LAST `Answer:` line only. A reply
              with no such line, with anything after it, with a `<think>` tag left in it, cut off at `max_tokens` (`TRUNCATED`) or that is the text of a failed
              request is rejected; an `Answer:` earlier in the steps counts for nothing.
  traces      round 12: up to `--max-traces` (3) DISTINCT accepted replies per kept task (RFT, Yuan et al. 2023: several correct reasoning paths per question), each
              stored verbatim (`<think>` blocks stripped, whitespace trimmed, nothing else: `teacher_target`), after the quality filters (R1 SFT-data style, `quality_drop`:
              mixed language, repetition loop, unreadable, self-contradiction) and the duplicate rule (`pick_traces`: identical normalized text, or token-set Jaccard >= 0.9).
  thinking    `--thinking` (a pilot): the server's reasoning is kept, `<think>reasoning</think>` + the visible reply (`thinking_reply`, `compose_trace`); only the visible text's last
              `Answer:` line is scored. Its files and stats say `thinking` (`-reasoning-think` in the tag): a run never mixes with a visible-reasoning one.
  sizing      `--quota kind=N,...` caps the tasks per family (`apply_quota`, round-robin so that a run stopped early is balanced) and `project` turns
              tasks x k x max_tokens into GPU hours and dollars before a GPU is rented (the assumptions are the constants below).
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re

from . import grammar
from .scoring import is_error_reply, score_strict, strict_task

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


def teacher_target(reply: str, t: dict, thinking: bool = False):
    """The record text of an accepted reply, or None: the reply verbatim (`clean_reply`), or in a thinking run its trace (`trace_text`: `<think>reasoning</think>` + the visible reply).
    {"text", "headline", "sentence", "template", "target_source"}; the headline is the text after `Answer:`, there is no sentence or template (nothing of it is rendered)."""
    if not score_reasoning(reply, t):
        return None
    return {"text": trace_text(reply, thinking), "headline": final_answer(reply), "sentence": None, "template": None, "target_source": TARGET_SOURCE}


# ---- the trace: what a record stores, in a visible-reasoning run and in a thinking run ------------------------------------------------

_THINK_INNER = re.compile(r"<think>(.*?)</think>", re.S)


def compose_trace(think: str, visible: str) -> str:
    """`<think>\\nreasoning\\n</think>\\n\\nvisible`: the form the Qwen3.x chat templates render an assistant turn that has reasoning; without reasoning, the visible text alone."""
    think = think.strip()
    return f"<think>\n{think}\n</think>\n\n{visible}" if think else visible


def think_text(reply: str) -> str:
    """The text inside the complete `<think>` blocks of a reply, blocks joined by a blank line ("" when there is none)."""
    return "\n\n".join(b.strip() for b in _THINK_INNER.findall(reply) if b.strip())


def trace_text(reply: str, thinking: bool = False) -> str:
    """What the record stores for an accepted reply. Visible-reasoning run: `clean_reply` (the `<think>` blocks of the reply are dropped, as in round 11). Thinking run: the reasoning
    of the reply's think blocks, then the visible reply (`compose_trace`); the visible reply is what `final_answer` and `score_reasoning` read."""
    visible = clean_reply(reply)
    return compose_trace(think_text(reply), visible) if thinking else visible


def thinking_reply(content, reasoning_content="") -> str:
    """The reply of a THINKING request as the checkpoint stores it: `reasoning_content` of the server (llama-server splits the reasoning off) and any `<think>...</think>` block
    left in `content`, both as the reasoning, then the visible rest, in `compose_trace` form. A `</think>` with no opening tag, and no `reasoning_content`, is the end of a reasoning whose
    opening tag the chat template put into the prompt: the text in front of it is the reasoning. With neither, there is no reasoning and the reply is the visible text alone
    (`quality_drop` rejects such a reply in a thinking run: `no_think`). An unterminated `<think>` stays in the text: the reply is malformed and `final_answer` rejects it."""
    text = content or ""
    parts = [(reasoning_content or "").strip()]
    parts += [b.strip() for b in _THINK_INNER.findall(text)]
    text = _THINK.sub("", text)
    if not any(parts) and "</think>" in text and "<think>" not in text:
        head, _, text = text.partition("</think>")
        parts.append(head.strip())
    return compose_trace("\n\n".join(p for p in parts if p), text.strip())


def reply_text(content, finish, reasoning_content="", thinking: bool = False) -> str:
    """One reply of a reasoning run as the checkpoint stores it, from the server's message: `content` and `reasoning_content` of the choice, its `finish_reason`. Visible-reasoning run:
    the `<think>` blocks stripped, whitespace trimmed; thinking run: `thinking_reply`. A reply the server cut off (finish "length") ends in `TRUNCATED`, which is not an `Answer:` line,
    so it is rejected whatever the cut left. The sampler and `kaggle_score.py` both build their replies here."""
    txt = thinking_reply(content, reasoning_content) if thinking else _THINK.sub("", content or "").strip()
    if finish == "length":
        txt += "\n" + TRUNCATED
    return txt


def user_message(t: dict) -> str:
    """The user message of a task: its pseudocode, the data it references, the question. The sample stage asks it, the SFT record holds it (modal_rea_data._messages)."""
    extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
    return f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"


def messages(t: dict) -> list:
    """The system and user message of a task in a reasoning run (the messages of both record sources)."""
    return [{"role": "system", "content": SYSTEM_REASONING}, {"role": "user", "content": user_message(t)}]


# ---- trace quality filters (DeepSeek-R1 SFT-data style), applied to ACCEPTED replies before the duplicate rule ----------------------------------

DEFAULT_MAX_TRACES = 3            # `--max-traces`: records of teacher reasoning per kept task (RFT keeps several correct paths per question)
JACCARD_DUPLICATE = 0.9           # two traces whose token sets overlap this much (intersection over union) are one trace
MIXED_LANGUAGE_MAX = 0.005        # more than this share of the letters of the prose (outside code and quotes) is not ASCII: CJK, Cyrillic, ... -> mixed language
LINE_REPEAT_MAX = 3               # the same line (12+ characters, not a row of numbers) more often than this: a loop
LINE_MIN_CHARS = 12
NGRAM = 8                         # words in a window...
NGRAM_REPEAT_MAX = 3              # ...the same window more often than this: a loop
QUALITY_DROPS = ("no_think", "mixed_language", "repetition", "unreadable", "contradiction")      # in the order they are tried; a reply is counted under the first that drops it
TRACE_DROPS = QUALITY_DROPS + ("duplicate", "over_cap")      # what can happen to an accepted reply of a kept task besides being stored

# What is NOT prose: fenced code, inline code, double-quoted strings, single-quoted strings that open and close on a word boundary (an apostrophe in `the key's` opens nothing).
_QUOTED = re.compile(r"```.*?```|`[^`\n]{0,200}`|\"[^\"\n]{0,200}\"|“[^”\n]{0,200}”|(?<!\w)'[^'\n]{1,200}'(?!\w)", re.S)
_THINK_TAG = re.compile(r"</?think>")


def _task_words(t) -> set:
    """The words of the prompt of a task (its pseudocode, the data it references, the question): a non-ASCII identifier or string of the task is not language mixing when a reply names it."""
    if not t:
        return set()
    return set(re.findall(r"\w+", " ".join(str(t.get(k) or "") for k in ("pseudocode", "refs", "question"))))


def mixed_language(text: str, t: dict = None) -> bool:
    """True when more than MIXED_LANGUAGE_MAX of the letters of the prose are not ASCII. The prose is the text without fenced code, inline code and quoted strings, and without the
    words that the task itself contains (a non-ASCII identifier of its pseudocode named in the reply). Only LETTERS count: arrows, comparison signs, `x`, curly quotes, dashes,
    digits and hex dumps (`e4 b8 ad`) are not letters or not non-ASCII, so they never count. 0.5% is 3 letters in a 600-letter reply: a stray CJK word is over it, one `pi` is not."""
    prose = _QUOTED.sub(" ", _THINK_TAG.sub(" ", text))
    exempt = _task_words(t)
    letters = foreign = 0
    for word in re.findall(r"\w+", prose):
        ls = [c for c in word if c.isalpha()]
        n = sum(ord(c) > 127 for c in ls)
        if n and word in exempt:
            continue
        letters += len(ls)
        foreign += n
    return letters > 0 and foreign / letters > MIXED_LANGUAGE_MAX


_HEXLIKE = re.compile(r"0x[0-9a-f]+|[0-9]+|[0-9a-f]{2}")      # a word of a row of numbers or a hex dump: such a window repeats legitimately (a table of zeros)


def repetition(text: str) -> bool:
    """True when the same line (lower case, bullet and spacing ignored, at least LINE_MIN_CHARS characters, not a row of numbers) occurs more than LINE_REPEAT_MAX times, or the same
    window of NGRAM words (not a window of numbers only) occurs more than NGRAM_REPEAT_MAX times: a loop (`Wait, let me recheck.` over and over). A self-correction that says new things
    is not a loop."""
    lines = collections.Counter()
    for line in text.splitlines():
        s = " ".join(re.sub(r"^\s*(?:[-*•>]+|\d+[.)])\s*", "", line.lower()).split())
        if len(s) >= LINE_MIN_CHARS and not all(_HEXLIKE.fullmatch(w) for w in s.split()):
            lines[s] += 1
    if lines and max(lines.values()) > LINE_REPEAT_MAX:
        return True
    words = re.findall(r"[a-z0-9_]+", _THINK_TAG.sub(" ", text.lower()))
    num = [bool(_HEXLIKE.fullmatch(w)) for w in words]
    grams = collections.Counter(tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1) if not all(num[i:i + NGRAM]))
    return bool(grams) and max(grams.values()) > NGRAM_REPEAT_MAX


_RESTATING = frozenset("the a an is are was be so thus therefore hence then and of to it its this that here answer final result value equals gives "
                       "analysis step steps reasoning explanation solution".split())


def _value_words(text: str) -> list:
    """Lower-case words of a text, a number (`0x20`, `32`) as its decimal value, so that `0x20` and `32` are the same word."""
    out = []
    for w in re.findall(r"0x[0-9a-f]+|[0-9]+|[a-z_][a-z0-9_]*", text.lower()):
        v = grammar.num_value(w)
        out.append(str(v) if v is not None else w)
    return out


def unreadable(analysis: str, answer: str) -> bool:
    """True when the analysis in front of the `Answer:` line is empty, or says nothing but the answer again: every word of it is a word of the answer or a restating word
    (`the answer is`, `so`, `therefore`, `result`, ...). A reply must have some reasoning before its `Answer:` line."""
    words = _value_words(analysis)
    return not words or all(w in set(_value_words(answer)) | _RESTATING for w in words)


# A statement that gives a final value: `the answer is X`, `final answer: X`, `result = X`, or a conclusion (`so the offset is X`, `therefore 0x14 + 0x20 = X`). A plain `= X` in the middle of
# a calculation and a plain `the size is X` are not statements: they may name a part of the answer.
_STATEMENT = re.compile(r"""(?ix)
    (?: \b(?:final\s+)?(?:answer|result)\s*(?:is|=|:)
      | \b(?:so|thus|therefore|hence|giving|gives)\b [^=\n]*? (?:\bis\b|\bare\b|=|\bequals?\b|\bgives\b)
    ) \s* (?P<x>.+?) [\s.!;,:]* $""")
_SENTENCE = re.compile(r"\n+|(?<=[.!?])\s+")
_CORRECTION = re.compile(r"\b(?:wait|actually|hmm+|oops|re-?check|re-?read|re-?examin\w*|re-?verify|double-?check|on second thought|correction|mistake|mis-?read|instead)\b|let me (?:re|check|verify|look again)|\bno,", re.I)


def contradicts(analysis: str, answer: str, t: dict) -> bool:
    """True only in the simple case: the LAST explicit statement of a final value in the analysis (`_STATEMENT`) gives numbers, none of which is a number of the task's answer, and nothing after
    it says the model is correcting itself (`wait`, `actually`, `let me recheck`: a self-correction is kept, as in R1). The numbers are read with the task's accept groups (`grammar.Spec`:
    `0x34` and `52` are one value); the statement may hold only numbers, units of the task and function words, anything else is not understood. The Answer line has passed `score_strict`, so
    "a value other than the Answer line's" is a number that is not in the task's accept groups. Names and strings are not looked at (the grammar of the Answer line already is), a statement with one right and one wrong number
    is a hedge, and anything unclear is kept: unsure means no drop."""
    s = strict_task(t)
    if s.get("score") != "tokens":
        return False
    sp = grammar.Spec(s)
    if not sp.values:
        return False
    sentences = [x for x in (" ".join(re.sub(r"[*`]|^[#>\-\s]+", "", c).split()) for c in _SENTENCE.split(analysis)) if x]
    for i in range(len(sentences) - 1, -1, -1):
        m = _STATEMENT.search(sentences[i])
        if not m:
            continue
        if any(_CORRECTION.search(later) for later in sentences[i + 1:]):
            return False
        nums = []
        ok = {"num", "ctx"}
        for tok in grammar.lex(grammar.normalise(m.group("x"))):
            if tok.kind in ok:
                nums.append(tok.val)
            elif tok.kind == "p" and tok.text in "(),":
                continue
            elif not (tok.kind == "word" and (tok.text in grammar.GLUE or tok.text in sp.units or tok.text in ("hex", "hexadecimal", "decimal"))):
                return False
        return bool(nums) and not any(n in sp.values for n in nums)
    return False


def reply_parts(reply: str, thinking: bool = False) -> tuple:
    """(analysis, answer) of a reply: the reasoning in front of the last `Answer:` line (in a thinking run the think blocks come first, then the visible text before the line), and the text
    after the label. ("", "") for a reply without a final `Answer:` line."""
    answer = final_answer(reply)
    if answer is None:
        return "", ""
    lines = clean_reply(reply).splitlines()
    visible = "\n".join(lines[:-1]).strip()
    return ((think_text(reply) + "\n\n" + visible).strip() if thinking else visible), answer


def quality_drop(reply: str, t: dict, thinking: bool = False):
    """Why an ACCEPTED reply is not stored as a trace, or None: "no_think" (thinking run: the reply has no reasoning block), "mixed_language", "repetition", "unreadable" or "contradiction"
    (QUALITY_DROPS, in this order; a reply is counted under the first that drops it). `wait` / `let me recheck` self-corrections are not a reason."""
    analysis, answer = reply_parts(reply, thinking)
    text = trace_text(reply, thinking)
    if thinking and not think_text(reply):
        return "no_think"
    if mixed_language(text, t):
        return "mixed_language"
    if repetition(text):
        return "repetition"
    if unreadable(analysis, answer):
        return "unreadable"
    if contradicts(analysis, answer, t):
        return "contradiction"
    return None


# ---- duplicates, and the traces a task keeps -------------------------------------------------------------------------------------------------------

def normalized(text: str) -> str:
    """The normal form two traces are compared in: lower case, think tags removed, every run of whitespace one space. Digits and punctuation are kept (`0x20` and `0x24` are different words)."""
    return " ".join(_THINK_TAG.sub(" ", text.lower()).split())


def token_set(text: str) -> set:
    """The words of the normal form (`\\w+`: letters, digits and `_`, so `0x20` and `FUN_00101234` are one word each; punctuation separates)."""
    return set(re.findall(r"\w+", normalized(text)))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def is_duplicate(a: str, b: str) -> bool:
    """Two traces are one when their normalized texts are identical, or the Jaccard similarity of their token sets (intersection over union of the words, so the same words in another order
    count as the same, and a difference in punctuation only is no difference) is at least JACCARD_DUPLICATE. Deterministic, no model: a trace is compared with the traces already kept, in sample order."""
    return normalized(a) == normalized(b) or jaccard(token_set(a), token_set(b)) >= JACCARD_DUPLICATE


def pick_traces(replies: list, t: dict, max_traces: int = DEFAULT_MAX_TRACES, thinking: bool = False) -> tuple:
    """([teacher targets], Counter of the replies that were not stored by reason) from the ACCEPTED replies of one task, in sample order: each reply passes the quality filters, then the duplicate
    rule against the traces already kept, then the cap (`over_cap`: a distinct trace beyond `max_traces`). The first `max_traces` distinct clean replies are the traces."""
    kept, dropped = [], collections.Counter()
    for reply in replies:
        why = quality_drop(reply, t, thinking)
        if why:
            dropped[why] += 1
            continue
        target = teacher_target(reply, t, thinking)
        if any(is_duplicate(target["text"], k["text"]) for k in kept):
            dropped["duplicate"] += 1
        elif len(kept) >= max_traces:
            dropped["over_cap"] += 1
        else:
            kept.append(target)
    return kept, dropped


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


def thinking_cost_line(p: dict, source: str = "") -> str:
    """The line of a THINKING run (printed before the GPU is rented). There is no expected cost: the 5x-8x length of the visible-reasoning mode does not hold when the model reasons first and the
    length of its reasoning is not measured. What is known is the worst case, every reply running to `--max-tokens`."""
    w = p["worst"]
    return (f"projected cost (thinking pilot): {p['tasks']:,} tasks x k={p['k']} = {p['replies']:,} replies. No estimate: the 5x-8x length assumption is for visible reasoning, and how long this model "
            f"reasons is not measured. Worst case, every reply runs to max_tokens={p['max_tokens']}: {w['gpu_hours']} GPU-h on one L40S = about US${w['usd']:.2f} at US${p['usd_per_h']:.2f}/h "
            f"(an overestimate: it scales the phase-A time per reply by length). Keep --quota small, then read `perf` and the share of replies cut off at max_tokens. The prompt and the reply share "
            f"one 8,192-token slot of the server. A sampler call times out after {p['call_hours']:g} h: the worst case takes ~{max(1, math.ceil(w['gpu_hours'] / p['call_hours']))} call(s), each after the first with "
            f"--resume." + (f" Tasks: {source}." if source else ""))


def thinking_budget_warning(p: dict, budget_usd: float):
    """A warning when even the pilot's worst case is over the budget (there is no expected cost to compare), else None."""
    if p["worst"]["usd"] > budget_usd:
        return (f"WARNING: the worst case US${p['worst']['usd']:.2f} of this thinking run is over the budget US${budget_usd:.2f}; the expected cost is not known: lower --quota, --k or --max-tokens, "
                f"or raise --budget-usd on purpose")
    return None

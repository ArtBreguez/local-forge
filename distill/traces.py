"""What the replies of a teacher become: the final answer, the reasoning trace, the quality filters, the duplicate rule, and the traces a task keeps. Stdlib only, no subject in it.

Moved from `rea_bench/reasoning.py` with its behaviour (rejection sampling in the sense of STaR and RFT, the R1 distills): **the reply's final answer is checked against the computed truth, the reasoning in
front of it is kept as it is and is not checked.** Some of its steps may be wrong while the answer is right. What a subject decides is passed in as `domain` (`distill.domain.Domain`): `verify` for the answer,
`messages` for the prompt words a reply may name in any script, and the two optional hooks `contradicts` and `canon_word` of the filters below.

  the reply     `final_answer`: the text after the LAST `Answer:` line, when that line is the last line of the reply. `verdict` / `score_reasoning`: accepted, or why not (error, truncated, no answer line, wrong).
  the trace     `teacher_target` stores an accepted reply verbatim (`<think>` blocks stripped, whitespace trimmed); in a thinking run `<think>reasoning</think>` + the visible reply (`thinking_reply`, `compose_trace`).
  the filters   `quality_drop` (DeepSeek-R1 SFT-data style: mixed language, a repetition loop, an unreadable analysis, a plain contradiction; `no_think` for a thinking run without reasoning), then the duplicate rule
                (`is_duplicate`: identical normalized text, or token-set Jaccard >= 0.9) and the cap, in `pick_traces`: up to `max_traces` distinct clean replies per task (RFT: several correct paths per question).
"""
from __future__ import annotations

import collections
import re

from .domain import mode_of

TARGET_SOURCE = "teacher-reasoning"
TRUNCATED = "[truncated at max_tokens]"  # appended by the sampler to a reply the server cut off (finish_reason "length"); it is not an `Answer:` line, so the reply is rejected
ERROR_PREFIX = "[error"       # what the sample stage writes for a failed request (`[error <exception repr>]`)


def is_error_reply(reply: str) -> bool:
    """The text of a failed request, not a reply of the model (a dead server, a timeout, a prompt over the slot size). Never scored, never stored."""
    return reply.lstrip().startswith(ERROR_PREFIX)


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


def is_truncated(reply: str) -> bool:
    """True for a reply the server cut off at max_tokens: the text ends in `TRUNCATED` (`reply_text` appends it for finish "length")."""
    return clean_reply(reply).endswith(TRUNCATED)


def verdict(reply: str, t: dict, *, domain) -> str:
    """Why a reply is or is not an answer: "error" (the text of a failed request), "truncated", "no_answer_line" (no final `Answer:` line, text after it, or an empty
    one), "wrong" (the line is there and `domain.verify` rejects its text) or "accepted"."""
    if is_error_reply(reply):
        return "error"
    if is_truncated(reply):
        return "truncated"
    answer = final_answer(reply)
    if answer is None:
        return "no_answer_line"
    return "accepted" if domain.verify(answer, t) else "wrong"


def score_reasoning(reply: str, t: dict, *, domain) -> bool:
    """DATA scorer of the reasoning mode: `domain.verify` on the text after the last `Answer:` line; every other reply is rejected (see `verdict`)."""
    return verdict(reply, t, domain=domain) == "accepted"


def teacher_target(reply: str, t: dict, thinking: bool = False, *, domain):
    """The record text of an accepted reply, or None: the reply verbatim (`clean_reply`), or in a thinking run its trace (`trace_text`: `<think>reasoning</think>` + the visible reply).
    {"text", "headline", "sentence", "template", "target_source"}; the headline is the text after `Answer:`, there is no sentence or template (nothing of it is rendered)."""
    if not score_reasoning(reply, t, domain=domain):
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
    so it is rejected whatever the cut left. The sample stage and `score_replies.py` both build their replies here."""
    txt = thinking_reply(content, reasoning_content) if thinking else _THINK.sub("", content or "").strip()
    if finish == "length":
        txt += "\n" + TRUNCATED
    return txt


# ---- trace quality filters (DeepSeek-R1 SFT-data style), applied to ACCEPTED replies before the duplicate rule ----------------------------------

DEFAULT_MAX_TRACES = 3            # `--max-traces`: records of teacher reasoning per kept task (RFT keeps several correct paths per question)
JACCARD_DUPLICATE = 0.9           # two traces whose token sets overlap this much (intersection over union) are one trace
MIXED_LANGUAGE_MAX = 0.005        # more than this share of the letters of the prose (outside code and quotes) is not ASCII: CJK, Cyrillic, ... -> mixed language
LINE_REPEAT_MAX = 3               # the same line (12+ characters, not a row of numbers) more often than this: a loop
LINE_MIN_CHARS = 12
NGRAM = 8                         # words in a window...
NGRAM_REPEAT_MAX = 3              # ...the same window more often than this: a loop
PROMPT_REPEAT_MAX = 12            # ...but a line or a window that is in the task's own prompt (the model re-reads its question) more often than THIS: 12, not 8 (4 of the 12 pilot replies that only quote the question do so 9-11 times)
QUALITY_DROPS = ("no_think", "mixed_language", "repetition", "unreadable", "contradiction")      # in the order they are tried; a reply is counted under the first that drops it
TRACE_DROPS = QUALITY_DROPS + ("duplicate", "over_cap")      # what can happen to an accepted reply of a kept task besides being stored

# What is NOT prose: fenced code, inline code, double-quoted strings, single-quoted strings that open and close on a word boundary (an apostrophe in `the key's` opens nothing).
_QUOTED = re.compile(r"```.*?```|`[^`\n]{0,200}`|\"[^\"\n]{0,200}\"|“[^”\n]{0,200}”|(?<!\w)'[^'\n]{1,200}'(?!\w)", re.S)
_THINK_TAG = re.compile(r"</?think>")


def _task_words(domain, t) -> set:
    """The words of the prompt of a task (`domain.task_words`): a non-ASCII identifier or string of the task is not language mixing when a reply names it."""
    if not t or domain is None:
        return set()
    return domain.task_words(t)


def mixed_language(text: str, t: dict = None, *, domain=None) -> bool:
    """True when more than MIXED_LANGUAGE_MAX of the letters of the prose are not ASCII. The prose is the text without fenced code, inline code and quoted strings, and without the
    words that the task's prompt itself contains (a non-ASCII identifier of its code named in the reply). Only LETTERS count: arrows, comparison signs, `x`, curly quotes, dashes,
    digits and hex dumps (`e4 b8 ad`) are not letters or not non-ASCII, so they never count. 0.5% is 3 letters in a 600-letter reply: a stray CJK word is over it, one `pi` is not."""
    prose = _QUOTED.sub(" ", _THINK_TAG.sub(" ", text))
    exempt = _task_words(domain, t)
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


def _line_form(line: str) -> str:
    """A line in the form the repetition counts compare it in: lower case, bullet or list number removed, spacing collapsed."""
    return " ".join(re.sub(r"^\s*(?:[-*\u2022>]+|\d+[.)])\s*", "", line.lower()).split())


def _gram_words(text: str) -> list:
    return re.findall(r"[a-z0-9_]+", _THINK_TAG.sub(" ", text.lower()))


def repetition(text: str, prompt: str = "") -> bool:
    """True when the same line (lower case, bullet and spacing ignored, at least LINE_MIN_CHARS characters, not a row of numbers) occurs more than LINE_REPEAT_MAX times, or the same
    window of NGRAM words (not a window of numbers only) occurs more than NGRAM_REPEAT_MAX times: a loop (`Wait, let me recheck.` over and over). A self-correction that says new things
    is not a loop. `prompt`: the task's own prompt (the system and the user message the model was asked): a model that reasons re-reads its question and quotes it (the thinking pilot: 12 of 13
    replies dropped as loops only quoted the question, 4 to 12 times), and quoting the question is not looping, so a line or a window that is in the prompt is counted against its own, higher limit,
    PROMPT_REPEAT_MAX, instead of the 3. What the model says about it, and every window that only partly overlaps the prompt, still counts with the strict one.

    THE HOLE, stated: what the prompt contains is not held to the strict limit, so a reply may re-read it more than three times and pass: up to PROMPT_REPEAT_MAX (12) times in any form (the pilot
    has three accepted replies that re-read the same instruction 10 or 11 times; they are kept, and whether such a trace belongs in the data is a length rule, not this filter; a limit of 8 would have
    dropped 4 of the 12 pilot replies that only quote the question). Beyond 12 a prompt line or window is a loop. One way of restating the prompt passes whatever the count: a SHORT line restated with a
    label that changes each round, in a number or in a word. With a constant label word and a number that changes (`check 3: <a line of the code>`) it is a line of up to NGRAM - 2 = 6 words: every
    8-word window of such a reply holds the number, which is different in each round, so no window repeats, and the line itself is never repeated as written. With a different label WORD each round
    (`alpha: <a line of the code>`, `bravo: <a line of the code>`, ...: a model does not write ordinals, but it can number its rounds with words) it is a line of up to NGRAM - 1 = 7 words, because no
    8-word window is free of a label word; it passes for as many rounds as the model has different label words. A line of 7 words behind the constant label word has a window (its words and the label's word)
    that repeats, a line of 8 words or more has windows of its own with no label in them, and a long one has windows inside it that are the prompt's: those are counted. A reply that restates short lines this
    way ends: it is a long restating, kept like the others, not a loop that is cut off.
    Not a hole: a loop in the model's own words, and a block of the question copied over and over (the windows across the copies are not in the prompt)."""
    in_prompt = {_line_form(line) for line in prompt.splitlines()}
    lines, quoted_lines = collections.Counter(), collections.Counter()
    for line in text.splitlines():
        s = _line_form(line)
        if len(s) >= LINE_MIN_CHARS and not all(_HEXLIKE.fullmatch(w) for w in s.split()):
            (quoted_lines if s in in_prompt else lines)[s] += 1
    if lines and max(lines.values()) > LINE_REPEAT_MAX or quoted_lines and max(quoted_lines.values()) > PROMPT_REPEAT_MAX:
        return True
    quoted_words = _gram_words(prompt)
    quoted = {tuple(quoted_words[i:i + NGRAM]) for i in range(len(quoted_words) - NGRAM + 1)}
    words = _gram_words(text)
    num = [bool(_HEXLIKE.fullmatch(w)) for w in words]
    grams, quoted_grams = collections.Counter(), collections.Counter()
    for i in range(len(words) - NGRAM + 1):
        if not all(num[i:i + NGRAM]):
            g = tuple(words[i:i + NGRAM])
            (quoted_grams if g in quoted else grams)[g] += 1
    return bool(grams) and max(grams.values()) > NGRAM_REPEAT_MAX or bool(quoted_grams) and max(quoted_grams.values()) > PROMPT_REPEAT_MAX


_RESTATING = frozenset("the a an is are was be so thus therefore hence then and of to it its this that here answer final result value equals gives "
                       "analysis step steps reasoning explanation solution".split())


def _value_words(text: str, domain=None) -> list:
    """Lower-case words of a text, a number (`0x20`, `32`) as its decimal value, so that `0x20` and `32` are the same word (the domain's `canon_word`)."""
    canon = domain.canon_word if domain is not None else (lambda w: w)
    return [canon(w) for w in re.findall(r"0x[0-9a-f]+|[0-9]+|[a-z_][a-z0-9_]*", text.lower())]


def unreadable(analysis: str, answer: str, *, domain=None) -> bool:
    """True when the analysis in front of the `Answer:` line is empty, or says nothing but the answer again: every word of it is a word of the answer or a restating word
    (`the answer is`, `so`, `therefore`, `result`, ...). A reply must have some reasoning before its `Answer:` line."""
    words = _value_words(analysis, domain)
    return not words or all(w in set(_value_words(answer, domain)) | _RESTATING for w in words)


def prompt_text(domain, t: dict, thinking: bool = False) -> str:
    """The task's own prompt as the sample stage asks it (`domain.messages` of the run's mode: the system and the user message), one text: what a reply may quote without looping. "" without a task, and
    for a record that lacks the fields its prompt is made of (the filters also run on a bare task: `domain.task_words` has the same case): nothing to quote, so the repetition rule is the strict one."""
    if not t:
        return ""
    try:
        return "\n".join(m["content"] for m in domain.messages(t, mode_of(True, thinking)))
    except KeyError:
        return ""


def reply_parts(reply: str, thinking: bool = False) -> tuple:
    """(analysis, answer) of a reply: the reasoning in front of the last `Answer:` line (in a thinking run the think blocks come first, then the visible text before the line), and the text
    after the label. ("", "") for a reply without a final `Answer:` line."""
    answer = final_answer(reply)
    if answer is None:
        return "", ""
    lines = clean_reply(reply).splitlines()
    visible = "\n".join(lines[:-1]).strip()
    return ((think_text(reply) + "\n\n" + visible).strip() if thinking else visible), answer


def quality_drop(reply: str, t: dict, thinking: bool = False, *, domain):
    """Why an ACCEPTED reply is not stored as a trace, or None: "no_think" (thinking run: the reply has no reasoning block), "mixed_language", "repetition", "unreadable" or "contradiction"
    (QUALITY_DROPS, in this order; a reply is counted under the first that drops it). `wait` / `let me recheck` self-corrections are not a reason."""
    analysis, answer = reply_parts(reply, thinking)
    text = trace_text(reply, thinking)
    if thinking and not think_text(reply):
        return "no_think"
    if mixed_language(text, t, domain=domain):
        return "mixed_language"
    if repetition(text, prompt_text(domain, t, thinking)):
        return "repetition"
    if unreadable(analysis, answer, domain=domain):
        return "unreadable"
    if domain.contradicts(analysis, answer, t):
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


def pick_traces(replies: list, t: dict, max_traces: int = DEFAULT_MAX_TRACES, thinking: bool = False, *, domain) -> tuple:
    """([teacher targets], Counter of the replies that were not stored by reason) from the ACCEPTED replies of one task, in sample order: each reply passes the quality filters, then the duplicate
    rule against the traces already kept, then the cap (`over_cap`: a distinct trace beyond `max_traces`). The first `max_traces` distinct clean replies are the traces."""
    kept, dropped = [], collections.Counter()
    for reply in replies:
        why = quality_drop(reply, t, thinking, domain=domain)
        if why:
            dropped[why] += 1
            continue
        target = teacher_target(reply, t, thinking, domain=domain)
        if any(is_duplicate(target["text"], k["text"]) for k in kept):
            dropped["duplicate"] += 1
        elif len(kept) >= max_traces:
            dropped["over_cap"] += 1
        else:
            kept.append(target)
    return kept, dropped

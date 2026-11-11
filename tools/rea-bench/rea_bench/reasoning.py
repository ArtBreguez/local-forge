"""What is REA's in the teacher-reasoning mode: the prompt, the kinds of task, and the two trace filters that read the answer's numbers. Stdlib only.

The mode itself (the `Answer:` line, the trace, the quality filters, the duplicate rule, the traces a task keeps, the sampling settings, the quota) is `distill/traces.py` and `distill/config.py`, moved from here
with its behaviour; what is left is what only REA decides, and `distill/domains/rea` hands it to them:

  SYSTEM_REASONING   the system message of a reasoning or thinking run (`SYSTEM_RECIPE` of modal_rea.py is the plain one and is untouched): a short step-by-step analysis, then a LAST line `Answer: <value>`.
  messages           the system and user message of a task (`user_message`: its pseudocode, the data it references, the question); both record sources and the sampler use them.
  contradicts        the `contradiction` filter: the last explicit statement of a final value in the analysis names numbers none of which is a number of the task's accept groups (read with `grammar.Spec`).
  task_words         the words of a task's prompt (a reply may name a non-ASCII word of it in any script: the `mixed_language` filter).
  value_word         the word form of the `unreadable` filter: `0x20` and `32` are one word (`grammar.num_value`).
  known_kinds        every `kind` a task can have: the ten new family names and the legacy template names.
"""
from __future__ import annotations

import re

from . import grammar
from .scoring import strict_task

SYSTEM_REASONING = ("You are a reverse engineer. You are given decompiled C pseudocode of one function from a stripped binary and a question. "
                    "First write a short step-by-step analysis, at most eight short lines: what the code reads, which constants it uses and why they give the answer. "
                    "Then write one last line of the form `Answer: <the concrete value or name asked for, in the form the question asks>` and nothing after it.")


def user_message(t: dict) -> str:
    """The user message of a task: its pseudocode, the data it references, the question. The sample stage asks it, the SFT record holds it (`Rea.messages`)."""
    extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
    return f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"


def messages(t: dict) -> list:
    """The system and user message of a task in a reasoning run (the messages of both record sources)."""
    return [{"role": "system", "content": SYSTEM_REASONING}, {"role": "user", "content": user_message(t)}]


def task_words(t) -> set:
    """The words of the prompt of a task (its pseudocode, the data it references, the question): a non-ASCII identifier or string of the task is not language mixing when a reply names it."""
    if not t:
        return set()
    return set(re.findall(r"\w+", " ".join(str(t.get(k) or "") for k in ("pseudocode", "refs", "question"))))


def value_word(w: str) -> str:
    """A word of the analysis or the answer in the form the `unreadable` filter compares: a number (`0x20`, `32`) as its decimal value, anything else as it is."""
    v = grammar.num_value(w)
    return str(v) if v is not None else w


def known_kinds() -> list:
    """Every `kind` a task can have: the ten new family names and the legacy template names (`xor-key`, `lcg`, ...)."""
    from . import families
    return sorted({m for m in families.modules() if m != "legacy"} | {t.kind for t in families.generate("legacy", 10, 7)})


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

"""Answer scoring against a task's truth (`score` + `accept` [+ `reject`, `cand`, `allow`]).

TWO SCORERS, ONE FOR EACH JOB (round 3: M1-M3, m2, m9, m10).

  score / score_any_line   EVALUATION (`modal_rea.py`): the fixed questions and the old templates keep their frozen whole-reply substring
                           scoring, so 34/60 -> 50/60 stays comparable. The generated families are scored on the headline only, by
                           the whitelist grammar of `grammar.py`.
  score_strict             DATA (`modal_rea_data.py --stage sample`, `sft_target`): EVERY family, legacy included, goes through the
                           whitelist grammar. A legacy task is converted first (`strict_task`: its accept strings become grammar groups,
                           `exact` keys become exact strings). Nothing that is trained on is scored by a substring match.

Legacy modes (the original ten templates, evaluation only):
  exact / any       one accepted spelling must appear in the answer (whitespace and case ignored)
  all               every accepted value must appear (each with its numeric spellings, see `variants`)
  all_groups        every group must have one spelling in the answer

Headline modes (the generated families, and every family on the data path):
  tokens            the HEADLINE must parse completely as the task's answer grammar (grammar.py): every accept group exactly once (the name
                    groups in any order, the numeric groups in the order of `accept`), separated by `, ; / and` or whitespace, with units,
                    `allow` phrases and a short vocabulary of function words around them.
                    Anything else is a leftover and rejects: any other word, operator, symbol or number, however it is spelled.
                    Bounds take `lo to hi`, `lo..hi`, `[lo, hi]`, `lo <= x <= hi` and nothing exclusive.
  exact_case        the whole headline is the key, byte for byte: after quotes, backticks, markdown, a lead-in that ends in a colon or one of
  exact_head        `is`/`writes`/`reads`/`prints`/`decodes to` ("The string is", "Alphabet:") and one trailing mark nothing but the key is left.
                    The reply is not normalised: case, spacing and every character are significant.

The headline (`parse_reply`): the first answer line (markdown removed, fences skipped; a bare lead-in line made of function words and the question's own
words, such as "Answer:", is skipped); if it
holds several sentences the first one is the headline. What follows must be prose (a sentence of 4+ words): a second answer-shaped line
("K\\nW", a code block with two values, "Candidates:\\n...") makes the reply a list and rejects it.

What is stored as an SFT target is NEVER the teacher's text (round 4): the teacher's headline is a filter (`accepted_headline`: does it parse as the
task's answer?), and `sft_target` returns the target rendered from the task's own truth (targets.py: the canonical headline and one factual
sentence of a per-variant template set). The raw headline of the teacher is returned as `teacher_headline`, for analysis only.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from . import grammar
from .grammar import Spec, headline_ok, is_exact_answer, lex, normalise, num_value  # noqa: F401  (num_value is re-exported)

LEGACY_MODES = ("exact", "any", "all", "all_groups")
HEADLINE_MODES = ("tokens", "exact_case", "exact_head")
MODES = LEGACY_MODES + HEADLINE_MODES


# ---- legacy modes (evaluation only, frozen) ----------------------------------------------------------------------------------

def variants(x: str) -> list:
    """An integer answer may be given in decimal or hex, signed or as the 32-bit pattern (0x97c59b7a == -0x683a6486); accept all."""
    out = [x]
    m = re.fullmatch(r"-?\d+", x)
    if m:
        v = int(x)
        if abs(v) > 9:
            out += [format(abs(v), "x") if v > 0 else "-0x" + format(-v, "x"), "0x" + format(v & 0xffffffff, "x")]
            if v > 0:
                out.append("0x" + format(v, "x"))
    elif re.fullmatch(r"[0-9a-f]{8}", x):
        v = int(x, 16)
        if v >= 0x80000000:
            out.append("-0x" + format(0x100000000 - v, "x"))
    return out


def _legacy_score(answer: str, t: dict) -> bool:
    a = re.sub(r"\s+", "", answer.lower())
    if t["score"] == "all_groups":
        return all(any(re.sub(r"\s+", "", x.lower()) in a for x in g) for g in t["accept"])
    acc = [[re.sub(r"\s+", "", v.lower()) for v in variants(x)] for x in t["accept"]]
    return all(any(v in a for v in g) for g in acc) if t["score"] == "all" else any(v in a for g in acc for v in g)


# ---- legacy tasks on the data path -------------------------------------------------------------------------------------------

# Which accept strings of a legacy task are hex constants (the question asks "(hex)"); every other numeric string is decimal.
_HEX_AT = {"magic": (0,), "rotate": (0,), "hash": (1,)}
# Names a reply may use for the legacy hash families (the frozen accept list holds the stem only: "fnv" must also be "FNV-1a"). Only names of THAT
# algorithm: FNV-1 (multiply, then xor) is not the FNV-1a of the template, MurmurHash3 and djb2a are other hashes (round 4, M2). The question asks
# "which well-known hash is this BASED ON", so the murmur template (a per-byte loop with MurmurHash2's multiplier and seed, not MurmurHash2 itself)
# accepts the reply "MurmurHash2" while the stored name is "Murmur-style" (round 5, B3).
_HASH_NAMES = {"fnv": ("fnv", "fnv-1a", "fnv1a", "fnv-32", "fnv32"), "djb2": ("djb2", "djb", "bernstein", "djb-2"),
               "sdbm": ("sdbm",), "murmur": ("murmur", "murmurhash", "murmur2", "murmurhash2", "murmur-ish", "murmur-style")}
# What a constant guess gets right on a legacy task whose answer set is tiny (the generated families carry `chance`; legacy tasks did not).
# skip: one of 7 byte values. The other templates hide a 32-bit constant (magic, hash, rotate), an exact string, or a pair that no guess gets.
_LEGACY_CHANCE = {"skip": 0.14}


def _bare_hex_ok(s: str) -> bool:
    """A hex spelling without 0x is only safe when no plain word or decimal number can be mistaken for it (core._bare_hex_ok)."""
    return set(s) == {"0"} or (len(s) >= 2 and any(c in "abcdef" for c in s) and any(c.isdigit() for c in s))


def _legacy_group(x: str, hex_const: bool) -> list:
    """The spellings of one legacy accept string as a grammar group. A hex constant: `0x..` and its bare spelling only where it holds a letter and a
    digit (never the all-digit text of `30312932`, which the grammar reads as the DECIMAL number 30,312,932). A decimal number: itself and `0x..`
    (and the 32-bit pattern of a negative one), never its bare hex digits (the hex digits of 39 are `27`, which is another number): round 4, B1.
    The base a group is asked in is enforced by the grammar (`bases`), not by this list."""
    if hex_const and re.fullmatch(r"[0-9a-f]{1,16}", x):
        v = int(x, 16)
        return list(dict.fromkeys([f"0x{v:x}"] + [b for b in (x, f"{v:x}") if _bare_hex_ok(b)]))
    if re.fullmatch(r"-?\d+", x):
        v = int(x)
        out = [x, f"0x{v:x}" if v >= 0 else f"-0x{-v:x}"]
        if v < 0:
            out.append(f"0x{v & 0xffffffff:x}")
        return list(dict.fromkeys(out))
    return [x]


def strict_task(t: dict) -> dict:
    """The task as the data path scores it: a task of the new families is returned as it is; a legacy task becomes a `tokens` /
    `exact_head` / `exact_case` task with the same key (`exact` -> an exact string, `any`/`all`/`all_groups` -> grammar groups in the
    order of its question), with the base each number is asked in (`bases`) and, where a guess can be right, a `chance`."""
    mode, acc, kind = t["score"], t["accept"], t.get("kind", "")
    if mode not in LEGACY_MODES:
        return t
    extra = {"chance": _LEGACY_CHANCE[kind]} if kind in _LEGACY_CHANCE and "chance" not in t else {}
    if mode == "exact":
        return {**t, **extra, "score": "exact_case" if kind == "alphabet" else "exact_head", "accept": acc[:1]}
    if mode == "any":                                    # skip: the value as hex; `acc` holds one value in three spellings ("0x0a", "0a", "10")
        v = int(acc[0], 16) if re.fullmatch(r"0x[0-9a-f]+", acc[0]) else int(acc[0])
        return {**t, **extra, "score": "tokens", "accept": [_legacy_group(f"{v:02x}", True)], "bases": ["hex"]}
    if mode == "all":
        hexes = _HEX_AT.get(kind)
        is_hex = [(i in hexes) if hexes is not None else bool(re.fullmatch(r"[0-9a-f]{8}", x) and not x.isdigit()) for i, x in enumerate(acc)]
        return {**t, **extra, "score": "tokens", "accept": [_legacy_group(x, h) for x, h in zip(acc, is_hex)], "bases": ["hex" if h else "dec" for h in is_hex]}
    names = [n for stem in acc[0] for n in (*_HASH_NAMES.get(stem, ()), stem)]
    return {**t, **extra, "score": "tokens", "accept": [list(dict.fromkeys(names))] + [_legacy_group(g[0], True) for g in acc[1:]],
            "bases": [None] + ["hex"] * (len(acc) - 1)}


# ---- the reply: lines, lead-in, headline, justification ----------------------------------------------------------------------

# Markdown the headline may start with: headings and bullets. `>` is never stripped (`> 40` is a comparison, not a quote), an ASCII bullet is
# not stripped before a digit (`- 40` is minus forty or a bullet: unknown, so it stays and rejects), numbered-list prefixes stay (`5. 40`).
_MD_LEAD = re.compile(r"^\s*(?:#{1,6}\s+|•\s*|[-*+]\s+(?![\d-]))+")
_SEPARATOR = re.compile(r"[-_=~\s]*")
_FENCE = re.compile(r"^\s*(?:```|~~~)")

_JUSTIFICATION_LEAD = frozenset("justification reason reasoning explanation because why note notes rationale".split())


def _strip_md(line: str) -> str:
    s = _MD_LEAD.sub("", line)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"(?<![\w*])\*(?=\S)([^*]*?\S)\*(?![\w*])", r"\1", s)        # *emphasis*; any other `*` stays (an operator is not markdown)
    return s.strip()


def _words(s: str) -> list:
    return re.findall(r"[a-z']+", normalise(s).lower())


def _is_leadin(s: str, t: dict = None) -> bool:
    """A bare lead-in line: ends with a colon, holds no digit and only function words and the words of the question ("Answer:", "Here is the string:",
    "Offset:"); for an exact-string task also the lead-in words of grammar.LEAD_WORDS ("Plaintext:"). Never a free noun such as `key` or `size`."""
    if not s.endswith(":") or re.search(r"\d", s):
        return False
    ok = set(grammar.LEADIN_LINE_WORDS)
    if t is not None:
        ok |= set(re.findall(r"[a-z]+", t.get("question", "").lower()))
        if t.get("score") in ("exact_case", "exact_head"):
            ok |= grammar.LEAD_WORDS
    return all(w in ok for w in _words(s))


def _is_prose(s: str) -> bool:
    """A sentence (4+ whitespace-separated words, 3+ of them plain words), as opposed to an answer line, a list item or a URL."""
    toks = s.split()
    return len(toks) >= 4 and sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]+[.,;:!?]?", w)) for w in toks) >= 3


def _answer_lines(answer: str) -> list:
    """[(raw line, markdown-stripped line)] of the non-empty lines; fences and rules are left out."""
    out = []
    for raw in answer.splitlines():
        if _FENCE.match(raw):
            continue
        s = _strip_md(raw)
        if s and not _SEPARATOR.fullmatch(s):
            out.append((raw, s))
    return out


_ABBREV = frozenset("e.g. i.e. etc. vs. approx. fig. figs. no. nos. cf. st. al. mr. mrs. ms. dr. eq. eqs. ch. sec. vol. pp. ca. esp. incl. "
                    "resp. ref. refs. a.k.a. aka.".split())
_END = re.compile(r"[.!?][\"')\]]*(?=\s+[A-Z\"'(\[]|\s*$)")


def _boundary(text: str):
    """Index just after the first sure sentence end of `text` (a `.`, `!` or `?` followed by a space and a capital letter, or by the end), or
    None. Abbreviations (`approx.`, `Fig.`, `No.`, `e.g.`), initials (`U.S.`), a mark inside an open parenthesis or quote, a decimal and a
    mark followed by a lower-case word are not sentence ends."""
    for m in _END.finditer(text):
        tok = re.search(r"(\S+)$", text[:m.start() + 1]).group(1).lstrip("([\"'").lower()
        if tok in _ABBREV or re.fullmatch(r"(?:[a-z]\.)+", tok):
            continue
        upto = text[:m.end()]                                  # the closing quote or parenthesis right after the mark belongs to the sentence
        if upto.count("(") > upto.count(")") or upto.count("[") > upto.count("]") or upto.count('"') % 2:
            continue
        return m.end()
    return None


class Reply(NamedTuple):
    lead: str          # the bare lead-in line that was skipped ("Answer:"), or ""
    head: str          # the headline: the first sentence of the first answer line, markdown removed
    pieces: list       # what follows: the rest of that line, then the other lines (all markdown removed)


def parse_reply(answer: str, t: dict = None):
    """The structure of a reply, or None when it has no headline: nothing but fences, a lead-in with nothing after it, or an answer line
    followed by something that is not a sentence (a second value, a list, a code block with two values: the reply is a candidate list)."""
    lines = _answer_lines(answer)
    if not lines:
        return None
    i = 1 if _is_leadin(lines[0][1], t) else 0
    if i >= len(lines):
        return None
    line = lines[i][1]
    b = _boundary(line)
    head, tail = (line, "") if b is None or not line[b:].strip() else (line[:b].strip(), line[b:].strip())
    pieces = ([tail] if tail else []) + [s for _, s in lines[i + 1:]]
    if pieces and pieces[0].endswith(":") and not re.search(r"\d", pieces[0]) and set(_words(pieces[0])) <= _JUSTIFICATION_LEAD:
        pieces = pieces[1:]                                                    # "Justification:" before the sentence
    if pieces and not _is_prose(pieces[0]):
        return None
    return Reply(lines[0][1] if i else "", head, pieces)


def headline(answer: str, t: dict = None) -> str:
    """The headline of the reply, markdown removed; "" if there is none (empty reply, a lead-in with nothing after it, a list). `t`: the task, whose
    question's words may stand in a lead-in line."""
    r = parse_reply(answer, t)
    return "" if r is None else r.head


# ---- scoring -----------------------------------------------------------------------------------------------------------------

ERROR_PREFIX = "[error"       # what modal_rea_data.sample_answers writes for a failed request (`[error <exception repr>]`)


def is_error_reply(reply: str) -> bool:
    """The text of a failed request, not a reply of the model (a dead server, a timeout, a prompt over the slot size). Never scored, never stored."""
    return reply.lstrip().startswith(ERROR_PREFIX)


def check_mode(t: dict) -> None:
    """Fail fast on a task whose score mode this module does not know (a GPU run must not die half way through on it)."""
    if t.get("score") not in MODES:
        raise ValueError(f"unknown score mode {t.get('score')!r} in task {t.get('id', '?')}; known: {', '.join(MODES)}")


def _headline_ok(head: str, t: dict) -> bool:
    if t["score"] == "tokens":
        return headline_ok(head, t)
    return any(is_exact_answer(head, k) for k in t["accept"])            # exact_case, exact_head


def _headline_score(answer: str, t: dict) -> bool:
    r = parse_reply(answer, t)
    return r is not None and _headline_ok(r.head, t)


def score(answer: str, t: dict) -> bool:
    """EVALUATION scorer: the frozen whole-reply substring scoring for the legacy modes (comparable with earlier numbers), the whitelist
    grammar on the headline for the modes of the generated families."""
    check_mode(t)
    if is_error_reply(answer):
        return False                                           # the text of a failed request is an answer to nothing, in either scorer
    if t["score"] in LEGACY_MODES:
        return _legacy_score(answer, t)
    return _headline_score(answer, t)


def score_strict(answer: str, t: dict) -> bool:
    """DATA scorer (sample stage, `sft_target`): the whitelist grammar for every family. A legacy task is converted (`strict_task`); the text
    of a failed request is never an answer."""
    check_mode(t)
    return not is_error_reply(answer) and _headline_score(answer, strict_task(t))


def score_any_line(answer: str, t: dict) -> bool:
    """Format-insensitive control: does ANY line of the reply (read as a reply of its own) score? A model that thinks aloud on
    line 1 and answers on line 4 scores 0 under `score` and 1 here; the gap is the share of misses that are formatting. Not
    used for training data, only to read a held-out score next to the strict one."""
    check_mode(t)
    if is_error_reply(answer):
        return False
    if t["score"] in LEGACY_MODES:
        return _legacy_score(answer, t)
    return score(answer, t) or any(score(line, t) for line in answer.splitlines() if line.strip())


# ---- what is stored ----------------------------------------------------------------------------------------------------------

def accepted_headline(reply: str, t: dict):
    """The teacher's headline (raw, markdown removed) when the data scorer accepts it as the task's answer, else None: no headline, a headline that
    does not parse as the answer (every family, legacy included: `strict_task`), or the text of a failed request. This is the FILTER of the sample
    stage; nothing of the reply is stored."""
    check_mode(t)
    if is_error_reply(reply):
        return None
    s = strict_task(t)
    r = parse_reply(reply, s)
    return r.head if r is not None and _headline_ok(r.head, s) else None


def check_target(t: dict) -> None:
    """Fail fast on a task whose truth template is missing or renders a text its own scorer rejects (a GPU run must not find that out at the end)."""
    from . import targets
    r = targets.render(t)
    if not score_strict(r["text"], t):
        raise ValueError(f"the truth template of task {t.get('id', '?')} ({r['key']}) renders a text that score_strict rejects: {r['text']!r}")


def sft_target(reply: str, t: dict):
    """The target to store for a reply, or None when the reply is not an answer (see accepted_headline). The text is RENDERED FROM THE TRUTH
    (targets.render): the canonical headline and one factual sentence; nothing of the reply is in it. {"text", "headline", "sentence", "template",
    "teacher_headline", "target_source"}; `teacher_headline` is the raw headline of the reply, kept for analysis only."""
    from . import targets
    head = accepted_headline(reply, t)
    if head is None:
        return None
    r = targets.render(t)
    if not score_strict(r["text"], t):
        raise ValueError(f"the truth template of task {t.get('id', '?')} ({r['key']}) renders a text that score_strict rejects: {r['text']!r}")
    return {"text": r["text"], "headline": r["headline"], "sentence": r["sentence"], "template": r["template"], "teacher_headline": head,
            "target_source": targets.TARGET_SOURCE}


def canonical(t: dict) -> str:
    """A correct answer written from the truth alone (first spelling of each group), for tests."""
    mode, acc = t["score"], t["accept"]
    if mode in ("tokens", "all_groups"):
        return ", ".join(g[0] for g in acc)
    if mode == "all":
        return ", ".join(acc)
    return acc[0]

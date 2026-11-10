"""Answer scoring against a task's truth (`score` + `accept` [+ `reject`, `cand`, `allow`]).

Legacy modes (the original ten templates; whole reply, substring based, semantics frozen so earlier numbers stay comparable):
  exact / any       one accepted spelling must appear in the answer (whitespace and case ignored)
  all               every accepted value must appear (each with its numeric spellings, see `variants`)
  all_groups        every group must have one spelling in the answer

Headline modes (the generator's new families). One rule for all of them: only the HEADLINE of the reply is scored, and the headline
must be an answer, not a hedge.

  headline          first non-empty line of the reply with markdown removed (`**`, backticks, `#`, bullets). A bare lead-in line
                    ("Answer:", ends with a colon, no digit) is skipped only when exactly ONE answer line follows before prose or the
                    end of the reply: "Possible offsets:\n0x8\n0x10\n0x18" is a list, not an answer.
  hedge             a headline that contains `or|either|nor|not|n't|instead|rather than|vs|maybe|perhaps|possibly|probably|likely|
                    might|could|...` fails in `tokens` mode (the whole-answer match of the exact modes leaves no room for one).
  second candidate  a headline that holds another value of the key's shape fails: a number the key does not contain, a `reject`
                    name (the family's name vocabulary plus common confusables), a `cand`-shaped token (fsm input strings,
                    environment variable names, base<N>). Numbers directly attached to a unit or a standard name that the question
                    did not ask for ("128-bit", "(88 bits)", "x86-64", "CRC-32") are not candidates.
  tokens            every group must match a whole token of the headline. Numbers are compared by value (`0x08` = `0x8`,
                    `16u` = 16, `250,000,000` = 250000000). With two numeric groups (lower, upper bound) they must appear in order.
                    `allow` lists phrases that are blanked first ("extended tea": XTEA may be explained, "TEA" alone may not).
  exact_case        the whole headline is the key: after quotes, backticks, markdown, a short lead-in phrase ("The string is",
                    "Alphabet:") and trailing punctuation nothing but the key is left. Case and spaces are significant (alphabets
                    and plaintexts: "aB" and "Ab" are different; "admin login 42" is not "adminlogin42"), and the key is delimited:
                    "xkey", "key7" and "/etc/key" are other strings.
  exact_head        the same whole-headline match for plaintexts and secrets.

What is stored as an SFT target is the headline and at most ONE justification sentence (`sft_target`, `sft.py`): the rest of the
reply is cut off (`truncated: true`), and a sentence that names another candidate value, hedges, or says wait / actually /
correction / however vetoes the reply. The stored text always scores True itself.

Substring modes are fine for long values (a key, a 32-bit constant) and vacuous for small ones ("0x18" contains "8", "XTEA"
contains "tea"); the families that ask for offsets, rounds, key lengths and names use `tokens`.
"""
from __future__ import annotations

import re

LEGACY_MODES = ("exact", "any", "all", "all_groups")
HEADLINE_MODES = ("tokens", "exact_case", "exact_head")
MODES = LEGACY_MODES + HEADLINE_MODES


# ---- legacy modes ------------------------------------------------------------------------------------------------------

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


# ---- headline ----------------------------------------------------------------------------------------------------------

_MD_LEAD = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+•]\s+)+")
_SEPARATOR = re.compile(r"[-_=~\s]*")
_FENCE = re.compile(r"^\s*(?:```|~~~)")

# A headline that hedges is not an answer. `n't` covers isn't / doesn't / can't. The key's own spellings never contain these words
# (a test walks every generated task), so the check can look at the whole headline.
HEDGE = re.compile(r"\b(?:or|either|nor|not|cannot|instead|rather\s+than|vs|versus|maybe|perhaps|possibly|probably|likely|unlikely|"
                   r"might|could|unsure|presumably)\b|n['’]t\b", re.I)
# Words of a reply that changes its mind: a justification sentence with one of them vetoes the reply.
HESITATION = re.compile(r"\b(?:wait|actually|correction|however|hmm+|oops|sorry|mistake|scratch\s+that|on\s+second\s+thought|let\s+me|"
                        r"re-?check|re-?compute|re-?evaluate|but)\b", re.I)
UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|possibly|probably|likely|might|could|guess|unsure|not\s+sure|i\s+think|i\s+believe)\b", re.I)


def _strip_md(line: str) -> str:
    s = _MD_LEAD.sub("", line)
    return s.replace("**", "").replace("__", "").replace("`", "").replace("*", " ").strip()


def _is_leadin(s: str) -> bool:
    """A bare lead-in line: ends with a colon and holds no digit ("Answer:", "Here it is:")."""
    return s.endswith(":") and not re.search(r"\d", s)


def _is_prose(s: str) -> bool:
    """A sentence (4+ words), as opposed to an answer line or a list item."""
    return len(re.findall(r"[A-Za-z]{2,}", s)) >= 4


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


def _split(answer: str) -> tuple:
    """(index of the headline in the lines, the lines); index None when there is no usable headline. A lead-in line is skipped only
    when exactly one answer line follows before prose or the end ("Possible offsets:" followed by a list has no headline)."""
    lines = _answer_lines(answer)
    if not lines:
        return None, lines
    if _is_leadin(lines[0][1]):
        if len(lines) < 2 or (len(lines) > 2 and not _is_prose(lines[2][1])):
            return None, lines
        return 1, lines
    return 0, lines


def headline(answer: str) -> str:
    """The headline of the reply, markdown removed; "" if there is none (empty reply, a lead-in followed by a list)."""
    i, lines = _split(answer)
    return "" if i is None else lines[i][1]


# ---- tokens ------------------------------------------------------------------------------------------------------------

# A token is a run of [0-9a-z_] with an optional leading minus that is not glued to a preceding word ("3-5" is 3 and 5,
# "x is -37" is -37). Hyphenated words ("x-tea", "base-64") are also offered as one token, and so is a run of single letters
# ("b b b", "b, c, d" -> "bbb", "bcd": the fsm answers are letter strings).
_TOK = re.compile(r"(?<![0-9a-z_])-?[0-9a-z_]+")
_HYPHEN_WORD = re.compile(r"[a-z_]+(?:-[0-9a-z_]+)+")
_LETTER_RUN = re.compile(r"(?<![0-9a-z_])[a-z](?:[\s,;>\-]+[a-z](?![0-9a-z_]))+(?![0-9a-z_])")
_NUM = re.compile(r"(-?)(0x[0-9a-f]+|\d+)(?:ull|llu|ul|lu|ll|u|l)?")
_THOUSANDS = re.compile(r"(?<![0-9a-z_,])\d{1,3}(?:,\d{3})+(?![0-9a-z_])")
# A number that is part of a unit or a standard name the question did not ask about is not a competing candidate:
# "(128-bit key)", "-byte aligned", "(88 bits)", "x86-64", "CRC-32", "AES-128", "base-64".
_UNIT_AFTER = re.compile(r"-(?:bit|byte)s?\b")
_PAREN_BITS = re.compile(r"\s+bits?\b")
_NAME_BEFORE = re.compile(r"(?:crc|x86|x64|aes|sha|md|rc|des|base)-$")


def num_value(tok: str):
    """Integer value of a decimal or 0x literal, C suffix allowed ("0x08" -> 8, "16ul" -> 16); None for anything else."""
    m = _NUM.fullmatch(tok)
    if not m:
        return None
    v = int(m.group(2), 16 if m.group(2).startswith("0x") else 10)
    return -v if m.group(1) else v


def _prep(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(?<=\d)'(?=[0-9a-f])|(?<=[0-9a-f])'(?=\d)", "", t)      # C23 digit separators: 1'000
    t = re.sub(r"(?<=\d)_(?=\d)", "", t)                                  # 1_000, 0x1234_5678
    return _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), t)       # 250,000,000


def _is_hexlike(tok: str) -> bool:
    """A bare hex constant: 2+ hex digits with a letter and a digit (3f, edb88320). Plain words never look like this."""
    return len(tok) >= 2 and re.fullmatch(r"[0-9a-f]+", tok) is not None and re.search(r"[a-f]", tok) is not None and re.search(r"\d", tok) is not None


def tokens(answer: str) -> set:
    return {t for t, _v, _p in _token_list(answer)}


def _token_list(text: str) -> list:
    """[(token, numeric value or None, position in _prep(text))]: the plain tokens, then hyphenated words and letter runs."""
    t = _prep(text)
    out = [(m.group(0), num_value(m.group(0)), m.start()) for m in _TOK.finditer(t)]
    out += [(m.group(0), None, m.start()) for m in _HYPHEN_WORD.finditer(t)]
    out += [(re.sub(r"[^a-z]", "", m.group(0)), None, m.start()) for m in _LETTER_RUN.finditer(t)]
    return out


def _ignorable(prep: str, pos: int, tok: str) -> bool:
    """A number that belongs to a unit or a standard name ("128-bit", "(88 bits)", "CRC-32"): not a candidate for the key."""
    end = pos + len(tok)
    return bool(_UNIT_AFTER.match(prep, end) or (prep[max(0, pos - 1):pos] == "(" and _PAREN_BITS.match(prep, end))
                or _NAME_BEFORE.search(prep[max(0, pos - 8):pos]))


def _split_group(group: list) -> tuple:
    nums = {num_value(a.lower()) for a in group} - {None}
    return nums, {a.lower() for a in group if num_value(a.lower()) is None}


def _blank_allowed(text: str, phrases) -> str:
    """The text with the task's `allow` phrases (an explanation of the key: "extended tea" for XTEA) replaced by a space."""
    for ph in phrases or []:
        text = re.sub(re.escape(ph), " ", text, flags=re.I)
    return text


def _competitor(text: str, t: dict) -> bool:
    """True when `text` names another value of the key's shape: a number the key does not contain, a bare hex constant, a `reject`
    name, or a `cand`-shaped token. A reply that lists candidates is not an answer."""
    prep = _prep(text)
    toks = _token_list(text)
    texts = {tok for tok, _v, _p in toks}
    if any(r.lower() in texts for r in t.get("reject", [])):
        return True
    groups = [_split_group(g) for g in t["accept"]]
    wanted = set().union(*(nums for nums, _ in groups)) if groups else set()
    literals = set().union(*(lits for _, lits in groups)) if groups else set()
    if wanted:                                          # numeric key: no number besides the key's own
        for tok, v, p in toks:
            if v is not None and v not in wanted and not _ignorable(prep, p, tok):
                return True
            if v is None and _is_hexlike(tok) and tok not in literals:
                return True
    if "cand" in t:                                     # non-numeric candidates of the same shape (fsm input strings)
        shape = re.compile(t["cand"])
        if any(shape.fullmatch(tok) and tok not in literals for tok, v, _p in toks if v is None):
            return True
    return False


def _arithmetic(head: str) -> bool:
    """Two numbers joined by an operator ("40 + 40", "7 * 8", "n == 8"): an expression, not an answer."""
    prep = _prep(head)
    nums = [m for m in _TOK.finditer(prep) if num_value(m.group(0)) is not None]
    return any(set(prep[a.end():b.start()].strip()) and set(prep[a.end():b.start()].strip()) <= set("+*/=^&|<>%")
               for a, b in zip(nums, nums[1:]))


def _tokens_score(head: str, t: dict) -> bool:
    head = _blank_allowed(head, t.get("allow"))
    if HEDGE.search(head) or _arithmetic(head):
        return False
    toks = _token_list(head)
    if not toks:
        return False
    groups = [_split_group(g) for g in t["accept"]]
    first, order = {}, []
    for gi, (nums, lits) in enumerate(groups):
        hits = [p for tok, v, p in toks if (v is not None and v in nums) or tok in lits]
        if not hits:
            return False
        first[gi] = min(hits)
        if nums:
            order.append(gi)
    if [first[g] for g in order] != sorted(first[g] for g in order) or len({first[g] for g in order}) != len(order):
        return False                                    # two numeric groups (lower, upper) must come in order
    return not _competitor(head, t)


# ---- exact strings -----------------------------------------------------------------------------------------------------

_WRAP = "\"'`“”‘’()[]<>"
_TRAIL = ".,;:!?" + _WRAP
_LEAD_WORDS = frozenset("""the a an this that it its it's proc function string text plaintext plain ascii secret alphabet result output
answer value key name path url writes write written writing returns return returned outputs produces produce builds copies decodes decoded
decode yields yield stores store is are was here here's equals equal argument buffer index order in of to into as""".split())
_PREFIX_PUNCT = re.compile(r"[\s:=,\-–>" + re.escape(_WRAP) + r"]+")


def _is_exact_answer(head: str, key: str) -> bool:
    """The whole headline is `key`: quotes, backticks, trailing punctuation and a short lead-in phrase made of function words
    ("The string is", "Alphabet:") are allowed around it; anything else (a second string, a prefix or suffix glued to the key,
    another case, other spacing) is not."""
    s = head.strip().rstrip(_TRAIL)
    if s == key:
        return True
    if not key or not s.endswith(key):
        return False
    prefix = s[:-len(key)]
    if prefix[-1] not in " \t:=-–>" + _WRAP:             # the key must start a word: "xkey", "/etc/key" are other strings
        return False
    words = re.findall(r"[A-Za-z']+", prefix)
    if _PREFIX_PUNCT.sub("", re.sub(r"[A-Za-z']+", "", prefix)):
        return False                                    # digits or other symbols in the lead-in
    return len(words) <= 6 and all(w.lower() in _LEAD_WORDS for w in words)


_STRING_SHAPE = re.compile(r"[A-Za-z0-9]+(?:[_./:@-][A-Za-z0-9]+)+|[A-Za-z0-9+/]{24,}")


def _string_competitor(text: str, t: dict) -> bool:
    """A plaintext-, path- or alphabet-shaped token other than the key in a justification."""
    for k in t["accept"]:
        text = text.replace(k, " ")
    return bool(_STRING_SHAPE.search(text))


# ---- entry points ------------------------------------------------------------------------------------------------------

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
        return _tokens_score(head, t)
    return any(_is_exact_answer(head, k) for k in t["accept"])           # exact_case, exact_head


def score(answer: str, t: dict) -> bool:
    check_mode(t)
    if t["score"] in LEGACY_MODES:
        return _legacy_score(answer, t)
    head = headline(answer)
    return bool(head) and _headline_ok(head, t)


def score_any_line(answer: str, t: dict) -> bool:
    """Format-insensitive control: does ANY line of the reply (read as a reply of its own) score? A model that thinks aloud on
    line 1 and answers on line 4 scores 0 under `score` and 1 here; the gap is the share of misses that are formatting. Not
    used for training data, only to read a held-out score next to the strict one."""
    check_mode(t)
    if t["score"] in LEGACY_MODES:
        return _legacy_score(answer, t)
    return score(answer, t) or any(score(line, t) for line in answer.splitlines() if line.strip())


# ---- what is stored ----------------------------------------------------------------------------------------------------

_DOT = "\u2024"        # stands in for the dots of e.g. / i.e. / etc. / vs. while the text is split into sentences
_ABBREV = {"e.g.": f"e{_DOT}g{_DOT}", "i.e.": f"i{_DOT}e{_DOT}", "etc.": f"etc{_DOT}", "vs.": f"vs{_DOT}"}
MAX_SENTENCE = 400


def _first_sentence(text: str) -> tuple:
    """(first sentence, rest) of the text; a sentence ends at . ! ? followed by a space or the end (e.g./i.e./etc. do not end one)."""
    for a, b in _ABBREV.items():
        text = text.replace(a, b)
    m = re.match(r"(.+?[.!?])(?:\s+|$)", text, flags=re.S)
    sent, rest = (m.group(1), text[m.end():]) if m else (text, "")
    return sent.replace(_DOT, ".").strip(), rest.replace(_DOT, ".").strip()


def justification_ok(sentence: str, t: dict) -> bool:
    """May this sentence be stored next to a right headline? It must not change its mind (wait / actually / correction / however /
    but), hedge or negate, or name another candidate value (a number the key does not contain, another cipher name, another
    string of the key's shape): such a sentence either contradicts the headline or teaches a wrong fact beside it."""
    if HESITATION.search(sentence):
        return False
    if t["score"] in LEGACY_MODES:                  # whole-reply scoring has no notion of a candidate; uncertainty words are the veto
        return not UNCERTAIN.search(sentence)
    sentence = _blank_allowed(sentence, t.get("allow"))
    if HEDGE.search(sentence):
        return False
    return not (_competitor(sentence, t) if t["score"] == "tokens" else _string_competitor(sentence, t))


def sft_target(reply: str, t: dict):
    """The text to store for a reply that scored: the headline (with its lead-in line, if it had one) and at most one justification
    sentence. {"text", "headline", "sentence", "truncated"}; None when the reply must not be stored: no right headline, a
    justification that hedges or names another candidate, a sentence longer than MAX_SENTENCE, or (legacy tasks, whose answer may sit
    anywhere in the reply) a cut reply that no longer scores. The text always scores True itself."""
    check_mode(t)
    if is_error_reply(reply):
        return None
    if t["score"] in LEGACY_MODES:
        lines = [ln.rstrip() for ln in reply.splitlines() if ln.strip()]
        if not lines:
            return None
        blocks, rest = [lines[0]], "\n".join(lines[1:])
        head = lines[0]
    else:
        i, al = _split(reply)
        if i is None or not _headline_ok(al[i][1], t):
            return None
        blocks, head = [raw.rstrip() for raw, _ in al[:i + 1]], al[i][1]
        rest_lines = [s for _, s in al[i + 1:]]
        if rest_lines and _is_leadin(rest_lines[0]):            # "Justification:" before the sentence
            rest_lines = rest_lines[1:]
        rest = " ".join(rest_lines)
    sent, more = _first_sentence(rest) if rest.strip() else ("", "")
    if len(sent) > MAX_SENTENCE or (sent and not justification_ok(sent, t)):
        return None
    text = "\n".join(blocks + ([sent] if sent else []))
    if not score(text, t):
        return None
    return {"text": text, "headline": head, "sentence": sent, "truncated": bool(more)}


def canonical(t: dict) -> str:
    """A correct answer written from the truth alone (first spelling of each group), for tests."""
    mode, acc = t["score"], t["accept"]
    if mode in ("tokens", "all_groups"):
        return ", ".join(g[0] for g in acc)
    if mode == "all":
        return ", ".join(acc)
    return acc[0]

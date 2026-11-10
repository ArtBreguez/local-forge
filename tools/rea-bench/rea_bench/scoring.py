"""Answer scoring against a task's truth (`score` + `accept` [+ `reject`, `cand`]).

Legacy modes (the original ten templates; whole reply, substring based, semantics frozen so earlier numbers stay comparable):
  exact / any       one accepted spelling must appear in the answer (whitespace and case ignored)
  all               every accepted value must appear (each with its numeric spellings, see `variants`)
  all_groups        every group must have one spelling in the answer

Headline modes (the generator's new families). Only the HEADLINE of the reply is scored: its first non-empty line with markdown
removed (`**`, backticks, `#`, bullets), skipping a bare lead-in such as "Answer:". The sampler's system prompt asks for the
answer in one short line and a justification after it, so numbers the justification restates ("n - 2", "states 0 to 7") can
never turn a wrong headline into a match, and the full reply is only worth keeping when its headline matched.
  tokens            every group must match a whole token of the headline; a `reject` token in the headline fails it.
                    Numbers are compared by value: `0x08` = `0x8`, `16u`/`16UL` = 16, `250,000,000` = 250000000 (`,` only as a
                    thousands separator, also `_` and `'`). A headline that holds a number the key does not contain (a competing
                    candidate) fails, so a reply listing several candidates ("0 1 2", "0x10 0x20") is not an answer. With two
                    numeric groups (lower and upper bound) they must appear in that order. Optional `cand` (a regex) names the
                    shape of non-numeric candidates (the fsm input strings); another such token in the headline fails too.
  exact_case        like exact on the headline but case-sensitive (alphabets: "aB" and "Ab" are different alphabets)
  exact_head        like exact on the headline (plaintexts, secrets)

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


def _strip_md(line: str) -> str:
    s = _MD_LEAD.sub("", line)
    return s.replace("**", "").replace("__", "").replace("`", "").replace("*", " ").strip()


def headline(answer: str) -> str:
    """First non-empty line of the reply, markdown removed. Fence lines, rules and a lead-in line ("Answer:", "Here it is:" - ends
    with a colon, holds no digit) are skipped; "" if nothing is left."""
    for line in answer.splitlines():
        s = _strip_md(line)
        if not s or _SEPARATOR.fullmatch(s) or (s.endswith(":") and not re.search(r"\d", s)):
            continue
        return s
    return ""


# ---- tokens ------------------------------------------------------------------------------------------------------------

# A token is a run of [0-9a-z_] with an optional leading minus that is not glued to a preceding word ("3-5" is 3 and 5,
# "x is -37" is -37). Hyphenated words ("x-tea", "base-64") are also offered as one token, and so is a run of single letters
# ("b b b", "b, c, d" -> "bbb", "bcd": the fsm answers are letter strings).
_TOK = re.compile(r"(?<![0-9a-z_])-?[0-9a-z_]+")
_HYPHEN_WORD = re.compile(r"[a-z_]+(?:-[0-9a-z_]+)+")
_LETTER_RUN = re.compile(r"(?<![0-9a-z_])[a-z](?:[\s,;>\-]+[a-z](?![0-9a-z_]))+(?![0-9a-z_])")
_NUM = re.compile(r"(-?)(0x[0-9a-f]+|\d+)(?:ull|llu|ul|lu|ll|u|l)?")
_THOUSANDS = re.compile(r"(?<![0-9a-z_,])\d{1,3}(?:,\d{3})+(?![0-9a-z_])")


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
    """A bare hex constant: 3+ hex digits with a letter and a digit (edb88320). Plain words never look like this."""
    return len(tok) >= 3 and re.fullmatch(r"[0-9a-f]+", tok) is not None and re.search(r"[a-f]", tok) is not None and re.search(r"\d", tok) is not None


def tokens(answer: str) -> set:
    return {t for t, _v, _p in _token_list(answer)}


def _token_list(text: str) -> list:
    """[(token, numeric value or None, position)]: the plain tokens, then hyphenated words and letter runs."""
    t = _prep(text)
    out = [(m.group(0), num_value(m.group(0)), m.start()) for m in _TOK.finditer(t)]
    out += [(m.group(0), None, m.start()) for m in _HYPHEN_WORD.finditer(t)]
    out += [(re.sub(r"[^a-z]", "", m.group(0)), None, m.start()) for m in _LETTER_RUN.finditer(t)]
    return out


def _split_group(group: list) -> tuple:
    nums = {num_value(a.lower()) for a in group} - {None}
    return nums, {a.lower() for a in group if num_value(a.lower()) is None}


def _tokens_score(head: str, t: dict) -> bool:
    toks = _token_list(head)
    if not toks:
        return False
    texts = {tok for tok, _v, _p in toks}
    if any(r.lower() in texts for r in t.get("reject", [])):
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
    wanted = set().union(*(nums for nums, _ in groups)) if groups else set()
    literals = set().union(*(lits for _, lits in groups)) if groups else set()
    if wanted:                                          # numeric key: no number besides the key's own
        for tok, v, _p in toks:
            if v is not None and v not in wanted:
                return False
            if v is None and _is_hexlike(tok) and tok not in literals:
                return False
    if "cand" in t:                                     # non-numeric candidates of the same shape (fsm input strings)
        shape = re.compile(t["cand"])
        if any(shape.fullmatch(tok) and tok not in literals for tok, v, _p in toks if v is None):
            return False
    return True


# ---- entry points ------------------------------------------------------------------------------------------------------

def check_mode(t: dict) -> None:
    """Fail fast on a task whose score mode this module does not know (a GPU run must not die half way through on it)."""
    if t.get("score") not in MODES:
        raise ValueError(f"unknown score mode {t.get('score')!r} in task {t.get('id', '?')}; known: {', '.join(MODES)}")


def score(answer: str, t: dict) -> bool:
    check_mode(t)
    mode = t["score"]
    if mode in LEGACY_MODES:
        return _legacy_score(answer, t)
    head = headline(answer)
    if mode == "tokens":
        return _tokens_score(head, t)
    if mode == "exact_case":
        return any(x in re.sub(r"\s+", "", head) for x in t["accept"])
    h = re.sub(r"\s+", "", head.lower())                # exact_head
    return bool(h) and any(re.sub(r"\s+", "", x.lower()) in h for x in t["accept"])


def canonical(t: dict) -> str:
    """A correct answer written from the truth alone (first spelling of each group), for tests."""
    mode, acc = t["score"], t["accept"]
    if mode in ("tokens", "all_groups"):
        return ", ".join(g[0] for g in acc)
    if mode == "all":
        return ", ".join(acc)
    return acc[0]


_score, _variants = score, variants   # names modal_rea_data.py used before the move

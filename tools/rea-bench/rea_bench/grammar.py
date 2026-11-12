"""The whitelist grammar of a headline (round 3: M1, M2, M3, m2, m9; round 4: M1, M4, m2, m3, m5).

Rounds 1 to 3 each found a new word that turned a wrong reply into an answer in a scorer built on a list of hedge words. This module
does not look for bad words. It asks whether the WHOLE headline is an answer, and rejects what it cannot parse.

    headline  :=  (filler | separator)*  group_1  (filler | separator)*  group_2  ...  group_n  (filler | separator)*

after normalisation: unicode quotes, apostrophes, dashes, the minus sign and full-width characters become ASCII, markdown and backticks
are gone, one trailing `. , ; : !` is removed (a `?` or `...` is not removed: it is hesitation). Then, token by token:

  group_i      a spelling of the i-th accept group (a number by value: `0x08` = `8` = `eight`; a name or a bare hex byte as written). Every
               group is written exactly once; a group may be written twice in a row in two spellings (`0x28 (40)`). The NAME groups (no
               number in `accept`: algorithm, direction, base, technique) of one run may come in any order (`decrypt, XTEA, 32` = `XTEA,
               decrypt, 32`, `Base64 encode` = `encode, base64`); the numeric groups keep the order of `accept` (bounds lo..hi), and a
               number stays where `accept` puts it relative to the names (`32, XTEA, decrypt` is still a leftover). Spec.block numbers the runs.
               A number ASKED IN HEX (`Spec.bases`: the question says "(hex)", a legacy task says it per group) is written `0x..` or as a bare hex with
               a letter: a plain decimal whose hex reading is another number (`18` for 0x12) is a leftover, unless it is the parenthesised respelling
               of the 0x form (`0x12 (18)`); a single digit reads the same in both bases.
  separator    `,` `;` `/` `:` the word `and`, whitespace, quotes, balanced parentheses.
  filler       a fixed vocabulary of function words (`the`, `is`, `with`, ...), the generic words `answer`/`result`, the question's own unit
               words (`bytes`, `rounds`, `iterations`, `states`, `ns`, ... per question, see UNITS), and, for a task with at most one
               numeric group, a noun that names the quantity (`offset`, `size`, `value`, ...) in the lead-in only, before the first group, and
               only if the QUESTION uses the word (after a group, or for a word the question does not use, it could label the wrong number:
               `TEA, encrypt, key 48`, `size 0x8` for an offset); nouns that describe a name (`TEA cipher`, `ptrace check`) must stand directly beside a
               name group (glue words and parentheses between, not a comma). `hex`/`hexadecimal` only beside a `0x` number or a bare hex with a
               letter, `decimal` only beside plain digits. `returns`/`return value` only where the question asks what proc returns, right before the
               number that is that value. `allow` phrases of the task ("extended tea" for XTEA, "main loop" where the question says it) are removed first.
  context      `128-bit`, `(88 bits)`, `CRC-32`: only where the question itself states that quantity (the key width, the block size, the CRC
               width) or, for a byte count, eight times a group's value; never as a hit for a group, never as a free pass; and with the label the
               question gives it (`128-bit key`, `8-byte block`: `128-bit block` and `32-bit key` are leftovers).
  bounds       a task with two numeric groups that is a closed interval (`bounds` family, legacy `threshold`) also takes `lo to hi`,
               `lo..hi`, `lo-hi`, `[lo, hi]`, `from lo to hi`, `lo <= x <= hi`, `lowest lo, highest hi`, a trailing `inclusive`. Nothing
               exclusive: `<`, `>`, `(lo, hi)`, `[lo, hi)`, `between`, `exclusive`, `strictly`, `than` are not in the vocabulary.

Everything else is a leftover and rejects: any other word (`not`, `or`, `at least`, `unknown`, `except`, a second name, a number word that is not
the key), operator, symbol or number. A number is a hit only where it is a group's value, so `40, forty-one`, `≠ 40`, `at least 40`,
`(1570, 1751)` and `CRC-40` fail for the same reason: something is left over. The price is yield: a headline with a clause after the
answer is rejected. Poison is not acceptable, yield is. Digit separators are the C23 groups only (`1'000`, `1_000`, `0xdead'beef`); a run of more
than MAX_DIGITS digits is not a number (and cannot crash the conversion).

Exact-string keys (`exact_case`, `exact_head`) are compared byte for byte (`is_exact_answer`): no normalisation, quotes and one trailing mark are stripped, and a
lead-in must END in a colon, `=` or one of a closed list of verbs (`is`, `writes`, `reads`, `prints`, `decodes to`).
"""
from __future__ import annotations

import re
from typing import NamedTuple

# ---- normalisation -----------------------------------------------------------------------------------------------------------


def _table() -> dict:
    t = {}
    for ch in "‘’‛´ʼʹʻˈ′‵":
        t[ord(ch)] = "'"
    for ch in "“”„‟″«»❝❞":
        t[ord(ch)] = '"'
    for ch in "‐‑‒–—―−﹣⁃⸺":
        t[ord(ch)] = "-"
    for ch in "               　  \t\r\f\v":
        t[ord(ch)] = " "
    for c in range(0xFF01, 0xFF5F):                     # full-width ASCII (including ＇ and －) -> ASCII
        t.setdefault(c, chr(c - 0xFEE0))
    t.update({0x2264: "<=", 0x2265: ">=", 0x2260: "!=", 0x2026: "...", 0xb5: "u", 0x3bc: "u"})
    return t


_TABLE = _table()


def normalise(s: str) -> str:
    """Unicode quotes, apostrophes, dashes, the minus sign, full-width characters and odd spaces -> ASCII, `\\x3f` -> `x3f`, one space."""
    s = s.translate(_TABLE)
    s = re.sub(r"\\x([0-9a-fA-F]{2})", r"x\1", s)
    return re.sub(r"\s+", " ", s).strip()


# ---- vocabulary --------------------------------------------------------------------------------------------------------------

GLUE = frozenset("the a an this that it its it's is are here here's there with of in at for as using uses use runs run running "
                 "performs implements equals equal exactly".split())
GENERIC = frozenset("answer result solution output final response".split())
# Nouns that name the QUANTITY asked for: they may only lead in (before the first group), only in a task with at most one numeric group, and (round 4,
# M1) only when the QUESTION itself uses the word (or its synonym below): `size 0x8` for an offset or `key 0x8` is a wrong label on a right number.
# Nouns that describe a NAME (`ptrace check`, `TEA cipher`) must stand directly beside a name group (a run of them, glue and punctuation allowed between).
NOUNS_Q = frozenset("""offset offsets size length count number value key constant field record struct structure
poly polynomial init initial register input string text plaintext secret alphabet proc""".split())
NOUNS_N = frozenset("cipher algorithm technique check hash stream block endian endianness order function family variable name".split())
RETURN_WORDS = frozenset("returns return returned".split())     # `returns 144` only where the question asks WHAT proc returns, right before that number
NOUN_SYNONYMS = {"poly": "polynomial", "polynomial": "poly", "init": "initial", "initial": "init", "offsets": "offset"}
NOUNS = NOUNS_Q | NOUNS_N
CTX_NOUNS = frozenset("key keys block blocks word words size width".split())     # the label of a context number, right behind it: `128-bit key`, `8-byte block`
CTX_LABELS = CTX_NOUNS | {"crc", "checksum"}

_ROUNDS = {"round", "rounds", "iteration", "iterations"}          # the question's own words for a round count; `64 main`, `64 cycles`, `64 times` label nothing
_BYTES = {"byte", "bytes"}
UNITS = {
    "structs": _BYTES, "pktsum": _BYTES, "cipher/rc4": _BYTES, "cipher": _ROUNDS, "multi/limit_helper": {"character", "characters", "char", "chars", "byte", "bytes"},
    "fsm/table_states": {"state", "states", "distinct", "different"}, "fsm/switch_states": {"state", "states", "distinct", "different"},
    "fsm/opcode_count": {"opcode", "opcodes", "case", "cases", "handler", "handlers", "different", "distinct"}, "fsm/opcode_value": {"opcode"},
    "antidebug/timing_ns": {"ns", "nanosecond", "nanoseconds"}, "antidebug/timing_us": {"us", "microsecond", "microseconds"},
    "bounds/char_class": {"value", "values", "byte", "bytes", "different", "distinct"},
    "lcg": _ROUNDS, "rotate": {"bit", "bits", "position", "positions"},
}
ANY_UNITS = frozenset(_ROUNDS | _BYTES | {"state", "states", "opcode", "opcodes", "value", "values", "char", "chars", "character", "characters",
                                           "ns", "us", "nanoseconds", "microseconds", "distinct", "different"})

LO_LABELS = frozenset("lowest lower low min minimum smallest from first start".split())
HI_LABELS = frozenset("highest upper high max maximum largest last end".split())
NEUTRAL_LABELS = frozenset("bound bounds limit limits range interval inclusive inclusively".split())

# Spellings that are one name in the question's own words ("little-endian" for 'little'), and phrases that only explain a name.
ALIASES = {"little": ("little-endian",), "big": ("big-endian",)}
EXPLAIN = {"tea": ("tiny encryption algorithm",)}

# The lead-in of an exact string ("The string is", "Alphabet:", "It writes"): function words only.
LEAD_WORDS = frozenset("""the a an this that it its it's proc function string text plaintext plain ascii secret alphabet result output
answer value key name path url writes write written writing reads read prints print returns return returned outputs produces produce builds copies decodes decoded
decode yields yield stores store is are was here here's equals equal argument buffer index order in of to into as input counts count
counted looks for checks final solution response""".split())
LEADIN_LINE_WORDS = GLUE | GENERIC        # a bare lead-in line: these words and the words of the question (scoring._is_leadin)

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_NUMWORD = {w: i for i, w in enumerate(_ONES)}
for _i, _tw in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2):
    _NUMWORD[_tw] = 10 * _i
    for _u in range(1, 10):
        _NUMWORD[f"{_tw}-{_ONES[_u]}"] = 10 * _i + _u


# ---- lexer -------------------------------------------------------------------------------------------------------------------

class Tok(NamedTuple):
    kind: str            # "num", "word", "ctx" (`N-bit`, `N-bytes`), "p" (one punctuation character)
    text: str
    val: int = 0         # num: value; ctx: N
    unit: str = ""       # ctx: bit | byte | char


_NUM = re.compile(r"(-?)(0x[0-9a-f]+|[0-9]+)(?:ull|llu|ul|lu|ll|u|l)?")
_THOUSANDS = re.compile(r"(?<![0-9a-z_,])[0-9]{1,3}(?:,[0-9]{3})+(?![0-9a-z_])")
_LEX = re.compile(r"""
    (?P<ctx>(?<![0-9a-z_])(?P<cn>[0-9]{1,18})-(?P<cu>bit|byte|character|char)s?(?![0-9a-z_]))
  | (?P<num>(?<![0-9a-z_])-?(?:0x[0-9a-f]+|[0-9]+)(?:ull|llu|ul|lu|ll|u|l)?(?![0-9a-z_]))
  | (?P<word>[a-z_][a-z0-9_]*(?:-[a-z0-9_]+)*(?:'[a-z]+)?|[0-9][0-9a-z_]*)
  | (?P<space>\s+)
  | (?P<other>.)
""", re.X | re.S)


MAX_DIGITS = 40          # a longer run of digits is never an answer; `int()` of thousands of digits raises (round 4, m3)


def num_value(tok: str):
    """Integer value of a decimal or 0x literal, C suffix allowed ("0x08" -> 8, "16ul" -> 16); None for anything else (and for a run of more than
    MAX_DIGITS digits: no answer is that long, and Python refuses to convert thousands of digits)."""
    m = _NUM.fullmatch(tok)
    if not m or len(m.group(2)) > MAX_DIGITS:
        return None
    v = int(m.group(2), 16 if m.group(2).startswith("0x") else 10)
    return -v if m.group(1) else v


_GROUP3 = re.compile(r"(?<![0-9a-z_'])[0-9]{1,3}(?:[_'][0-9]{3})+(?![0-9a-z_'])")
_GROUP4 = re.compile(r"(?<![0-9a-z_'])0x[0-9a-f]{1,4}(?:[_'][0-9a-f]{4})+(?![0-9a-z_'])")


def prep(text: str) -> str:
    """Lower case; digit separators only where C23 writes them (`1'000`, `1_000`: groups of three; `0xdead'beef`, `0xdead_beef`: groups of four hex
    digits) and thousands commas (250,000,000) removed. `4'0`, `4_0` and `0x2'8` are not numbers (round 4, m5)."""
    t = text.lower()
    t = _GROUP3.sub(lambda m: re.sub(r"[_']", "", m.group(0)), t)
    t = _GROUP4.sub(lambda m: re.sub(r"[_']", "", m.group(0)), t)
    return _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), t)


def lex(text: str, letter_runs: bool = False) -> list:
    """Tokens of an already normalised text. Number words (zero..ninety-nine) are numbers. With `letter_runs`, single letters separated only
    by spaces or commas are one word ("b, b, b" -> "bbb"): the fsm answers are letter strings."""
    out = []
    for m in _LEX.finditer(prep(text)):
        g = m.lastgroup
        if g == "space":
            continue
        s = m.group(0)
        if g == "ctx":
            out.append(Tok("ctx", s, int(m.group("cn")), m.group("cu")))
        elif g == "num":
            out.append(Tok("num", s, num_value(s)))
        elif g == "word":
            out.append(Tok("num", s, _NUMWORD[s]) if s in _NUMWORD else Tok("word", s))
        else:
            out.append(Tok("p", s))
    if letter_runs:
        merged, i = [], 0
        while i < len(out):
            j = i
            while j < len(out) and out[j].kind == "word" and len(out[j].text) == 1:
                k = j + 1
                if k < len(out) and out[k].kind == "p" and out[k].text == ",":
                    k += 1
                if k < len(out) and out[k].kind == "word" and len(out[k].text) == 1:
                    j = k
                else:
                    break
            if j > i:
                merged.append(Tok("word", "".join(t.text for t in out[i:j + 1] if t.kind == "word")))
                i = j + 1
            else:
                merged.append(out[i])
                i += 1
        out = merged
    return out


# ---- what a task allows ------------------------------------------------------------------------------------------------------

_QUESTION_CTX = re.compile(r"([0-9]+)-(bit|byte|character|char)s?\b(?:\s+([a-z]+))?", re.I)
_QUESTION_CRC = re.compile(r"([0-9]+)-bit\s+crc", re.I)
_QUESTION_NUM = re.compile(r"\b(?:0x[0-9a-f]+|[0-9]+)\b", re.I)
_QUESTION_RETURNS = re.compile(r"\bwhat\s+(?:does\s+)?proc\s+return", re.I)      # "what does proc return when ...": the answer IS a return value


def singular(w: str) -> str:
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


class Spec:
    """Everything a headline may contain for one task, derived from its truth dict (`accept`, `allow`, `question`, `kind`, `variant`)."""

    def __init__(self, t: dict):
        self.groups = []
        for g in t["accept"]:
            nums = {num_value(a.lower()) for a in g} - {None}
            lits = {a.lower() for a in g}
            for a in list(lits):
                lits |= {x for x in ALIASES.get(a, ())}
            self.groups.append((nums, lits))
        self.numeric = [i for i, (nums, _l) in enumerate(self.groups) if nums]
        self.multi_num = len(self.numeric) >= 2
        # A block is a run of consecutive NAME groups (algorithm, direction, base, technique) or one numeric group. The groups of a block may be
        # written in any order; the blocks, and with them every number, keep the order of `accept`.
        self.block, prev_name, b = [], False, -1
        for i in range(len(self.groups)):
            is_name = i not in self.numeric
            if not (is_name and prev_name):
                b += 1
            self.block.append(b)
            prev_name = is_name
        kind, variant = t.get("kind", ""), t.get("variant", "")
        self.interval = self.multi_num and (kind in ("bounds", "threshold") or bool(t.get("interval")))
        self.nouns_ok = not self.multi_num
        key = f"{kind}/{variant}"
        self.units = set(UNITS.get(key) or UNITS.get(kind) or (ANY_UNITS if not kind else ()))
        q = t.get("question", "")
        self.qwords = set(re.findall(r"[a-z]+", q.lower()))
        self.q_numbers = {num_value(m.group(0).lower()) for m in _QUESTION_NUM.finditer(q)} - {None}
        self.ctx, self.ctx_labels = set(), {}
        for n, u, noun in _QUESTION_CTX.findall(q):
            key = (int(n), u.lower()[:4] if u.lower().startswith("char") else u.lower())
            self.ctx.add(key)
            labels = self.ctx_labels.setdefault(key, set())
            if singular(noun.lower()) in {singular(w) for w in CTX_LABELS}:
                labels.add(singular(noun.lower()))
        self.crc = {int(n) for n in _QUESTION_CRC.findall(q)}
        values = {v for nums, _l in self.groups for v in nums}
        self.bits_ok = {n for n, u in self.ctx if u == "bit"}
        if kind == "structs" or key == "cipher/rc4" or not kind:     # a byte count whose size in bits may be stated: RC4, N bytes (8N bits)
            self.bits_ok |= {8 * v for v in values if v > 0}
        # The base each numeric group is asked in. A legacy task says so itself (`bases`: "hex" | "dec" | None per accept group, its question mixes
        # both); a generated task asks one base for all its numbers, in its own words ("(hex, like 0x1c)", "(decimal)").
        if t.get("bases"):
            self.bases = list(t["bases"])
        else:
            asked = "hex" if ("hex" in self.qwords or "hexadecimal" in self.qwords) and "decimal" not in self.qwords else \
                "dec" if "decimal" in self.qwords and "hex" not in self.qwords and "hexadecimal" not in self.qwords else None
            self.bases = [asked if i in self.numeric else None for i in range(len(self.groups))]
        self.fill = set()                # `hex 0x12`, `decimal 40`: only beside a number that is spelled that way (see headline_ok)
        for words, marker in ((("decimal", "dec"), "decimal"), (("hex", "hexadecimal"), "hex")):
            if marker in self.qwords and self.numeric:
                self.fill |= set(words)
        self.allow = [p.lower() for p in t.get("allow", [])]
        if "main loop" in q.lower():
            self.allow.append("main loop")          # "iterations of the main loop": a phrase of the question, not a free `main`
        for nums_lits in self.groups:
            for lit in nums_lits[1]:
                self.allow += list(EXPLAIN.get(lit, ()))
        self.returns = bool(_QUESTION_RETURNS.search(q))
        self.letter_runs = bool(t.get("cand"))
        self.values = values
        self.literals = {l for _n, ls in self.groups for l in ls}
        self.kind_variant = key
        self.name_groups = {i for i in range(len(self.groups)) if i not in self.numeric}

    def noun_in_question(self, w: str) -> bool:
        """A quantity noun is a lead-in only where the question itself uses the word (or its synonym: poly/polynomial, init/initial)."""
        return w in self.qwords or singular(w) in {singular(x) for x in self.qwords} or NOUN_SYNONYMS.get(w) in self.qwords

    def ctx_allowed(self, tok) -> set:
        """The labels a context number may carry (`128-bit key`): the one the question gives that (N, unit); for a bit count derived from the truth
        (RC4's `72-bit key`) `key`; no label otherwise."""
        u = tok.unit[:4] if tok.unit.startswith("char") else tok.unit
        if (tok.val, u) in self.ctx:
            return self.ctx_labels.get((tok.val, u), set())
        return {"key"} if self.kind_variant == "cipher/rc4" else set()


def _blank(text: str, phrases) -> str:
    for ph in phrases:
        text = re.sub(r"(?<![0-9a-z_-])" + re.escape(ph) + r"(?![0-9a-z_-])", " ", text, flags=re.I)
    return text


def _ctx_ok(tok: Tok, sp: Spec) -> bool:
    u = tok.unit[:4] if tok.unit.startswith("char") else tok.unit
    return tok.val in sp.bits_ok if u == "bit" else (tok.val, u) in sp.ctx


_CLOSERS = "\"'"
_CHAIN = re.compile(r"(-?(?:0x[0-9a-f]+|[0-9]+))\s*<=\s*(?:x|v|n|i|idx|index|value)\s*<=\s*(-?(?:0x[0-9a-f]+|[0-9]+))")
_PLAIN_DEC = re.compile(r"-?([0-9]+)(?:ull|llu|ul|lu|ll|u|l)?")
_CTX_LABEL_WORDS = frozenset(singular(w) for w in CTX_LABELS) | NOUNS


def _is_0x(tok: Tok) -> bool:
    return tok.kind == "num" and re.match(r"-?0x", tok.text) is not None


def _is_plain_dec(tok: Tok) -> bool:
    return tok.kind == "num" and _PLAIN_DEC.fullmatch(tok.text) is not None


def _dec_reads_differently(tok: Tok) -> bool:
    """A plain digit string whose hex reading is another number (`18` is 0x12 as a decimal and 0x18 as hex; `8` and `0` read the same either way)."""
    m = _PLAIN_DEC.fullmatch(tok.text)
    return m is not None and int(m.group(1), 10) != int(m.group(1), 16)


def _is_hex_word(tok: Tok) -> bool:
    """A bare hex spelling with a letter and a digit (`3f`, `edb88320`): the only bare hex that cannot be a decimal number or a plain word."""
    w = tok.text
    return tok.kind == "word" and len(w) >= 2 and re.fullmatch(r"[0-9a-f]+", w) is not None and any(c in "abcdef" for c in w) and any(c.isdigit() for c in w)


def _sig(toks: list, k: int, step: int):
    """The nearest token from index k in direction `step` that is not punctuation, or None."""
    while 0 <= k < len(toks) and toks[k].kind == "p":
        k += step
    return toks[k] if 0 <= k < len(toks) else None


def _paired_hex(toks: list, here: int) -> bool:
    """`0x12 (18)` and `18 (0x12)`: a decimal respelling in parentheses right behind (or in front of) the 0x spelling of the same value."""
    me = toks[here]
    def hexed(k):
        return 0 <= k < len(toks) and _is_0x(toks[k]) and toks[k].val == me.val
    def p(k, c):
        return 0 <= k < len(toks) and toks[k].kind == "p" and toks[k].text == c
    return (p(here - 1, "(") and hexed(here - 2) and p(here + 1, ")")) or (p(here + 1, "(") and hexed(here + 2) and p(here + 3, ")"))


def headline_ok(head: str, t: dict, sp: Spec = None) -> bool:
    """Does the whole headline parse as the answer grammar of the task (see the module docstring)?"""
    sp = sp or Spec(t)
    s = normalise(head).rstrip(_CLOSERS + " ")
    s = re.sub(r"[.,;:!]$", "", s).rstrip(_CLOSERS + " ")
    if sp.interval:
        m = re.fullmatch(r"\[(.*)\]", s)
        if m:
            s = m.group(1)
        s = _CHAIN.sub(r"\1 to \2", s.lower())
        s = re.sub(r"(?<=[0-9a-z])\s*\.\.\s*(?=-?[0-9a-z])", " to ", s)
    s = _blank(s.lower(), sp.allow)
    toks = lex(s, sp.letter_runs)
    groups, n = sp.groups, len(sp.groups)
    if not toks or n == 0:
        return False
    def groups_of(tok) -> list:
        return [g for g, (nums, lits) in enumerate(groups)
                if (tok.kind == "num" and (tok.val in nums or tok.text in lits)) or (tok.kind == "word" and tok.text in lits)]

    def is_name_tok(k: int) -> bool:
        return any(g in sp.name_groups for g in groups_of(toks[k]))

    def noun_n_ok(k: int) -> bool:
        """A noun that describes a name (`TEA cipher`, `ptrace check`, `variable name X`): the run of such nouns it belongs to stands directly beside a name group
        (glue words and parentheses may sit between, a comma may not). `block 0x8`, `decrypt, block 64` or `TEA, 48, a block cipher` label nothing."""
        lo = hi = k
        while lo > 0 and toks[lo - 1].kind == "word" and toks[lo - 1].text in NOUNS_N:
            lo -= 1
        while hi + 1 < len(toks) and toks[hi + 1].kind == "word" and toks[hi + 1].text in NOUNS_N:
            hi += 1
        def skip(j, step):                                     # glue words and parentheses only: a comma ends the noun's reach (`decrypt, block 64`)
            while 0 <= j < len(toks) and ((toks[j].kind == "p" and toks[j].text in "()") or (toks[j].kind == "word" and (toks[j].text in GLUE or toks[j].text == "and"))):
                j += step
            return j
        return any(0 <= j < len(toks) and is_name_tok(j) for j in (skip(lo - 1, -1), skip(hi + 1, 1)))

    def returns_ok(k: int) -> bool:
        """`returns 144`: the number that follows is a numeric group (the return value the question asks for), not a name or another count."""
        while k < len(toks) and ((toks[k].kind == "p" and toks[k].text in ":,=") or (toks[k].kind == "word" and toks[k].text in ("value", "is", "of"))):
            k += 1
        return k < len(toks) and toks[k].kind == "num" and any(g in sp.numeric for g in groups_of(toks[k]))

    def fill_ok(w: str, k: int) -> bool:
        """`hex` stands beside a number written with 0x (or a bare hex with a letter), `decimal` beside plain digits: `hex 18` and `decimal 0x6f` mislabel."""
        near = [x for x in (_sig(toks, k - 1, -1), _sig(toks, k + 1, 1)) if x is not None]
        if w in ("hex", "hexadecimal"):
            return any(_is_0x(x) or _is_hex_word(x) for x in near)
        return any(_is_plain_dec(x) for x in near)

    used, expected = set(), 0                                      # the groups written so far, and the lowest group not written yet
    current, depth, pending, last_hit, last_ctx, last_ret = None, 0, None, -9, -9, -9
    ctx_labels = set()                                             # indices of label words consumed by the context number before them
    lo_i = sp.numeric[0] if sp.interval else None
    hi_i = sp.numeric[-1] if sp.interval else None
    i = 0
    while i < len(toks):
        tok = toks[i]
        i += 1
        here = i - 1                                               # index of `tok`
        if tok.kind == "p":
            c = tok.text
            if c in ",;/:\"'":
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    return False
                depth -= 1
            elif c == "-" and sp.interval and current == lo_i and expected == hi_i and pending is None and last_hit == here - 1 \
                    and i < len(toks) and hi_i in groups_of(toks[i]):
                continue                                           # lo-hi: the hyphen directly after lo and directly before hi
            else:
                return False
            continue
        cands = groups_of(tok)
        if cands:
            if sp.interval and depth > 0:
                ok = [g for g in cands if g == current]            # in parentheses only a second spelling of the same bound
            else:                                              # a group not written yet in the block of the lowest one (a number: that group only), or a respelling
                ok = [g for g in cands if g not in used and expected < n and sp.block[g] == sp.block[expected]] or [g for g in cands if g == current]
            if not ok:
                return False
            g = ok[0]
            if pending is not None and not (g == expected and g == (lo_i if pending == "lo" else hi_i)):
                return False                                       # "lowest 1751": a label must sit on its own bound
            if sp.bases[g] == "hex" and tok.kind == "num" and _dec_reads_differently(tok) and not _paired_hex(toks, here):
                return False                                       # asked in hex: `18` is the wrong number to a model that means 0x18 (round 4, m2)
            if g not in used:
                used.add(g)
                while expected in used:
                    expected += 1
            current, pending, last_hit = g, None, here
            continue
        if tok.kind == "ctx":
            if not _ctx_ok(tok, sp):
                return False
            allowed = sp.ctx_allowed(tok)                          # `128-bit key`: the label is the question's own for that quantity (round 4, M1)
            before = toks[here - 1] if here and (here - 1) not in ctx_labels else None
            if before is not None and before.kind == "word" and singular(before.text) in _CTX_LABEL_WORDS and singular(before.text) not in allowed:
                return False
            after = toks[i] if i < len(toks) else None
            if after is not None and after.kind == "word" and singular(after.text) in _CTX_LABEL_WORDS:
                if singular(after.text) not in allowed:
                    return False
                ctx_labels.add(i)
                i += 1
            last_ctx = here
            continue
        if tok.kind == "num":
            nxt = toks[i] if i < len(toks) else None
            if depth > 0 and nxt is not None and nxt.kind == "word" and nxt.text in ("bit", "bits") and tok.val in sp.bits_ok:
                i += 1                                             # "(88 bits)"
                continue
            return False
        w = tok.text
        if w in sp.units and (not sp.multi_num or current == sp.numeric[-1]):
            continue                                               # `8 bits` after the rotation amount, not after the XOR constant
        if w in RETURN_WORDS and sp.returns and returns_ok(i):
            last_ret = here
            continue
        if w == "and" or w in GLUE or w in GENERIC or (w in sp.fill and fill_ok(w, here)) or (w == "value" and last_ret == here - 1):
            continue                                               # `return value 111`: the value of what the question asked proc to return
        if (w in NOUNS_Q and sp.nouns_ok and current is None and sp.noun_in_question(w)) or (w in NOUNS_N and noun_n_ok(here)):
            continue                                               # a quantity noun only leads in, and only in the question's own words (`key 48` would mislabel the count)
        if sp.interval:
            if w in LO_LABELS or w in HI_LABELS:
                if pending is not None or expected != (lo_i if w in LO_LABELS else hi_i):
                    return False                                   # two labels in a row, or a label for a bound that is already written
                pending = "lo" if w in LO_LABELS else "hi"
                continue
            if w in NEUTRAL_LABELS:
                continue
            if w == "to" and current == lo_i and expected == hi_i and pending is None and last_hit == here - 1 \
                    and i < len(toks) and hi_i in groups_of(toks[i]):
                continue                                           # `lo to hi`: directly between the two bounds ("lo, to hi", "lo to lo" are not a range)
        m = re.fullmatch(r"crc-?([0-9]+)", w)
        if m and int(m.group(1)) in sp.crc:
            continue
        return False
    return len(used) == n and depth == 0 and pending is None


# ---- exact strings -----------------------------------------------------------------------------------------------------------

_OPEN = "\"'`([“‘"        # what may stand in front of a key without a lead-in word
_CLOSE = "\"'`)]”’"       # what may follow it; angle brackets are comparison operators, not quotes
_WORD = r"[A-Za-z]+(?:'[A-Za-z]+)*"
LEAD_VERBS = frozenset("is writes reads prints".split())     # the closed list that may end a lead-in without a colon (and `decodes to`, and `=`)


def _lead_in_ok(prefix: str) -> bool:
    """`prefix` is what stands before an exact key. It is empty, only opening quotes, or a lead-in that ENDS in a colon, `=` or one of a closed list of
    verbs (`is`, `writes`, `reads`, `prints`, `decodes to`) after function words and nouns of LEAD_WORDS: `The string is`, `Alphabet:`, `It writes`.
    A noun directly in front of the key (`secret admin status 26`, `path ...`) is a word of the string, not a lead-in (round 4, M4)."""
    core = prefix.rstrip(_OPEN + " ")
    if not core:
        return True
    if core[-1] in ":=":
        label = core[:-1].strip()
        words = re.findall(_WORD, label)
        return 1 <= len(words) <= 6 and re.fullmatch(r"(?:%s| )*" % _WORD, label) is not None and all(w.lower() in LEAD_WORDS for w in words)
    if re.fullmatch(r"(?:%s| )+" % _WORD, core) is None:
        return False
    words = [w.lower() for w in re.findall(_WORD, core)]
    if len(words) > 6 or not all(w in LEAD_WORDS for w in words):
        return False
    return words[-1] in LEAD_VERBS or words[-2:] == ["decodes", "to"]


def is_exact_answer(head: str, key: str) -> bool:
    """The whole headline is `key`, byte for byte: quotes, backticks, one trailing `. , ; : !` and a lead-in that ends in a colon, `=` or one of the
    verbs `is`/`writes`/`reads`/`prints`/`decodes to` ("The string is", "Alphabet:") are allowed around it; anything else (a second string, a noun in
    front of the key, a prefix or suffix glued to it, another case, other spacing, a full-width or escaped spelling) is not. The reply is NOT
    normalised: case, spaces and every character are significant (round 4, M4)."""
    s = head.strip()
    if s != key:
        s = s.rstrip(_CLOSE + " ")                         # closing quotes and brackets, then ONE terminal mark ("key..." and "key?" are not the key)
        s = re.sub(r"[.,;:!]$", "", s).rstrip(_CLOSE + " ")
    if s == key:
        return True
    if not key or not s.endswith(key):
        return False
    prefix = s[:-len(key)]
    if prefix[-1] not in " :=" + _OPEN:                    # the key must start a word: "xkey", "/etc/key" are other strings
        return False
    return _lead_in_ok(prefix)

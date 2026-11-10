"""The SFT target, rendered from the GROUND TRUTH (round 4: B2, M4 stored text, m9; round 5: only semantic sentences).

Rounds 1 to 4 each found free prose that carried a wrong fact past every filter: a sentence that contradicted its own headline
("xtea, decrypt, 64" over "It encrypted the data."), numbers in swapped roles, the family's opposite word in another inflection. Every
filter was a list of bad words or shapes, and the next review found the word that was not on it. The answer is not a better filter:

    The teacher's role is a FILTER. A task enters the dataset only when the teacher's headline is accepted by `score_strict` (with the
    K-agreement rule of sft.py): that proves the task is answerable from the pseudocode. What is STORED as the assistant message is never
    the teacher's text. It is rendered here from the task's own truth: the canonical headline plus ONE factual justification sentence,
    chosen from a per-variant set of 3-6 phrasings by a seeded RNG (`random.Random("truth-template/<task id>/<variant>")`, so a task is
    always rendered the same way and the set of phrasings is spread over the tasks of a variant).

THE RULE FOR A SENTENCE (round 5, owner decision, applies to every template of every family). A sentence states only SEMANTIC facts:

  * what `proc` computes, returns, accepts or writes, as a function of its inputs (`proc returns 1 for every integer from 3 to 9`, `the result is
    XORed with 0x..`), including what it asks of the operating system (`asks to be traced by its parent`, `reads the environment variable X`);
  * the values of the task's true parameters (the key, the round count, an offset, the polynomial constant), each a named entry of the dict
    `facts(t)` returns: a value of the task's own truth (`accept`), a number or name the task's QUESTION states, or a constant of the generator's
    template for that variant, true for every task of it (the TEA/XTEA delta 0x9E3779B9, the ChaCha rotations 16/12/8/7, the XOR mask 0x7fffffff).

It does NOT describe code structure or compiler output: no "on the stack", "byte by byte", "loop", "comparison", "table lookup", "array",
"switch", "branch", "copied", "shifts right", "unsigned comparison", no helper that "calls" another. The compiler decides which of those the
model is shown: gcc turns a stack string into direct stores, clang folds an XOR loop, both rewrite a bound. A structural word is allowed only
where it is guaranteed to be visible at every cc/opt of the task AND a mechanical check proves it; there is none, structure was dropped instead.
A specific algorithm is named only where the code computes exactly that reference algorithm (TEA, XTEA, RC4, ChaCha, Salsa, base64, base32,
base58, FNV-1a, djb2, sdbm: compared with a reference implementation in tests/rea_claims.py); the legacy murmur loop is "Murmur-style".

WHAT A SENTENCE MAY CONTAIN (round 6, M1 and m4). Round 5 listed the words a template types in and a lint listed the words it must not use. Both lists were
incomplete: round 6 found `nanoseconds`, `NULL`, `CRC`, `byte offset`, `environment variable`, `debugger` typed and unchecked, and 66 of 79 false one-edit
sentences (two numbers in swapped roles, a flipped direction word, a wrong name, a wrong unit) passed the whole suite, because a checker verified the FACT that
rendered a sentence and nothing tied the words of the sentence to it. The rule is now a whitelist, written independently of this file in tests/rea_sentences.py:

  * every sentence has a PATTERN: its literal text, and a named group wherever it says something that can be wrong (a number, an algorithm, a call, a file, a
    field, an environment variable, a unit, a byte order, a direction word);
  * the checker (tests/rea_claims.py) parses the RENDERED sentence, takes the values of the groups FROM THE SENTENCE and verifies each against the task: its C
    compiled and run (a Python reference, a sweep of the domain, `offsetof`/`sizeof`, a fake libc call that records the request, the path, the variable, the clock),
    or its C source. A sentence that matches no pattern ("wording the table does not know"), a group whose value is false, and a group that no claim reads each
    fail. So a sentence contains only its pattern's literal text and its groups: `union`, `slicing-by-8`, `via a hash` are not in any pattern.
  * a sentence uses a universal word (any, every, all, always, never, only, each, none, no other, exactly) only where tests/test_rea_quantifiers.py registers the
    claim that backs it and the domain it covers (round 6, B1: "changing any byte ... makes the checksum fail" was false for Fletcher-16: a byte 0x00 -> 0xFF is not seen);
  * every edit of a template shows in tests/golden/rea_sentences.txt (every headline and sentence with a named sentinel per placeholder), and
    tests/test_rea_claims_mutations.py applies the reviewers' false sentences, false headlines and false facts (79 + 13 of round 6, 109 of round 7: 201 edits) to a
    copy of this file: the claims machinery, without the golden file, must reject each one.

THE HEADLINE IS READ LIKE THE SENTENCE (round 7, M1 and B1). The first line of the stored message is rendered from the truth too, and it says things: `It writes: <s>`
over a function that only compares its input (decode_secret shared the strobf tuple), `{n} values` where the count is of bytes, `lowest {hi}, highest {lo}`. Each
headline phrasing has a pattern in tests/rea_sentences.py (`HEADS`, one per phrasing of every key), is parsed from the RENDERED headline, and its claims read the groups
(the same claims as the sentences; a string, a label, a unit or a byte order that a headline states is verified against the compiled task like a number). What a
headline does not state but a claim needs is implied by the question, a constant of its meaning, or the groups that were read. A headline tuple shared by two keys is
true for both: every literal word of a headline is registered with the keys it is true for (`HEAD_WORDS`) and the lint rejects any other use.

EVERY SENTENCE AND EVERY HEADLINE HAS A CHECKER. tests/test_rea_claims.py holds the claims table: for every template sentence and headline of every key, its pattern and the
names of the claims it makes, and for every claim a function that verifies the parsed values against the task's C source and its compiled behaviour, over at least 200
generated tasks of every variant at two seeds, with EVERY sentence and EVERY headline of the template rendered for every task. A template without an entry, an entry
without a checker, a pattern that does not match its own template, a placeholder that the pattern does not capture: each fails the test. The rendered text of a template
stays semantic: no code structure.

`render(t)` works on the task record as the sample stage has it (`task.meta()` plus the pseudocode): `id`, `kind`, `variant`, `question`, `accept`.
No generator parameter is needed, so a decompile run made before this module existed can still be sampled. tests/test_rea_targets.py renders
2,000 tasks per family and checks every rendered text against the truth AND every fact against the generator's own `params`.
"""
from __future__ import annotations

import random
import re
from typing import Callable, NamedTuple

from . import grammar

TARGET_SOURCE = "truth-template"


class Spec(NamedTuple):
    facts: Callable            # (task record) -> {name: value}: everything a phrasing may state
    heads: tuple               # canonical headlines (format strings over the facts); phrasing i uses heads[i % len(heads)]
    sentences: tuple           # one factual sentence per phrasing (3 to 6)
    consts: tuple = ()         # names of facts that are constants of the template, true for every task of the variant (checked against the C source)


SPECS: dict = {}


def spec(key: str, heads, sentences, consts=()):
    def deco(fn):
        SPECS[key] = Spec(fn, tuple(heads), tuple(sentences), tuple(consts))
        return fn
    return deco


def _val(group) -> int:
    """The number a group stands for (its first spelling that reads as a number)."""
    for s in group:
        v = grammar.num_value(s.lower())
        if v is not None:
            return v
    raise ValueError(f"no number in {group}")


def _q(t: dict, pattern: str, what: str):
    m = re.search(pattern, t.get("question", ""), re.I)
    if m is None:
        raise ValueError(f"task {t.get('id', '?')}: the question has no {what}")
    return m.group(1)


def _qint(t: dict, pattern: str, what: str) -> int:
    s = _q(t, pattern, what)
    return int(s, 16) if s.lower().startswith("0x") else int(s)


# Headline tuples shared by several keys must be TRUE FOR EVERY KEY that uses them (round 7, B1: `multi/decode_secret` shared the strobf tuple and said
# "It writes: <secret>" of a function that only compares its input). A verb or label belongs to the keys whose `proc` does it: `tests/rea_sentences.py`
# HEAD_VERBS registers each verb with the keys it is true for, and the lint in tests/test_rea_claims.py rejects any other use.
EXACT = ("{s}", "The string is: {s}", "It writes: {s}", "Plaintext: {s}")                 # strobf: proc WRITES the string into its argument
ACCEPTED = ("{s}", "The string is: {s}", "Input: {s}", "The input is: {s}")               # decode_secret, password: proc COMPARES its input with the string
NUM = ("{n}", "It is {n}.", "Answer: {n}", "{n}.")
OFFSET = ("{off:#x}", "Offset {off:#x}", "The offset is {off:#x}.", "byte offset {off:#x}")


# ---- structs: the offsets and sizes of the x86-64 layout model ----------------------------------------------------------------------

@spec("structs/array_stride", ("{n} bytes", "{n}", "The record size is {n} bytes.", "Size: {n} bytes"),
      ("A record that proc reads is {n} bytes long, so consecutive records start {n} bytes apart.",
       "proc reads its input as a sequence of records of {n} bytes.",
       "The distance between a field of one record and the same field of the next record is {n} bytes."))
def _(t):
    return {"n": _val(t["accept"][0])}


@spec("structs/field_read", OFFSET,
      ("Once its flag field is non-zero, proc returns the field at byte offset {off:#x} of the structure, plus a constant.",
       "The value that proc returns is the field at byte offset {off:#x}, plus a constant, when the flag field is non-zero.",
       "The field at byte offset {off:#x} of the structure is read by proc and, with a constant added, returned for a non-zero flag field."))
def _(t):
    return {"off": _val(t["accept"][0])}


@spec("structs/init_magic", OFFSET,
      ("After proc has run, the constant {magic:#x} is stored at byte offset {off:#x} of the structure.",
       "proc writes {magic:#x} into the field at byte offset {off:#x}, together with the other fields that it initialises.",
       "Among the fields that proc initialises, the field that receives {magic:#x} is at byte offset {off:#x}."))
def _(t):
    return {"off": _val(t["accept"][0]), "magic": _qint(t, r"constant\s+(0x[0-9a-f]+|\d+)", "constant")}


@spec("structs/alloc_size", ("{n} bytes", "{n}", "The structure is {n} bytes.", "Size: {n} bytes"),
      ("proc allocates memory for a structure of {n} bytes.",
       "A structure that proc allocates is {n} bytes long, whether it allocates a single structure or several.",
       "The memory that proc allocates is made of structures of {n} bytes."))
def _(t):
    return {"n": _val(t["accept"][0])}


@spec("structs/nested_offset", OFFSET,
      ("The value that proc returns is a field of a nested structure, and counted from the start of the outer structure it is at byte offset {off:#x}.",
       "proc returns a field of a nested structure, which is at byte offset {off:#x} of the outer structure.",
       "Adding the offset of the nested structure to the offset of the field inside it gives byte offset {off:#x} of the outer structure."))
def _(t):
    return {"off": _val(t["accept"][0])}


@spec("structs/linked_next", OFFSET,
      ("proc follows the pointer at byte offset {off:#x} of a node to reach the next node of the list, and stops at a node whose pointer is NULL.",
       "The pointer to the next node is at byte offset {off:#x} of a node, and the list ends where that pointer is NULL.",
       "A node leads to its successor through the pointer stored at byte offset {off:#x}."))
def _(t):
    return {"off": _val(t["accept"][0])}


# ---- fsm ---------------------------------------------------------------------------------------------------------------------------

def _fsm_input(t):
    s = t["accept"][0][0]
    return {"s": s, "d": len(s), "final": _qint(t, r"returns\s+(\d+)\s*\.", "target state")}


_FSM_IN = ("{s}", "Input: {s}", "The input is {s}.", "\"{s}\"")
_FSM_IN_S = ("Starting in state 0, the {d} letters of this input take the machine to state {final}, which is what proc returns.",
             "The machine reads the {d} letters of this string in turn and ends in state {final}.",
             "This input of {d} letters moves the machine from state 0 to state {final}, and no shorter input does.")
spec("fsm/table_input", _FSM_IN, _FSM_IN_S)(_fsm_input)
spec("fsm/switch_input", _FSM_IN, _FSM_IN_S)(_fsm_input)


def _fsm_states(t):
    n = _val(t["accept"][0])
    return {"n": n, "last": n - 1}


_FSM_ST_H = ("{n} states", "{n}", "{n} distinct states", "It has {n} states.")
_FSM_ST_S = ("The machine has {n} states, numbered from 0, and proc returns the state it ends in.",
             "There are {n} distinct states, numbered 0 to {last}, and the machine starts in state 0.",
             "Starting from state 0, the machine can reach all of its {n} states.")
spec("fsm/table_states", _FSM_ST_H, _FSM_ST_S)(_fsm_states)
spec("fsm/switch_states", _FSM_ST_H, _FSM_ST_S)(_fsm_states)


@spec("fsm/opcode_value", ("{n}", "opcode {n}", "It is opcode {n}.", "{n} (opcode)"),
      ("proc returns {expr} when its first argument is {n}.",
       "When the first argument is {n}, the function returns {expr}, and no other opcode does.",
       "Opcode {n} is the opcode for which proc returns {expr}."))
def _(t):
    return {"n": _val(t["accept"][0]), "expr": _q(t, r"makes it return (.+?)\?", "returned expression")}


@spec("fsm/opcode_count", ("{n}", "{n} opcodes", "It is {n}.", "{n} cases"),
      ("proc handles {n} different opcodes, each with a result of its own, and returns {dflt} for the other values of the first argument.",
       "Exactly {n} values of the first argument are handled by proc; the other values return {dflt}.",
       "proc tells {n} different opcodes apart and gives {dflt} for the rest."),
      consts=("dflt",))
def _(t):
    return {"n": _val(t["accept"][0]), "dflt": 0}


# ---- crc ---------------------------------------------------------------------------------------------------------------------------

def _crc_width(t) -> int:
    return _qint(t, r"(\d+)-bit CRC", "CRC width")


def _crc_poly(t):
    return {"w": _crc_width(t), "lit": _val(t["accept"][0]), "dir": t["accept"][1][0], "refl": t["accept"][1][0] == "reflected"}


_CRC_POLY = ("{lit:#x}, {dir}", "{lit:#x} {dir}", "The polynomial is {lit:#x}, {dir}.", "polynomial {lit:#x}, {dir}")
_CRC_POLY_S = ("proc computes a {w}-bit CRC with the polynomial constant {lit:#x}, in {dir} form.",
               "The {w}-bit CRC that proc returns is {dir}, and its polynomial constant is {lit:#x}.",
               "The polynomial of this {w}-bit CRC is given by the constant {lit:#x}, and the CRC is {dir}.")
spec("crc/bitwise_poly", _CRC_POLY, _CRC_POLY_S)(_crc_poly)
spec("crc/table_poly", _CRC_POLY, _CRC_POLY_S)(_crc_poly)


def _crc_val(t):
    return {"w": _crc_width(t), "x": _val(t["accept"][0])}


_CRC_VAL = ("{x:#x}", "It is {x:#x}.", "Answer: {x:#x}", "value {x:#x}")
_CRC_INIT_S = ("The initial value of this {w}-bit CRC is {x:#x}.",
               "proc starts the {w}-bit CRC computation from the value {x:#x}.",
               "Before the first byte is processed, the {w}-bit CRC has the value {x:#x}.")
_CRC_XOR_S = ("After the last byte, the {w}-bit CRC is XORed with {x:#x} before it is returned.",
              "The final XOR value of this {w}-bit CRC is {x:#x}.",
              "The result of the {w}-bit CRC computation is XORed with {x:#x} as its last step.")
spec("crc/bitwise_init", _CRC_VAL, _CRC_INIT_S)(_crc_val)
spec("crc/table_init", _CRC_VAL, _CRC_INIT_S)(_crc_val)
spec("crc/bitwise_xorout", _CRC_VAL, _CRC_XOR_S)(_crc_val)
spec("crc/table_xorout", _CRC_VAL, _CRC_XOR_S)(_crc_val)


# ---- cipher ------------------------------------------------------------------------------------------------------------------------

_ALG = {"tea": "TEA", "xtea": "XTEA", "rc4": "RC4", "chacha": "ChaCha", "salsa": "Salsa"}
TEA_DELTA = 0x9E3779B9
_CHACHA_ROT, _SALSA_ROT = (16, 12, 8, 7), (7, 9, 13, 18)


def _tea(t):
    d = t["accept"][1][0]
    return {"alg": _ALG[t["accept"][0][0]], "dir": d, "verb": d + "s", "done": d + "ed", "rounds": _val(t["accept"][2]), "delta": TEA_DELTA,
            "key_bits": _qint(t, r"(\d+)-bit key", "key width"), "block": _qint(t, r"(\d+)-byte block", "block size")}


_TEA_HEADS = ("{alg}, {dir}, {rounds}", "{alg}, {dir}, {rounds} iterations", "{dir}, {alg}, {rounds} iterations", "{alg} {dir}, {rounds} iterations")
_TEA_S = ("proc {verb} a block of {block} bytes with a {key_bits}-bit key, in {rounds} iterations.",
          "This {alg} implementation runs {rounds} iterations and uses the delta constant 0x{delta:08X}.",
          "The block of {block} bytes is {done} with the {key_bits}-bit key over {rounds} iterations, and the delta constant is 0x{delta:08X}.")
for _k in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec"):
    spec(f"cipher/{_k}", _TEA_HEADS, _TEA_S, consts=("delta",))(_tea)


@spec("cipher/rc4", ("{alg}, {n}", "{alg}, {n} bytes", "{alg}: {n} bytes", "It is {alg}, {n} bytes."),
      ("proc applies {alg} with a key of {n} bytes to its buffer.",
       "The key of this {alg} implementation is {n} bytes long.",
       "The keystream comes from a key of {n} bytes and a state of {state} byte values."),
      consts=("state",))
def _(t):
    return {"alg": "RC4", "n": _val(t["accept"][1]), "state": 256}


def _stream(rot):
    def f(t):
        r = _val(t["accept"][1])
        return {"alg": _ALG[t["accept"][0][0]], "rounds": r, "half": r // 2, "rot": rot, "key_bits": _qint(t, r"(\d+)-bit key", "key width"),
                "block": _qint(t, r"(\d+)-byte block", "block size")}
    return f


_STREAM_HEADS = ("{alg}, {rounds}", "{alg}, {rounds} rounds", "{alg}: {rounds} rounds", "It is {alg}, {rounds} rounds.")
_STREAM_S = ("proc computes the {alg} block function with {rounds} rounds, which is {half} double rounds.",
             "The function expands a {key_bits}-bit key into a block of {block} bytes with {rounds} rounds of {alg}.",
             "The rotation amounts are {rot[0]}, {rot[1]}, {rot[2]} and {rot[3]} bits, as in {alg}, and the function runs {rounds} rounds.")
spec("cipher/chacha", _STREAM_HEADS, _STREAM_S, consts=("rot",))(_stream(_CHACHA_ROT))
spec("cipher/salsa", _STREAM_HEADS, _STREAM_S, consts=("rot",))(_stream(_SALSA_ROT))


# ---- encoding ----------------------------------------------------------------------------------------------------------------------

_GROUPS = {"base64": (3, 4, 6), "base32": (5, 8, 5), "base58": (0, 0, 0)}      # input bytes, output characters, bits per character


def _enc_base(t) -> dict:
    base, direction = t["variant"].split("_")
    gin, gout, bits = _GROUPS[base]
    return {"base": base, "dir": direction, "radix": {"base64": 64, "base32": 32, "base58": 58}[base], "gin": gin, "gout": gout, "bits": bits}


def _enc(t):
    f = _enc_base(t)
    assert t["accept"][0][0] == f["dir"] and t["accept"][1][0] == f["base"], "the variant does not match the truth"
    return f


_ENC_HEADS = ("{dir}, {base}", "{base}, {dir}", "{base} {dir}", "{Dir}s {base}")


def _enc_facts(t):
    f = _enc(t)
    f["Dir"] = f["dir"].capitalize()
    return f


_ENC_BITS_E = ("{gin} input bytes become {gout} output characters, chosen from an alphabet of {radix} characters.",
               "proc turns groups of {gin} bytes into {gout} characters of a {radix}-character alphabet.",
               "An output character stands for {bits} bits of the input and comes from an alphabet of {radix} characters.")
_ENC_BITS_D = ("proc turns an input character back into its value in a {radix}-character alphabet and packs the values into bytes.",
               "An input character stands for {bits} bits of the decoded data, and the characters come from an alphabet of {radix} characters.",
               "The decoded bytes are packed from the values of characters of a {radix}-character alphabet, {bits} bits per character.")
spec("encoding/base64_encode", _ENC_HEADS, _ENC_BITS_E)(_enc_facts)
spec("encoding/base64_decode", _ENC_HEADS, _ENC_BITS_D)(_enc_facts)
spec("encoding/base32_encode", _ENC_HEADS, _ENC_BITS_E)(_enc_facts)
spec("encoding/base32_decode", _ENC_HEADS, _ENC_BITS_D)(_enc_facts)
spec("encoding/base58_encode", _ENC_HEADS,
     ("proc converts its input to base {radix} and writes the digits as single characters of an alphabet of {radix} characters.",
      "The output is the input written in radix {radix}, and the digits are characters of an alphabet of {radix} characters.",
      "The result has digits of radix {radix}, which are characters of an alphabet of {radix} characters."))(_enc_facts)
spec("encoding/base58_decode", _ENC_HEADS,
     ("proc reads its input as a number written in base {radix}, a single character of an alphabet of {radix} characters per digit, and writes the bytes of that number into its second argument.",
      "An input character is a digit of radix {radix}, taken from an alphabet of {radix} characters.",
      "The decoded number has the characters of the input as its digits in radix {radix}, and the characters come from an alphabet of {radix} characters."))(_enc_facts)


def _alpha(t):
    f = _enc_base(t)
    f["s"] = t["accept"][0]
    f["alen"] = _qint(t, r"(\d+)-character alphabet", "alphabet size")
    f["agent"] = "encoder" if f["dir"] == "encode" else "decoder"
    return f


_ALPHA_HEADS = ("{s}", "The alphabet is: {s}", "Alphabet: {s}", "\"{s}\"")
for _b in ("base64", "base32", "base58"):
    for _d in ("encode", "decode"):
        spec(f"encoding/{_b}_{_d}#alphabet", _ALPHA_HEADS,
             ("This is the alphabet of {alen} characters that the {agent} uses, in index order starting at the value 0.",
              "The {agent} works with this string of {alen} characters as its alphabet: its first character stands for the value 0, and the characters after it stand for the following values in order.",
              "The {alen} characters are listed in index order: the first character stands for the value 0."))(_alpha)


# ---- strobf: exact strings ---------------------------------------------------------------------------------------------------------
# Every variant says the same thing: what proc writes. How the string is stored (on a stack, XOR-encoded, rotated, reversed) is code structure that the
# compiler decides (it decodes a constant array at compile time, or builds a stack string straight into the output); the `object_gate` of
# survive.py only keeps the tasks whose compiled object still holds the string encoded.

def _plain(t):
    return {"s": t["accept"][0], "len": len(t["accept"][0])}


_STROBF_S = ("proc writes this string of {len} characters into its argument, followed by a terminating NUL byte.",
             "The {len} characters of this string, followed by a NUL byte, are what proc writes into its argument.",
             "After proc has run, its argument holds exactly this string of {len} characters and a terminating NUL byte.")
for _v in ("stack_string", "xor_const", "xor_rolling", "alt_xor", "sub_const", "reverse_xor", "rot_n"):
    spec(f"strobf/{_v}", EXACT, _STROBF_S)(_plain)


# ---- pktsum ------------------------------------------------------------------------------------------------------------------------

def _pkt(t):
    q = t["question"]
    if "little-endian or big-endian" in q:
        e = t["accept"][0][0]
        return {"ask": "endian", "e": e, "first": "low" if e == "little" else "high", "kind": t["variant"]}
    if "is that length byte" in q:
        return {"ask": "len", "n": _val(t["accept"][0]), "kind": t["variant"]}
    return {"ask": "start", "n": _val(t["accept"][0]), "kind": t["variant"]}


def _pkt_spec(variant, ends, lens, starts):
    """One spec per variant; the facts say which question was asked and the phrasing is picked from the matching set (3 phrasings each)."""
    heads = {"endian": ("{e}", "{e}-endian", "It is {e}-endian.", "{e} endian"), "start": ("{n}", "Offset {n}", "It is {n}.", "byte {n}"),
             "len": ("{n}", "Offset {n}", "It is at offset {n}.", "byte {n}")}
    SPECS[f"pktsum/{variant}"] = Spec(_pkt, heads["endian"] + heads["start"] + heads["len"], ends + starts + lens, ())
    PKT_SETS[variant] = {"endian": (0, len(ends)), "start": (len(ends), len(ends) + len(starts)),
                         "len": (len(ends) + len(starts), len(ends) + len(starts) + len(lens))}
    PKT_HEADS[variant] = heads


PKT_SETS: dict = {}
PKT_HEADS: dict = {}
_END_S = ("proc accepts the packet when its last two bytes hold the checksum as a 16-bit value stored {e}-endian, with the {first} byte first.",
          "The stored 16-bit checksum is read {e}-endian: the {first} byte comes first.",
          "The checksum that the function accepts is stored {e}-endian, with the {first} byte first.")
_START_S = ("The checksummed range starts at byte offset {n} of the packet and runs up to the stored checksum.",
            "The bytes before offset {n} do not take part in the checksum.",
            "The bytes from offset {n} up to the stored checksum are the bytes that the checksum is computed over.")
_LEN_S = ("The length byte is read from offset {n} of the packet, and the checksum follows that many payload bytes.",
          "The payload length is stored in the byte at offset {n}, and the checksum comes right after the payload.",
          "The byte at offset {n} says how many bytes the checksum covers, and the checksum is stored behind them.")
for _v in ("sum8", "xor8", "sum16_le", "sum16_be", "fletcher16", "crc16", "len_field"):
    _pkt_spec(_v, _END_S, _LEN_S, _START_S)


# ---- bounds ------------------------------------------------------------------------------------------------------------------------

def _bounds(t):
    return {"lo": _val(t["accept"][0]), "hi": _val(t["accept"][1])}


_BOUND_HEADS = ("{lo} to {hi}", "lowest {lo}, highest {hi}", "[{lo}, {hi}]", "from {lo} to {hi}")
spec("bounds/range_chain", _BOUND_HEADS,
     ("proc returns 1 for every integer from {lo} to {hi}, both included, and for no other integer.",
      "The function accepts exactly the integers between {lo} and {hi} inclusive and returns 1 for them.",
      "The lowest integer for which proc returns 1 is {lo}, and the highest is {hi}."))(_bounds)


@spec("bounds/window", _BOUND_HEADS,
      ("proc returns 1 for a window of {w} consecutive integers: it starts at {lo} and ends at {hi}.",
       "The accepted values begin at {lo}, end at {hi} and make up {w} integers.",
       "proc returns 1 for the {w} integers from {lo} to {hi}, and for no other integer."))
def _(t):
    f = _bounds(t)
    f["w"] = f["hi"] - f["lo"] + 1
    return f


spec("bounds/index_limit", _BOUND_HEADS,
     ("proc returns -1 for every index outside the range from {lo} to {hi}, and a non-negative value for every index inside it.",
      "The indexes from {lo} to {hi} are accepted: for them proc returns a byte value, and the other indexes give -1.",
      "The smallest index for which proc returns something other than -1 is {lo}, and the largest is {hi}."))(_bounds)
spec("bounds/clamp", _BOUND_HEADS,
     ("proc limits its argument to the interval from {lo} to {hi}, both included.",
      "A value below {lo} becomes {lo} and a value above {hi} becomes {hi}.",
      "The result lies between the lower bound {lo} and the upper bound {hi}."))(_bounds)


@spec("bounds/loop_iterations", ("{n}", "It returns {n}.", "returns {n}", "Answer: {n}"),
      ("For the argument {n0}, proc returns {n}.",
       "Calling proc with {n0} gives the return value {n}.",
       "The value that proc returns when it is called with {n0} is {n}."))
def _(t):
    return {"n": _val(t["accept"][0]), "n0": _qint(t, r"called with\s+(\d+)", "argument")}


@spec("bounds/char_class", ("{n}", "{n} values", "It is {n}.", "{n} byte values"),
      ("proc returns 1 for {n} of the {total} possible byte values.",
       "Counting the byte values from 0 to {top} for which proc returns 1 gives {n}.",
       "Exactly {n} byte values make proc return 1; for the other byte values it does not."),
      consts=("total", "top"))
def _(t):
    return {"n": _val(t["accept"][0]), "total": 256, "top": 255}


# ---- antidebug ---------------------------------------------------------------------------------------------------------------------

def _ad_rv(t):
    return {"tech": t["accept"][0][0], "rv": _val(t["accept"][1])}


_AD_HEADS = ("{tech}, {rv}", "{tech}, returns {rv}", "It is a {tech} check; returns {rv}.", "{tech} check, {rv}")
_PTRACE_S = ("proc asks to be traced by its parent with ptrace(PTRACE_TRACEME) and returns {rv} when that request fails, which shows that a debugger is attached.",
             "The check is a ptrace request to be traced by the parent, and the function returns {rv} if the request is refused.",
             "When ptrace(PTRACE_TRACEME) fails because a debugger is already attached, proc returns {rv}.")
spec("antidebug/ptrace", _AD_HEADS, _PTRACE_S)(_ad_rv)
spec("antidebug/ptrace_raw", _AD_HEADS, _PTRACE_S)(_ad_rv)
spec("antidebug/tracerpid", _AD_HEADS,
     ("proc reads the TracerPid line of /proc/self/status and returns {rv} when it names a tracer.",
      "The function reads the status file of its own process for TracerPid, and a tracer makes it return {rv}.",
      "A TracerPid that shows a tracer makes proc return {rv}."))(_ad_rv)


@spec("antidebug/env", ("{tech}, {NAME}", "{tech} check, {NAME}", "{NAME}, {tech}", "getenv, {NAME}"),
      ("proc looks at the environment variable {NAME}.",
       "The function tests the environment variable {NAME} as a debugging variable.",
       "The check reads the environment variable {NAME}, and that is what makes it an environment-based test."))
def _(t):
    return {"tech": t["accept"][0][0], "NAME": t["accept"][1][0].upper()}


def _timing(unit, scale):
    def f(t):
        return {"tech": t["accept"][0][0], "thr": _val(t["accept"][1]), "unit": unit, "scale": scale}
    return f


spec("antidebug/timing_ns", ("{tech}, {thr}", "{tech}, {thr} ns", "{tech}: {thr} ns", "{thr} ns, {tech}"),
     ("proc measures the time between two clock readings and reports a debugger when it is above {thr} nanoseconds.",
      "A debugger is reported when more than {thr} nanoseconds pass between the two clock readings.",
      "The largest time between the two clock readings that proc still lets pass is {thr} nanoseconds."))(_timing("nanoseconds", 10 ** 9))
spec("antidebug/timing_us", ("{tech}, {thr}", "{tech}, {thr} us", "{tech}: {thr} us", "{thr} us, {tech}"),
     ("proc measures the time between two clock readings and reports a debugger when it is above {thr} microseconds.",
      "A debugger is reported when more than {thr} microseconds pass between the two clock readings.",
      "The largest time between the two clock readings that proc still lets pass is {thr} microseconds."))(_timing("microseconds", 10 ** 6))


# ---- multi -------------------------------------------------------------------------------------------------------------------------
# The pseudocode of a helper is shown with proc's, but whether the compiler inlined it is its own business: a sentence says what proc returns.

@spec("multi/eval_arith", NUM,
      ("For the argument {n0}, proc returns {n}.",
       "Calling proc with {n0} gives {n}.",
       "The value that proc returns when it is called with {n0} is {n}."))
def _(t):
    return {"n": _val(t["accept"][0]), "n0": _qint(t, r"called with\s+(\d+)", "argument")}


@spec("multi/key_helper", ("{k:#x}", "It is {k:#x}.", "Answer: {k:#x}", "byte {k:#x}"),
      ("When it is called with the seed {seed}, proc XORs the byte {k:#x} into every byte of the buffer.",
       "The byte that proc XORs into each byte of the buffer for the seed {seed} is {k:#x}.",
       "For the seed {seed}, every byte of the buffer is XORed with {k:#x}."))
def _(t):
    return {"k": _val(t["accept"][0]), "seed": _qint(t, r"seed\s*=\s*(\d+)", "seed")}


@spec("multi/decode_secret", ACCEPTED,
      ("proc returns 1 when its input is exactly this string of {len} characters, and for other inputs it does not.",
       "Only this string of {len} characters makes proc return 1.",
       "The secret that proc accepts has {len} characters, and it is this string."))
def _(t):
    return _plain(t)


@spec("multi/chain", NUM,
      ("For the argument {n0}, proc returns {n}.",
       "Calling proc with {n0} gives {n}.",
       "The value that proc returns when it is called with {n0} is {n}."))
def _(t):
    return {"n": _val(t["accept"][0]), "n0": _qint(t, r"called with\s+(\d+)", "argument")}


@spec("multi/limit_helper", ("{n}", "{n} characters", "It is {n}.", "{n} chars"),
      ("With the mode {mode}, proc returns 1 for strings of up to {n} characters and does not for longer ones.",
       "proc accepts a string with the mode {mode} only while its length is at most {n} characters.",
       "The longest string that passes with the mode {mode} has {n} characters."))
def _(t):
    return {"n": _val(t["accept"][0]), "mode": _qint(t, r"mode\s*=\s*(\d+)", "mode")}


# ---- legacy: the original ten templates ---------------------------------------------------------------------------------------------

@spec("xor-key", ("{s}", "The key is: {s}", "Key: {s}", "XOR key: {s}"),
      ("The function XORs every byte of the buffer with this key, and the key repeats every {len} bytes.",
       "Each byte of the buffer is XORed with a character of the key string, taken in turn.",
       "The key is a string of {len} characters that is repeated along the whole buffer."))
def _(t):
    return _plain(t)


@spec("password", ACCEPTED,
      ("The function returns 1 only when its argument is equal to this string.",
       "Only an input of exactly {len} characters that equals this string makes the function return 1.",
       "The password check succeeds for this string and for no other input."))
def _(t):
    return _plain(t)


@spec("keycount", ("{s}", "The key is: {s}", "Key: {s}", "The counted key is: {s}"),
      ("The function counts the lines that start with this key followed by an equals sign and a value that is not empty, and it skips comment lines.",
       "A line is counted when it begins with this key and an equals sign and has something after it.",
       "The lines that are counted begin with this {len}-character key and an equals sign, and have something after it."))
def _(t):
    return _plain(t)


@spec("alphabet", ("{s}", "The alphabet is: {s}", "Alphabet: {s}", "\"{s}\""),
      ("The decoder returns the position of its argument in this string of {len} characters, or {miss} for any other printable character.",
       "The index that the function returns is the position in this alphabet of {len} characters.",
       "The position of a character in this string of {len} characters is the value that proc returns for it."),
      consts=("miss",))
def _(t):
    return {**_plain(t), "miss": -1}


@spec("magic", ("0x{m:08x}, {v}", "0x{m:08x} and {v}", "0x{m:08x}; {v}", "0x{m:08x} / {v}"),
      ("A header is valid only if its first four bytes, read as a little-endian 32-bit word, are 0x{m:08x}, and a version above {v} is rejected.",
       "The 32-bit magic word at the start of the header, read little-endian, must be 0x{m:08x}, and the highest version that proc accepts is {v}.",
       "A header is valid when its first four bytes, read as a little-endian 32-bit value, are 0x{m:08x} and its version is at most {v}."))
def _(t):
    return {"m": int(t["accept"][0], 16), "v": int(t["accept"][1])}


@spec("lcg", ("{a}, {r}", "{a} and {r} rounds", "{a}, {r} rounds", "{a}; {r} rounds"),
      ("The seed goes through {r} rounds of a linear congruential step with the multiplier {a}, masked with 0x{mask:08x}.",
       "A new seed is computed from the previous seed in {r} rounds, with the multiplier {a} and the mask 0x{mask:08x} that keeps the low bits.",
       "This is a linear congruential generator with the multiplier {a}, applied {r} times."),
      consts=("mask",))
def _(t):
    return {"a": int(t["accept"][0]), "r": int(t["accept"][1]), "mask": 0x7fffffff}


# The reference algorithm of a name only where the legacy C computes exactly it (tests/rea_claims.py compares the C with a Python port of each):
# FNV-1a, djb2 and sdbm do; the "murmur-ish" per-byte loop with the MurmurHash2 multiplier and seed does NOT (MurmurHash2 reads 32-bit blocks,
# starts from seed ^ length, mixes a tail and finishes with two more steps), so its name is "Murmur-style" (round 5, B3).
_HASH_NAME = {"fnv": "FNV-1a", "djb2": "djb2", "sdbm": "sdbm", "murmur": "Murmur-style"}
_HASH_INIT = {"fnv": 2166136261, "djb2": 5381, "sdbm": 0, "murmur": 0x9747b28c}      # what `h` starts from in legacy.py's template of each hash


@spec("hash", ("{name}, 0x{c:08x}", "{name} and 0x{c:08x}", "It is {name}, 0x{c:08x}.", "{name}: 0x{c:08x}"),
      ("The hash starts from the value {init:#x}, and the result is XORed with 0x{c:08x} before it is returned.",
       "The hash is {name}, and the final step XORs the result with the constant 0x{c:08x}.",
       "The function returns the hash of its input string XORed with 0x{c:08x}."),
      consts=("init",))
def _(t):
    stem = t["accept"][0][0]
    return {"name": _HASH_NAME[stem], "c": int(t["accept"][1][0], 16), "init": _HASH_INIT[stem]}


@spec("skip", ("0x{b:02x}", "It is 0x{b:02x}.", "The value is 0x{b:02x}.", "byte 0x{b:02x}"),
      ("The checksum adds up every byte of the input except those equal to 0x{b:02x}, and then adds the length.",
       "Bytes with the value 0x{b:02x} are left out of the sum, and the length of the input is added at the end.",
       "Exactly one byte value does not count in the sum, and it is 0x{b:02x}."))
def _(t):
    return {"b": int(t["accept"][0], 16)}


@spec("rotate", ("0x{k:08x}, {r}", "0x{k:08x} and {r}", "0x{k:08x}; {r}", "0x{k:08x}, {r} bits"),
      ("The function XORs its argument with 0x{k:08x} and then rotates the result left by {r} bits.",
       "The constant 0x{k:08x} is XORed in first, and the left rotation that follows is by {r} bits.",
       "A rotation to the left by {r} bits comes after the XOR with 0x{k:08x}."),
      consts=())
def _(t):
    return {"k": int(t["accept"][0], 16), "r": int(t["accept"][1])}


@spec("threshold", ("{lo} to {hi}", "lowest {lo}, highest {hi}", "[{lo}, {hi}]", "from {lo} to {hi}"),
      ("The function returns {inside} for every v from {lo} to {hi}, {below} for a smaller v and {above} for a larger v.",
       "Below {lo} it returns {below}, above {hi} it returns {above}, and in between, both bounds included, it returns {inside}.",
       "The lower bound is {lo} and the upper bound is {hi}, both inclusive."),
      consts=("below", "inside", "above"))
def _(t):
    return {"lo": int(t["accept"][0]), "hi": int(t["accept"][1]), "below": -1, "inside": 0, "above": 1}


# ---- the renderer ------------------------------------------------------------------------------------------------------------------

def key_of(t: dict) -> str:
    """Which spec renders this task: `kind/variant` (`#alphabet` for the alphabet question of an encoding task), or the kind of a legacy task."""
    k = f"{t['kind']}/{t['variant']}" if t.get("variant") else t["kind"]
    return k + ("#alphabet" if t["kind"] == "encoding" and t.get("score") == "exact_case" else "")


def _template_set(key: str, f: dict) -> tuple:
    """(heads, sentences) of the phrasings that apply to this task: pktsum has three kinds of question inside one variant."""
    s = SPECS[key]
    if key.startswith("pktsum/"):
        lo, hi = PKT_SETS[key.split("/")[1]][f["ask"]]
        return PKT_HEADS[key.split("/")[1]][f["ask"]], s.sentences[lo:hi]
    return s.heads, s.sentences


def known(t: dict) -> bool:
    return key_of(t) in SPECS


def render(t: dict, index: int | None = None, head: int | None = None) -> dict:
    """{"text", "headline", "sentence", "facts", "key", "template", "templates", "head_template", "head_templates"}: the canonical headline and one factual
    sentence for task `t`, from the truth alone. The teacher's reply plays no part here. `index` and `head` (tests only) render phrasing `index` of the
    sentences, and phrasing `head` of the headlines, of the set that applies to the task instead of the ones the seeded RNG picks: the claims table checks
    every sentence and every headline of a template against every task."""
    key = key_of(t)
    if key not in SPECS:
        raise ValueError(f"no truth template for task {t.get('id', '?')} ({key}); add one to rea_bench/targets.py")
    f = SPECS[key].facts(t)
    heads, sentences = _template_set(key, f)
    i = random.Random(f"{TARGET_SOURCE}/{t.get('id', '')}/{key}").randrange(len(sentences)) if index is None else index
    k = i % len(heads) if head is None else head
    h = heads[k].format(**f)
    sentence = sentences[i].format(**f)
    return {"text": f"{h}\n{sentence}", "headline": h, "sentence": sentence, "facts": f, "key": key, "template": i, "templates": len(sentences),
            "head_template": k, "head_templates": len(heads)}

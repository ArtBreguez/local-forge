"""The SFT target, rendered from the GROUND TRUTH (round 4: B2, M4 stored text, m9).

Rounds 1 to 4 each found free prose that carried a wrong fact past every filter: a sentence that contradicted its own headline
("xtea, decrypt, 64" over "It encrypted the data."), numbers in swapped roles, the family's opposite word in another inflection. Every
filter was a list of bad words or shapes, and the next review found the word that was not on it. The answer is not a better filter:

    The teacher's role is a FILTER. A task enters the dataset only when the teacher's headline is accepted by `score_strict` (with the
    K-agreement rule of sft.py): that proves the task is answerable from the pseudocode. What is STORED as the assistant message is never
    the teacher's text. It is rendered here from the task's own truth: the canonical headline plus ONE factual justification sentence,
    chosen from a per-variant set of 3-6 phrasings by a seeded RNG (`random.Random("truth-template/<task id>/<variant>")`, so a task is
    always rendered the same way and the set of phrasings is spread over the tasks of a variant).

What a sentence may say: only facts that are (1) a value of the task's own truth (`accept`: the key, the round count, the offset, ...), (2) a
number or name the task's QUESTION states about this very task (the constant it stores, the argument it is called with, the mode, the bit width
of the CRC), or (3) a constant of the generator's template for that variant, true for every task of it (the TEA/XTEA delta 0x9E3779B9, the
ChaCha rotations 16/12/8/7, the 256-entry RC4 state, the FNV prime). Every fact is a named entry of the dict `facts(t)` returns, and the
templates only use `{name}` placeholders of that dict: nothing in a template is typed in as a number or as the name of an algorithm.

`render(t)` works on the task record as the sample stage has it (`task.meta()` plus the pseudocode): `id`, `kind`, `variant`, `question`, `accept`.
No generator parameter is needed, so a decompile run made before this module existed can still be sampled. tests/test_rea_targets.py renders
2,000 tasks per family and checks every rendered text against the truth AND every fact against the generator's own `params` (and the constants
against the C source), so a template that states the wrong direction, the wrong round count or a constant the code does not have fails there.
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


EXACT = ("{s}", "The string is: {s}", "It writes: {s}", "Plaintext: {s}")
NUM = ("{n}", "It is {n}.", "Answer: {n}", "{n}.")
OFFSET = ("{off:#x}", "Offset {off:#x}", "The offset is {off:#x}.", "byte offset {off:#x}")


# ---- structs: the offsets and sizes of the x86-64 layout model ----------------------------------------------------------------------

@spec("structs/array_stride", ("{n} bytes", "{n}", "The record size is {n} bytes.", "Size: {n} bytes"),
      ("The loop steps through the array a record at a time, and consecutive records are {n} bytes apart.",
       "Each iteration reads a record of the array, so the stride of the loop is {n} bytes.",
       "A record is {n} bytes long: that is the distance between the fields the loop reads in turn."))
def _(t):
    return {"n": _val(t["accept"][0])}


@spec("structs/field_read", OFFSET,
      ("The value that proc returns is the field at byte offset {off:#x}, plus a constant, once a flag field has been tested against zero.",
       "After testing a flag field against zero, proc returns the field at byte offset {off:#x} of the structure plus a constant.",
       "The returned field sits at byte offset {off:#x} of the structure."))
def _(t):
    return {"off": _val(t["accept"][0])}


@spec("structs/init_magic", OFFSET,
      ("proc stores the constant {magic:#x} at byte offset {off:#x} of the structure.",
       "The constant {magic:#x} is written to the field at byte offset {off:#x}, among the other fields that proc initialises.",
       "Among the stores that initialise the structure, the store that writes {magic:#x} goes to byte offset {off:#x}."))
def _(t):
    return {"off": _val(t["accept"][0]), "magic": _qint(t, r"constant\s+(0x[0-9a-f]+|\d+)", "constant")}


@spec("structs/alloc_size", ("{n} bytes", "{n}", "The structure is {n} bytes.", "Size: {n} bytes"),
      ("The size that proc passes to the allocator for a structure is {n} bytes.",
       "proc allocates memory for structures of {n} bytes each.",
       "Each structure that proc allocates is {n} bytes long, whether it allocates a single structure or an array of them."))
def _(t):
    return {"n": _val(t["accept"][0])}


@spec("structs/nested_offset", OFFSET,
      ("The field that proc returns lives in a nested structure, and counted from the start of the outer structure it sits at byte offset {off:#x}.",
       "proc returns a field of a nested structure, which is at byte offset {off:#x} of the outer structure.",
       "Adding the offset of the nested structure to the offset of the field inside it gives byte offset {off:#x} of the outer structure."))
def _(t):
    return {"off": _val(t["accept"][0])}


@spec("structs/linked_next", OFFSET,
      ("The loop follows the pointer at byte offset {off:#x} of each node until it reaches NULL.",
       "Each iteration moves to the next node through the pointer stored at byte offset {off:#x}.",
       "The pointer to the next node of the list is at byte offset {off:#x}, and the loop ends when it is NULL."))
def _(t):
    return {"off": _val(t["accept"][0])}


# ---- fsm ---------------------------------------------------------------------------------------------------------------------------

def _fsm_input(t):
    s = t["accept"][0][0]
    return {"s": s, "d": len(s), "final": _qint(t, r"returns\s+(\d+)\s*\.", "target state")}


_FSM_IN = ("{s}", "Input: {s}", "The input is {s}.", "\"{s}\"")
spec("fsm/table_input", _FSM_IN,
     ("Starting in state 0, the {d} letters of this input drive the transition table into state {final}, which is what proc returns.",
      "The machine reads the {d} letters of this string in turn through its table and ends in state {final}.",
      "This input of {d} letters moves the machine from state 0 to state {final}, the shortest way there."))(_fsm_input)
spec("fsm/switch_input", _FSM_IN,
     ("Starting in state 0, the {d} letters of this input drive the switch into state {final}, which is what proc returns.",
      "The machine reads the {d} letters of this string in turn through its switch and ends in state {final}.",
      "This input of {d} letters moves the machine from state 0 to state {final}, the shortest way there."))(_fsm_input)


@spec("fsm/table_states", ("{n} states", "{n}", "{n} distinct states", "It has {n} states."),
      ("The transition table has a row for each of the {n} states, numbered from 0, and proc returns the state it ends in.",
       "There are {n} distinct states, numbered 0 to {last}, and the machine starts in state 0.",
       "The table is indexed by the state, and it has {n} rows: a row for each state."))
def _(t):
    n = _val(t["accept"][0])
    return {"n": n, "last": n - 1}


@spec("fsm/switch_states", ("{n} states", "{n}", "{n} distinct states", "It has {n} states."),
      ("The switch has a case for each of the {n} states, numbered from 0, and proc returns the state it ends in.",
       "There are {n} distinct states, numbered 0 to {last}, and the machine starts in state 0.",
       "Each state has its own case in the switch, and there are {n} of them."))
def _(t):
    n = _val(t["accept"][0])
    return {"n": n, "last": n - 1}


@spec("fsm/opcode_value", ("{n}", "opcode {n}", "It is opcode {n}.", "{n} (opcode)"),
      ("The case for opcode {n} returns {expr}.",
       "When the first argument is {n}, the switch returns {expr}.",
       "Opcode {n} is the opcode whose handler returns {expr}."))
def _(t):
    return {"n": _val(t["accept"][0]), "expr": _q(t, r"makes it return (.+?)\?", "returned expression")}


@spec("fsm/opcode_count", ("{n}", "{n} opcodes", "It is {n}.", "{n} cases"),
      ("The switch has {n} cases besides the default, so the function handles {n} different opcodes.",
       "Counting the cases of the switch without the default gives {n} opcodes.",
       "proc handles {n} different opcodes and falls through to the default for every other value."))
def _(t):
    return {"n": _val(t["accept"][0])}


# ---- crc ---------------------------------------------------------------------------------------------------------------------------

def _crc_width(t) -> int:
    return _qint(t, r"(\d+)-bit CRC", "CRC width")


def _crc_poly(t):
    refl = t["accept"][1][0] == "reflected"
    return {"w": _crc_width(t), "lit": _val(t["accept"][0]), "dir": t["accept"][1][0], "refl": refl,
            "shift": "right, least significant bit first" if refl else "left, most significant bit first"}


_CRC_POLY = ("{lit:#x}, {dir}", "{lit:#x} {dir}", "The polynomial is {lit:#x}, {dir}.", "polynomial {lit:#x}, {dir}")
spec("crc/bitwise_poly", _CRC_POLY,
     ("The {w}-bit CRC loop shifts the register {shift}, bit by bit, and XORs in the constant {lit:#x}.",
      "Each bit of every byte is processed in a loop that shifts the register {shift} and XORs in {lit:#x}.",
      "The code of this {w}-bit CRC contains the polynomial constant {lit:#x}, and its register shifts {shift}."))(_crc_poly)
spec("crc/table_poly", _CRC_POLY,
     ("The {entries}-entry table of this {w}-bit CRC is built with the constant {lit:#x} and a register that shifts {shift}.",
      "The table lookup of this {w}-bit CRC works on a register that shifts {shift}, and the table is made from {lit:#x}.",
      "The constant {lit:#x} appears in the loop that fills the table of the {w}-bit CRC, which shifts {shift}."))(lambda t: {**_crc_poly(t), "entries": 256})


def _crc_val(t):
    return {"w": _crc_width(t), "x": _val(t["accept"][0])}


_CRC_VAL = ("{x:#x}", "It is {x:#x}.", "Answer: {x:#x}", "value {x:#x}")
spec("crc/bitwise_init", _CRC_VAL,
     ("The register of this {w}-bit CRC starts at {x:#x} before the first byte is processed.",
      "Before the loop over the bytes begins, the {w}-bit CRC register is set to {x:#x}.",
      "The initial value of the {w}-bit CRC register is {x:#x}."))(_crc_val)
spec("crc/table_init", _CRC_VAL,
     ("The register of this table-driven {w}-bit CRC starts at {x:#x} before the first byte is looked up.",
      "Before the loop over the bytes begins, the {w}-bit CRC register is set to {x:#x}.",
      "The initial value of the {w}-bit CRC register is {x:#x}."))(_crc_val)
spec("crc/bitwise_xorout", _CRC_VAL,
     ("After the last byte, the function XORs the {w}-bit register with {x:#x} before it returns it.",
      "The final value XORed into the register of this {w}-bit CRC is {x:#x}.",
      "Before returning, the bitwise {w}-bit CRC is XORed with the constant {x:#x}."))(_crc_val)
spec("crc/table_xorout", _CRC_VAL,
     ("After the last byte, the function XORs the {w}-bit register with {x:#x} before it returns it.",
      "The final value XORed into the register of this table-driven {w}-bit CRC is {x:#x}.",
      "Before returning, the table-driven {w}-bit CRC is XORed with the constant {x:#x}."))(_crc_val)


# ---- cipher ------------------------------------------------------------------------------------------------------------------------

_ALG = {"tea": "TEA", "xtea": "XTEA", "rc4": "RC4", "chacha": "ChaCha", "salsa": "Salsa"}
TEA_DELTA = 0x9E3779B9
_CHACHA_ROT, _SALSA_ROT = (16, 12, 8, 7), (7, 9, 13, 18)


def _tea(t):
    return {"alg": _ALG[t["accept"][0][0]], "dir": t["accept"][1][0], "rounds": _val(t["accept"][2]), "delta": TEA_DELTA,
            "key_bits": _qint(t, r"(\d+)-bit key", "key width"), "block": _qint(t, r"(\d+)-byte block", "block size")}


_TEA_HEADS = ("{alg}, {dir}, {rounds}", "{alg}, {dir}, {rounds} iterations", "{dir}, {alg}, {rounds} rounds", "{alg} {dir}, {rounds} iterations")
spec("cipher/tea_enc", _TEA_HEADS,
     ("In each of its {rounds} iterations the loop adds the delta 0x{delta:08X} to the running sum.",
      "The main loop runs {rounds} iterations, and each of them adds the delta 0x{delta:08X} to the sum.",
      "The block of {block} bytes is mixed with the {key_bits}-bit key over {rounds} iterations that each add 0x{delta:08X} to the sum."),
     consts=("delta",))(_tea)
spec("cipher/tea_dec", _TEA_HEADS,
     ("In each of its {rounds} iterations the loop subtracts the delta 0x{delta:08X} from the running sum.",
      "The main loop runs {rounds} iterations, and each of them subtracts the delta 0x{delta:08X} from the sum.",
      "The block of {block} bytes is mixed with the {key_bits}-bit key over {rounds} iterations that each subtract 0x{delta:08X} from the sum."),
     consts=("delta",))(_tea)
spec("cipher/xtea_enc", _TEA_HEADS,
     ("In each of its {rounds} iterations the loop adds the delta 0x{delta:08X} to the sum and picks key words with the bits of the sum.",
      "The main loop runs {rounds} iterations, and each of them adds the delta 0x{delta:08X} to the sum.",
      "The block of {block} bytes is mixed with the {key_bits}-bit key over {rounds} iterations that each add 0x{delta:08X} to the sum, and the key word is chosen by the sum."),
     consts=("delta",))(_tea)
spec("cipher/xtea_dec", _TEA_HEADS,
     ("In each of its {rounds} iterations the loop subtracts the delta 0x{delta:08X} from the sum and picks key words with the bits of the sum.",
      "The main loop runs {rounds} iterations, and each of them subtracts the delta 0x{delta:08X} from the sum.",
      "The block of {block} bytes is mixed with the {key_bits}-bit key over {rounds} iterations that each subtract 0x{delta:08X} from the sum, and the key word is chosen by the sum."),
     consts=("delta",))(_tea)


@spec("cipher/rc4", ("{alg}, {n}", "{alg}, {n} bytes", "{alg}: {n} bytes", "It is {alg}, {n} bytes."),
      ("The function sets up a state array of {state} bytes from a key of {n} bytes, which is the key schedule of {alg}.",
       "A permutation of {state} byte values is initialised and then shuffled with the {n} bytes of the key.",
       "The key has {n} bytes, and the state array that it scrambles has {state} entries."),
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
spec("cipher/chacha", _STREAM_HEADS,
     ("Every iteration of the loop is a double round of eight quarter-rounds, so {rounds} rounds take {half} iterations.",
      "The quarter-rounds rotate by {rot[0]}, {rot[1]}, {rot[2]} and {rot[3]} bits, and the loop counts a double round as two of the {rounds} rounds.",
      "The function expands a {key_bits}-bit key into a block of {block} bytes with {rounds} rounds of the quarter-round."),
     consts=("rot",))(_stream(_CHACHA_ROT))
spec("cipher/salsa", _STREAM_HEADS,
     ("Every iteration of the loop is a double round of eight quarter-rounds, so {rounds} rounds take {half} iterations.",
      "The quarter-rounds rotate by {rot[0]}, {rot[1]}, {rot[2]} and {rot[3]} bits, and the loop counts a double round as two of the {rounds} rounds.",
      "The function expands a {key_bits}-bit key into a block of {block} bytes with {rounds} rounds of the quarter-round."),
     consts=("rot",))(_stream(_SALSA_ROT))


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


spec("encoding/base64_encode", _ENC_HEADS,
     ("Every {gin} input bytes become {gout} characters, each looked up in a table of {radix} characters.",
      "The encoder turns groups of {gin} bytes into {gout} characters of a {radix}-character alphabet.",
      "Each output character is an index of {bits} bits into an alphabet of {radix} characters."))(_enc_facts)
spec("encoding/base64_decode", _ENC_HEADS,
     ("The decoder maps each character back to its index in a {radix}-character alphabet and stops at the padding character.",
      "Each input character stands for {bits} bits of an alphabet of {radix} characters, and the loop stops at the padding character.",
      "Characters of a {radix}-character alphabet are turned back into values and packed into bytes, until the padding character."))(_enc_facts)
spec("encoding/base32_encode", _ENC_HEADS,
     ("Every {gin} input bytes become {gout} characters, each looked up in a table of {radix} characters.",
      "The encoder turns groups of {gin} bytes into {gout} characters of a {radix}-character alphabet.",
      "Each output character is an index of {bits} bits into an alphabet of {radix} characters."))(_enc_facts)
spec("encoding/base32_decode", _ENC_HEADS,
     ("The decoder maps each character back to its index in a {radix}-character alphabet and stops at the padding character.",
      "Each input character stands for {bits} bits of an alphabet of {radix} characters, and the loop stops at the padding character.",
      "Characters of a {radix}-character alphabet are turned back into values and packed into bytes, until the padding character."))(_enc_facts)
spec("encoding/base58_encode", _ENC_HEADS,
     ("The encoder divides the input as a big number by {radix} again and again, and each remainder is a character of a {radix}-character alphabet.",
      "The input is converted to radix {radix}, and every digit is looked up in an alphabet of {radix} characters.",
      "Remainders of repeated divisions by {radix} are the indexes into the alphabet of {radix} characters."))(_enc_facts)
spec("encoding/base58_decode", _ENC_HEADS,
     ("The decoder multiplies a big number by {radix} for every character and adds the index of that character in the {radix}-character alphabet.",
      "Each input character is a digit of radix {radix}, looked up in an alphabet of {radix} characters.",
      "The accumulated number grows by a factor of {radix} per input character, and the characters come from an alphabet of {radix} characters."))(_enc_facts)


def _alpha(t):
    f = _enc_base(t)
    f["s"] = t["accept"][0]
    f["alen"] = _qint(t, r"(\d+)-character alphabet", "alphabet size")
    f["agent"] = "encoder" if f["dir"] == "encode" else "decoder"
    f["use"] = "looks up each output character in" if f["dir"] == "encode" else "looks up each input character in"
    return f


_ALPHA_HEADS = ("{s}", "The alphabet is: {s}", "Alphabet: {s}", "\"{s}\"")
for _b in ("base64", "base32", "base58"):
    for _d in ("encode", "decode"):
        spec(f"encoding/{_b}_{_d}#alphabet", _ALPHA_HEADS,
             ("This is the alphabet of {alen} characters that the {agent} indexes, in index order starting at the value 0.",
              "The {agent} {use} this string of {alen} characters, whose first character stands for the value 0.",
              "The {alen} characters are listed in index order: the first character stands for the value 0."))(_alpha)


# ---- strobf: exact strings ---------------------------------------------------------------------------------------------------------

def _plain(t):
    return {"s": t["accept"][0], "len": len(t["accept"][0])}


for _v, _sentences in {
        "stack_string": ("The string is built byte by byte on the stack and then copied to the output buffer.",
                         "Each of the {len} characters is stored into a local array first, and the array is copied into the argument.",
                         "proc assembles the string on the stack with a store for each of its {len} characters."),
        "xor_const": ("Every stored byte is XORed with the same constant to decode the string.",
                      "The {len} bytes of the stored data are decoded with a single XOR constant and written into the argument.",
                      "A single-byte XOR key turns the stored bytes back into this string of {len} characters."),
        "xor_rolling": ("Every stored byte is XORed with a key that grows by a fixed step at each position.",
                        "The XOR key changes with the position: it advances by the same step for each of the {len} bytes.",
                        "The stored bytes are decoded with a rolling XOR key that starts at a constant and increases with the index."),
        "alt_xor": ("The even and the odd positions are XORed with two different constants.",
                    "The {len} stored bytes are decoded with an XOR key that alternates between two constants.",
                    "Two XOR constants are used in turn: a constant for the even positions and another for the odd positions of the string."),
        "sub_const": ("A constant is subtracted from every stored byte to recover the string.",
                      "The {len} stored bytes are decoded by subtracting a constant from each of them.",
                      "Each character is the stored byte minus the same constant."),
        "reverse_xor": ("The stored bytes are read from the end and XORed with a constant.",
                        "The string is written in reverse order of the stored data, and each byte is XORed with a constant.",
                        "The loop walks the stored bytes backwards, XORs them with a constant and writes the {len} characters of the string."),
        "rot_n": ("The stored text is a lowercase string that is shifted by a fixed amount, modulo {modulus}, to recover the plaintext.",
                  "Each lowercase letter of the stored string is rotated by the same number of places in the alphabet.",
                  "The decoder shifts every lowercase letter of the stored string by a constant and leaves the other characters alone."),
}.items():
    spec(f"strobf/{_v}", EXACT, _sentences, consts=())(lambda t: {**_plain(t), "modulus": 26} if t["variant"] == "rot_n" else _plain(t))


# ---- pktsum ------------------------------------------------------------------------------------------------------------------------

def _pkt(t):
    q = t["question"]
    if "little-endian or big-endian" in q:
        e = t["accept"][0][0]
        return {"ask": "endian", "e": e, "first": "low" if e == "little" else "high", "kind": t["variant"]}
    if "is that length byte" in q:
        return {"ask": "len", "n": _val(t["accept"][0]), "kind": t["variant"]}
    return {"ask": "start", "n": _val(t["accept"][0]), "kind": t["variant"]}


def _pkt_spec(variant, sums, ends, lens, starts):
    """One spec per variant; the facts say which question was asked and the phrasing is picked from the matching set (3 phrasings each)."""
    heads = {"endian": ("{e}", "{e}-endian", "It is {e}-endian.", "{e} endian"), "start": ("{n}", "Offset {n}", "It is {n}.", "byte {n}"),
             "len": ("{n}", "Offset {n}", "It is at offset {n}.", "byte {n}")}
    SPECS[f"pktsum/{variant}"] = Spec(_pkt, heads["endian"] + heads["start"] + heads["len"], ends + starts + lens, ())
    PKT_SETS[variant] = {"endian": (0, len(ends)), "start": (len(ends), len(ends) + len(starts)),
                         "len": (len(ends) + len(starts), len(ends) + len(starts) + len(lens))}
    PKT_HEADS[variant] = heads


PKT_SETS: dict = {}
PKT_HEADS: dict = {}
_END_S = ("The two checksum bytes at the end of the packet are combined with the {first} byte first, so the checksum is stored {e}-endian.",
          "The stored 16-bit checksum is read {e}-endian: the {first} byte comes first.",
          "The checksum that the function compares against is stored {e}-endian, with the {first} byte first.")
_START_S = ("The checksummed range starts at byte offset {n} of the packet and runs up to the stored checksum.",
            "The loop over the bytes of the packet begins at offset {n}.",
            "The bytes before offset {n} are not part of the checksum.")
_LEN_S = ("The length byte is read from offset {n} of the packet, and the checksum follows that many payload bytes.",
          "The payload length is stored in the byte at offset {n}, and the checksum comes right after the payload.",
          "The byte at offset {n} says how many bytes are summed, and the checksum is stored behind them.")
for _v in ("sum8", "xor8", "sum16_le", "sum16_be", "fletcher16", "crc16", "len_field"):
    _pkt_spec(_v, None, _END_S, _LEN_S, _START_S)


# ---- bounds ------------------------------------------------------------------------------------------------------------------------

def _bounds(t):
    return {"lo": _val(t["accept"][0]), "hi": _val(t["accept"][1])}


_BOUND_HEADS = ("{lo} to {hi}", "lowest {lo}, highest {hi}", "[{lo}, {hi}]", "from {lo} to {hi}")
spec("bounds/range_chain", _BOUND_HEADS,
     ("proc returns 1 for every integer from {lo} to {hi}, both included, however the compiler rewrote the comparisons.",
      "The function accepts the integers between {lo} and {hi} inclusive and returns 1 for them.",
      "The tests of the chain describe a single interval of integers: its lowest value is {lo} and its highest is {hi}."))(_bounds)


@spec("bounds/window", _BOUND_HEADS,
      ("A single unsigned comparison accepts a window of {w} integers: it starts at {lo} and ends at {hi}.",
       "The window of accepted values begins at {lo}, ends at {hi} and holds {w} integers.",
       "proc returns 1 for the {w} integers from {lo} to {hi}, and for no other integer."))
def _(t):
    f = _bounds(t)
    f["w"] = f["hi"] - f["lo"] + 1
    return f


spec("bounds/index_limit", _BOUND_HEADS,
     ("The function returns -1 for an index outside the range from {lo} to {hi}, and a table value for an index inside it.",
      "Indexes from {lo} to {hi} are looked up in the table, and any other index is rejected with -1.",
      "The smallest index that is accepted is {lo} and the largest is {hi}."))(_bounds)
spec("bounds/clamp", _BOUND_HEADS,
     ("proc limits its argument to the interval from {lo} to {hi}, both included.",
      "A value below {lo} becomes {lo} and a value above {hi} becomes {hi}.",
      "The result always lies between the lower bound {lo} and the upper bound {hi}."))(_bounds)


@spec("bounds/loop_iterations", ("{n}", "It returns {n}.", "returns {n}", "Answer: {n}"),
      ("For the argument {n0}, the loop runs to its end and proc returns the counter, which is {n}.",
       "Calling proc with {n0} makes the counter end at {n}, and that is the return value.",
       "The loop is evaluated for {n0}, and the value that comes out is {n}."))
def _(t):
    return {"n": _val(t["accept"][0]), "n0": _qint(t, r"called with\s+(\d+)", "argument")}


@spec("bounds/char_class", ("{n}", "{n} values", "It is {n}.", "{n} byte values"),
      ("The tests of the function accept {n} of the {total} possible byte values.",
       "Counting the byte values from 0 to {top} that make proc return 1 gives {n}.",
       "Exactly {n} byte values pass the character class, and the other values fail it."),
      consts=("total", "top"))
def _(t):
    return {"n": _val(t["accept"][0]), "total": 256, "top": 255}


# ---- antidebug ---------------------------------------------------------------------------------------------------------------------

def _ad_rv(t):
    return {"tech": t["accept"][0][0], "rv": _val(t["accept"][1])}


_AD_HEADS = ("{tech}, {rv}", "{tech}, returns {rv}", "It is a {tech} check; returns {rv}.", "{tech} check, {rv}")
spec("antidebug/ptrace", _AD_HEADS,
     ("proc calls ptrace with PTRACE_TRACEME and returns {rv} when that call shows that a debugger is attached.",
      "The check is a ptrace call that asks to be traced by the parent, and the function returns {rv} if it fails.",
      "When ptrace(PTRACE_TRACEME) fails because a debugger is already attached, proc returns {rv}."))(_ad_rv)
spec("antidebug/ptrace_raw", _AD_HEADS,
     ("proc issues the ptrace system call directly with PTRACE_TRACEME and returns {rv} when it shows that a debugger is attached.",
      "The check is a raw ptrace system call that asks to be traced by the parent, and the function returns {rv} if it fails.",
      "When the raw ptrace system call fails because a debugger is already attached, proc returns {rv}."))(_ad_rv)
spec("antidebug/tracerpid", _AD_HEADS,
     ("proc reads /proc/self/status, looks for the TracerPid line and returns {rv} when a tracer is attached.",
      "The function scans the status file of its own process for TracerPid, and a tracer makes it return {rv}.",
      "A TracerPid that shows a tracer makes proc return {rv}."))(_ad_rv)


@spec("antidebug/env", ("{tech}, {NAME}", "{tech} check, {NAME}", "{NAME}, {tech}", "getenv, {NAME}"),
      ("proc looks at the environment variable {NAME} with getenv.",
       "The function calls getenv(\"{NAME}\") to see whether a debugging variable is set.",
       "The check reads the environment variable {NAME}, and that is what makes it an environment-based test."))
def _(t):
    return {"tech": t["accept"][0][0], "NAME": t["accept"][1][0].upper()}


def _timing(unit, call, scale):
    def f(t):
        return {"tech": t["accept"][0][0], "thr": _val(t["accept"][1]), "unit": unit, "call": call, "scale": scale}
    return f


spec("antidebug/timing_ns", ("{tech}, {thr}", "{tech}, {thr} ns", "{tech}: {thr} ns", "{thr} ns, {tech}"),
     ("The function reads clock_gettime twice and reports a debugger when the elapsed time is above {thr} nanoseconds.",
      "A debugger is reported when more than {thr} nanoseconds pass between the two clock readings.",
      "proc compares the time between two calls of clock_gettime with {thr} nanoseconds."))(_timing("nanoseconds", "clock_gettime", 10 ** 9))
spec("antidebug/timing_us", ("{tech}, {thr}", "{tech}, {thr} us", "{tech}: {thr} us", "{thr} us, {tech}"),
     ("The function reads gettimeofday twice and reports a debugger when the elapsed time is above {thr} microseconds.",
      "A debugger is reported when more than {thr} microseconds pass between the two clock readings.",
      "proc compares the time between two calls of gettimeofday with {thr} microseconds."))(_timing("microseconds", "gettimeofday", 10 ** 6))


# ---- multi -------------------------------------------------------------------------------------------------------------------------

@spec("multi/eval_arith", NUM,
      ("Evaluating proc by hand for the argument {n0}: it calls the helper, uses its result, and returns {n}.",
       "For the argument {n0}, proc calls the helper, uses what it returns, and the value that proc returns is {n}.",
       "Calling proc with {n0} gives {n}, once the result of the helper is used."))
def _(t):
    return {"n": _val(t["accept"][0]), "n0": _qint(t, r"called with\s+(\d+)", "argument")}


@spec("multi/key_helper", ("{k:#x}", "It is {k:#x}.", "Answer: {k:#x}", "byte {k:#x}"),
      ("With the seed {seed}, the helper derives the byte {k:#x}, and proc XORs it into every byte of the buffer.",
       "proc calls the helper for the seed {seed} and XORs the byte it gets, {k:#x}, into each byte of the buffer.",
       "The key byte for the seed {seed} is {k:#x}, and the loop XORs it into the whole buffer."))
def _(t):
    return {"k": _val(t["accept"][0]), "seed": _qint(t, r"seed\s*=\s*(\d+)", "seed")}


@spec("multi/decode_secret", EXACT,
      ("The helper decodes a XOR-encoded copy of this string, and proc returns 1 when its input is equal to it.",
       "proc compares its input with the decoded secret using strcmp, and only this string makes it return 1.",
       "The secret never appears in the code in clear: the helper decodes it, and the comparison succeeds for this {len}-character string."))
def _(t):
    return _plain(t)


@spec("multi/chain", NUM,
      ("For the argument {n0}, proc passes the value through a helper that calls a second helper, masks the result, and returns {n}.",
       "Calling proc with {n0} goes through two helpers, where the first calls the second, and returns {n}.",
       "The chain of three functions turns the argument {n0} into {n}."))
def _(t):
    return {"n": _val(t["accept"][0]), "n0": _qint(t, r"called with\s+(\d+)", "argument")}


@spec("multi/limit_helper", ("{n}", "{n} characters", "It is {n}.", "{n} chars"),
      ("For the mode {mode}, the helper returns the limit {n}, and proc returns 1 for strings of up to {n} characters.",
       "proc accepts a string while its length is at most the limit that the helper computes from the mode {mode}, which is {n}.",
       "The longest string that passes with the mode {mode} has {n} characters."))
def _(t):
    return {"n": _val(t["accept"][0]), "mode": _qint(t, r"mode\s*=\s*(\d+)", "mode")}


# ---- legacy: the original ten templates ---------------------------------------------------------------------------------------------

@spec("xor-key", ("{s}", "The key is: {s}", "Key: {s}", "It writes: {s}"),
      ("The function XORs every byte of the buffer with this key, and the key repeats every {len} bytes.",
       "Each byte of the buffer is XORed with a character of the key string, taken in turn.",
       "The key is a string of {len} characters that is applied to the buffer again and again."))
def _(t):
    return _plain(t)


@spec("password", ("{s}", "The string is: {s}", "Input: {s}", "The input is: {s}"),
      ("The function compares its argument with this string and returns 1 only when they are equal.",
       "Only an input of exactly {len} characters that equals this string makes the function return 1.",
       "The password check succeeds for this string and for no other input."))
def _(t):
    return _plain(t)


@spec("keycount", ("{s}", "The key is: {s}", "Key: {s}", "It counts: {s}"),
      ("The function counts the lines that start with the key followed by an equals sign, and it skips comment lines.",
       "A line is counted when it begins with this key and an equals sign and has something after it.",
       "The key that is compared at the start of each line is this {len}-character string."))
def _(t):
    return _plain(t)


@spec("alphabet", ("{s}", "The alphabet is: {s}", "Alphabet: {s}", "\"{s}\""),
      ("The decoder returns the position of its argument in this string of {len} characters, or {miss} when it is not there.",
       "The index that the function returns is the position in this alphabet of {len} characters.",
       "These {len} characters are searched with strchr, and the position of a match is the value."),
      consts=("miss",))
def _(t):
    return {**_plain(t), "miss": -1}


@spec("magic", ("0x{m:08x}, {v}", "0x{m:08x} and {v}", "0x{m:08x}; {v}", "0x{m:08x} / {v}"),
      ("The header must start with the magic value 0x{m:08x}, and a version above {v} is rejected.",
       "proc compares the magic field at the start of the header with 0x{m:08x}, and the highest version it accepts is {v}.",
       "A header is valid when it starts with 0x{m:08x} and its version is at most {v}."))
def _(t):
    return {"m": int(t["accept"][0], 16), "v": int(t["accept"][1])}


@spec("lcg", ("{a}, {r}", "{a} and {r} rounds", "{a}, {r} rounds", "{a}; {r} rounds"),
      ("The loop multiplies the seed by {a} and runs {r} rounds, and the result is masked with 0x{mask:08x} after each round.",
       "Each of the {r} rounds computes seed * {a} plus a constant and keeps the low bits with the mask 0x{mask:08x}.",
       "This is a linear congruential generator with the multiplier {a}, applied {r} times."),
      consts=("mask",))
def _(t):
    return {"a": int(t["accept"][0]), "r": int(t["accept"][1]), "mask": 0x7fffffff}


_HASH_NAME = {"fnv": "FNV-1a", "djb2": "djb2", "sdbm": "sdbm", "murmur": "MurmurHash2"}
_HASH_INIT = {"fnv": 2166136261, "djb2": 5381, "sdbm": 0, "murmur": 0x9747b28c}      # what `h` starts from in legacy.py's template of each hash


@spec("hash", ("{name}, 0x{c:08x}", "{name} and 0x{c:08x}", "It is {name}, 0x{c:08x}.", "{name}: 0x{c:08x}"),
      ("The hash starts from the value {init:#x}, and the result is XORed with 0x{c:08x} before it is returned.",
       "The hash is {name}, and the final step XORs the result with the constant 0x{c:08x}.",
       "After the loop over the characters, the function returns the hash XORed with 0x{c:08x}."),
      consts=("init",))
def _(t):
    stem = t["accept"][0][0]
    return {"name": _HASH_NAME[stem], "c": int(t["accept"][1][0], 16), "init": _HASH_INIT[stem]}


@spec("skip", ("0x{b:02x}", "It is 0x{b:02x}.", "The value is 0x{b:02x}.", "byte 0x{b:02x}"),
      ("The checksum adds up every byte of the input except those equal to 0x{b:02x}, and then adds the length.",
       "Bytes with the value 0x{b:02x} are left out of the sum, and the length of the input is added at the end.",
       "The comparison in the loop skips a single byte value, which is 0x{b:02x}."))
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


def render(t: dict) -> dict:
    """{"text", "headline", "sentence", "facts", "key", "template", "templates"}: the canonical headline and one factual sentence for task `t`,
    from the truth alone. The teacher's reply plays no part here."""
    key = key_of(t)
    if key not in SPECS:
        raise ValueError(f"no truth template for task {t.get('id', '?')} ({key}); add one to rea_bench/targets.py")
    f = SPECS[key].facts(t)
    heads, sentences = _template_set(key, f)
    i = random.Random(f"{TARGET_SOURCE}/{t.get('id', '')}/{key}").randrange(len(sentences))
    head = heads[i % len(heads)].format(**f)
    sentence = sentences[i].format(**f)
    return {"text": f"{head}\n{sentence}", "headline": head, "sentence": sentence, "facts": f, "key": key, "template": i, "templates": len(sentences)}

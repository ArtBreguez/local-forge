"""Round 6, M1: the sentence table. A claim checks what a SENTENCE says, not what the fact that rendered it says.

Every sentence of every truth template has a PATTERN here, written by hand and independent of tools/rea-bench/rea_bench/targets.py: its literal text, and a
named group wherever the sentence says something that can be wrong. The checker parses the RENDERED sentence with its pattern, takes the numbers, names and
direction words FROM THE SENTENCE, and verifies each against the task's compiled behaviour or C source (tests/rea_claims.py). So:

  * a sentence whose wording the table does not know (a word edited, added or dropped) matches no pattern and fails;
  * a number in the wrong role, a flipped direction word, a wrong name, a wrong unit is a captured group with a wrong value, and the claim that reads it fails;
  * a captured group that no claim reads fails too (`unread`): a sentence cannot say something that nothing checks.
  The whitelist is the pattern: a sentence contains only its literal text and its captured groups; nothing else gets through. The literal text is reviewed
  with the table (and shows in tests/golden/rea_sentences.txt); the words that carry meaning (names, units, byte order, directions, the numbers) are groups.

Pattern syntax: literal text, and `{name:kind}` for a group (a later `{name}` must repeat the same value). kinds: `i` an integer, `h` a hex number `0x..`,
`p` a path, `q` a PTRACE request, `e` an expression over a and b, `w` an identifier, `l` letters, `t` a string that a claim compares with the compiled behaviour
(the string of a headline: it is checked byte for byte, so the group may be as loose as the string is), or `a|b|c` one of these words.

HEADLINES (round 7, M1). The first line of the stored message gets the same treatment as the sentence: a pattern per headline phrasing (`HEADS`, `H(pattern,
claims..., implied...)`), parsed from the RENDERED headline, and the claims of its row read the groups and verify them against the compiled task (the claims
of the sentences, a few of them extended to read a string or a label). What a headline does not state but a claim needs (the width of the CRC, the call
argument, the direction of a rotation) is IMPLIED: a constant of the question's meaning, a value the QUESTION states (`Q`), or a function of the groups that
were read; an implied group is never "unread" and never comes from the parameter that rendered the headline.
"""
from __future__ import annotations

import re

KINDS = {"i": r"-?\d+", "h": r"0x[0-9a-fA-F]+", "p": r"/[A-Za-z0-9_/.]+", "q": r"PTRACE_[A-Z_]+", "e": r"[ab0-9 ^&|+*()<>-]+?", "w": r"[A-Za-z_][A-Za-z0-9_]*",
         "l": r"[A-Za-z]+", "t": r".+"}
FIELD = re.compile(r"\{(\w+)(?::([^{}]*))?\}")


class Caps(dict):
    """The groups of a parsed sentence. Reading one (`[]`, `get`) is recorded: a group that no claim read is a failure of the table."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read: set = set()
        self.implied: set = set()            # groups that the text does not state (a headline): never "unread"

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)

    def unread(self) -> set:
        return set(self) - self.read - self.implied


def compile_pattern(text: str):
    """(compiled regex, {group: kind}) of a pattern."""
    out, pos, seen = [], 0, {}
    for m in FIELD.finditer(text):
        out.append(re.escape(text[pos:m.start()]))
        name, kind = m.group(1), m.group(2)
        if name in seen:
            assert kind is None, f"{{{name}}} is defined twice in {text!r}"
            out.append(f"(?P={name})")
        else:
            assert kind, f"{{{name}}} has no kind in {text!r}"
            seen[name] = kind
            rx = KINDS.get(kind) or "|".join(re.escape(a) for a in kind.split("|"))
            out.append(f"(?P<{name}>{rx})")
        pos = m.end()
    out.append(re.escape(text[pos:]))
    return re.compile("".join(out)), seen


def parse(text: str, sentence: str):
    """Caps of the sentence under the pattern, or None when the pattern does not match the whole sentence."""
    rx, kinds = compile_pattern(text)
    m = rx.fullmatch(sentence)
    if m is None:
        return None
    out = Caps()
    for name, kind in kinds.items():
        v = m.group(name)
        out[name] = int(v) if kind == "i" else (int(v, 16) if kind == "h" else v)
    return out


def groups(text: str) -> set:
    return set(compile_pattern(text)[1])


def P(pattern: str, *claims: str) -> tuple:
    return (pattern, tuple(claims))


# ---- structs ---------------------------------------------------------------------------------------------------------------------

_FLAG = "non-zero|zero"
TABLE: dict = {
    "structs/array_stride": [
        P("A record that proc reads is {n:i} bytes long, so consecutive records start {n} bytes apart.", "stride_records"),
        P("proc reads its input as a sequence of records of {n:i} bytes.", "stride_records"),
        P("The distance between a field of one record and the same field of the next record is {n:i} bytes.", "stride_records")],
    "structs/field_read": [
        P("Once its flag field is {flag:%s}, proc returns the field at byte offset {off:h} of the structure, plus a constant." % _FLAG, "field_read_offset"),
        P("The value that proc returns is the field at byte offset {off:h}, plus a constant, when the flag field is {flag:%s}." % _FLAG, "field_read_offset"),
        P("The field at byte offset {off:h} of the structure is read by proc and, with a constant added, returned for a {flag:%s} flag field." % _FLAG,
          "field_read_offset")],
    "structs/init_magic": [
        P("After proc has run, the constant {magic:h} is stored at byte offset {off:h} of the structure.", "init_magic_offset"),
        P("proc writes {magic:h} into the field at byte offset {off:h}, together with the other fields that it initialises.", "init_magic_offset", "init_other_fields"),
        P("Among the fields that proc initialises, the field that receives {magic:h} is at byte offset {off:h}.", "init_magic_offset", "init_other_fields")],
    "structs/alloc_size": [
        P("proc allocates memory for a structure of {n:i} bytes.", "alloc_structure_size"),
        P("A structure that proc allocates is {n:i} bytes long, whether it allocates a single structure or several.", "alloc_structure_size"),
        P("The memory that proc allocates is made of structures of {n:i} bytes.", "alloc_structure_size")],
    "structs/nested_offset": [
        P("The value that proc returns is a field of a nested structure, and counted from the start of the outer structure it is at byte offset {off:h}.",
          "nested_field_offset"),
        P("proc returns a field of a nested structure, which is at byte offset {off:h} of the outer structure.", "nested_field_offset"),
        P("Adding the offset of the nested structure to the offset of the field inside it gives byte offset {off:h} of the outer structure.",
          "nested_field_offset")],
    "structs/linked_next": [
        P("proc follows the pointer at byte offset {off:h} of a node to reach the next node of the list, and stops at a node whose pointer is NULL.",
          "linked_pointer_offset"),
        P("The pointer to the next node is at byte offset {off:h} of a node, and the list ends where that pointer is NULL.", "linked_pointer_offset"),
        P("A node leads to its successor through the pointer stored at byte offset {off:h}.", "linked_pointer_offset")],
}

# ---- fsm -------------------------------------------------------------------------------------------------------------------------

_FSM_IN = [
    P("Starting in state {z:i}, the {d:i} letters of this input take the machine to state {final:i}, which is what proc returns.", "fsm_input_final"),
    P("The machine reads the {d:i} letters of this string in turn and ends in state {final:i}.", "fsm_input_final"),
    P("This input of {d:i} letters moves the machine from state {z:i} to state {final:i}, and no shorter input does.", "fsm_input_final", "fsm_input_shortest")]
_FSM_ST = [
    P("The machine has {n:i} states, numbered from {z:i}, and proc returns the state it ends in.", "fsm_states_count"),
    P("There are {n:i} distinct states, numbered {z:i} to {last:i}, and the machine starts in state {z}.", "fsm_states_count"),
    P("Starting from state {z:i}, the machine can reach all of its {n:i} states.", "fsm_states_reachable", "fsm_states_count")]
TABLE.update({
    "fsm/table_input": _FSM_IN, "fsm/switch_input": _FSM_IN, "fsm/table_states": _FSM_ST, "fsm/switch_states": _FSM_ST,
    "fsm/opcode_value": [
        P("proc returns {expr:e} when its first argument is {n:i}.", "opcode_returns_expr"),
        P("When the first argument is {n:i}, the function returns {expr:e}, and no other opcode does.", "opcode_returns_expr", "opcode_unique"),
        P("Opcode {n:i} is the opcode for which proc returns {expr:e}.", "opcode_returns_expr", "opcode_unique")],
    "fsm/opcode_count": [
        P("proc handles {n:i} different opcodes, each with a result of its own, and returns {dflt:i} for the other values of the first argument.",
          "opcode_count_handled", "opcode_default"),
        P("Exactly {n:i} values of the first argument are handled by proc; the other values return {dflt:i}.", "opcode_count_handled", "opcode_default"),
        P("proc tells {n:i} different opcodes apart and gives {dflt:i} for the rest.", "opcode_count_handled", "opcode_default")],
})

# ---- crc -------------------------------------------------------------------------------------------------------------------------

_DIR = "normal|reflected"
_CRC_POLY = [
    P("proc computes a {w:i}-bit CRC with the polynomial constant {lit:h}, in {dir:%s} form." % _DIR, "crc_poly_form"),
    P("The {w:i}-bit CRC that proc returns is {dir:%s}, and its polynomial constant is {lit:h}." % _DIR, "crc_poly_form"),
    P("The polynomial of this {w:i}-bit CRC is given by the constant {lit:h}, and the CRC is {dir:%s}." % _DIR, "crc_poly_form")]
_CRC_INIT = [
    P("The initial value of this {w:i}-bit CRC is {x:h}.", "crc_init_value"),
    P("proc starts the {w:i}-bit CRC computation from the value {x:h}.", "crc_init_value"),
    P("Before the first byte is processed, the {w:i}-bit CRC has the value {x:h}.", "crc_init_value")]
_CRC_XOR = [
    P("After the last byte, the {w:i}-bit CRC is XORed with {x:h} before it is returned.", "crc_xorout_value"),
    P("The final XOR value of this {w:i}-bit CRC is {x:h}.", "crc_xorout_value"),
    P("The result of the {w:i}-bit CRC computation is XORed with {x:h} as its last step.", "crc_xorout_value")]
TABLE.update({"crc/bitwise_poly": _CRC_POLY, "crc/table_poly": _CRC_POLY, "crc/bitwise_init": _CRC_INIT, "crc/table_init": _CRC_INIT,
              "crc/bitwise_xorout": _CRC_XOR, "crc/table_xorout": _CRC_XOR})

# ---- cipher ----------------------------------------------------------------------------------------------------------------------

_TEA = [
    P("proc {verb:encrypts|decrypts} a block of {block:i} bytes with a {key_bits:i}-bit key, in {rounds:i} iterations.", "tea_reference"),
    P("This {alg:TEA|XTEA|RC4|ChaCha|Salsa} implementation runs {rounds:i} iterations and uses the delta constant {delta:h}.", "tea_reference"),
    P("The block of {block:i} bytes is {done:encrypted|decrypted} with the {key_bits:i}-bit key over {rounds:i} iterations, and the delta constant is {delta:h}.",
      "tea_reference")]
for _k in ("tea_enc", "tea_dec", "xtea_enc", "xtea_dec"):
    TABLE[f"cipher/{_k}"] = _TEA
_ALGS = "TEA|XTEA|RC4|ChaCha|Salsa"
TABLE["cipher/rc4"] = [
    P("proc applies {alg:%s} with a key of {n:i} bytes to its buffer." % _ALGS, "rc4_reference"),
    P("The key of this {alg:%s} implementation is {n:i} bytes long." % _ALGS, "rc4_reference"),
    P("The keystream comes from a key of {n:i} bytes and a state of {state:i} byte values.", "rc4_reference")]
_STREAM = [
    P("proc computes the {alg:%s} block function with {rounds:i} rounds, which is {half:i} double rounds." % _ALGS, "stream_reference"),
    P("The function expands a {key_bits:i}-bit key into a block of {block:i} bytes with {rounds:i} rounds of {alg:%s}." % _ALGS, "stream_reference"),
    P("The rotation amounts are {rot0:i}, {rot1:i}, {rot2:i} and {rot3:i} bits, as in {alg:%s}, and the function runs {rounds:i} rounds." % _ALGS,
      "stream_reference")]
TABLE["cipher/chacha"] = _STREAM
TABLE["cipher/salsa"] = _STREAM

# ---- encoding --------------------------------------------------------------------------------------------------------------------

_ENC_E = [
    P("{gin:i} input bytes become {gout:i} output characters, chosen from an alphabet of {radix:i} characters.", "encoding_reference"),
    P("proc turns groups of {gin:i} bytes into {gout:i} characters of a {radix:i}-character alphabet.", "encoding_reference"),
    P("An output character stands for {bits:i} bits of the input and comes from an alphabet of {radix:i} characters.", "encoding_reference")]
_ENC_D = [
    P("proc turns an input character back into its value in a {radix:i}-character alphabet and packs the values into bytes.", "encoding_reference"),
    P("An input character stands for {bits:i} bits of the decoded data, and the characters come from an alphabet of {radix:i} characters.", "encoding_reference"),
    P("The decoded bytes are packed from the values of characters of a {radix:i}-character alphabet, {bits:i} bits per character.", "encoding_reference")]
_ENC_58E = [
    P("proc converts its input to base {radix:i} and writes the digits as single characters of an alphabet of {radix} characters.", "encoding_reference"),
    P("The output is the input written in radix {radix:i}, and the digits are characters of an alphabet of {radix} characters.", "encoding_reference"),
    P("The result has digits of radix {radix:i}, which are characters of an alphabet of {radix} characters.", "encoding_reference")]
_ENC_58D = [
    P("proc reads its input as a number written in base {radix:i}, a single character of an alphabet of {radix} characters per digit, and writes the bytes of "
      "that number into its second argument.", "encoding_reference"),
    P("An input character is a digit of radix {radix:i}, taken from an alphabet of {radix} characters.", "encoding_reference"),
    P("The decoded number has the characters of the input as its digits in radix {radix:i}, and the characters come from an alphabet of {radix} characters.",
      "encoding_reference")]
_ALPHA = [
    P("This is the alphabet of {alen:i} characters that the {agent:encoder|decoder} uses, in index order starting at the value {z:i}.", "alphabet_semantics"),
    P("The {agent:encoder|decoder} works with this string of {alen:i} characters as its alphabet: its first character stands for the value {z:i}, and the "
      "characters after it stand for the following values in order.", "alphabet_semantics"),
    P("The {alen:i} characters are listed in index order: the first character stands for the value {z:i}.", "alphabet_semantics")]
TABLE.update({"encoding/base64_encode": _ENC_E, "encoding/base32_encode": _ENC_E, "encoding/base64_decode": _ENC_D, "encoding/base32_decode": _ENC_D,
              "encoding/base58_encode": _ENC_58E, "encoding/base58_decode": _ENC_58D})
for _b in ("base64", "base32", "base58"):
    for _d in ("encode", "decode"):
        TABLE[f"encoding/{_b}_{_d}#alphabet"] = _ALPHA

# ---- strobf ----------------------------------------------------------------------------------------------------------------------

_STROBF = [
    P("proc writes this string of {len:i} characters into its argument, followed by a terminating NUL byte.", "writes_exact_string"),
    P("The {len:i} characters of this string, followed by a NUL byte, are what proc writes into its argument.", "writes_exact_string"),
    P("After proc has run, its argument holds exactly this string of {len:i} characters and a terminating NUL byte.", "writes_exact_string")]
for _v in ("stack_string", "xor_const", "xor_rolling", "alt_xor", "sub_const", "reverse_xor", "rot_n"):
    TABLE[f"strobf/{_v}"] = _STROBF

# ---- pktsum: three sets of three (ends, starts, lens) ----------------------------------------------------------------------------

_PKT = [
    P("proc accepts the packet when its last two bytes hold the checksum as a 16-bit value stored {e:little|big}-endian, with the {first:low|high} byte first.",
      "pkt_endian_behaviour"),
    P("The stored 16-bit checksum is read {e:little|big}-endian: the {first:low|high} byte comes first.", "pkt_endian_behaviour"),
    P("The checksum that the function accepts is stored {e:little|big}-endian, with the {first:low|high} byte first.", "pkt_endian_behaviour"),
    P("The checksummed range starts at byte offset {n:i} of the packet and runs up to the stored checksum.", "pkt_range_start"),
    P("The bytes before offset {n:i} do not take part in the checksum.", "pkt_range_start"),
    P("The bytes from offset {n:i} up to the stored checksum are the bytes that the checksum is computed over.", "pkt_range_start"),
    P("The length byte is read from offset {n:i} of the packet, and the checksum follows that many payload bytes.", "pkt_length_byte"),
    P("The payload length is stored in the byte at offset {n:i}, and the checksum comes right after the payload.", "pkt_length_byte"),
    P("The byte at offset {n:i} says how many bytes the checksum covers, and the checksum is stored behind them.", "pkt_length_byte")]
for _v in ("sum8", "xor8", "sum16_le", "sum16_be", "fletcher16", "crc16", "len_field"):
    TABLE[f"pktsum/{_v}"] = _PKT

# ---- bounds ----------------------------------------------------------------------------------------------------------------------

TABLE.update({
    "bounds/range_chain": [
        P("proc returns {ret:i} for every integer from {lo:i} to {hi:i}, both included, and for no other integer.", "returns_one_exactly_in_range"),
        P("The function accepts exactly the integers between {lo:i} and {hi:i} inclusive and returns {ret:i} for them.", "returns_one_exactly_in_range"),
        P("The {lw:lowest|highest} integer for which proc returns {ret:i} is {lo:i}, and the {hw:highest|lowest} is {hi:i}.", "returns_one_exactly_in_range")],
    "bounds/window": [
        P("proc returns {ret:i} for a window of {w:i} consecutive integers: it starts at {lo:i} and ends at {hi:i}.", "returns_one_for_window"),
        P("The accepted values begin at {lo:i}, end at {hi:i} and make up {w:i} integers.", "returns_one_for_window"),
        P("proc returns {ret:i} for the {w:i} integers from {lo:i} to {hi:i}, and for no other integer.", "returns_one_for_window")],
    "bounds/index_limit": [
        P("proc returns {neg:i} for every index {o1:outside|inside} the range from {lo:i} to {hi:i}, and a non-negative value for every index {o2:inside|outside} it.",
          "index_limit_range"),
        P("The indexes from {lo:i} to {hi:i} are accepted: for them proc returns a byte value, and the other indexes give {neg:i}.", "index_limit_range"),
        P("The {lw:smallest|largest} index for which proc returns something other than {neg:i} is {lo:i}, and the {hw:largest|smallest} is {hi:i}.",
          "index_limit_range")],
    "bounds/clamp": [
        P("proc limits its argument to the interval from {lo:i} to {hi:i}, both included.", "clamp_interval"),
        P("A value {d1:below|above} {lo:i} becomes {lo} and a value {d2:above|below} {hi:i} becomes {hi}.", "clamp_interval"),
        P("The result lies between the {lb:lower|upper} bound {lo:i} and the {ub:upper|lower} bound {hi:i}.", "clamp_interval")],
    "bounds/loop_iterations": [
        P("For the argument {n0:i}, proc returns {n:i}.", "returns_loop_value"),
        P("Calling proc with {n0:i} gives the return value {n:i}.", "returns_loop_value"),
        P("The value that proc returns when it is called with {n0:i} is {n:i}.", "returns_loop_value")],
    "bounds/char_class": [
        P("proc returns {ret:i} for {n:i} of the {total:i} possible byte values.", "char_class_count"),
        P("Counting the byte values from {z:i} to {top:i} for which proc returns {ret:i} gives {n:i}.", "char_class_count"),
        P("Exactly {n:i} byte values make proc return {ret:i}; for the other byte values it does not.", "char_class_count")],
})

# ---- antidebug -------------------------------------------------------------------------------------------------------------------

_PTRACE = [
    P("proc asks to be traced by its parent with {call:ptrace|syscall|prctl}({req:q}) and returns {rv:i} when that request {outcome:fails|succeeds}, which shows "
      "that a debugger is attached.", "ptrace_request"),
    P("The check is a {call:ptrace|syscall|prctl} request to be traced by the parent, and the function returns {rv:i} if the request is {outcome:refused|granted}.",
      "ptrace_request"),
    P("When {call:ptrace|syscall|prctl}({req:q}) {outcome:fails|succeeds} because a debugger is already attached, proc returns {rv:i}.", "ptrace_request")]
TABLE.update({
    "antidebug/ptrace": _PTRACE, "antidebug/ptrace_raw": _PTRACE,
    "antidebug/tracerpid": [
        P("proc reads the {field:w} line of {file:p} and returns {rv:i} when it names a tracer.", "tracerpid_status"),
        P("The function reads the status file of its own process for {field:w}, and a tracer makes it return {rv:i}.", "tracerpid_status"),
        P("A {field:w} that shows a tracer makes proc return {rv:i}.", "tracerpid_status")],
    "antidebug/env": [
        P("proc looks at the environment variable {NAME:w}.", "env_variable_read"),
        P("The function tests the environment variable {NAME:w} as a debugging variable.", "env_variable_read"),
        P("The check reads the environment variable {NAME:w}, and that is what makes it an environment-based test.", "env_variable_read")],
})
_UNIT = "nanoseconds|microseconds|milliseconds|seconds"
_TIMING = [
    P("proc measures the time between two clock readings and reports a debugger when it is {cmp:above|below} {thr:i} {unit:%s}." % _UNIT, "timing_threshold"),
    P("A debugger is reported when more than {thr:i} {unit:%s} pass between the two clock readings." % _UNIT, "timing_threshold"),
    P("The largest time between the two clock readings that proc still lets pass is {thr:i} {unit:%s}." % _UNIT, "timing_threshold")]
TABLE["antidebug/timing_ns"] = _TIMING
TABLE["antidebug/timing_us"] = _TIMING

# ---- multi -----------------------------------------------------------------------------------------------------------------------

_ARG = [P("For the argument {n0:i}, proc returns {n:i}.", "returns_for_argument"),
        P("Calling proc with {n0:i} gives {n:i}.", "returns_for_argument"),
        P("The value that proc returns when it is called with {n0:i} is {n:i}.", "returns_for_argument")]
TABLE.update({
    "multi/eval_arith": _ARG, "multi/chain": _ARG,
    "multi/key_helper": [
        P("When it is called with the seed {seed:i}, proc XORs the byte {k:h} into every byte of the buffer.", "xors_key_byte"),
        P("The byte that proc XORs into each byte of the buffer for the seed {seed:i} is {k:h}.", "xors_key_byte"),
        P("For the seed {seed:i}, every byte of the buffer is XORed with {k:h}.", "xors_key_byte")],
    "multi/decode_secret": [
        P("proc returns {ret:i} when its input is exactly this string of {len:i} characters, and for other inputs it does not.", "accepts_only_the_secret"),
        P("Only this string of {len:i} characters makes proc return {ret:i}.", "accepts_only_the_secret"),
        P("The secret that proc accepts has {len:i} characters, and it is this string.", "accepts_only_the_secret")],
    "multi/limit_helper": [
        P("With the mode {mode:i}, proc returns {ret:i} for strings of up to {n:i} characters and does not for longer ones.", "limit_for_mode"),
        P("proc accepts a string with the mode {mode:i} only while its length is at most {n:i} characters.", "limit_for_mode"),
        P("The longest string that passes with the mode {mode:i} has {n:i} characters.", "limit_for_mode")],
})

# ---- legacy ----------------------------------------------------------------------------------------------------------------------

_LR = "left|right"
TABLE.update({
    "xor-key": [
        P("The function XORs every byte of the buffer with this key, and the key repeats every {len:i} bytes.", "xor_key_repeats"),
        P("Each byte of the buffer is XORed with a character of the key string, taken in turn.", "xor_key_repeats"),
        P("The key is a string of {len:i} characters that is repeated along the whole buffer.", "xor_key_repeats")],
    "password": [
        P("The function returns {ret:i} only when its argument is equal to this string.", "password_equal_only"),
        P("Only an input of exactly {len:i} characters that equals this string makes the function return {ret:i}.", "password_equal_only"),
        P("The password check succeeds for this string and for no other input.", "password_equal_only")],
    "keycount": [
        P("The function counts the lines that start with this key followed by an equals sign and a value that is not empty, and it skips comment lines.",
          "keycount_lines"),
        P("A line is counted when it begins with this key and an equals sign and has something after it.", "keycount_lines"),
        P("The lines that are counted begin with this {len:i}-character key and an equals sign, and have something after it.", "keycount_lines")],
    "alphabet": [
        P("The decoder returns the position of its argument in this string of {len:i} characters, or {miss:i} for any other printable character.",
          "alphabet_position"),
        P("The index that the function returns is the position in this alphabet of {len:i} characters.", "alphabet_position"),
        P("The position of a character in this string of {len:i} characters is the value that proc returns for it.", "alphabet_position")],
    "magic": [
        P("A header is valid only if its first four bytes, read as a {e:little|big}-endian 32-bit word, are {m:h}, and a version above {v:i} is rejected.",
          "magic_header"),
        P("The 32-bit magic word at the start of the header, read {e:little|big}-endian, must be {m:h}, and the highest version that proc accepts is {v:i}.",
          "magic_header"),
        P("A header is valid when its first four bytes, read as a {e:little|big}-endian 32-bit value, are {m:h} and its version is at most {v:i}.",
          "magic_header")],
    "lcg": [
        P("The seed goes through {r:i} rounds of a linear congruential step with the multiplier {a:i}, masked with {mask:h}.", "lcg_reference"),
        P("A new seed is computed from the previous seed in {r:i} rounds, with the multiplier {a:i} and the mask {mask:h} that keeps the low bits.", "lcg_reference"),
        P("This is a linear congruential generator with the multiplier {a:i}, applied {r:i} times.", "lcg_reference")],
    "hash": [
        P("The hash starts from the value {init:h}, and the result is XORed with {c:h} before it is returned.", "hash_start_value", "hash_final_xor"),
        P("The hash is {name:FNV-1a|djb2|sdbm|Murmur-style|MurmurHash2|Murmur}, and the final step XORs the result with the constant {c:h}.",
          "hash_name_is_the_algorithm", "hash_final_xor"),
        P("The function returns the hash of its input string XORed with {c:h}.", "hash_final_xor")],
    "skip": [
        P("The checksum adds up every byte of the input except those equal to {b:h}, and then adds the length.", "skip_one_value"),
        P("Bytes with the value {b:h} are left out of the sum, and the length of the input is added at the end.", "skip_one_value"),
        P("Exactly one byte value does not count in the sum, and it is {b:h}.", "skip_one_value")],
    "rotate": [
        P("The function XORs its argument with {k:h} and then rotates the result {dir:%s} by {r:i} bits." % _LR, "rotate_reference"),
        P("The constant {k:h} is XORed in first, and the {dir:%s} rotation that follows is by {r:i} bits." % _LR, "rotate_reference"),
        P("A rotation to the {dir:%s} by {r:i} bits comes after the XOR with {k:h}." % _LR, "rotate_reference")],
    "threshold": [
        P("The function returns {inside:i} for every v from {lo:i} to {hi:i}, {below:i} for a {c1:smaller|larger} v and {above:i} for a {c2:larger|smaller} v.",
          "threshold_values"),
        P("{d1:Below|Above} {lo:i} it returns {below:i}, {d2:above|below} {hi:i} it returns {above:i}, and in between, both bounds included, it returns {inside:i}.",
          "threshold_values"),
        P("The {lb:lower|upper} bound is {lo:i} and the {ub:upper|lower} bound is {hi:i}, both inclusive.", "threshold_values")],
})


# ---- headlines: the words a headline may use, and the keys each is TRUE for (round 7, B1) -------------------------------------------------------
#
# A headline is the first line of the stored message. `multi/decode_secret` shared the strobf tuple and said "It writes: <secret>" of a function that only
# compares its input: a verb or a label that is true for one key was reused for another. Every literal word of every headline template is registered
# here with the keys it is true for (prefixes of `kind/variant`, or the legacy kind); a word used by a key outside its prefixes fails
# `HeadWording` in tests/test_rea_claims.py. A neutral word (a copula, a connective) is registered for every key.

_ANY = ("",)
HEAD_WORDS = {
    # neutral: they say nothing about proc
    "a": _ANY, "and": _ANY, "answer": _ANY, "are": _ANY, "at": _ANY, "from": _ANY, "is": _ANY, "it": _ANY, "the": _ANY, "there": _ANY, "to": _ANY,
    "s": _ANY, "x": _ANY,                   # the plural suffix after a placeholder (`{Dir}s`), the hex prefix of `0x{m:08x}`
    # labels and units, each true for the keys whose question and proc they describe
    "alphabet": ("encoding/", "alphabet"),
    "bits": ("rotate",),
    "byte": ("structs/", "pktsum/", "bounds/char_class"),
    "bytes": ("structs/", "cipher/rc4"),
    "cases": ("fsm/opcode_count",),
    "characters": ("multi/limit_helper",),
    "chars": ("multi/limit_helper",),
    "check": ("antidebug/",),
    "counted": ("keycount",),
    "distinct": ("fsm/table_states", "fsm/switch_states"),
    "endian": ("pktsum/",),
    "getenv": ("antidebug/env",),
    "highest": ("bounds/range_chain", "bounds/window", "bounds/index_limit", "bounds/clamp", "threshold"),
    "input": ("fsm/table_input", "fsm/switch_input", "password", "multi/decode_secret"),
    "iterations": ("cipher/tea_", "cipher/xtea_"),
    "key": ("xor-key", "keycount"),
    "lowest": ("bounds/range_chain", "bounds/window", "bounds/index_limit", "bounds/clamp", "threshold"),
    "ns": ("antidebug/timing_ns",),
    "offset": ("structs/", "pktsum/"),
    "opcode": ("fsm/opcode_",),
    "opcodes": ("fsm/opcode_",),
    "plaintext": ("strobf/",),
    "polynomial": ("crc/",),
    "record": ("structs/array_stride",),
    "returns": ("antidebug/ptrace", "antidebug/tracerpid", "bounds/loop_iterations"),
    "rounds": ("cipher/chacha", "cipher/salsa", "lcg"),
    "size": ("structs/array_stride", "structs/alloc_size"),
    "states": ("fsm/table_states", "fsm/switch_states"),
    "string": ("strobf/", "multi/decode_secret", "password", "xor-key"),
    "structure": ("structs/alloc_size",),
    "us": ("antidebug/timing_us",),
    "value": ("crc/", "skip", "multi/key_helper"),
    "values": ("bounds/char_class",),
    "writes": ("strobf/",),                  # proc WRITES the string into its argument; decode_secret and password compare, xor-key writes the buffer
}


def head_words(template: str) -> list:
    """The literal words of a headline template (every placeholder removed), lower-cased."""
    lit = FIELD.sub(" ", template.replace("{{", " ").replace("}}", " "))
    lit = re.sub(r"\{[^{}]*\}", " ", lit)
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", lit)]


def head_word_problems(heads: dict) -> list:
    """[problem] for {key: tuple of headline templates}: a word that is not registered, or registered for other keys than `key`."""
    out = []
    for key, templates in heads.items():
        for i, t in enumerate(templates):
            for w in head_words(t):
                prefixes = HEAD_WORDS.get(w)
                if prefixes is None:
                    out.append(f"{key} head[{i}] {t!r}: the word {w!r} is not registered in HEAD_WORDS (tests/rea_sentences.py)")
                elif not any(key.startswith(pre) for pre in prefixes):
                    out.append(f"{key} head[{i}] {t!r}: the word {w!r} is registered for {sorted(prefixes)}, not for this key")
    return out


# ---- headlines: one pattern per phrasing (round 7, M1) ------------------------------------------------------------------------------------------

class Q:
    """A value that the QUESTION of the task states (`rec["question"]`), used where a headline does not state it but a claim needs it: the CRC width, the
    argument the question calls proc with, the direction of a rotation. `kind`: `i` an integer (decimal or 0x), `w` a word."""

    def __init__(self, pattern: str, kind: str = "i"):
        self.rx, self.kind = re.compile(pattern, re.I), kind

    def __call__(self, question: str):
        m = self.rx.search(question)
        if m is None:
            raise ValueError(f"the question has no {self.rx.pattern!r}: {question!r}")
        v = m.group(1)
        return (int(v, 16) if v.lower().startswith(("0x", "-0x")) else int(v)) if self.kind == "i" else v


def H(pattern: str, *claims: str, **implied) -> tuple:
    """A headline phrasing: its pattern, the claims that check what it states, and the groups it implies (a constant, a `Q`, or a function of the Caps)."""
    return (pattern, tuple(claims), implied)


def parse_head(row: tuple, headline: str, question: str):
    """Caps of a rendered headline under the pattern of its row (the implied groups added), or None when the pattern does not match the whole headline."""
    pattern, _claims, implied = row
    caps = parse(pattern, headline)
    if caps is None:
        return None
    for name, how in implied.items():
        caps[name] = how(question) if isinstance(how, Q) else (how(caps) if callable(how) else how)
        caps.implied.add(name)
    return caps


def _hr(patterns, *claims, **implied) -> list:
    return [H(p, *claims, **implied) for p in patterns]


_ALGS_ = "TEA|XTEA|RC4|ChaCha|Salsa"
_Q_WIDTH = Q(r"(\d+)-bit CRC")
_Q_ARG = Q(r"called with\s+(\d+)")
_OFFSET_H = ("{off:h}", "Offset {off:h}", "The offset is {off:h}.", "byte offset {off:h}")
_NUM_H = ("{n:i}", "It is {n:i}.", "Answer: {n:i}", "{n:i}.")
_BOUND_H = ("{lo:i} to {hi:i}", "lowest {lo:i}, highest {hi:i}", "[{lo:i}, {hi:i}]", "from {lo:i} to {hi:i}")
_ACCEPTED_H = ("{s:t}", "The string is: {s:t}", "Input: {s:t}", "The input is: {s:t}")
_FSM_IN_H = ("{s:l}", "Input: {s:l}", "The input is {s:l}.", '"{s:l}"')
_FSM_ST_H = ("{n:i} states", "{n:i}", "{n:i} distinct states", "There are {n:i} states.")
_Q_FINAL = Q(r"returns\s+(\d+)\s*\.")
_CRC_POLY_H = ("{lit:h}, {dir:normal|reflected}", "{lit:h} {dir:normal|reflected}", "The polynomial is {lit:h}, {dir:normal|reflected}.",
               "polynomial {lit:h}, {dir:normal|reflected}")
_CRC_VAL_H = ("{x:h}", "It is {x:h}.", "Answer: {x:h}", "value {x:h}")
_TEA_H = ["{alg:%s}, {dir:encrypt|decrypt}, {rounds:i}" % _ALGS_, "{alg:%s}, {dir:encrypt|decrypt}, {rounds:i} iterations" % _ALGS_,
          "{dir:encrypt|decrypt}, {alg:%s}, {rounds:i} iterations" % _ALGS_, "{alg:%s} {dir:encrypt|decrypt}, {rounds:i} iterations" % _ALGS_]
_STREAM_H = ["{alg:%s}, {rounds:i}" % _ALGS_, "{alg:%s}, {rounds:i} rounds" % _ALGS_, "{alg:%s}: {rounds:i} rounds" % _ALGS_, "It is {alg:%s}, {rounds:i} rounds." % _ALGS_]
_ENC_H = ["{dir:encode|decode}, {base:base64|base32|base58}", "{base:base64|base32|base58}, {dir:encode|decode}", "{base:base64|base32|base58} {dir:encode|decode}",
          "{Dir:Encode|Decode}s {base:base64|base32|base58}"]
_ALPHA_H = ("{s:t}", "The alphabet is: {s:t}", "Alphabet: {s:t}", '"{s:t}"')
_TECH_UNIT = "ns|us"
_Q_UNIT = Q(r"above how many (nanoseconds|microseconds|milliseconds|seconds)", "w")


def _ad(tech: str, claim: str, **implied) -> list:
    return _hr(("{tech:%s}, {rv:i}" % tech, "{tech:%s}, returns {rv:i}" % tech, "It is a {tech:%s} check; returns {rv:i}." % tech, "{tech:%s} check, {rv:i}" % tech),
               claim, **implied)


def _timing_h(tech: str = "timing") -> list:
    return ([H("{tech:%s}, {thr:i}" % tech, "timing_threshold", unit=_Q_UNIT)]
            + _hr(("{tech:%s}, {thr:i} {unit:%s}" % (tech, _TECH_UNIT), "{tech:%s}: {thr:i} {unit:%s}" % (tech, _TECH_UNIT), "{tech:%s} check, {thr:i} {unit:%s}" % (tech, _TECH_UNIT)),
                  "timing_threshold"))


HEADS: dict = {
    # structs
    "structs/array_stride": _hr(("{n:i} bytes", "{n:i}", "The record size is {n:i} bytes.", "Size: {n:i} bytes"), "stride_records"),
    "structs/field_read": _hr(_OFFSET_H, "field_read_offset", flag="non-zero"),
    "structs/init_magic": _hr(_OFFSET_H, "init_magic_offset", magic=Q(r"constant\s+(0x[0-9a-f]+|\d+)")),
    "structs/alloc_size": _hr(("{n:i} bytes", "{n:i}", "The structure is {n:i} bytes.", "Size: {n:i} bytes"), "alloc_structure_size"),
    "structs/nested_offset": _hr(_OFFSET_H, "nested_field_offset"),
    "structs/linked_next": _hr(_OFFSET_H, "linked_pointer_offset"),
    # fsm
    "fsm/table_input": _hr(_FSM_IN_H, "fsm_input_string", final=_Q_FINAL),
    "fsm/switch_input": _hr(_FSM_IN_H, "fsm_input_string", final=_Q_FINAL),
    "fsm/table_states": _hr(_FSM_ST_H, "fsm_states_count", z=0),
    "fsm/switch_states": _hr(_FSM_ST_H, "fsm_states_count", z=0),
    "fsm/opcode_value": _hr(("{n:i}", "opcode {n:i}", "It is opcode {n:i}.", "{n:i} (opcode)"), "opcode_returns_expr", expr=Q(r"makes it return (.+?)\?", "w")),
    "fsm/opcode_count": _hr(("{n:i}", "{n:i} opcodes", "It is {n:i}.", "{n:i} cases"), "opcode_count_handled"),
    # crc
    "crc/bitwise_poly": _hr(_CRC_POLY_H, "crc_poly_form", w=_Q_WIDTH),
    "crc/table_poly": _hr(_CRC_POLY_H, "crc_poly_form", w=_Q_WIDTH),
    "crc/bitwise_init": _hr(_CRC_VAL_H, "crc_init_value", w=_Q_WIDTH),
    "crc/table_init": _hr(_CRC_VAL_H, "crc_init_value", w=_Q_WIDTH),
    "crc/bitwise_xorout": _hr(_CRC_VAL_H, "crc_xorout_value", w=_Q_WIDTH),
    "crc/table_xorout": _hr(_CRC_VAL_H, "crc_xorout_value", w=_Q_WIDTH),
    # cipher
    "cipher/tea_enc": _hr(_TEA_H, "tea_reference"), "cipher/tea_dec": _hr(_TEA_H, "tea_reference"),
    "cipher/xtea_enc": _hr(_TEA_H, "tea_reference"), "cipher/xtea_dec": _hr(_TEA_H, "tea_reference"),
    "cipher/rc4": _hr(["{alg:%s}, {n:i}" % _ALGS_, "{alg:%s}, {n:i} bytes" % _ALGS_, "{alg:%s}: {n:i} bytes" % _ALGS_, "It is {alg:%s}, {n:i} bytes." % _ALGS_], "rc4_reference"),
    "cipher/chacha": _hr(_STREAM_H, "stream_reference"), "cipher/salsa": _hr(_STREAM_H, "stream_reference"),
    # strobf: proc writes the string
    **{f"strobf/{v}": _hr(("{s:t}", "The string is: {s:t}", "It writes: {s:t}", "Plaintext: {s:t}"), "writes_exact_string")
       for v in ("stack_string", "xor_const", "xor_rolling", "alt_xor", "sub_const", "reverse_xor", "rot_n")},
    # pktsum: the heads of the three questions in a row: endianness, start, length byte
    **{f"pktsum/{v}": (_hr(("{e:little|big}", "{e:little|big}-endian", "It is {e:little|big}-endian.", "{e:little|big} endian"), "pkt_endian_behaviour")
                       + _hr(("{n:i}", "Offset {n:i}", "It is {n:i}.", "byte {n:i}"), "pkt_range_start")
                       + _hr(("{n:i}", "Offset {n:i}", "It is at offset {n:i}.", "byte {n:i}"), "pkt_length_byte"))
       for v in ("sum8", "xor8", "sum16_le", "sum16_be", "fletcher16", "crc16", "len_field")},
    # bounds
    "bounds/range_chain": _hr(_BOUND_H, "returns_one_exactly_in_range", ret=Q(r"does proc return (-?\d+)")),
    "bounds/window": _hr(_BOUND_H, "returns_one_for_window", ret=Q(r"does proc return (-?\d+)"), w=lambda s: s["hi"] - s["lo"] + 1),
    "bounds/index_limit": _hr(_BOUND_H, "index_limit_range", neg=Q(r"returns (-?\d+) for an index")),
    "bounds/clamp": _hr(_BOUND_H, "clamp_interval"),
    "bounds/loop_iterations": _hr(("{n:i}", "It returns {n:i}.", "returns {n:i}", "Answer: {n:i}"), "returns_loop_value", n0=_Q_ARG),
    "bounds/char_class": _hr(("{n:i}", "{n:i} values", "It is {n:i}.", "{n:i} byte values"), "char_class_count", ret=Q(r"make proc return (-?\d+)")),
    # antidebug: the label is the technique of the question's list
    "antidebug/ptrace": _ad("ptrace", "ptrace_request", call="ptrace", outcome="fails"),
    "antidebug/ptrace_raw": _ad("ptrace", "ptrace_request", call="ptrace", outcome="fails"),
    "antidebug/tracerpid": _ad("tracerpid", "tracerpid_status", field="TracerPid"),
    "antidebug/env": _hr(("{tech:env}, {NAME:w}", "{tech:env} check, {NAME:w}", "{NAME:w}, {tech:env}", "getenv, {NAME:w}"), "env_variable_read"),
    "antidebug/timing_ns": _timing_h(), "antidebug/timing_us": _timing_h(),
    # multi
    "multi/eval_arith": _hr(_NUM_H, "returns_for_argument", n0=_Q_ARG), "multi/chain": _hr(_NUM_H, "returns_for_argument", n0=_Q_ARG),
    "multi/key_helper": _hr(("{k:h}", "It is {k:h}.", "Answer: {k:h}", "The value is {k:h}."), "xors_key_byte", seed=Q(r"seed\s*=\s*(\d+)")),
    "multi/decode_secret": _hr(_ACCEPTED_H, "accepts_only_the_secret"),
    "multi/limit_helper": _hr(("{n:i}", "{n:i} characters", "It is {n:i}.", "{n:i} chars"), "limit_for_mode", mode=Q(r"mode\s*=\s*(\d+)")),
    # legacy
    "xor-key": _hr(("{s:t}", "The key is: {s:t}", "Key: {s:t}", "The key string is: {s:t}"), "xor_key_repeats"),
    "password": _hr(_ACCEPTED_H, "password_equal_only"),
    "keycount": _hr(("{s:t}", "The key is: {s:t}", "Key: {s:t}", "The counted key is: {s:t}"), "keycount_lines"),
    "alphabet": _hr(_ALPHA_H, "alphabet_position"),
    "magic": _hr(("{m:h}, {v:i}", "{m:h} and {v:i}", "{m:h}; {v:i}", "{m:h} / {v:i}"), "magic_header"),
    "lcg": _hr(("{a:i}, {r:i}", "{a:i} and {r:i} rounds", "{a:i}, {r:i} rounds", "{a:i}; {r:i} rounds"), "lcg_reference"),
    "hash": _hr(["{name:FNV-1a|djb2|sdbm|Murmur-style|MurmurHash2|Murmur}, {c:h}", "{name:FNV-1a|djb2|sdbm|Murmur-style|MurmurHash2|Murmur} and {c:h}",
                 "It is {name:FNV-1a|djb2|sdbm|Murmur-style|MurmurHash2|Murmur}, {c:h}.", "{name:FNV-1a|djb2|sdbm|Murmur-style|MurmurHash2|Murmur}: {c:h}"],
                "hash_name_is_the_algorithm", "hash_final_xor"),
    "skip": _hr(("{b:h}", "It is {b:h}.", "The value is {b:h}.", "Answer: {b:h}"), "skip_one_value"),
    "rotate": _hr(("{k:h}, {r:i}", "{k:h} and {r:i}", "{k:h}; {r:i}", "{k:h}, {r:i} bits"), "rotate_reference", dir=Q(r"(left|right)-rotation", "w")),
    "threshold": _hr(_BOUND_H, "threshold_values", inside=Q(r"for which this returns (-?\d+)")),
}
# the alphabet questions of the six encoding variants and the identity questions of the six
for _b in ("base64", "base32", "base58"):
    for _d in ("encode", "decode"):
        HEADS[f"encoding/{_b}_{_d}"] = _hr(_ENC_H[:3], "encoding_identity") + [H(_ENC_H[3], "encoding_identity", dir=lambda s: s["Dir"].lower())]
        HEADS[f"encoding/{_b}_{_d}#alphabet"] = _hr(_ALPHA_H, "alphabet_semantics", alen=lambda s: len(s["s"]), z=0)

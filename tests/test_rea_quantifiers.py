"""Round 6, B1 and the quantifier audit: a sentence that says `any`, `every`, `all`, `always`, `never`, `only`, `each`, `none`, `no other` or `exactly`
claims something about a whole domain, and a checker that tries one value of it proves nothing (round 6, B1: "Changing any byte ... makes the checksum
fail" was false for every Fletcher-16 task, because the checker flipped bytes with one XOR mask).

  * the sentences that still use such a word are exactly the keys of QUANTIFIED, each with the claim that backs it and the domain that claim covers;
  * a sentence that is not in the list uses none (so a new quantifier cannot come in without an entry and a checker);
  * the B1 counterexample is pinned: the Fletcher-16 checksum does not see a covered byte go from 0x00 to 0xFF.

Domains: `all256` every byte value; `printable` every printable ASCII character; `machine` the transitions of the whole machine, proven equal to the compiled
one, then a breadth-first search; `memory` the whole output buffer; `sweep` the integers from -2**17 to 2**17 one by one, then 2**32 in strides of 8191, then
the extremes; `positions` every byte value at every position of a long buffer; `lengths` every length from 0 to 89 with three fill bytes; `neighbours` every
input one edit away from the accepted one (each position x each byte: replaced, inserted, deleted), the empty string and an extension; `bits` every single-bit
change of the accepted header. A domain of 2**32 integers is swept, not enumerated: the one that is enumerated is reported by the claim's own message.
"""
from __future__ import annotations

import re
import shutil
import sys
import unittest

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families, f_pktsum, targets  # noqa: E402

WORDS = re.compile(r"\b(any|every|all|always|never|only|each|none|no|exactly|nothing|everything)\b", re.I)

# sentence (as stored in targets.py) -> (claim that backs it, domain it covers)
QUANTIFIED = {
    "This input of {d} letters moves the machine from state 0 to state {final}, and no shorter input does.": ("fsm_input_shortest", "machine"),
    "Starting from state 0, the machine can reach all of its {n} states.": ("fsm_states_reachable", "machine"),
    "When the first argument is {n}, the function returns {expr}, and no other opcode does.": ("opcode_unique", "sweep"),
    "proc handles {n} different opcodes, each with a result of its own, and returns {dflt} for the other values of the first argument.":
        ("opcode_count_handled", "sweep"),
    "Exactly {n} values of the first argument are handled by proc; the other values return {dflt}.": ("opcode_count_handled", "sweep"),
    "After proc has run, its argument holds exactly this string of {len} characters and a terminating NUL byte.": ("writes_exact_string", "memory"),
    "proc returns 1 for every integer from {lo} to {hi}, both included, and for no other integer.": ("returns_one_exactly_in_range", "sweep"),
    "The function accepts exactly the integers between {lo} and {hi} inclusive and returns 1 for them.": ("returns_one_exactly_in_range", "sweep"),
    "proc returns 1 for the {w} integers from {lo} to {hi}, and for no other integer.": ("returns_one_for_window", "sweep"),
    "proc returns -1 for every index outside the range from {lo} to {hi}, and a non-negative value for every index inside it.":
        ("index_limit_range", "sweep"),
    "Exactly {n} byte values make proc return 1; for the other byte values it does not.": ("char_class_count", "all256"),
    "When it is called with the seed {seed}, proc XORs the byte {k:#x} into every byte of the buffer.": ("xors_key_byte", "positions"),
    "The byte that proc XORs into each byte of the buffer for the seed {seed} is {k:#x}.": ("xors_key_byte", "positions"),
    "For the seed {seed}, every byte of the buffer is XORed with {k:#x}.": ("xors_key_byte", "positions"),
    "proc returns 1 when its input is exactly this string of {len} characters, and for other inputs it does not.": ("accepts_only_the_secret", "neighbours"),
    "Only this string of {len} characters makes proc return 1.": ("accepts_only_the_secret", "neighbours"),
    "proc accepts a string with the mode {mode} only while its length is at most {n} characters.": ("limit_for_mode", "lengths"),
    "The function XORs every byte of the buffer with this key, and the key repeats every {len} bytes.": ("xor_key_repeats", "positions"),
    "Each byte of the buffer is XORed with a character of the key string, taken in turn.": ("xor_key_repeats", "positions"),
    "The function returns 1 only when its argument is equal to this string.": ("password_equal_only", "neighbours"),
    "Only an input of exactly {len} characters that equals this string makes the function return 1.": ("password_equal_only", "neighbours"),
    "The password check succeeds for this string and for no other input.": ("password_equal_only", "neighbours"),
    "The decoder returns the position of its argument in this string of {len} characters, or {miss} for any other printable character.":
        ("alphabet_position", "printable"),
    "A header is valid only if its first four bytes, read as a little-endian 32-bit word, are 0x{m:08x}, and a version above {v} is rejected.":
        ("magic_header", "bits"),
    "The checksum adds up every byte of the input except those equal to 0x{b:02x}, and then adds the length.": ("skip_one_value", "all256"),
    "Exactly one byte value does not count in the sum, and it is 0x{b:02x}.": ("skip_one_value", "all256"),
    "The function returns {inside} for every v from {lo} to {hi}, {below} for a smaller v and {above} for a larger v.": ("threshold_values", "sweep"),
}
DOMAINS = {"all256", "printable", "machine", "memory", "sweep", "positions", "lengths", "neighbours", "bits"}


def quantified_sentences() -> dict:
    out = {}
    for key, spec in targets.SPECS.items():
        for i, s in enumerate(spec.sentences):
            if WORDS.search(s):
                out.setdefault(s, []).append((key, i))
    return out


class QuantifierAudit(unittest.TestCase):
    def test_the_sentences_with_a_quantifier_are_exactly_the_registered_ones(self):
        found = quantified_sentences()
        self.assertEqual(sorted(set(found) - set(QUANTIFIED)), [], "a sentence uses a universal word and has no entry (remove the word or register its checker)")
        self.assertEqual(sorted(set(QUANTIFIED) - set(found)), [], "an entry for a sentence that no longer has a universal word")

    def test_every_entry_names_a_claim_that_exists_and_a_known_domain(self):
        from . import rea_claims as RC
        for sentence, (claim, domain) in QUANTIFIED.items():
            self.assertIn(claim, RC.CLAIMS, sentence)
            self.assertIn(domain, DOMAINS, sentence)

    def test_every_registered_claim_is_made_by_the_sentence_it_backs(self):
        from . import test_rea_claims as T
        for sentence, (claim, _domain) in QUANTIFIED.items():
            keys = quantified_sentences()[sentence]
            for key, i in keys:
                self.assertIn(claim, T.claims_of(key, i), f"{key}[{i}]: the claim {claim} is not among the claims of this sentence")

    def test_no_pktsum_sentence_quantifies_over_bytes(self):
        """B1: the five checksums do not all detect every change of one byte, so no pktsum sentence may say that any byte, every byte or each byte matters."""
        for key, spec in targets.SPECS.items():
            if key.startswith("pktsum/"):
                for s in spec.sentences:
                    self.assertIsNone(re.search(r"\b(any|every|each|all)\b", s, re.I), s)


@unittest.skipUnless(shutil.which("gcc"), "gcc is needed")
class FletcherCounterexample(unittest.TestCase):
    """The fact that made B1 a blocker: a covered byte that goes from 0x00 to 0xFF leaves a Fletcher-16 checksum unchanged (it is taken mod 255)."""

    def test_the_model_does_not_see_00_become_ff(self):
        self.assertEqual(f_pktsum.csum("fletcher16", bytes(8)), f_pktsum.csum("fletcher16", bytes(7) + b"\xff"))
        for kind in ("sum8", "xor8", "sum16", "crc16"):
            self.assertNotEqual(f_pktsum.csum(kind, bytes(8)), f_pktsum.csum(kind, bytes(7) + b"\xff"), kind)

    def test_the_compiled_proc_accepts_both_packets(self):
        from . import rea_claims as RC
        task = next(t for t in families.generate("pktsum", 7 * 40, 99) if t.variant == "fletcher16" and t.params["ask"] == "start")
        p = task.params
        H, S, W = p["H"], p["S"], p["W"]

        head = bytearray(1 for _ in range(H))
        head[0] = p["magic"]
        for i in range(S, H):
            head[i] = 0
        body = bytes(head) + bytes(8)                  # every covered byte is 0x00
        a = body + f_pktsum.csum("fletcher16", body[S:]).to_bytes(W, p["endian"] or "little")
        b = a[:-W - 1] + b"\xff" + a[-W:]                # the same stored checksum, the last covered byte 0x00 -> 0xFF
        arr = lambda x: RC.c_data(bytes(x))
        out = RC.ints(RC.gcc_run(task.source(f"int main(void) {{ static const uint8_t a[] = {arr(a)}, b[] = {arr(b)}; printf(\"%d %d\\n\", proc(a, sizeof a), proc(b, sizeof b)); return 0; }}")))
        self.assertEqual(out, [1, 1], "a covered byte 0x00 -> 0xFF is not detected by Fletcher-16: 'any byte makes the checksum fail' is false")


if __name__ == "__main__":
    unittest.main()

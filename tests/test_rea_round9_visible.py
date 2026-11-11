"""Round 9, B1: the model must be SHOWN the bytes the answer is read from.

`strobf/reverse_xor` compiled (gcc -O2/-Os) to a pointer that runs down from the LAST byte of the encoded array, and `refdata` prints 64 bytes FORWARD from an address: 15 of the 16
bytes were never shown (20 of 24 production tasks). Two layers fix it, each tested here:
  (a) the generator writes the loop forward and indexes the OUTPUT backwards;
  (b) `survive.check_visible` (every family): every byte of a `static const` array that the linked binary keeps in .rodata is inside a window refdata prints for a data reference.
The byte-flip scan (`rea_bench.bytescan`, `python3 tools/rea-bench bytescan`) is the independent check: it finds the bytes the PROGRAM needs, and it reproduces B1 on the old loop.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP
from .test_rea_modal import ModalCase
from .test_rea_round8_modal import private_mkdtemp

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import bytescan, core, families, f_strobf, survive, visible  # noqa: E402

CLANG = bool(shutil.which("clang"))
CRC_TABLE_VARIANTS = {"crc/bitwise_init", "crc/bitwise_xorout", "crc/table_init", "crc/table_xorout", "pktsum/crc16"}
OLD_LOOP, NEW_LOOP = "out[i] = (char)(enc[n - 1 - i] ^", "out[n - 1 - i] = (char)(enc[i] ^"


def reverse_tasks(n: int = 24, cc: str = "gcc", opt: str = "-Os", seed: int = 9091) -> list:
    """n reverse_xor tasks of the generator, forced to one compiler and -O level."""
    out = []
    for t in families.generate("strobf", 7 * n, seed, "mix", None):
        if t.variant == "reverse_xor":
            t.cc, t.opt = cc, opt
            out.append(t)
    return out[:n]


def with_old_loop(t):
    assert NEW_LOOP in t.body
    t.body = t.body.replace(NEW_LOOP, OLD_LOOP)
    return t


class StaticArrays(unittest.TestCase):
    def test_the_arrays_of_a_body_as_the_compiler_lays_them_out(self):
        body = '''void proc(void) {
    static const uint8_t a[] = {0x01, 0x02, 255, 'A', '\\n'};
    static const uint16_t b[2] = {0x0102, 3};
    static const uint32_t c[4] = {0xdeadbeefu, 1u};
    static const char s[] = "a\\tb\\x41\\101\\\\\\"";
    static const uint8_t T[2][3] = {{1, 2, 3}, {4, 5, 6}};
    static const uint8_t tiny[2] = {1, 2};
    static uint8_t not_const[8] = {1, 2, 3, 4, 5, 6, 7, 8};
}'''
        got = dict(visible.static_arrays(body))
        self.assertEqual(got["a"], bytes([1, 2, 255, 65, 10]))
        self.assertEqual(got["b"], bytes([2, 1, 3, 0]))
        self.assertEqual(got["c"], bytes([0xef, 0xbe, 0xad, 0xde, 1, 0, 0, 0] + [0] * 8), "zero fill up to the declared size")
        self.assertEqual(got["s"], b"a\tbAA\\\"\0")
        self.assertEqual(got["T"], bytes([1, 2, 3, 4, 5, 6]))
        self.assertNotIn("tiny", got, "fewer than 4 bytes cannot be told from the rest of .rodata")
        self.assertNotIn("not_const", got, "a writable array is not in .rodata")

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_every_array_of_every_family_is_found_whole_in_the_rodata_at_gcc_O0(self):
        """The extraction is faithful: at -O0 gcc keeps every static const array, contiguously and byte for byte."""
        bad = []
        with tempfile.TemporaryDirectory() as d:
            for fam, m in families.modules().items():
                n = 2 * (len(m.VARIANTS) or 1)
                for t in families.generate(fam, n, 31, "gcc", None):
                    arrays = visible.static_arrays(t.body)
                    if not arrays:
                        continue
                    exe = os.path.join(d, "x")
                    r = visible.link(t.source(), "gcc", "-O0", exe)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    rod = visible.Exe(exe).rodata
                    bad += [(t.id, name) for name, arr in arrays if rod.find(arr) < 0]
        self.assertEqual(bad, [])


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class ReverseXor(unittest.TestCase):
    def test_the_loop_runs_forward_over_the_stored_array_and_writes_the_output_backwards(self):
        for t in reverse_tasks(8):
            self.assertIn(NEW_LOOP, t.body)
            self.assertNotIn("enc[n - 1 - i]", t.body, "no backward index into the stored array: the compiler would reference its last byte")
            p, k = t.params["plain"], t.params["key"]
            stored = bytes.fromhex(t.object_gate["stored"])
            self.assertEqual(stored, bytes(ord(c) ^ k for c in reversed(p)), "the array is still the plaintext reversed")
            self.assertEqual(len(stored), len(p))

    def test_the_old_loop_hides_the_array_at_gcc_Os_and_the_gate_says_so(self):
        hidden = 0
        for t in reverse_tasks(24):
            ok, why = survive.check_visible(with_old_loop(t).source(), "gcc", "-Os")
            hidden += not ok
            if not ok:
                self.assertIn("refdata prints", why)
                self.assertIn("`enc`", why)
        self.assertGreaterEqual(hidden, 12, "the reviewer measured 20 of 24 at the tasks' own cc/opt")

    def test_the_new_loop_is_shown_at_every_compiler_and_level(self):
        for cc in ("gcc",) + (("clang",) if CLANG else ()):
            for opt in core.OPTS:
                for t in reverse_tasks(8, cc, opt):
                    self.assertEqual(survive.check_visible(t.source(), cc, opt), (True, ""), f"{t.id} {cc} {opt}")

    def test_the_gate_rejects_a_draw_whose_array_is_hidden_and_counts_it(self):
        gate = survive.Gate(warn=lambda m: None)
        verdicts = [gate(with_old_loop(t)) for t in reverse_tasks(12)]
        self.assertIn(False, verdicts)
        self.assertEqual(gate.unshown, verdicts.count(False))
        self.assertIn("a byte of a table is not in any window refdata prints", gate.summary())
        gate = survive.Gate(warn=lambda m: None)
        self.assertTrue(all(gate(t) for t in reverse_tasks(12)))
        self.assertEqual(gate.unshown, 0)

    def test_a_table_beyond_the_first_window_is_hidden_and_one_referenced_from_its_start_is_not(self):
        def src(n):
            body = ("void proc(uint8_t *out) {\n    static const uint8_t T[%d] = {%s};\n    for (size_t i = 0; i < %d; i++) out[i] = T[i] ^ 0x5a;\n}" %
                    (n, ", ".join(str((i * 37 + 11) & 0xff) for i in range(n)), n))
            return core.HDR + body + core.MAIN
        self.assertEqual(survive.check_visible(src(48), "gcc", "-O0"), (True, ""))
        ok, why = survive.check_visible(src(160), "gcc", "-O0")
        self.assertFalse(ok)
        self.assertIn("of the 160 bytes of `T`", why, "a 160-byte table: 64 bytes are shown, the rest is not")

    def test_a_task_without_a_static_array_needs_no_compile(self):
        with mock.patch.object(visible, "link", side_effect=AssertionError("compiled")):
            self.assertEqual(survive.check_visible(core.HDR + "int proc(int x) { return x + 1; }" + core.MAIN, "gcc", "-O2"), (True, ""))


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class ByteFlipScan(unittest.TestCase):
    """The scanner is the independent check: it flips bytes of the program, not of the arrays the gate knows."""

    def scan(self, t):
        with tempfile.TemporaryDirectory() as d:
            return bytescan.scan_task(t, t.cc, t.opt, {"dir": d, "env": os.environ})

    def test_it_reproduces_b1_on_the_old_loop_and_finds_nothing_on_the_new_one(self):
        old = [self.scan(with_old_loop(t)) for t in reverse_tasks(12)]
        new = [self.scan(t) for t in reverse_tasks(12)]
        self.assertTrue(all(r["base_ok"] for r in old + new), "the harness proves the truth on every build")
        self.assertGreaterEqual(sum(bool(r["uncovered_written"]) for r in old), 6, [r["uncovered"] for r in old])
        self.assertEqual([r["uncovered"] for r in new], [0] * 12)
        self.assertTrue(all(r["needed"] >= len(t.params["plain"]) for r, t in zip(new, reverse_tasks(12))), "every byte of the encoded array is one the program needs")

    def test_a_gated_population_of_every_family_has_no_uncovered_byte(self):
        """The gate keeps what the scan then finds clean: one task per variant of every family (REA_BYTESCAN_N), at its own compiler and -O level."""
        n = int(os.environ.get("REA_BYTESCAN_N", "1"))
        notes = []
        for fam in families.modules():
            table = bytescan.scan(fam, 9091, n, own=True, gated=True, out=lambda line: notes.append(line) if line.startswith("NOTE") else self.fail(line[:400]))
            for v, r in table.items():
                self.assertEqual((r["written_tasks"], r["base_bad"], r["errors"]), (0, 0, 0), (fam, v))
                if r["made_only_tasks"]:         # data the compiler made: gcc 15 and clang turn a bitwise CRC loop into a 256-entry table that no question asks about
                    self.assertIn(f"{fam}/{v}", CRC_TABLE_VARIANTS)
                    self.assertFalse(v.endswith("_poly"), "a polynomial question is the one that WOULD ask for the table: `no_table` keeps those tasks out")


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class LegacyAndModal(ModalCase):
    def setUp(self):
        private_mkdtemp(self)

    def test_a_legacy_task_whose_data_is_hidden_is_dropped_on_a_training_seed_only(self):
        gate = survive.Gate(warn=lambda m: None)
        t = [x for x in families.generate("legacy", 60, 7) if x.kind == "alphabet"][0]
        with mock.patch.object(gate, "legacy_shown", return_value=(False, "hidden")):
            self.assertIn("not shown to the model", families.untrainable(t, 7, gate))
            self.assertIsNone(families.untrainable(t, core.HELDOUT_SEED, gate), "the held-out seed is not filtered")
        self.assertIsNone(families.untrainable(t, 7, gate), "a frozen string is shown: the task stays")

    def test_the_frozen_legacy_strings_are_shown(self):
        gate = survive.Gate(warn=lambda m: None)
        tasks = [t for t in families.generate("legacy", 200, 7) if survive.has_arrays(t)]
        self.assertGreaterEqual(len(tasks), 20)
        self.assertEqual([t.id for t in tasks if not gate.legacy_shown(t)[0]], [])

    def test_the_modal_side_recheck_refuses_a_hidden_table_and_passes_the_helpers(self):
        t = next(x for x in (with_old_loop(y) for y in reverse_tasks(24)) if not survive.check_visible(x.source(), "gcc", "-Os")[0])        # a task whose old loop hides the array
        with mock.patch.object(self.data, "_sh_rc", return_value=(0, "")), mock.patch.object(self.data, "_decompile_fn", side_effect=KeyError("reached the decompile")):
            rec = self.data._decompile_task(t.meta(), t.source(), "seed9091-strobf")
            self.assertTrue(rec["error"].startswith("data not shown to the model"), rec)
            self.assertEqual(self.data.outputs.error_reason(rec), "table not shown (Modal compilers)")
            shown = reverse_tasks(1, "gcc", "-O0")[0]
            with self.assertRaises(KeyError):
                self.data._decompile_task(shown.meta(), shown.source(), "seed9091-strobf")
        seen: list = []
        task = {"id": "x-1", "kind": "multi", "variant": "decode_secret", "opt": "-O2", "cc": "gcc", "helpers": ["helper"], "score": "tokens", "accept": [["1"]]}
        with mock.patch.object(self.data, "_sh_rc", return_value=(0, "")), mock.patch.object(self.data.survive, "check_visible", lambda *a: (seen.append(a), (False, "gone"))[1]):
            rec = self.data._decompile_task(task, "SRC", "seed7-multi")
        self.assertEqual(seen, [("SRC", "gcc", "-O2", ["helper"])])
        self.assertIn("data not shown to the model", rec["error"])


if __name__ == "__main__":
    unittest.main()

"""Round 5, the generators: B1 (alt_xor draws two different constants), M1 (the strobf object gate), M2 (legacy skip with b = 0x00 is not trained on).

B1  `alt_xor` drew its two constants independently; 0.46% of the tasks had the same one twice. Now k1 != k, and only those draws changed.
M1  the compiler decodes a constant array at compile time (the string becomes immediates): such a task shows the answer in the code and no
    decoding. A strobf task of an encoded variant carries an `object_gate` (survive.py): the plaintext must not be in the object compiled at
    its own cc/opt, the stored bytes (or a key) must be.
M2  a legacy `skip` task with b = 0x00 asks for a compare that no optimiser leaves in the code. The frozen generator is unchanged; the training
    path (families.generate_many / dedupe.training_tasks) drops those tasks for every seed but the held-out one.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import pathlib

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, dedupe, f_strobf, families, survive  # noqa: E402

GCC_OBJDUMP = bool(shutil.which("gcc") and shutil.which("objdump"))
VARIANTS = list(f_strobf.VARIANTS)
ENCODED = [v for v in VARIANTS if v != "stack_string"]


class AltXorConstantsDiffer(unittest.TestCase):
    PAIR = re.compile(r"\(i & 1\) \? (0x[0-9a-f]+) : (0x[0-9a-f]+)\)")

    def test_no_alt_xor_task_has_the_same_constant_twice(self):
        i0 = VARIANTS.index("alt_xor")
        seen = 0
        for seed in (99, 1001, 4242, 7, 53):
            for i in range(i0, 7 * 2500, 7):
                t = core.draw("strobf", f_strobf.VARIANTS, i, seed, "mix", 0)
                a, b = self.PAIR.search(t.body).groups()
                self.assertNotEqual(a, b, t.id)
                self.assertEqual((int(b, 16), int(a, 16)), (t.params["key"], t.params["k1"]))
                seen += 1
        self.assertEqual(seen, 12500)

    def test_only_the_draws_that_were_equal_changed(self):
        """The rng stream is untouched: the plaintext, k and k1 come out in the same order as before; k1 is the next value only when it equalled k."""
        i0 = VARIANTS.index("alt_xor")
        changed = 0
        for seed in (99, 1001, 4242):
            for i in range(i0, 7 * 1500, 7):
                rng = core.TaskRng(f"strobf/{seed}/{i}")
                rng.held = seed == core.HELDOUT_SEED
                p = f_strobf.plaintext(rng)
                k, k1 = rng.randint(1, 255), rng.randint(1, 255)       # the old code: two independent draws
                t = core.draw("strobf", f_strobf.VARIANTS, i, seed, "mix", 0)
                self.assertEqual((t.params["plain"], t.params["key"]), (p, k))
                self.assertEqual(t.params["k1"], k1 if k1 != k else k1 % 255 + 1, t.id)
                changed += k1 == k
        self.assertGreater(changed, 0, "the sample must contain draws that were equal")
        self.assertLess(changed, 60)

    def test_the_held_out_list_is_still_the_generator_s(self):
        now = {line.split()[1] for line in dedupe.heldout_lines()}
        self.assertEqual(dedupe.load_heldout() - now, set(), "no pinned held-out key is lost: the fix changed no held-out task")


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class ObjectGate(unittest.TestCase):
    @staticmethod
    def build(src: str, cc: str = "gcc", opt: str = "-O0", extra=()) -> pathlib.Path:
        d = pathlib.Path(tempfile.mkdtemp())
        out = d / "o.o"
        r = subprocess.run([cc, opt, "-w", "-c", *extra, "-o", str(out), "-x", "c", "-"], input=src, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return out

    def test_the_blob_holds_code_and_data_and_no_symbols_or_debug_info(self):
        src = 'const char data_str[] = "visible_in_data";\nint zzsymbolname(int qxdebugvar) { return qxdebugvar + 7; }\n'
        blob = survive.object_blob(self.build(src, extra=("-g",)))
        self.assertIn(b"visible_in_data", blob)
        self.assertNotIn(b"zzsymbolname", blob, "symbol names are no part of the program")
        self.assertNotIn(b"qxdebugvar", blob, "neither is debug info")

    def test_plaintext_window_needs_letters(self):
        self.assertTrue(survive.plaintext_visible("admin_cache42", b"..xx_cach.."))
        self.assertFalse(survive.plaintext_visible("admin_cache42", b"..42..11..__.."), "digits and punctuation are in every object")
        self.assertFalse(survive.plaintext_visible("ab", b"ab"), "a string shorter than a window has none")

    def test_every_encoded_variant_passes_at_gcc_O0_and_the_plaintext_in_clear_fails(self):
        for t in families.generate("strobf", 7 * 4, 31, "gcc"):
            if t.variant == "stack_string":
                self.assertIsNone(t.object_gate, "a stack string has no encoded form: no gate")
                continue
            self.assertTrue(set(t.object_gate) == {"hidden", "stored", "keys"} and t.object_gate["hidden"] == t.params["plain"], t.id)
            self.assertEqual(survive.check_task(t, "gcc", "-O0"), (True, ""), t.id)
            clear = t.source().replace(t.body, f'void proc(char *out) {{ strcpy(out, "{t.params["plain"]}"); }}')
            ok, why = survive.check_object(t.object_gate, clear, "gcc", "-O0")
            self.assertFalse(ok, t.id)
            self.assertIn("plaintext is in the object", why)

    def test_a_program_with_neither_the_stored_bytes_nor_the_key_fails(self):
        t = next(t for t in families.generate("strobf", 7 * 3, 31, "gcc") if t.variant == "sub_const")
        other = 'void proc(char *out) { for (int i = 0; i < 9; i++) out[i] = (char)(i * 31 + 5); out[9] = 0; }'
        gate = {**t.object_gate, "keys": [251]}
        ok, why = survive.check_object(gate, t.source().replace(t.body, other), "gcc", "-O0")
        self.assertFalse(ok)
        self.assertIn("neither the stored bytes nor a key", why)

    def test_the_gate_is_in_the_task_json_and_the_modal_side_can_check_it(self):
        t = next(t for t in families.generate("strobf", 7 * 3, 31, "gcc") if t.variant == "xor_const")
        m = t.meta()
        self.assertEqual(m["object_gate"], t.object_gate)
        self.assertEqual(survive.check_object(m["object_gate"], t.source(), "gcc", "-O0"), (True, ""))
        self.assertNotIn("object_gate", families.generate("crc", 3, 31)[0].meta())
        self.assertNotIn("object_gate", families.generate("legacy", 10, 1001)[0].meta(), "legacy output is frozen")

    def test_the_compiler_does_fold_some_encoded_strings_and_the_gate_redraws(self):
        """Without the gate some draws show the plaintext in the object (gcc -O2/-O3, clang -O1 and up); gated generation keeps only the ones that do not."""
        free = families.generate("strobf", 7 * 14, 5, "mix")
        folded = [t for t in free if t.object_gate and not survive.check_task(t)[0]]
        self.assertGreater(len(folded), 0, "the sample must hold folded draws, else this test shows nothing")
        gate = survive.Gate()
        gated = families.generate("strobf", 7 * 14, 5, "mix", gate)
        self.assertGreater(gate.rejected, 0)
        self.assertEqual(len(gated), len(free))
        for t in gated:
            self.assertTrue(survive.check_task(t)[0], f"{t.id} {t.variant} {t.cc} {t.opt}")
        self.assertIn("strobf/xor_const", gate.by_variant)


class LegacySkipZero(unittest.TestCase):
    def zero(self, t):
        return t.kind == "skip" and t.truth["accept"][0] == "0x00"

    def test_the_frozen_generator_still_makes_them_and_the_training_path_drops_them(self):
        all_ = families.generate("legacy", 3000, 7)
        zeros = [t.id for t in all_ if self.zero(t)]
        self.assertGreater(len(zeros), 20, "about one skip task in seven draws b = 0x00")
        dropped: list = []
        kept = families.generate_many(["legacy"], 3000, 7, dropped=dropped)
        self.assertEqual([t.id for t, _why in dropped], zeros)
        self.assertEqual([t.id for t in kept], [t.id for t in all_ if not self.zero(t)])
        self.assertEqual([t.meta() for t in kept], [t.meta() for t in all_ if not self.zero(t)], "the other tasks are byte for byte what they were")

    def test_the_held_out_seed_is_not_filtered(self):
        a, b = families.generate("legacy", 3000, core.HELDOUT_SEED), families.generate_many(["legacy"], 3000, core.HELDOUT_SEED)
        self.assertTrue(any(self.zero(t) for t in a))
        self.assertEqual([t.meta() for t in a], [t.meta() for t in b])

    def test_training_tasks_reports_the_count_and_the_other_families_are_untouched(self):
        report: dict = {}
        kept, held = dedupe.training_tasks(["legacy", "structs"], 600, 7, report=report)
        n = sum(self.zero(t) for t in families.generate("legacy", 300, 7))
        self.assertEqual(sum(report["untrainable"].values()), n)
        self.assertEqual(sum(t.kind == "structs" for t in kept) + sum(t.kind == "structs" for t in held), 300)
        self.assertTrue(all(not self.zero(t) for t in kept))
        self.assertIsNone(families.untrainable(families.generate("crc", 1, 7)[0], 7))

    def test_the_cli_generator_and_the_pin_are_the_frozen_ones(self):
        self.assertTrue(any(self.zero(t) for t in families.generate("legacy", 3000, 7)), "families.generate and `gen` are not filtered")


if __name__ == "__main__":
    unittest.main()

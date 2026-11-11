"""Round 10 (the last gate before Modal).

M1  the local gate must not RAISE on a draw: "no free probe marker" is a failed probe, the draw is drawn again (`cipher-7171-0090`, salsa at clang -O0, aborted the generation).
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import unittest

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, families, survive  # noqa: E402

CLANG = bool(shutil.which("clang"))


def objdump_of(src: str, cc: str, opt: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        obj = pathlib.Path(d, "o.o")
        survive.compile_object(src, cc, opt, obj)
        return survive._objdump(["-d", "--no-show-raw-insn", "-M", "intel"], obj)


# ---- M1 -------------------------------------------------------------------------------------------------------------------------------

def proc_with_every_marker() -> str:
    """A `proc` whose code holds every marker of every table of survive.MARKERS as an immediate (a product with a volatile: the compiler cannot fold it), a loop bound of 40
    and two hex constants to ask for."""
    products = "".join(f"    s += v * {m}u;\n" for ms in survive.MARKERS.values() for m in ms)
    return ("unsigned proc(unsigned x) {\n    volatile unsigned v = x;\n    unsigned i, s = x;\n    for (i = 0; i < 40; i++) s += v;\n    s ^= 0x1f2e;\n    s ^= 0x7ac3d915;\n"
            + products + "    return s;\n}\n")


PROBES = {"decimal, exact": ([39, 40, 41], {"at": r"i < (\d+);", "kind": "exact"}),
          "decimal, near": ([39, 40, 41], {"at": r"i < (\d+);", "kind": "near"}),
          "decimal, half": ([39, 40, 41], {"at": r"i < (\d+);", "kind": "half"}),
          "16-bit hex": ([0x1f2e], {"at": r"\^= (0x1f2e);", "kind": "exact", "hex": True, "bits": 16}),
          "32-bit hex": ([0x7ac3d915], {"at": r"\^= (0x7ac3d915);", "kind": "exact", "hex": True, "bits": 32})}


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class NoFreeMarker(unittest.TestCase):
    def test_a_source_that_holds_every_marker_fails_its_probe_and_does_not_raise(self):
        src = core.HDR + proc_with_every_marker() + core.MAIN
        nums = survive.numbers(objdump_of(src, "gcc", "-O0"))
        for table, markers in survive.MARKERS.items():
            self.assertTrue(set(markers) <= nums, f"the test source holds every {table} marker (else it proves nothing)")
        for name, (alts, probe) in PROBES.items():
            with self.subTest(name):
                ok, why = survive.check([[*alts, probe]], src, "gcc", "-O0")
                self.assertFalse(ok)
                self.assertIn("no free probe marker", why)

    def test_the_gate_rejects_and_counts_such_a_draw_instead_of_raising(self):
        body = proc_with_every_marker()
        t = core.Task(kind="cipher", variant="synthetic", body=body, question="q", truth={"score": "tokens", "accept": [["40"]]}, id="cipher-test-0001", opt="-O0", cc="gcc",
                      facts=[[39, 40, 41, PROBES["decimal, half"][1]]])
        gate = survive.Gate(warn=lambda m: None)
        self.assertIs(gate(t), False)
        self.assertEqual((gate.tried, gate.rejected), (1, 1))
        self.assertIn("1 of 1 draws rejected", gate.summary())

    def test_the_marker_table_is_longer_and_the_first_eight_are_the_old_ones(self):
        self.assertEqual(survive.MARKERS["dec"][:8], [1237, 1693, 2087, 2411, 3041, 3673, 4349, 5003], "a draw that passed keeps its marker")
        self.assertGreaterEqual(len(survive.MARKERS["dec"]), 16)
        self.assertEqual(len(set(survive.MARKERS["dec"])), len(survive.MARKERS["dec"]))
        self.assertIsNone(survive._marker({"kind": "exact"}, set(survive.MARKERS["dec"])))
        self.assertEqual(survive._marker({"kind": "exact"}, set(survive.MARKERS["dec"][:8])), survive.MARKERS["dec"][8])

    @unittest.skipUnless(CLANG, "clang is needed")
    def test_the_draw_that_aborted_the_generation_is_gated_without_an_exception(self):
        """cipher-7171-0090 (salsa, clang -O0, 40 rounds): `families.generate("cipher", 168, 7171, "mix", Gate())` died with every probe marker already in the code."""
        t = next(x for x in families.generate("cipher", 91, 7171, "mix", None) if x.id == "cipher-7171-0090")
        self.assertEqual((t.variant, t.cc, t.opt), ("salsa", "clang", "-O0"), "the draw is the reviewer's")
        ok, why = survive.check_task(t)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(why, str)


if __name__ == "__main__":
    unittest.main()

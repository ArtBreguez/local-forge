"""Round 10 (the last gate before Modal).

M1  the local gate must not RAISE on a draw: "no free probe marker" is a failed probe, the draw is drawn again (`cipher-7171-0090`, salsa at clang -O0, aborted the generation).
m1  `survive.numbers` does not read the target of a branch or the displacement of a rip-relative operand as a decimal constant (`structs-7171-0174`, gcc -O3, stride 40).
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import unittest

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP
from .test_rea_round9_data import literal_lines

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


# ---- m1 -------------------------------------------------------------------------------------------------------------------------------

class BranchTargets(unittest.TestCase):
    DIS = ("0000000000000000 <proc>:\n"
           "   0:\tlea    rcx,[rdx+rdx*4]\n"
           "   4:\tlea    r8,[rcx*8+0x0]\n"
           "   c:\tjne    40 <proc+0x40>\n"
           "   e:\tjmp    1a <proc+0x1a>\n"
           "  10:\tcall   1050 <helper>\n"
           "  15:\tlea    rax,[rip+0x2e5f]        # 4010 <tbl>\n"
           "  1c:\tjmp    QWORD PTR [rdi+0x18]\n"
           "  20:\tnotrack jmp rax\n"
           "  22:\tbnd jmp 2a <proc+0x2a>\n"
           "  24:\tloop   5e <proc+0x5e>\n"
           "  26:\tjrcxz  72 <proc+0x72>\n"
           "  28:\tcall   QWORD PTR [rip+0x2fe2]        # 3fd0 <puts@got>\n"
           "  2e:\tadd    rdi,0x38\n")

    def test_the_target_of_a_branch_and_a_rip_relative_displacement_are_no_numbers_of_the_program(self):
        self.assertEqual(survive.numbers(self.DIS), {4, 8, 0, 0x18, 0x38}, "the scales and displacements stay; 40, 1a, 1050, 2a, 5e, 72, 0x2e5f, 0x2fe2 are addresses")
        self.assertEqual(survive.numbers(self.DIS, stack=False), {4, 8, 0, 0x18, 0x38})

    def test_an_indirect_branch_keeps_its_displacement_and_a_constant_is_still_read(self):
        dis = "0000000000000000 <proc>:\n   0:\tcall   QWORD PTR [rdi+0x18]\n   4:\tcmp    eax,0x28\n   7:\tjne    28 <proc+0x28>\n"
        self.assertEqual(survive.numbers(dis), {0x18, 0x28}, "0x28 from the cmp, not from the jne")
        self.assertEqual(survive.numbers("0000000000000000 <proc>:\n   0:\tjne    28 <proc+0x28>\n"), set())

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_the_stride_40_task_of_the_reviewer_is_rejected(self):
        """structs-7171-0174: gcc -O3, stride 40, a vectorised loop that scales with `lea rcx,[rdx+rdx*4]` and `lea r8,[rcx*8]`: its only 40 was `jne 40 <proc+0x40>`."""
        t = next(x for x in families.generate("structs", 175, 7171, "mix", None) if x.id == "structs-7171-0174")
        self.assertEqual((t.variant, t.cc, t.opt, t.params["size"]), ("array_stride", "gcc", "-O3", 40), "the draw is the reviewer's")
        dis = objdump_of(t.source(), "gcc", "-O3")
        body = dis.split("<proc>:")[1].split("<main>:")[0]
        self.assertTrue(any("jne    40 <proc+0x40>" in x for x in body.splitlines()), "the code has the branch to 0x40 the reviewer named")
        self.assertEqual(literal_lines(body, 40), [], "no instruction of the code names 40 or 0x28 other than as a branch target")
        self.assertNotIn(40, survive.numbers(dis, stack=False))
        ok, why = survive.check(t.facts, t.source(), "gcc", "-O3")
        self.assertFalse(ok)
        self.assertIn("outside the stack frame", why)
        self.assertIs(survive.Gate(warn=lambda m: None)(t), False)


if __name__ == "__main__":
    unittest.main()

"""Round 10 (the last gate before Modal).

M1  the local gate must not RAISE on a draw: "no free probe marker" is a failed probe, the draw is drawn again (`cipher-7171-0090`, salsa at clang -O0, aborted the generation).
m1  `survive.numbers` does not read the target of a branch or the displacement of a rip-relative operand as a decimal constant (`jne 40 <proc+0x40>` was the stride 40 of
    `structs-7171-0174`, gcc -O3, which has no literal 40 in its code).
m2  the gaps the round-9 mutations found in the tests of the visibility gate (V03 V05 V08 V10, G02 G03 G06 G09) and of the length rule (L03 L09): each test below fails when
    its mutation is applied (the mutations are listed in tools/modal/README.md, "round 10").
"""
from __future__ import annotations

import io
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP
from .test_rea_modal import ModalCase
from .test_rea_round9_data import literal_lines

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, families, sft, survive, verify, visible  # noqa: E402

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


# ---- m2: the visibility gate ----------------------------------------------------------------------------------------------------------

def table_proc(n: int, name: str = "T") -> str:
    rows = ", ".join(str((i * 37 + 11) & 0xff) for i in range(n))
    return ("void proc(uint8_t *out) {\n    static const uint8_t %s[%d] = {%s};\n    for (size_t i = 0; i < %d; i++) out[i] = %s[i] ^ 0x5a;\n}" % (name, n, rows, n, name))


def arrays_only_task(n: int, body: str | None = None, **kw):
    """A task of an arrays-only family (no fact, no object gate: an fsm table, an encoding alphabet, multi/decode_secret) whose only table has n bytes, referenced at its start."""
    return core.Task(kind="encoding", variant="synthetic", body=body or table_proc(n), question="Which bytes does proc write?", truth={"score": "tokens", "accept": [["1"]]},
                     id="encoding-test-0001", opt="-O0", cc="gcc", **kw)


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class EveryArrayIsChecked(unittest.TestCase):
    """V05: `hidden_bytes` looked at the first array only."""

    def test_the_second_array_is_checked_too(self):
        body = ("void proc(uint8_t *out) {\n    static const uint8_t A[16] = {%s};\n    static const uint8_t B[160] = {%s};\n"
                "    for (size_t i = 0; i < 16; i++) out[i] = A[i] ^ 0x21;\n    for (size_t i = 0; i < 160; i++) out[16 + i] = B[i] ^ 0x5a;\n}"
                % (", ".join(str((i * 7 + 3) & 0xff) for i in range(16)), ", ".join(str((i * 37 + 11) & 0xff) for i in range(160))))
        ok, why = survive.check_visible(core.HDR + body + core.MAIN, "gcc", "-O0")
        self.assertFalse(ok, "A is shown whole, B is referenced at its start only: 96 of its 160 bytes are not shown")
        self.assertIn("of the 160 bytes of `B`", why)
        self.assertNotIn("`A`", why)


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class ArraysOnlyFamilies(unittest.TestCase):
    """G02, G03, G09: a family whose only check is the visibility of its tables (fsm, encoding, multi/decode_secret: no fact, no object gate) is still gated. Today none of
    their ~835 draws (120 per variant, seed 7171) has a hidden table, so only a synthetic draw can show that the gate is there."""

    def test_a_table_only_task_is_gated(self):
        t = arrays_only_task(160)
        self.assertFalse(t.facts or t.object_gate)
        self.assertTrue(survive.has_arrays(t))
        self.assertTrue(survive.gated(t), "G02")

    def test_the_gate_rejects_a_hidden_table_of_such_a_task_and_keeps_a_shown_one(self):
        gate = survive.Gate(warn=lambda m: None)
        self.assertIs(gate(arrays_only_task(160)), False, "G03: 64 of 160 bytes are shown")
        self.assertEqual((gate.tried, gate.rejected, gate.unshown), (1, 1, 1))
        self.assertIs(gate(arrays_only_task(48)), True)
        self.assertEqual((gate.tried, gate.rejected, gate.unshown), (2, 1, 1))

    def test_check_task_and_the_check_command_refuse_it_too(self):
        ok, why = survive.check_task(arrays_only_task(160))
        self.assertFalse(ok)
        self.assertIn("of the 160 bytes of `T`", why)
        problems = verify.check_task(arrays_only_task(160), ubsan=False)
        self.assertTrue(any("fact does not survive" in p and "refdata prints" in p for p in problems), f"G09: {problems}")
        self.assertFalse(any("refdata prints" in p for p in verify.check_task(arrays_only_task(48), ubsan=False)))
        self.assertFalse(any("refdata prints" in p for p in verify.check_task(arrays_only_task(160), ubsan=False, survive_check=False)))


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class HelpersAndClones(unittest.TestCase):
    def test_a_table_referenced_by_a_helper_is_shown_when_the_task_declares_it(self):
        """G06: the model is shown the pseudocode of `proc` AND of its helpers; the table is referenced by the helper only."""
        rows = ", ".join(str((i * 37 + 11) & 0xff) for i in range(48))
        body = ("static const uint8_t T[48] = {%s};\n__attribute__((noinline)) static void helper(uint8_t *out) {\n    for (size_t i = 0; i < 48; i++) out[i] = T[i] ^ 0x5a;\n}\n"
                "void proc(uint8_t *out) {\n    helper(out);\n}" % rows)
        src = core.HDR + body + core.MAIN
        self.assertEqual(survive.check_visible(src, "gcc", "-O0", ["helper"]), (True, ""))
        self.assertFalse(survive.check_visible(src, "gcc", "-O0")[0], "no reference in `proc`: nothing of T is shown without the helper")
        t = arrays_only_task(0, body=body, helpers=["helper"])
        self.assertEqual(survive.check_task(t), (True, ""))
        t.helpers = []
        self.assertFalse(survive.check_task(t)[0])

    def test_the_part_the_compiler_split_off_proc_counts_as_proc(self):
        """V08: gcc -O3 moves the `abort` path of this proc into `proc.cold`, and the only reference to T is there."""
        rows = ", ".join(str((i * 37 + 11) & 0xff) for i in range(48))
        body = ("void proc(uint8_t *out, unsigned x) {\n    static const uint8_t T[48] = {%s};\n    if (__builtin_expect(x > 1000u, 0)) {\n"
                "        for (size_t i = 0; i < 48; i++) out[i] = T[i] ^ 0x5a;\n        abort();\n    }\n    out[0] = (uint8_t)x;\n}" % rows)
        src = core.HDR + "#include <stdlib.h>\n" + body + core.MAIN
        with tempfile.TemporaryDirectory() as d:
            exe = pathlib.Path(d, "t")
            self.assertEqual(visible.link(src, "gcc", "-O3", exe).returncode, 0)
            names = sorted(visible.Exe(exe).functions(("proc",)))
        self.assertEqual(names, ["proc", "proc.cold"], "gcc split the cold path off (another gcc: adapt the source until it does)")
        self.assertEqual(survive.check_visible(src, "gcc", "-O3"), (True, ""))


class FakeExe:
    """The part of visible.Exe that visible.Model reads: a .rodata at an address, and the function symbols."""

    def __init__(self, rodata: bytes, vma: int, functions: dict):
        self.path, self.rodata, self.rodata_vma, self._functions = "fake", rodata, vma, functions

    def functions(self, bases):
        return self._functions


def model_of(rodata: bytes, instructions: list, vma: int = 0x2000, lo: int = 0x1000, hi: int = 0x1100) -> visible.Model:
    with mock.patch.object(visible, "_disassemble", return_value=instructions):
        return visible.Model(FakeExe(rodata, vma, {"proc": (lo, hi - lo)}), ("proc",))


class WhatIsShown(unittest.TestCase):
    """V03, V10: what the model prints, on a made-up binary (no compiler: the layout is exactly the one asserted)."""

    def test_a_jump_table_shows_its_entries_and_nothing_after_them(self):
        """V10: Ghidra turns the table into the cases of a switch; it prints no window at the table, so the bytes after the last entry are not shown."""
        entries = b"".join((target - 0x2000).to_bytes(4, "little", signed=True) for target in (0x1010, 0x1020, 0x1030, 0x1040))
        rodata = entries + b"\xaa" * 100
        ins = [(0x1000, "lea    rax,[rip+0x1000]        # 2000 <tbl>"), (0x1007, "movsxd rdx,DWORD PTR [rax+rdx*4]"), (0x100b, "add    rax,rdx"), (0x100e, "jmp    rax")]
        model = model_of(rodata, ins)
        self.assertEqual(model.jump_tables, [(0x2000, 0x1000, 0x1100)])
        self.assertEqual(model.refs, [])
        self.assertEqual(model.shown, set(range(0x2000, 0x2010)))
        self.assertEqual(model.hidden(0x2000, 0x2014), [0x2010, 0x2011, 0x2012, 0x2013])

    def test_a_string_is_shown_up_to_its_nul_and_not_beyond(self):
        """V03: a printable run is shown whole, however long (more than the 64-byte window), but the bytes after its NUL are another datum."""
        rodata = b"A" * 100 + b"\0" + b"\xff" * 200
        model = model_of(rodata, [(0x1000, "lea    rdi,[rip+0xfff]        # 2000 <s>")])
        self.assertEqual(model.refs, [0x2000])
        self.assertEqual(model.shown, set(range(0x2000, 0x2000 + 101)), "the 100 characters and the NUL")
        self.assertEqual(model.hidden(0x2000, 0x2000 + 105), [0x2000 + i for i in range(101, 105)])

    def test_a_table_that_is_not_a_string_is_shown_for_one_window(self):
        rodata = bytes(range(1, 200))
        model = model_of(rodata, [(0x1000, "lea    rdi,[rip+0xfff]        # 2000 <t>")])
        self.assertEqual(model.shown, set(range(0x2000, 0x2000 + visible.WINDOW)))


# ---- m2: the length rule ----------------------------------------------------------------------------------------------------------------

class LengthRule(ModalCase):
    def test_the_chat_template_is_counted_on_top_of_the_messages(self):
        """L03: `LENGTH_OVERHEAD` = 0 kept a record the template makes longer than the window (it adds about 20 tokens: three role markers, the empty thinking block, the end of turn)."""
        self.assertGreaterEqual(sft.LENGTH_OVERHEAD, 24)
        window = 100
        record = lambda n: {"task": "t0", "kind": "a", "messages": [{"role": "system", "content": "w " * (n // 2)}, {"role": "user", "content": "w " * (n - n // 2)},
                                                                    {"role": "assistant", "content": "x"}]}
        stats = {"tasks": 2, "kept_tasks": 2, "kept_records": 2, "by_kind": {"a": {"tasks": 2, "kept_tasks": 2, "kept_records": 2}}}
        kept, st = sft.drop_over_length([record(window - 20), {**record(window - 40), "task": "t1"}], stats, lambda text: len(text.split()), window)
        self.assertEqual([r["task"] for r in kept], ["t1"], f"{window - 20} + 1 measured tokens plus the template do not fit in {window}")
        self.assertEqual(st["over_length"]["dropped"], 1)

    def test_the_sampling_servers_tokenizer_is_asked_for_the_text_without_special_tokens(self):
        """L09: the training run tokenizes the text and adds the template itself, so the count must not add BOS/EOS on top (`add_special` false)."""
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps({"tokens": [1, 2, 3]}).encode())) as urlopen:
            self.assertEqual(self.data._server_token_count("a b c"), 3)
        req = urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8080/tokenize")
        self.assertEqual(json.loads(req.data), {"content": "a b c", "add_special": False})


if __name__ == "__main__":
    unittest.main()

"""Round 8, the generators, the gates and the data stage: M2 (1)-(3), M3, m1, m11, m13, m14, m15.

M2   (1) the fact gate and (2) the object gate compile with the TASK's compiler, (3) `Task.meta()` keeps the facts and the object gate.
M3   `refdata` prints every byte it reads (the hex of the first 32 hid the second half of a clang state-machine table); the end-to-end half flips every `.rodata`
     byte of two clang state machines and asks whether `refdata` prints each one the behaviour depends on.
m1   a legacy lcg task whose multiplier the compiler removed is not trained on.
m11  the answers of the free-string variants are not shared with the held-out split. m14 a slot that needs more than 60 draws resamples.
m15  a second checkpoint of the same second does not replace the first. m13 the 11 survivors of the reviewer's code mutations get their tests.
"""
from __future__ import annotations

import collections
import itertools
import os
import pathlib
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP, refdata_module as _refdata, tasks_of

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, dedupe, families, outputs, scoring, survive, targets  # noqa: E402


# ---- M2 (1)-(3): the gates run at the task's own compiler, the task record keeps its facts -------------------------------------------------

def _recording_run(real, seen: list):
    """subprocess.run that records the compiler of every compile and, where `clang` is not installed, runs `gcc` in its place (the argv recorded is the one asked for)."""
    def run(argv, *a, **k):
        if isinstance(argv, (list, tuple)) and argv and os.path.basename(str(argv[0])) in ("gcc", "clang", "cc"):
            seen.append(os.path.basename(str(argv[0])))
            if argv[0] == "clang" and not shutil.which("clang"):
                argv = ["gcc", *argv[1:]]
        return real(argv, *a, **k)
    return run


@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class GatesCompileWithTheTasksCompiler(unittest.TestCase):
    """Round 8, M2, C28 and C29: a gate that always compiled with gcc passed 15% of the clang crc tasks and 37% of the clang strobf tasks whose fact is not in the clang
    object. The shipped code passes `cc`; nothing noticed when it did not."""

    def compilers(self, fn):
        seen: list = []
        with mock.patch.object(survive.subprocess, "run", _recording_run(survive.subprocess.run, seen)), mock.patch.object(survive, "toolchain", lambda cc: "/usr/bin/" + cc):
            fn()
        return seen

    def test_the_fact_gate_compiles_every_object_with_the_tasks_cc_and_its_probe_marker_too(self):
        for cc in ("clang", "gcc"):
            ts = [t for t in families.generate("crc", 36, 7, cc=cc) if t.variant.endswith("init")][:3]
            self.assertTrue(ts and all(t.cc == cc and t.facts for t in ts))
            for t in ts:
                seen = self.compilers(lambda: survive.check_task(t))
                self.assertGreaterEqual(len(seen), 2, "the object and the marker build")
                self.assertEqual(set(seen), {cc}, f"{t.id}: the fact gate compiled with {set(seen)}, the task is {cc}")

    def test_the_gate_object_of_the_local_pipeline_gets_the_same_compiler(self):
        t = next(t for t in families.generate("crc", 36, 7, cc="clang") if t.facts)
        seen = self.compilers(lambda: survive.Gate()(t))
        self.assertEqual(set(seen), {"clang"})

    def test_the_object_gate_compiles_with_the_tasks_cc(self):
        for cc in ("clang", "gcc"):
            ts = [t for t in families.generate("strobf", 70, 7, cc=cc) if t.object_gate][:3]
            self.assertTrue(ts)
            for t in ts:
                seen = self.compilers(lambda: survive.check_task(t))
                self.assertEqual(set(seen), {cc}, f"{t.id}: the object gate compiled with {set(seen)}, the task is {cc}")

    def test_the_gate_uses_the_opt_level_of_the_task_too(self):
        t = next(t for t in families.generate("crc", 60, 7, cc="gcc") if t.facts and t.opt == "-Os")
        argv: list = []
        real = survive.subprocess.run

        def run(a, *args, **k):
            argv.append(list(a))
            return real(a, *args, **k)

        with mock.patch.object(survive.subprocess, "run", run):
            survive.check_task(t)
        self.assertTrue(any(a[1] == "-Os" for a in argv if a[0] in ("gcc", "clang")))


class TaskRecordKeepsItsGates(unittest.TestCase):
    def test_meta_has_the_facts_and_the_object_gate_of_every_task_that_has_them(self):
        """Round 8, M2, C27: `Task.meta()` without `facts` makes the Modal-side re-check check nothing."""
        seen = collections.Counter()
        for fam in families.modules():
            if fam == "legacy":
                continue
            for t in families.generate(fam, 60, 7):
                m = t.meta()
                if t.facts:
                    seen["facts"] += 1
                    self.assertEqual(m.get("facts"), t.facts, t.id)
                else:
                    self.assertNotIn("facts", m)
                if t.object_gate:
                    seen["gate"] += 1
                    self.assertEqual(m.get("object_gate"), t.object_gate, t.id)
        self.assertGreater(seen["facts"], 100)
        self.assertGreater(seen["gate"], 10)


# ---- M3 --------------------------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(GCC_OBJDUMP and shutil.which("readelf"), "gcc, objdump and readelf are needed")
class RefData(unittest.TestCase):
    def test_every_byte_it_reads_is_printed(self):
        """Round 8, M3: it read 64 bytes and printed the first 32: of a 40-byte table of ints (clang -O3 keeps one per state) the entries of index 8 and 9 were never shown."""
        rd = _refdata()
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d, "t.c")
            src.write_text("const unsigned char T[64] = {" + ",".join(str(i + 1) for i in range(64)) + "};\n"
                           "int proc(int i) { return T[i]; }\nint main(void) { return proc(0); }\n")
            exe = pathlib.Path(d, "t")
            subprocess.run(["gcc", "-O0", "-pie", "-fPIE", "-o", str(exe), str(src)], check=True)
            nm = subprocess.run(["nm", str(exe)], capture_output=True, text=True).stdout
            addr = int(next(l.split()[0] for l in nm.splitlines() if l.endswith(" T")), 16)
            line = rd.dump(str(exe), 0x100000 + addr)
        got = [int(x, 16) for x in line.split(":", 1)[1].split("|")[0].split()]
        self.assertEqual(got, list(range(1, 65)), "all 64 bytes of the window, in order")
        self.assertEqual(rd.N, 64)


class FsmTablesAreShown(unittest.TestCase):
    """Round 8, M3 (the end-to-end half): which bytes of `.rodata` does the behaviour of a state machine depend on (flip one byte of the linked binary, run every string up to
    the shortest accepted length), and are they all in what `refdata` prints for the data labels that proc's code refers to (`lea reg, [rip + X]`; a jump table is a
    switch that Ghidra resolves)? `fsm-3131-0019` (clang -O3) keeps a 40-byte table per state: with 32 printed bytes, 24 needed bytes were never shown."""

    CASES = (("fsm-3131-0019", 3131, "clang", "-O3"), ("fsm-4049-0037", 4049, "clang", "-Os"))
    # no string literal and no printf: the harness has no data of its own, so every byte of `.rodata` that changes the output belongs to the function under test
    HARNESS = ("#include <stdio.h>\n#include <signal.h>\n#include <unistd.h>\nextern int proc(const char *);\nstatic void die(int s) { (void)s; _exit(120); }\n"
               "int main(void) {\n    char b[64];\n    int c, n = 0;\n    signal(SIGSEGV, die); signal(SIGBUS, die); signal(SIGILL, die); signal(SIGFPE, die);\n    for (;;) {\n        c = getchar();\n"
               "        if (c == EOF) break;\n        if (c == 10) {\n            b[n] = 0;\n            int r = proc(b);\n            unsigned u = r < 0 ? (unsigned)-r : (unsigned)r;\n"
               "            char o[16];\n            int k = 0;\n            if (r < 0) putchar(45);\n            do { o[k++] = (char)(48 + u % 10); u /= 10; } while (u);\n"
               "            while (k) putchar(o[--k]);\n            putchar(10);\n            n = 0;\n            continue;\n        }\n        b[n++] = (char)c;\n    }\n    return 0;\n}\n")

    def analyse(self, t, cc, opt) -> tuple:
        """(needed addresses, the needed ones that refdata does not print, the needed ones it would not print with only 32 bytes of hex)."""
        rd = _refdata()
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "t.c").write_text(t.source())
            (d / "h.c").write_text(self.HARNESS)
            subprocess.run([cc, opt, "-w", "-fPIE", "-Dmain=stub_main", "-c", "-o", str(d / "t.o"), str(d / "t.c")], check=True, capture_output=True)
            subprocess.run(["gcc", "-O0", "-w", "-fPIE", "-c", "-o", str(d / "h.o"), str(d / "h.c")], check=True, capture_output=True)
            exe = d / "t"
            subprocess.run(["gcc", "-pie", "-o", str(exe), str(d / "t.o"), str(d / "h.o")], check=True, capture_output=True)
            dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-M", "intel", str(exe)], capture_output=True, text=True).stdout
            ins, cur = [], None
            for ln in dis.splitlines():
                m = re.match(r"^[0-9a-f]+ <([^>]+)>:$", ln.strip())
                if m:
                    cur = m.group(1)
                elif cur == "proc" and "\t" in ln:
                    ins.append(ln.split("\t", 1)[1].strip())
            shown, shown32, tables = set(), set(), []
            for k, tx in enumerate(ins):
                m = re.match(r"lea\s+(\w+),\[rip\+0x[0-9a-f]+\]\s+# ([0-9a-f]+)", tx)
                if not m:
                    continue
                rest = " ".join(re.sub(r"#.*", "", x) for x in ins[k + 1:])
                if re.search(r"movsxd\s+\w+,DWORD PTR \[" + m.group(1) + r"\+\w+\*4\]", rest) and re.search(r"jmp\s+r[a-z0-9]+\b", rest):
                    tables.append(int(m.group(2), 16))
                else:
                    addr = int(m.group(2), 16)
                    line = rd.dump(str(exe), 0x100000 + addr)
                    n = len(line.split(":", 1)[1].split("|")[0].split())
                    shown |= set(range(addr, addr + n))
                    shown32 |= set(range(addr, addr + min(n, 32)))
            hdr = subprocess.run(["objdump", "-h", str(exe)], capture_output=True, text=True).stdout
            m = re.search(r"\.rodata\s+([0-9a-f]+)\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)", hdr)
            size, vma, foff = (int(x, 16) for x in m.groups())
            raw = exe.read_bytes()
            pstart, pend = None, None
            for base in tables:                       # the entries of a jump table that point into proc: Ghidra turns them into the cases of a switch
                n = 0
                while base - vma + 4 * n + 4 <= size:
                    off = int.from_bytes(raw[foff + base - vma + 4 * n:foff + base - vma + 4 * n + 4], "little", signed=True)
                    if not (0x1000 <= base + off < vma):             # a code address (.text lies below .rodata in this layout)
                        break
                    n += 1
                shown |= set(range(base, base + 4 * n))
                shown32 |= set(range(base, base + 4 * n))
            alpha = t.params["alphabet"]
            strs = [""] + ["".join(p) for n in range(1, {1: 8, 2: 7, 3: 5, 4: 5}[len(alpha)] + 1) for p in itertools.product(alpha, repeat=n)]
            inp = "".join(x + "\n" for x in strs).encode()
            nocore = lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0))          # a flipped byte may crash the program: no core dump
            base_out = subprocess.run([str(exe)], input=inp, capture_output=True, preexec_fn=nocore).stdout
            needed = []
            for off in range(size):
                data = bytearray(raw)
                data[foff + off] ^= 0xff
                (d / "p").write_bytes(data)
                (d / "p").chmod(0o755)
                out = subprocess.run([str(d / "p")], input=inp, capture_output=True, preexec_fn=nocore, timeout=20).stdout
                if out != base_out:
                    needed.append(vma + off)
            return needed, sorted(a for a in needed if a not in shown), sorted(a for a in needed if a not in shown32)

    @unittest.skipUnless(GCC_OBJDUMP and shutil.which("clang") and shutil.which("readelf"), "gcc, clang, objdump and readelf are needed")
    def test_every_data_byte_the_behaviour_depends_on_is_in_what_the_model_is_shown(self):
        for tid, seed, cc, opt in self.CASES:
            t = next(t for t in families.generate("fsm", 60, seed) if t.id == tid)
            self.assertEqual(t.variant, "switch_input")
            needed, missing, would_miss = self.analyse(t, cc, opt)
            self.assertGreater(len(needed), 40, "the machine keeps its transitions in data at this cc/opt, else this test shows nothing")
            self.assertEqual(missing, [], f"{tid} ({cc} {opt}): {len(missing)} of {len(needed)} needed bytes are not in what refdata prints")
            self.assertGreater(len(would_miss), 0, f"{tid}: with 32 printed bytes it was 20-24 bytes short; the test must see that")

# ---- m1 --------------------------------------------------------------------------------------------------------------------------------

class LcgWithoutAMultiplier(unittest.TestCase):
    FOLDED = [("lcg-2027-0013", 2027), ("lcg-2027-0453", 2027), ("lcg-4049-1293", 4049), ("lcg-7-1043", 7), ("lcg-7-1833", 7), ("lcg-7-2043", 7)]

    def legacy(self, seed):
        return {t.id: t for t in families.generate("legacy", 3000, seed)}

    def test_the_tasks_whose_rounds_the_compiler_multiplied_out_are_not_trained_on(self):
        """Round 8, m1: `lcg-2027-0013` (gcc -O2) is `imul eax, edi, 0x979e791` = 1664525^4 mod 2^31: no multiplier and no round count in the code."""
        by = {s: self.legacy(s) for s in (2027, 4049, 7)}
        for tid, seed in self.FOLDED:
            why = families.untrainable(by[seed][tid], seed)
            self.assertIn("legacy lcg", why or "", tid)
        held = by[2027]["lcg-2027-0013"]
        self.assertIsNone(families.untrainable(held, core.HELDOUT_SEED), "the held-out seed is not filtered")
        self.assertIsNone(families.untrainable(by[2027]["lcg-2027-0173"], 2027), "an increment of 1013904223: three imul show the rounds")

    def test_generate_many_drops_them_and_reports_why(self):
        dropped: list = []
        kept = families.generate_many(["legacy"], 3000, 7, dropped=dropped)
        ids = {t.id for t, why in dropped if "lcg" in why}
        self.assertTrue({"lcg-7-1043", "lcg-7-1833", "lcg-7-2043"} <= ids)
        self.assertFalse(ids & {t.id for t in kept})

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_with_the_gate_no_kept_lcg_task_lacks_its_multiplier_in_the_compiled_code(self):
        gate = survive.Gate()
        dropped: list = []
        kept = families.generate_many(["legacy"], 1500, 4049, gate=gate, dropped=dropped)
        lcgs = [t for t in kept if t.kind == "lcg"]
        self.assertGreater(len(lcgs), 100)
        with tempfile.TemporaryDirectory() as d:
            for t in lcgs:
                obj = survive._compile(t.source(), "gcc", t.opt, d, "o.o")
                nums = survive.numbers(survive._objdump(["-d", "--no-show-raw-insn", "-M", "intel"], obj))
                self.assertTrue(core.forms(int(t.truth["accept"][0])) & nums, f"{t.id}: the multiplier is not in the code")

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_the_gate_itself_refuses_a_folded_task_whatever_the_rule_says(self):
        t = self.legacy(2027)["lcg-2027-0013"]
        ok, why = survive.Gate().legacy_visible(t)
        self.assertFalse(ok, why)
        self.assertEqual(survive.legacy_facts("lcg", t.truth["accept"]), [[1664525]])
        self.assertEqual(survive.legacy_facts("hash", ["fnv"]), [], "only lcg has a fact")


# ---- m11 -------------------------------------------------------------------------------------------------------------------------------

class AnswersOfFreeStringsAreHeldOut(unittest.TestCase):
    def test_a_training_task_never_shares_its_answer_with_a_held_out_task_of_the_variant(self):
        """Round 8, m11: strobf 4 of 120, decode_secret 3 of 24, antidebug/env 9 of 20, lcg 6 of 12 held-out answers were also the answer of a kept training task."""
        held = dedupe.heldout_answers()
        self.assertGreater(len(held), 1000)
        names = ["strobf", "multi", "antidebug", "legacy"]
        kept, dropped = dedupe.training_tasks(names, 4400, 7)
        scoped = [t for t in kept if dedupe.answer_scoped(t)]
        self.assertGreater(len(scoped), 800)
        self.assertFalse([t.id for t in scoped if dedupe.answer_key(t) in held], "a kept task has a held-out answer")
        by = collections.Counter(f"{t.kind}/{t.variant}" if t.variant else t.kind for t in dropped if dedupe.answer_key(t) in held and dedupe.task_key(t) not in dedupe.heldout_keys())
        for key in ("antidebug/env", "lcg"):
            self.assertGreater(by[key], 0, f"{key}: nothing was dropped for its answer")

    def test_the_numeric_variants_of_a_small_answer_space_are_not_thinned(self):
        """An offset, a size or a count is in every training set (that is what `chance` measures): the rule does not touch them."""
        for fam in ("structs", "pktsum", "fsm", "cipher", "crc"):
            tasks = families.generate_many([fam], 600, 7)
            kept, dropped = dedupe.training_tasks([fam], 600, 7)
            self.assertEqual(len(tasks), len(kept) + len(dropped))
            self.assertFalse([t for t in dropped if dedupe.answer_scoped(t)], fam)
        self.assertFalse(any(dedupe.answer_scoped(t) for t in families.generate("structs", 60, 7)))

    def test_drop_heldout_without_the_answers_would_keep_them(self):
        ts = [t for t in families.generate("antidebug", 600, 7) if t.variant == "env"]
        held = dedupe.heldout_answers()
        clash = [t for t in ts if dedupe.answer_key(t) in held]
        self.assertTrue(clash)
        kept, _ = dedupe.drop_heldout(ts, set())
        self.assertEqual(len(kept), len(ts))
        kept, dropped = dedupe.drop_heldout(ts, set(), held)
        self.assertEqual(len(dropped), len(clash))

    def test_the_decompile_run_says_how_many_it_dropped_for_the_answer(self):
        report: dict = {}
        dedupe.training_tasks(["antidebug"], 600, 7, report=report)
        self.assertGreater(report["answer_overlap"].get("antidebug/env", 0), 0)


# ---- m14, m15 --------------------------------------------------------------------------------------------------------------------------

class Resample(unittest.TestCase):
    def test_a_slot_that_needs_more_than_60_draws_resamples_instead_of_failing(self):
        """Round 8, m14: bounds/char_class rejects 75% of the draws of a cc/opt cell; 60 failed draws in a row (3e-8 per task) raised RuntimeError and stopped the run."""
        from rea_bench import f_bounds
        calls = []

        def gate(t):
            calls.append(t.id)
            return len(calls) > 150

        out = core.build("bounds", f_bounds.VARIANTS, 1, 7, gate=gate)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(calls), 151)
        again = core.build("bounds", f_bounds.VARIANTS, 1, 7, gate=lambda t, c=[0]: (c.__setitem__(0, c[0] + 1), c[0] > 150)[1])
        self.assertEqual(out[0].body, again[0].body, "the same draw is kept: the resampling is deterministic")

    def test_a_gate_that_no_draw_can_pass_still_ends_with_an_error(self):
        from rea_bench import f_bounds
        calls = []
        with self.assertRaisesRegex(RuntimeError, "no draw in"):
            core.build("bounds", f_bounds.VARIANTS, 1, 7, gate=lambda t: calls.append(1) and False)
        self.assertEqual(len(calls), core.MAX_TRIES)
        self.assertGreaterEqual(core.MAX_TRIES, 1000)


class CheckpointAside(unittest.TestCase):
    def test_two_old_checkpoints_of_the_same_second_are_both_kept(self):
        """Round 8, m15: `.old-<mtime>` of two checkpoints written in one second was one name, and `rename` replaced the first with the second."""
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d, "sft-m-seed7-legacy.partial.jsonl")
            for text in ("first\n", "second\n", "third\n"):
                p.write_text(text)
                os.utime(p, (1_700_000_000, 1_700_000_000))
                self.assertEqual(outputs.load_partial(p, resume=False), {})
                self.assertFalse(p.exists())
            kept = sorted(q.name for q in pathlib.Path(d).glob("*.old-*"))
            self.assertEqual(len(kept), 3, kept)
            self.assertEqual(sorted(pathlib.Path(d, k).read_text() for k in kept), ["first\n", "second\n", "third\n"])


# ---- m13: the survivors of the reviewer's code mutations (nothing is wrong today; a regression would not have been noticed) ---------------------------------------

class Hardening(unittest.TestCase):
    def test_c02_no_decoy_literal_is_an_instruction_form_of_an_answer_or_a_fact_alternative(self):
        """C02: the ban was tested for the value only; a decoy `1` is `forms(0xffff)` (the 16-bit pattern of -0xffff) in an instruction."""
        lit = re.compile(r"\b(0x[0-9a-fA-F]+|\d+)[uUlL]*\b")
        seen = 0
        for fam in [f for f in families.modules() if f != "legacy"]:
            m = families.modules()[fam]
            for t in families.generate(fam, 60 * len(m.VARIANTS), 5):
                dl = "\n".join(l for l in t.body.split("\n") if l.rstrip().endswith(core.DECOY_MARK))
                if not dl:
                    continue
                seen += 1
                dlits = {int(x, 16) if x.lower().startswith("0x") else int(x) for x in lit.findall(dl)}
                keys = {scoring.num_value(a.lower()) for g in t.truth["accept"] for a in (g if isinstance(g, list) else [])} - {None}
                keys |= {a for g in t.facts for a in survive.split_group(g)[0] if isinstance(a, int)}
                banned = set().union(*(core.forms(v) | {v} for v in keys)) if keys else set()
                self.assertFalse(dlits & banned, f"{t.id}: a decoy literal {sorted(dlits & banned)} is an answer or one of its instruction forms")
        self.assertGreater(seen, 800)

    def test_c03_a_redraw_keeps_the_flags_of_the_first_draw_for_the_first_attempts(self):
        for fam, seed, i in (("bounds", 7, 3), ("crc", 9, 11), ("strobf", 3, 40)):
            first = core.pick_flags(fam, seed, i)
            for a in range(1, 8):
                self.assertEqual(core.pick_flags(fam, seed, i, attempt=a), first, (fam, a))
        self.assertEqual(core.FLAG_PATIENCE, 8)
        flags = {core.pick_flags("bounds", 7, 3, attempt=a) for a in range(8, 38)}
        self.assertGreater(len(flags), 3, "after the patience the flags are drawn again")

    def test_c09_cheap_constant_is_the_set_of_powers_of_two_their_neighbours_and_3_5_9_times_a_power(self):
        cheap = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 20, 24, 31}          # 2^k-1, 2^k, 2^k+1, 3*2^k, 5*2^k, 9*2^k up to 31
        self.assertEqual({v for v in range(1, 32) if survive.cheap_constant(v)}, cheap)
        self.assertEqual({v for v in range(1, 32) if not survive.cheap_constant(v)}, {11, 13, 14, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30})

    def test_c16_a_held_out_run_of_one_task_more_than_the_listed_range_is_refused(self):
        dedupe.check_heldout_range(["crc"], dedupe.HELDOUT_N, core.HELDOUT_SEED)
        with self.assertRaises(SystemExit):
            dedupe.check_heldout_range(["crc"], dedupe.HELDOUT_N + 1, core.HELDOUT_SEED)
        dedupe.check_heldout_range(["crc"], 5000, 7)

    def test_c21_c23_the_held_out_keys_hold_the_pinned_list_and_every_attempt_of_every_slot(self):
        """Each alone is masked by the other (a missing pinned list, a `draws` that starts at attempt 1): together they leak."""
        self.assertLessEqual(dedupe.load_heldout(), dedupe.heldout_keys())
        for fam in ("crc", "structs", "strobf"):
            k = 3
            gen = [dedupe.task_key(t) for t in families.generate(fam, 12, core.HELDOUT_SEED)]
            drawn = families.draws(fam, 12, core.HELDOUT_SEED, k)
            self.assertEqual(len(drawn), 12 * k)
            self.assertEqual([dedupe.task_key(t) for t in drawn[::k]], gen, "attempt 0 of every slot is the ungated task")
            self.assertTrue(set(gen) <= dedupe.heldout_keys())

    def test_d06_no_vocabulary_set_holds_a_negation_and_a_negated_exact_answer_is_not_an_answer(self):
        from rea_bench import grammar
        for name in ("GLUE", "LEAD_WORDS", "NOUNS_Q", "NOUNS_N"):
            words = getattr(grammar, name)
            self.assertFalse(words & {"not", "no", "never", "except", "unless", "or", "nor", "but"}, name)
        for head in ("Not: abc", "This is not: abc", "The string is not: abc", "not abc"):
            self.assertFalse(grammar.is_exact_answer(head, "abc"), head)
        self.assertTrue(grammar.is_exact_answer("The string is: abc", "abc"))

    def test_d40_a_second_line_of_numbers_is_a_candidate_list_not_prose(self):
        t = next(t.meta() for t in families.generate("bounds", 60, 7) if t.variant == "loop_iterations")
        key = scoring.canonical(scoring.strict_task(t))
        self.assertTrue(scoring.score_strict(key, t))
        for reply in (f"{key}\n41 42 43 44", f"{key}\n{key} 41 42 43", f"{key}\n1 2 3 4 5"):
            self.assertFalse(scoring.score_strict(reply, t), reply)
        self.assertTrue(scoring.score_strict(f"{key}\nThis is what the function returns for that argument.", t))

    def test_d10_check_target_raises_on_a_text_its_scorer_rejects(self):
        t = tasks_of("bounds", "loop_iterations")[0].meta()
        scoring.check_target(t)
        bad = {"text": "nothing like the answer", "key": "bounds/loop_iterations", "headline": "x", "sentence": "y"}
        with mock.patch.object(targets, "render", lambda *a, **k: bad):
            with self.assertRaisesRegex(ValueError, "score_strict rejects"):
                scoring.check_target(t)

    def test_d09_d32_an_error_text_with_leading_whitespace_is_not_a_reply_and_a_hex_answer_has_no_decimal_spelling(self):
        for reply in ("[error URLError('x')]", " \n[error x]", "\t[error x]"):
            self.assertTrue(scoring.is_error_reply(reply), reply)
        self.assertFalse(scoring.is_error_reply("0x12 [error"))
        self.assertNotIn("18", core.hx(0x12))          # (D32 is equivalent as written: hx filters every bare spelling that is not hex-only-by-letter, so it cannot be killed)
        self.assertIn("0x12", core.hx(0x12))
        self.assertIn("18", core.dec(18))

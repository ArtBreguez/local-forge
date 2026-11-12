"""Round 9, the data and the sample stage: m1 (no teacher text in the SFT record), m2 (the length rule at the sample stage), m3 (array_stride: the stride is a literal outside the stack).

m1  `teacher_headline` was a field of every SFT record: the raw first headline of the teacher, in the file that is uploaded as the dataset. It is gone: a record holds
    `messages, task, kind, model, target_source`, and `sft.check_records` refuses anything else.
m2  `modal_lora.encode_ids` drops a pair over 2,048 tokens at training time; the sample stage now measures the same thing (the sampling server's tokenizer, the same window) before the
    file is written, drops those records and prints the count.
m3  gcc -O0 scales the index of a 24-byte record with `lea` and `shl` and has a frame slot at [rbp-0x18]: the presence test took the slot for the stride (11 of 14 gcc -O0 tasks that
    passed it). The stride must be a literal of an instruction that does not name the stack or frame pointer.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP
from .rea_tasks import meta as real_task, right
from .test_rea_modal import ModalCase, StubbedSampleStage

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from distill.backends import modal_sample  # noqa: E402
from rea_bench import families, scoring, survive, targets  # noqa: E402
from .rea_bound import sft  # noqa: E402


# ---- m1 -------------------------------------------------------------------------------------------------------------------------------

class NoTeacherText(unittest.TestCase):
    MARK = "ZZQQMARKXY"

    def replies(self, t):
        k = right(t)
        return [f"**{k}**\n\n{self.MARK} says the pointer is read twice. It is a doubly linked list.", f"Answer:\n`{k}`\nIt is the {self.MARK} field.", f"The answer is {k}."]

    def test_a_record_holds_the_rendered_text_and_no_word_of_the_teacher(self):
        t = real_task("structs", "linked_next", chance=0.01)
        outs = [(r, True) for r in self.replies(t)]
        recs, why = sft.select(outs, t)
        self.assertEqual(why, "kept")
        kept, stats = sft.assemble([t], {t["id"]: {"outs": [list(o) for o in outs]}}, "m", lambda task: [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
        for obj in (recs, kept, stats):
            self.assertNotIn(self.MARK, json.dumps(obj))
            self.assertNotIn("teacher", json.dumps(obj).lower())
        self.assertEqual(sorted(kept[0]), sorted(sft.RECORD_KEYS))
        self.assertEqual(kept[0]["messages"][-1]["content"], targets.render(t)["text"])
        self.assertNotIn("**", json.dumps(kept), "the markdown of the teacher's headline is not stored")
        self.assertNotIn("`", json.dumps(kept))

    def test_the_target_of_a_reply_has_no_teacher_field(self):
        t = real_task("structs", "linked_next")
        r = scoring.sft_target(self.replies(t)[0], t)
        self.assertEqual(sorted(r), ["headline", "sentence", "target_source", "template", "text"])

    def test_check_records_refuses_a_record_with_another_key(self):
        good = {"messages": [], "task": "x", "kind": "k", "model": "m", "target_source": "truth-template"}
        sft.check_records([good])
        with self.assertRaises(ValueError):
            sft.check_records([{**good, "teacher_headline": "0x8"}])
        with self.assertRaises(ValueError):
            sft.check_records([{k: v for k, v in good.items() if k != "model"}])

    def test_assemble_checks_what_it_returns(self):
        self.assertEqual(sft.RECORD_KEYS, ("messages", "task", "kind", "model", "target_source"))
        self.assertIn("check_records(kept)", __import__("inspect").getsource(sft.assemble), "every path to the SFT file goes through assemble")


class SampleStageLengthRule(StubbedSampleStage):
    def make_long(self, tid, words):
        from rea_bench import outputs
        p = pathlib.Path(self.root, "data", outputs.decompiled_stem(self.tag), f"{tid}.json")
        m = json.loads(p.read_text())
        m["pseudocode"] = f"/* {tid} */ void f(void) {{ /* " + "word " * words + "*/ }"          # the stub server finds the task in the prompt by its id
        p.write_text(json.dumps(m))

    def test_a_record_over_the_window_is_dropped_and_counted(self):
        before = self.run_stage(resume=False)["sft"]
        tid = before[3]["task"]
        self.make_long(tid, 2100)
        res = self.run_stage(resume=False)
        ids = [r["task"] for r in res["sft"]]
        self.assertNotIn(tid, ids)
        self.assertEqual(len(ids), len(before) - 1)
        o = res["stats"]["over_length"]
        kind = self.tasks[tid]["kind"]
        self.assertEqual(o["max_len"], self.data.MAX_LEN)           # one assertion each: `self.fail` is a lambda in StubbedSampleStage, a tuple diff would crash in it
        self.assertEqual(o["dropped"], 1)
        self.assertEqual(o["by_kind"], {kind: 1})
        self.assertNotIn("error", o)
        self.assertEqual(res["stats"]["kept_records"], len(before) - 1)
        self.assertEqual(res["stats"]["by_kind"][kind]["kept_records"], sum(r["kind"] == kind for r in before) - 1)
        self.assertGreater(o["longest_tokens"], self.data.MAX_LEN)
        self.assertIn(f"1 of {len(before)} records are longer than 2048 tokens", sft.over_length_line(res["stats"]))
        self.assertTrue(any("word word" in t for t in self.tokenized), "the measurement went through the sampling server's /tokenize")

    def test_the_window_is_the_training_one_and_a_short_set_loses_nothing(self):
        import modal_lora
        self.assertIs(self.data.MAX_LEN, modal_lora.MAX_LEN)
        self.assertEqual(self.data.MAX_LEN, 2048)
        res = self.run_stage(resume=False)
        self.assertEqual(res["stats"]["kept_records"], len(res["sft"]))
        self.assertGreaterEqual(len(res["sft"]), 6)
        self.assertEqual(res["stats"]["over_length"]["dropped"], 0)
        self.assertEqual(res["stats"]["over_length"]["by_kind"], {})

    def test_a_measurement_that_fails_keeps_the_records_and_says_so(self):
        base = len(self.run_stage(resume=False)["sft"])
        with mock.patch.object(modal_sample, "_server_token_count", side_effect=OSError("tokenize refused")):
            res = self.run_stage(resume=False)
        self.assertEqual(len(res["sft"]), base)
        self.assertIn("tokenize refused", res["stats"]["over_length"]["error"])
        self.assertTrue(sft.over_length_line(res["stats"]).startswith("WARNING"))

    def test_the_entrypoint_prints_the_count_after_a_sample_run(self):
        stats = {"tasks": 5, "answers": 20, "errors": 0, "correct": 4, "solved_tasks": 4, "kept_tasks": 3, "kept_records": 3, "agreement": 0, "by_kind": {},
                 "over_length": {"max_len": 2048, "dropped": 2, "near_limit": 1, "longest_tokens": 2600, "by_kind": {"legacy": 2}}}
        with tempfile.TemporaryDirectory() as d:
            self.run_main(pathlib.Path(d), remote_result={"sft": [], "stats": stats}, stage="sample", model="heretic-9b", seed=7, family="legacy")
        self.assertIn("over-length: 2 of 5 records are longer than 2048 tokens", self.printed)
        self.assertIn("(legacy 2)", self.printed)


class LengthRulePure(unittest.TestCase):
    def records(self, sizes):
        return [{"task": f"t{i}", "kind": "a" if i % 2 else "b", "messages": [{"role": "system", "content": "w " * (n // 2)}, {"role": "user", "content": "w " * (n - n // 2)},
                                                                                {"role": "assistant", "content": "x"}]} for i, n in enumerate(sizes)]

    def stats(self, n):
        row = lambda c: {"tasks": c, "kept_tasks": c, "kept_records": c}
        return {**row(n), "by_kind": {"a": row(n // 2), "b": row(n - n // 2)}}

    def test_the_window_is_inclusive_and_what_is_dropped_leaves_the_counts(self):
        count = lambda text: len(text.split())
        edge = 100 - sft.LENGTH_OVERHEAD - 1                       # + the one token of the assistant message + the overhead = exactly 100
        kept, st = sft.drop_over_length(self.records([edge, edge + 1, 500, 10]), self.stats(4), count, 100)
        self.assertEqual([r["task"] for r in kept], ["t0", "t3"])
        o = st["over_length"]
        self.assertEqual((o["dropped"], o["by_kind"], o["near_limit"], o["longest_tokens"]), (2, {"a": 1, "b": 1}, 1, 500 + 1 + sft.LENGTH_OVERHEAD))
        self.assertEqual((st["kept_tasks"], st["kept_records"], st["by_kind"]["a"]["kept_records"], st["by_kind"]["b"]["kept_records"]), (2, 2, 1, 1))
        self.assertEqual(self.stats(4)["kept_records"], 4, "the stats that came in are not changed")


# ---- m3 -------------------------------------------------------------------------------------------------------------------------------

_PREFIX_WORDS = ("bnd", "notrack", "rep", "repz", "repnz", "lock", "data16", "cs", "ds")


def literal_lines(body: str, size: int) -> list:
    """The lines of a `proc` disassembly (objdump -d -M intel) that name `size` as a number of the program, read independently of survive.numbers: no instruction that touches the
    stack or frame pointer, and (round 10, m1) not the target of a direct jump, call or loop (`jne 40 <proc+0x40>` is a branch to 0x40) and not a rip-relative displacement."""
    out = []
    for line in body.splitlines():
        if "\t" not in line:
            continue
        insn = re.sub(r"#.*|<[^>]*>", "", line.split("\t", 1)[1]).strip()
        words = [w for w in insn.split() if w not in _PREFIX_WORDS]
        if len(words) == 2 and re.fullmatch(r"j\w+|call|loop\w*", words[0]) and re.fullmatch(r"[0-9a-f]+|0x[0-9a-f]+", words[1]):
            continue
        insn = re.sub(r"\[rip[^\]]*\]", "", insn)
        if re.search(r"\b[re]?[sb]p\b", insn):
            continue
        if re.search(rf"(?<![\w.])-?(0x{size:x}|{size})(?![\w.])", insn):
            out.append(line)
    return out


class ArrayStride(unittest.TestCase):
    def test_a_frame_offset_is_not_a_constant_of_the_program(self):
        dis = ("0000000000000000 <proc>:\n   0:\tmov    QWORD PTR [rbp-0x18],rdi\n   4:\tsub    rsp,0x30\n   8:\tadd    rdi,0x38\n   c:\tmov    eax,DWORD PTR [rsp+0x20]\n"
               "  10:\timul   rax,rax,0x28\n")
        self.assertEqual(survive.numbers(dis), {0x18, 0x30, 0x38, 0x20, 0x28})
        self.assertEqual(survive.numbers(dis, stack=False), {0x38, 0x28})

    def test_the_facts_of_an_array_stride_task_ask_for_a_literal_outside_the_frame(self):
        for t in [t for t in families.generate("structs", 120, 9) if t.variant == "array_stride"]:
            self.assertEqual(t.facts, [[t.params["size"], {"no_stack": True}]])
            json.dumps(t.meta())
        alloc = [t for t in families.generate("structs", 120, 9) if t.variant == "alloc_size"]
        self.assertTrue(alloc and all(t.facts == [[t.params["size"]]] for t in alloc), "alloc_size is untouched")

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_the_stack_slot_that_looked_like_the_stride_no_longer_passes(self):
        """gcc -O0, ungated draws of seed 9091: the old presence test passed 11 of the tasks of which the literal is a frame slot only."""
        spurious, kept = [], []
        for t in [t for t in families.generate("structs", 6 * 120, 9091, "gcc", None) if t.variant == "array_stride"]:
            if t.opt != "-O0":
                continue
            size = t.params["size"]
            old = survive.check([[size]], t.source(), "gcc", "-O0")[0]
            new = survive.check(t.facts, t.source(), "gcc", "-O0")[0]
            (spurious if old and not new else kept).append((t, new))
        self.assertGreaterEqual(len(spurious), 2, "the defect exists in this sample")
        for t, _new in spurious:                                      # independent look at the object: no instruction that is off the stack names the stride
            with tempfile.TemporaryDirectory() as d:
                o = pathlib.Path(d, "o.o")
                survive.compile_object(t.source(), "gcc", "-O0", o)
                dis = survive._objdump(["-d", "--no-show-raw-insn", "-M", "intel"], o)
            body = dis.split("<proc>:")[1].split("<main>:")[0]
            size = t.params["size"]
            self.assertEqual(literal_lines(body, size), [], t.id)

    @unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
    def test_every_kept_stride_is_a_literal_of_an_instruction_off_the_stack(self):
        gate = survive.Gate(warn=lambda m: None)
        tasks = [t for t in families.generate("structs", 6 * 30, 9091, "mix", gate) if t.variant == "array_stride"]
        self.assertGreaterEqual(len(tasks), 25)
        for t in tasks:
            with tempfile.TemporaryDirectory() as d:
                o = pathlib.Path(d, "o.o")
                survive.compile_object(t.source(), t.cc, t.opt, o)
                dis = survive._objdump(["-d", "--no-show-raw-insn", "-M", "intel"], o)
            size = t.params["size"]
            body = dis.split("<proc>:")[1].split("<main>:")[0]
            self.assertTrue(literal_lines(body, size), f"{t.id} {t.cc} {t.opt}: stride {size} is not a literal of an instruction off the stack (a branch target is not one: round 10, m1)")


if __name__ == "__main__":
    unittest.main()

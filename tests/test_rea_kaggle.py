"""Round 12: the standalone, Modal-free scorer of a Kaggle replies.jsonl (rea_bench/kaggle_score.py, `python3 tools/rea-bench kaggle-score`). It must build the SAME records and stats as the Modal sample stage
from the same replies, so the main test runs the stubbed stage and the scorer on the same server answers and compares them. The rest: the file formats, the per-family table, the thinking flag, the refusals.

  SameAsTheStage    records and stats equal to `sample_answers` on the same replies, visible-reasoning and thinking
  Table             the per-family numbers of a small fixture, worked out by hand, and replies = traces + drops
  Files             malformed lines, repeated ids, ids that are not tasks, failed requests, cut-off replies, the held-out seed, the output names, overwrite
  Command           the CLI subcommand and the command line of the README
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

from .helpers import ROOT
from .rea_tasks import meta as real_task, right
from .test_rea_modal import StubbedSampleStage
from .test_rea_reasoning import STEPS, answer_body, reply, wrong_for
from .test_rea_traces import distinct, think_body

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import cli, kaggle_score, outputs  # noqa: E402
from .rea_bound import rsn, sft  # noqa: E402

WRONG = "0x0badc0de"          # not the answer of any task, numeric or string
CJK = "这个函数读取指针并加上偏移量。"


def out(text, finish="stop", tokens=10, secs=1.0, **extra):
    return {"text": text, "finish": finish, "tokens": tokens, "secs": secs, **extra}


def write_tasks(dirpath, tasks, name="decompiled-seed7-all.jsonl"):
    p = pathlib.Path(dirpath) / name
    p.write_text("\n".join(json.dumps({**t, "pseudocode": t.get("pseudocode", f"/* {t['id']} */ void f(void) {{}}")}) for t in tasks) + "\n")
    return p


def write_replies(dirpath, lines, name="replies.jsonl"):
    p = pathlib.Path(dirpath) / name
    p.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return p


class SameAsTheStage(StubbedSampleStage):
    """The stubbed Modal stage and the standalone scorer, on the same answers of the same fake server."""

    def setUp(self):
        super().setUp()
        self.ids = list(self.tasks)

        def server(tid, i, body):
            n = self.ids.index(tid) % 4
            good = distinct(self.tasks[tid], 3)
            if i == 0:
                return good[0]
            if i == 1:
                return good[0].replace("The function", "the function") if n == 0 else good[1]       # a duplicate for a quarter of the tasks
            return [good[2], reply(self.tasks[tid], answer=WRONG), reply(self.tasks[tid], f"Step for {tid}.\n{CJK}"), "Cut off"][n]       # the last one: the server cuts it off (see `answer`)
        self.server = server

    def answer(self, tid, i, thinking):
        text = self.server(tid, i, None)
        finish = "length" if text.endswith("Cut off") else "stop"
        if finish == "length":
            text = "Step for it.\nThe constant is"
        if thinking:
            return think_body(text, f"Reasoning about {tid}, sample {i}: the code reads it, then returns it.", finish)
        return answer_body(text, finish)

    def kaggle_lines(self, thinking):
        lines = []
        for tid, t in self.tasks.items():
            outs = []
            for i in range(self.k):
                ch = self.answer(tid, i, thinking)["choices"][0]
                text = ch["message"].get("content") or ""
                if ch["message"].get("reasoning_content"):
                    text = f"<think>{ch['message']['reasoning_content']}</think>\n\n{text}"
                outs.append(out(text, ch["finish_reason"], tokens=17 + i))
            lines.append({"id": tid, "kind": t["kind"], "outs": outs})
        return lines

    def compare(self, thinking):
        self.reply = lambda tid, i, body: self.answer(tid, i, thinking)
        kw = dict(reasoning=True, max_len=4096, **({"thinking": True, "max_tokens": 900} if thinking else {}))
        res = self.run_stage(False, **kw)
        run_info = {"thinking": thinking, "sampling": rsn.sampling_for(thinking), "max_tokens": 900 if thinking else rsn.DEFAULT_MAX_TOKENS, "seed_base": 1234, "ctx_per_slot": 8192, "slots": 4}
        with tempfile.TemporaryDirectory() as d:
            tasks = write_tasks(d, list(self.tasks.values()))
            replies = write_replies(d, self.kaggle_lines(thinking))
            records, stats = kaggle_score.score(replies, tasks, "m", thinking=thinking, run_info=run_info)
        self.assertEqual(records, res["sft"], "the same records, in the same order")
        skip = ("over_length", "perf", "families", "kaggle", "thinking", "slot", "prompts", "projected_fit", "run")          # the Modal stage's own measurements, and what only the Kaggle scorer reads
        self.assertEqual({k: v for k, v in stats.items() if k not in skip}, {k: v for k, v in res["stats"].items() if k not in skip}, "the same stats, the recorded sampling included (round 13)")
        self.assertEqual(stats["sampling"], res["stats"]["sampling"])
        return records, stats

    def test_visible_reasoning_gives_the_records_and_stats_of_the_modal_stage(self):
        records, stats = self.compare(False)
        self.assertGreater(len(records), 2 * len(self.tasks))
        self.assertEqual(stats["reasoning"]["mode"], "reasoning")
        self.assertTrue(stats["reasoning"]["dropped"]["duplicate"] and stats["reasoning"]["dropped"]["mixed_language"], "the replies exercised more than the happy path")
        self.assertTrue(stats["reasoning"]["wrong"] and stats["reasoning"]["truncated"])

    def test_thinking_gives_the_records_and_stats_of_the_modal_stage(self):
        records, stats = self.compare(True)
        self.assertEqual(stats["reasoning"]["mode"], "thinking")
        self.assertTrue(all(r["messages"][-1]["content"].startswith("<think>\n") for r in records if r["target_source"] == "teacher-reasoning"))
        self.assertTrue(stats["reasoning"]["dropped"]["mixed_language"])

    def test_the_messages_are_the_ones_the_sampler_asks(self):
        t = next(iter(self.tasks.values()))
        self.assertEqual(rsn.messages(t), self.data._messages(t, True))
        self.assertEqual(rsn.messages({**t, "refs": "00 01"})[1]["content"], self.data._messages({**t, "refs": "00 01"}, True)[1]["content"])
        self.assertIn("Data referenced by the function", rsn.user_message({**t, "refs": "00 01"}))


class Table(unittest.TestCase):
    def setUp(self):
        self.a = real_task("structs", "linked_next", chance=0.01)
        self.b = real_task("pktsum", "sum8", chance=0.01)
        self.h = real_task("crc", "bitwise_init", chance=0.55)
        d = distinct(self.a, 3)
        self.lines = [
            {"id": self.a["id"], "kind": "structs", "outs": [out(d[0], tokens=10), out(d[1], tokens=20), out(d[2], tokens=30), out(d[0].replace("The function", "the function"), tokens=40),
                                                             out(reply(self.a, answer=wrong_for(self.a)), tokens=50), out("Step one.\nThe constant is", "length", 60)]},
            {"id": self.b["id"], "kind": "pktsum", "outs": [out(reply(self.b, "\n".join(["Wait, let me recheck the code again."] * 4)), tokens=100), out(reply(self.b, f"A loop adds bytes.\n{CJK}"), tokens=200),
                                                            out(reply(self.b, "The checksum is stored after the last byte.\nA loop adds every byte."), tokens=300), out("", "error", 0), out("[error TimeoutError()]", "stop", 5)]},
            {"id": self.h["id"], "kind": "crc", "outs": [out(reply(self.h, "The register is folded once per byte."), tokens=70), out(reply(self.h, answer="0x7"), tokens=90)]}]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tasks = write_tasks(self.tmp.name, [self.a, self.b, self.h])
        self.replies = write_replies(self.tmp.name, self.lines)
        self.records, self.stats = kaggle_score.score(self.replies, self.tasks, "teacher-x")

    def test_the_family_rows(self):
        f = self.stats["families"]
        self.assertEqual(sorted(f), ["ALL", "crc", "pktsum", "structs"])
        s = f["structs"]
        self.assertEqual((s["tasks"], s["answers"], s["accepted"], s["kept_tasks"], s["traces"]), (1, 6, 4, 1, 3))
        self.assertEqual({k: v for k, v in s["drops"].items() if v}, {"truncated": 1, "wrong": 1, "duplicate": 1})
        self.assertEqual((s["truncated_share"], s["tokens"]), (round(1 / 6, 4), {"n": 6, "mean": 35.0, "median": 35.0}))
        p = f["pktsum"]
        self.assertEqual((p["tasks"], p["answers"], p["accepted"], p["kept_tasks"], p["traces"]), (1, 5, 3, 1, 1))
        self.assertEqual({k: v for k, v in p["drops"].items() if v}, {"repetition": 1, "mixed_language": 1, "error": 2})
        self.assertEqual(p["tokens"], {"n": 5, "mean": 121.0, "median": 100})
        c = f["crc"]
        self.assertEqual((c["tasks"], c["answers"], c["accepted"], c["kept_tasks"], c["traces"]), (1, 2, 1, 0, 0))
        self.assertEqual({k: v for k, v in c["drops"].items() if v}, {"wrong": 1, "in_lost_tasks": 1}, "one right answer for a chance 0.55 task: the agreement rule drops the task")
        a = f["ALL"]
        self.assertEqual((a["tasks"], a["answers"], a["accepted"], a["kept_tasks"], a["traces"]), (3, 13, 8, 2, 4))
        self.assertEqual(a["tokens"], {"n": 13, "mean": round((10 + 20 + 30 + 40 + 50 + 60 + 100 + 200 + 300 + 0 + 5 + 70 + 90) / 13, 1), "median": 50})
        self.assertEqual(a["truncated_share"], round(1 / 13, 4))

    def test_every_reply_is_a_trace_or_in_exactly_one_drop_reason(self):
        for name, row in self.stats["families"].items():
            self.assertEqual(row["answers"], row["traces"] + sum(row["drops"].values()), name)
        self.assertEqual(set(self.stats["families"]["ALL"]["drops"]), {"error", "truncated", "no_answer_line", "wrong", *rsn.TRACE_DROPS, "in_lost_tasks"})

    def test_the_records_are_the_assemble_records(self):
        self.assertEqual(sum(r["target_source"] == "truth-template" for r in self.records), 2)
        self.assertEqual(sum(r["target_source"] == "teacher-reasoning" for r in self.records), 4)
        self.assertTrue(all(sorted(r) == sorted(sft.RECORD_KEYS) and r["model"] == "teacher-x" for r in self.records))
        self.assertTrue(all(r["messages"][0] == {"role": "system", "content": rsn.SYSTEM_REASONING} for r in self.records))
        sft.check_records(self.records)
        a = {**self.a, "pseudocode": f"/* {self.a['id']} */ void f(void) {{}}"}
        direct, _ = sft.assemble([a], {a["id"]: {"outs": [[o["text"] if o["finish"] != "length" else o["text"] + "\n" + rsn.TRUNCATED, False] for o in self.lines[0]["outs"]]}}, "teacher-x",
                                 rsn.messages, True)
        self.assertEqual([r for r in self.records if r["task"] == self.a["id"]], direct)

    def test_the_stats_file_content_is_json_and_says_what_was_read(self):
        s = json.loads(json.dumps(self.stats))
        self.assertEqual(s["kaggle"], {"lines": 3, "malformed": 0, "duplicate_ids": 0, "unknown_ids": 0, "tasks_in_file": 3, "tasks_without_replies": 0, "replies_file": "replies.jsonl",
                                       "tasks_file": "decompiled-seed7-all.jsonl"})
        self.assertEqual((s["thinking"], s["reasoning"]["max_traces"], s["reasoning"]["mode"]), (False, 3, "reasoning"))
        self.assertEqual(s["tasks"], 3)

    def test_max_traces_and_the_cap(self):
        _, stats = kaggle_score.score(self.replies, self.tasks, "m", max_traces=1)
        self.assertEqual(stats["families"]["structs"]["traces"], 1)
        self.assertEqual(stats["families"]["structs"]["drops"]["over_cap"], 2)
        with self.assertRaisesRegex(SystemExit, "--max-traces"):
            kaggle_score.score(self.replies, self.tasks, "m", max_traces=0)


class Files(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.u = real_task("pktsum", "sum8", chance=0.01)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name
        self.tasks = write_tasks(self.d, [self.t, self.u])

    def good(self, t, n=1):
        return {"id": t["id"], "kind": t["kind"], "outs": [out(r) for r in distinct(t, n)]}

    def score(self, lines, **kw):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            kept, stats = kaggle_score.score(write_replies(self.d, lines), self.tasks, "m", **kw)
        return kept, stats, err.getvalue()

    def test_a_cut_line_a_repeated_id_and_an_unknown_id_are_counted_not_fatal(self):
        stale = {"id": self.t["id"], "kind": "structs", "outs": [out(reply(self.t, answer=wrong_for(self.t)))]}
        lines = [stale, self.good(self.u, 2), '{"id": "structs-7-9999", "ki', self.good(self.t, 2), {"id": "ghost-7-0001", "kind": "crc", "outs": [out("x")]}, "", {"no": "id"}, '[1, 2]']
        kept, stats, err = self.score(lines)
        self.assertEqual(stats["kaggle"]["lines"], 7)
        self.assertEqual((stats["kaggle"]["malformed"], stats["kaggle"]["duplicate_ids"], stats["kaggle"]["unknown_ids"]), (3, 1, 1))
        self.assertIn("ghost-7-0001", err)
        self.assertEqual(stats["tasks"], 2)
        self.assertEqual(stats["families"]["structs"]["answers"], 2, "the last line of an id wins: the stale first one is gone")
        self.assertEqual(stats["families"]["structs"]["drops"]["wrong"], 0)

    def test_a_task_without_replies_is_not_a_task_of_the_run(self):
        kept, stats, _ = self.score([self.good(self.t, 2)])
        self.assertEqual((stats["tasks"], stats["kaggle"]["tasks_without_replies"], stats["kaggle"]["tasks_in_file"]), (1, 1, 2))
        self.assertEqual({r["task"] for r in kept}, {self.t["id"]})

    def test_none_of_the_ids_in_the_tasks_file_is_an_error(self):
        with self.assertRaisesRegex(SystemExit, "none of the 1 task ids"):
            self.score([{"id": "ghost-7-0001", "kind": "crc", "outs": [out("x")]}])

    def test_failed_requests_and_cut_off_replies_are_never_scored_and_are_counted(self):
        good = distinct(self.t, 1)[0]
        lines = [{"id": self.t["id"], "kind": "structs", "outs": [out(good, "length"), out(good, "error"), out("", "error", 0), out(f"[error URLError()]\nAnswer: {right(self.t)}"), out(good), out(None, "stop", 3)]}]
        kept, stats, _ = self.score(lines)
        r = stats["reasoning"]
        self.assertEqual((r["truncated"], r["error"], r["accepted"], r["no_answer_line"], r["teacher_records"]), (1, 3, 1, 1, 1))
        self.assertEqual(stats["errors"], 3)
        self.assertEqual(kaggle_score.reply_of(out(good, "length")), good + "\n" + rsn.TRUNCATED)
        self.assertTrue(kaggle_score.reply_of(out("x", "error")).startswith("[error"))
        self.assertEqual(kaggle_score.reply_of(out("[error boom]", "error")), "[error boom]")
        self.assertTrue(kaggle_score.reply_of("not a dict").startswith("[error"))

    def test_the_thinking_flag_keeps_the_reasoning_and_without_it_the_blocks_are_stripped(self):
        visible = f"{STEPS}\nAnswer: {right(self.t)}"
        lines = [{"id": self.t["id"], "kind": "structs", "outs": [out(f"<think>\nThe node struct has a next pointer right after its first field.\n</think>\n\n{visible}"),
                                                                  out(f"Walking the list means loading the link field, which the code reads at a fixed displacement.\n</think>\n{visible}"),
                                                                  out(visible, reasoning_content="Looking at the struct, one field comes first."), out(visible)]}]
        kept, stats, _ = self.score(lines, thinking=True)
        texts = [r["messages"][-1]["content"] for r in kept if r["target_source"] == "teacher-reasoning"]
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(t.startswith("<think>\n") and t.endswith(visible) for t in texts))
        self.assertEqual((stats["reasoning"]["dropped"]["no_think"], stats["reasoning"]["mode"], stats["thinking"]), (1, "thinking", True))
        kept, stats, _ = self.score(lines)
        texts = [r["messages"][-1]["content"] for r in kept if r["target_source"] == "teacher-reasoning"]
        self.assertEqual(texts[0], visible, "without --thinking the think block is dropped, as the visible-reasoning run does")
        self.assertEqual(stats["reasoning"]["mode"], "reasoning")
        self.assertEqual(len(texts), 1, "and the four replies are one visible text")

    def test_the_held_out_seed_is_refused(self):
        held = write_tasks(self.d, [self.t], "decompiled-seed1001-all.jsonl")
        with self.assertRaisesRegex(SystemExit, "held-out"):
            kaggle_score.score(write_replies(self.d, [self.good(self.t)]), held, "m")
        renamed = write_tasks(self.d, [{**self.t, "id": "structs-1001-0001"}], "tasks.jsonl")
        with self.assertRaisesRegex(SystemExit, "held-out"):
            kaggle_score.score(write_replies(self.d, [{**self.good(self.t), "id": "structs-1001-0001"}]), renamed, "m")

    def test_a_task_without_a_truth_template_stops_the_run_like_the_sample_stage(self):
        """Before anything is scored: the stage checks every task before it asks, so a task whose replies are all wrong (nothing would ask for its truth record) stops the run too."""
        broken = write_tasks(self.d, [{**self.t, "variant": "bogus"}], "decompiled-seed7-x.jsonl")
        wrong = {"id": self.t["id"], "kind": "structs", "outs": [out(reply(self.t, answer=WRONG))]}
        with self.assertRaisesRegex(ValueError, "no truth template"):
            kaggle_score.score(write_replies(self.d, [wrong]), broken, "m")
        unknown_mode = write_tasks(self.d, [{**self.t, "score": "nonsense"}], "decompiled-seed7-y.jsonl")
        with self.assertRaisesRegex(ValueError, "unknown score mode"):
            kaggle_score.score(write_replies(self.d, [wrong]), unknown_mode, "m")


class Command(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = write_tasks(self.d, [self.t])
        self.replies = write_replies(self.d, [{"id": self.t["id"], "kind": "structs", "outs": [out(r) for r in distinct(self.t, 3)]}])

    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["kaggle-score", *map(str, args)])
        return buf.getvalue()

    def test_the_default_output_is_named_like_the_modal_stage_and_sits_next_to_the_tasks(self):
        text = self.run_cli("--replies", self.replies, "--tasks", self.tasks, "--model", "teacher-x")
        out_path = self.d / "sft-teacher-x-seed7-all-reasoning.jsonl"
        self.assertTrue(out_path.exists())
        self.assertTrue((self.d / "sft-teacher-x-seed7-all-reasoning.stats.json").exists())
        rows = [json.loads(l) for l in out_path.read_text().splitlines()]
        self.assertEqual([r["target_source"] for r in rows], ["truth-template"] + ["teacher-reasoning"] * 3)
        stats = json.loads((self.d / "sft-teacher-x-seed7-all-reasoning.stats.json").read_text())
        self.assertEqual(stats["families"]["structs"]["traces"], 3)
        self.assertIn("1 tasks, 3 replies, 1 kept tasks, 4 records (3 teacher-reasoning)", text)
        self.assertIn("structs", text)
        self.assertEqual(outputs.sft_stem("teacher-x", "seed7-all-reasoning"), out_path.stem)

    def test_thinking_output_says_thinking_and_an_existing_output_is_not_overwritten(self):
        self.replies = write_replies(self.d, [{"id": self.t["id"], "kind": "structs", "outs": [out(f"<think>\nThe node has a next pointer.\n</think>\n\n{reply(self.t)}")]}])
        self.run_cli("--replies", self.replies, "--tasks", self.tasks, "--model", "teacher-x", "--thinking")
        self.assertTrue((self.d / "sft-teacher-x-seed7-all-reasoning-think.jsonl").exists())
        self.assertFalse((self.d / "sft-teacher-x-seed7-all-reasoning.jsonl").exists())
        with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
            self.run_cli("--replies", self.replies, "--tasks", self.tasks, "--model", "teacher-x", "--thinking")
        self.run_cli("--replies", self.replies, "--tasks", self.tasks, "--model", "teacher-x", "--thinking", "--force")

    def test_out_and_max_traces(self):
        mine = self.d / "mine.jsonl"
        self.run_cli("--replies", self.replies, "--tasks", self.tasks, "--model", "m", "--out", mine, "--max-traces", "2")
        self.assertEqual(len(mine.read_text().splitlines()), 3)
        self.assertTrue((self.d / "mine.stats.json").exists())

    def test_the_module_runs_without_modal(self):
        code = ("import sys; sys.path.insert(0, %r); import rea_bench.kaggle_score, rea_bench.cli; "
                "assert 'modal' not in sys.modules and not any(m.startswith('modal_') for m in sys.modules), sorted(sys.modules)" % str(ROOT / "tools" / "rea-bench"))
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual((r.returncode, r.stderr), (0, ""))

    def test_python_dash_m_works_too(self):
        r = subprocess.run([sys.executable, "-m", "rea_bench.kaggle_score", "--replies", str(self.replies), "--tasks", str(self.tasks), "--model", "m", "--out", str(self.d / "x.jsonl")],
                           capture_output=True, text=True, cwd=ROOT / "tools" / "rea-bench")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.d / "x.jsonl").exists())

    def test_the_readme_names_a_command_the_parser_takes(self):
        readme = (ROOT / "tools" / "modal" / "README.md").read_text()
        cmds = re.findall(r"python3 tools/rea-bench kaggle-score ((?:[^\n`\\]|\\\n)*)", readme)
        self.assertTrue(cmds, "the README documents the Kaggle scorer")
        import argparse
        ap = argparse.ArgumentParser()
        kaggle_score.add_arguments(ap)
        for c in cmds:
            args = ap.parse_args(c.replace("\\\n", " ").split())
            self.assertTrue(args.model and args.replies)


if __name__ == "__main__":
    unittest.main()

"""The core of `distill/` does not know REA: the whole pipeline runs on the toy subject `distill/domains/example_arith` (a test fixture), and the contract of a `Domain` is checked for it and for REA.

  Independence   the core imports nothing of REA, Modal or a model library; it is the standard library only
  Pipeline       records (plain, reasoning, thinking), the Kaggle scorer and kit, and the Modal sample stage on a scripted server, for the toy subject, with the same rules as for REA: agreement, traces, duplicates, the
                 length rule, held-out tasks refused, a reply to another prompt not stored
  Contract       what every `Domain` must guarantee, run against the toy subject and against REA's frozen tasks
  Config         a run config is a file; the shipped REA configs are today's runs
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from . import distill_fixtures as fx
from .helpers import ROOT

sys.path.insert(0, str(ROOT))

from distill import config, outputs, records, score_replies, teachers, traces  # noqa: E402
from distill.backends import modal_sample  # noqa: E402
from distill.backends.kaggle import build_tasks  # noqa: E402
from distill.domain import Domain, MODES, mode_of  # noqa: E402
from distill.domains import load  # noqa: E402
from distill.domains.example_arith import DOMAIN as ARITH, HELDOUT_SEED, make_tasks  # noqa: E402

CORE = sorted(p for p in (ROOT / "distill").glob("*.py"))
TASKS = make_tasks(7, 12)
LOOP = "Wait, let me recheck the sum.\n" * 5


def steps(t, n=0):
    return [f"{t['question']} I add the two numbers and get the total.", f"First the operands are read, then they are combined for {t['id']}.", f"Counting it out one by one for {t['id']} gives the result."][n]


def reasoning_reply(t, n=0, answer=None):
    return f"{steps(t, n)}\nAnswer: {t['answer'] if answer is None else answer}"


class Independence(unittest.TestCase):
    def test_the_core_imports_without_rea_or_modal(self):
        code = ("import sys; sys.path.insert(0, %r); import distill.domain, distill.traces, distill.records, distill.config, distill.outputs, distill.score_replies, distill.teachers, distill.domains; "
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('rea_bench', 'modal', 'huggingface_hub', 'torch', 'transformers', 'datasets', 'peft') or m.startswith('modal_')); assert not bad, bad" % str(ROOT))
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stderr), (0, ""))

    def test_the_core_is_the_standard_library_only(self):
        self.assertGreaterEqual(len(CORE), 8)
        for path in CORE:
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
                for name in names:
                    self.assertTrue(name in sys.stdlib_module_names or name == "distill", f"{path.name} imports {name}")

    def test_no_word_of_rea_is_decided_in_the_core(self):
        for path in CORE:
            text = path.read_text()
            for word in ("rea_bench", "Ghidra", "pseudocode", "score_strict"):
                self.assertNotIn(word, text.replace("`rea_bench/reasoning.py`", "").replace("rea_bench/sft.py", "").replace("rea_bench/kaggle_score.py", "").replace("rea_bench/outputs.py", "").replace("the REA decompile stage", ""),
                                 f"{path.name} names {word}")

    def test_the_modal_sample_stage_has_no_modal_import(self):
        text = (ROOT / "distill" / "backends" / "modal_sample.py").read_text()
        self.assertNotRegex(text, r"(?m)^(import modal|from modal)")


class Pipeline(unittest.TestCase):
    def results(self, mode):
        """3 samples per task: right, a duplicate or another right one, then a wrong answer for some tasks."""
        out = {}
        for i, t in enumerate(TASKS):
            if mode == "plain":
                outs = [str(t["answer"]), f"{t['answer']}\nbecause", str(t["answer"] + 1)]
            elif mode == "reasoning":
                outs = [reasoning_reply(t, 0), reasoning_reply(t, 0 if i % 4 == 0 else 1), reasoning_reply(t, 2, t["answer"] + 1 if i % 3 == 0 else None)]
            else:
                outs = [traces.thinking_reply(reasoning_reply(t, 0), f"Reasoning one about {t['id']}."), traces.thinking_reply(reasoning_reply(t, 1), f"A second thought about {t['id']}, differently."),
                        traces.thinking_reply(reasoning_reply(t, 2, t["answer"] + 1 if i % 3 == 0 else None), "")]
            out[t["id"]] = {"outs": [[o, False] for o in outs]}
        return out

    def assemble(self, mode, **kw):
        return records.assemble(TASKS, self.results(mode), "m", lambda t: ARITH.messages(t, mode), mode != "plain", mode == "thinking", domain=ARITH, **kw)

    def test_plain_mode_stores_the_truth_and_never_the_teacher(self):
        kept, stats = self.assemble("plain")
        big = [t for t in TASKS if not ARITH.needs_agreement(t)]
        self.assertEqual([r["messages"][-1]["content"] for r in kept], [str(t["answer"]) for t in big], "one record per task that is kept: the truth, whatever the teacher wrote")
        self.assertTrue(all(r["target_source"] == "truth-template" for r in kept))
        self.assertEqual(sorted(kept[0]), sorted(records.RECORD_KEYS))
        self.assertEqual((stats["tasks"], stats["answers"], stats["kept_tasks"], stats["agreement"]), (12, 36, len(big), 12 - len(big)),
                         "one sample is right per task; a task a guess solves often needs two, and `7\\nbecause` is not a number")
        self.assertNotIn("reasoning", stats)

    def test_agreement_counts_for_the_tasks_the_domain_marks(self):
        t = next(t for t in TASKS if ARITH.needs_agreement(t))
        one = {t["id"]: {"outs": [[str(t["answer"]), False], [str(t["answer"] + 1), False]]}}
        kept, stats = records.assemble([t], one, "m", lambda x: [], domain=ARITH)
        self.assertEqual((kept, stats["agreement"]), ([], 1))
        two = {t["id"]: {"outs": [[str(t["answer"]), False], [str(t["answer"]) + ".", False]]}}
        self.assertEqual(len(records.assemble([t], two, "m", lambda x: [], domain=ARITH)[0]), 1)
        big = next(t for t in make_tasks(8, 60) if not ARITH.needs_agreement(t))
        self.assertEqual(len(records.assemble([big], {big["id"]: {"outs": [[str(big["answer"]), False]]}}, "m", lambda x: [], domain=ARITH)[0]), 1, "one right sample keeps a task a guess does not solve")

    def test_reasoning_mode_keeps_distinct_traces_beside_the_truth_record(self):
        kept, stats = self.assemble("reasoning")
        r = stats["reasoning"]
        self.assertEqual(r["mode"], "reasoning")
        self.assertGreater(r["dropped"]["duplicate"], 0, "a quarter of the tasks answered twice with the same steps")
        self.assertEqual(r["replies"], r["accepted"] + r["wrong"], "every reply is judged")
        by = {}
        for rec in kept:
            by.setdefault(rec["task"], []).append(rec)
        for tid, rows in by.items():
            self.assertEqual([x["target_source"] for x in rows][0], "truth-template")
            self.assertEqual(len({x["messages"][-1]["content"] for x in rows}), len(rows), "distinct")
            self.assertTrue(all(x["messages"][:-1] == ARITH.messages(next(t for t in TASKS if t["id"] == tid), "reasoning") for x in rows), "the same messages")
        records.check_records(kept)
        kept1, _ = self.assemble("reasoning", max_traces=1)
        self.assertLess(len(kept1), len(kept))
        self.assertEqual(sum(x["target_source"] == "teacher-reasoning" for x in kept1), len(by))

    def test_thinking_mode_keeps_the_reasoning_as_the_trace_and_drops_a_reply_without_it(self):
        kept, stats = self.assemble("thinking")
        self.assertEqual(stats["reasoning"]["mode"], "thinking")
        self.assertGreater(stats["reasoning"]["dropped"]["no_think"], 0, "the third sample had no reasoning")
        self.assertTrue(all(r["messages"][-1]["content"].startswith("<think>\n") for r in kept if r["target_source"] == "teacher-reasoning"))

    def test_the_filters_use_the_domains_hooks_and_its_defaults(self):
        t = TASKS[0]
        for text, why in ((reasoning_reply(t, 0, t["answer"]).replace(steps(t, 0), "That is it."), "unreadable"), (reasoning_reply(t, 0).replace(steps(t, 0), steps(t, 0) + " 这个函数读取指针并加上偏移量。"), "mixed_language"),
                          (reasoning_reply(t, 0).replace(steps(t, 0), LOOP), "repetition")):
            self.assertEqual(traces.quality_drop(text, t, domain=ARITH), why, text)
        self.assertIsNone(traces.quality_drop(reasoning_reply(t, 0), t, domain=ARITH))
        self.assertFalse(ARITH.contradicts("so the answer is 99", "7", t), "a domain without the hook never drops for a contradiction")
        self.assertEqual(ARITH.canon_word("0x20"), ARITH.canon_word("32"))
        self.assertIn("apples", ARITH.task_words(TASKS[0]))

    def test_a_non_ascii_word_of_the_prompt_is_not_language_mixing(self):
        t = {**TASKS[0], "question": "Zoë has 3 apples and buys 4 more. How many?", "answer": 7}
        self.assertFalse(traces.mixed_language("Zoë has 3 apples and buys 4 more, so Zoë has 7 in all of them together.", t, domain=ARITH))
        self.assertTrue(traces.mixed_language("Zoë has 3 apples and buys 4 more, so Zoë has 7 in all of them together.", {**t, "question": "Zo has 3"}, domain=ARITH))

    def test_the_length_rule_is_the_same_for_any_subject(self):
        kept, stats = self.assemble("reasoning")
        short, st = records.drop_over_length(kept, stats, lambda text: len(text.split()), 40, report_windows=config.REPORT_WINDOWS)
        self.assertEqual(len(short) + st["over_length"]["dropped"], len(kept))
        self.assertGreater(st["over_length"]["dropped"], 0)
        self.assertIn("over-length:", records.over_length_line(st))

    def write(self, d, lines, name="replies.jsonl"):
        p = pathlib.Path(d) / name
        p.write_text("".join(json.dumps(l) + "\n" for l in lines))
        return p

    def replies_lines(self, mode):
        return [{"id": t["id"], "kind": t["kind"], "prompt_sha": outputs.prompt_sha(ARITH.messages(t, mode)),
                 "outs": [{"text": ("<think>\n" + f"Reasoning about {t['id']}.\n</think>\n\n" if mode == "thinking" else "") + reasoning_reply(t, n), "finish": "stop", "tokens": 20 + n, "prompt_tokens": 50, "secs": 1.0} for n in range(3)]}
                for t in TASKS]

    def test_the_scorer_of_a_replies_file_gives_what_assemble_gives(self):
        for mode in ("reasoning", "thinking"):
            with tempfile.TemporaryDirectory() as d:
                tasks = self.write(d, TASKS, "tasks.jsonl")
                replies = self.write(d, self.replies_lines(mode))
                kept, stats = score_replies.score(replies, tasks, "m", thinking=mode == "thinking", domain=ARITH)
                want, _ = records.assemble(TASKS, {l["id"]: {"outs": [[score_replies.reply_of(o, mode == "thinking"), False] for o in l["outs"]]} for l in self.replies_lines(mode)}, "m",
                                           lambda t: ARITH.messages(t, mode), True, mode == "thinking", domain=ARITH)
                self.assertEqual(kept, want, mode)
                self.assertEqual(stats["reasoning"]["mode"], mode)
                self.assertEqual(stats["prompts"], {"checked": 12, "mismatch": 0, "mismatched_ids": []})
                self.assertEqual(stats["families"]["ALL"]["tasks"], 12)
                self.assertEqual(list(stats["projected_fit"]["windows"]), ["4096", "8192"] if mode == "thinking" else ["2048", "3072", "4096"])

    def test_the_command_line_runs_for_any_domain_and_names_its_files_from_the_tasks_file(self):
        with tempfile.TemporaryDirectory() as d:
            tasks = self.write(d, TASKS, "arith-seed7.jsonl")
            replies = self.write(d, self.replies_lines("reasoning"))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                score_replies.main(["--domain", "example_arith", "--replies", str(replies), "--tasks", str(tasks), "--model", "m"])
            self.assertTrue((pathlib.Path(d) / "sft-m-arith-seed7-reasoning.jsonl").exists(), sorted(p.name for p in pathlib.Path(d).iterdir()))
            self.assertIn("12 tasks, 36 replies", out.getvalue())
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                score_replies.main(["--domain", "example_arith", "--replies", str(replies), "--tasks", str(tasks), "--model", "m"])

    def test_held_out_tasks_and_replies_to_another_prompt_are_never_stored(self):
        with tempfile.TemporaryDirectory() as d:
            held = make_tasks(HELDOUT_SEED, 3)
            tasks = self.write(d, held, "tasks.jsonl")
            replies = self.write(d, [{"id": t["id"], "kind": t["kind"], "outs": [{"text": reasoning_reply(t), "finish": "stop", "tokens": 5}]} for t in held])
            with self.assertRaisesRegex(SystemExit, "held-out"):
                score_replies.score(replies, tasks, "m", domain=ARITH)
            tasks = self.write(d, TASKS, "tasks2.jsonl")
            lines = self.replies_lines("reasoning")
            lines[0]["prompt_sha"] = "0" * 16
            with contextlib.redirect_stderr(io.StringIO()):
                _kept, stats = score_replies.score(self.write(d, lines, "r2.jsonl"), tasks, "m", domain=ARITH)
            self.assertEqual((stats["prompts"]["mismatch"], stats["tasks"]), (1, 11))

    def test_the_kaggle_kit_builds_its_inputs_for_any_domain(self):
        with tempfile.TemporaryDirectory() as d:
            tasks = self.write(d, TASKS, "arith-seed7.jsonl")
            cfg = build_tasks.build([tasks], pathlib.Path(d) / "kit", "someone", quota="add=2,mul=1", k=3, thinking=False, teacher="qwen35-9b", domain=ARITH)
            kit = pathlib.Path(d) / "kit"
            rows = [json.loads(l) for l in (kit / "data" / "tasks.jsonl").read_text().splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["add", "mul", "add"])
            self.assertEqual(rows[0]["messages"], ARITH.messages(TASKS[0], "reasoning"))
            self.assertNotIn("answer", rows[0], "the truth stays here")
            self.assertEqual((cfg["model_repo"], cfg["k"], cfg["sampling"]), (teachers.find("qwen35-9b").repo, 3, {"temperature": 0.7}))
            self.assertEqual(json.loads((kit / "kernel" / "kernel-metadata.json").read_text())["id"], "someone/example-arith-teacher")
            held = self.write(d, make_tasks(HELDOUT_SEED, 3), "held.jsonl")
            with self.assertRaisesRegex(SystemExit, "held-out"):
                build_tasks.build([held], pathlib.Path(d) / "kit2", "someone", domain=ARITH)
            with self.assertRaisesRegex(SystemExit, "unknown model"):
                build_tasks.build([tasks], pathlib.Path(d) / "kit3", "someone", teacher="nope", domain=ARITH)


class ToyStage(unittest.TestCase):
    """The Modal sample stage on the toy subject: a scripted llama-server, a temporary /data, no `rea_bench`."""

    def setUp(self):
        import urllib.request
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "data").mkdir()
        self.src = self.root / "tasks.jsonl"
        self.src.write_text("".join(json.dumps(t) + "\n" for t in TASKS))
        self.bodies, self.lock, self.vol = [], threading.Lock(), types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
        real = pathlib.Path
        shim = types.SimpleNamespace(Path=lambda p, *a: real(str(self.root / "data" / str(p)[len("/data/"):]) if str(p).startswith("/data/") else p, *a))
        popen = types.SimpleNamespace(Popen=lambda *a, **k: types.SimpleNamespace(kill=lambda: None), DEVNULL=subprocess.DEVNULL)

        class Resp:
            def __init__(self, body):
                self.body, self.status = body, 200

            def read(self, *a):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=None):
            if isinstance(req, str):
                return Resp(b"ok")
            body = json.loads(req.data)
            if req.full_url.endswith("/tokenize"):
                return Resp(json.dumps({"tokens": [0] * len(body["content"].split())}).encode())
            with self.lock:
                self.bodies.append(body)
            t = next(t for t in TASKS if t["question"] == body["messages"][-1]["content"])
            i = body["seed"] - 1234
            msg = {"content": reasoning_reply(t, i)}
            if body["chat_template_kwargs"]["enable_thinking"]:
                msg["reasoning_content"] = f"Reasoning {i} about {t['id']}."
            return Resp(json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}).encode())

        for p in (mock.patch.object(modal_sample, "pathlib", shim), mock.patch.object(modal_sample, "subprocess", popen), mock.patch.object(urllib.request, "urlopen", urlopen),
                  mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=lambda *a, **k: "/tmp/fake.gguf")})):
            p.start()
            self.addCleanup(p.stop)

    def run_stage(self, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            res = modal_sample.sample_answers(ARITH, self.vol, self.src, "m", "repo", "f.gguf", 3, "seed7", **kw)
        self.printed = printed.getvalue()
        return res

    def test_plain_run_asks_the_plain_messages_and_stores_the_truth(self):
        res = self.run_stage(max_len=0)
        self.assertEqual(len(self.bodies), 36)
        self.assertEqual(self.bodies[0]["messages"][0]["content"], "Answer with the number only.")
        self.assertEqual({k: self.bodies[0][k] for k in ("temperature", "max_tokens")}, {"temperature": 0.7, "max_tokens": 300})
        self.assertEqual(res["stats"]["kept_tasks"], 0, "the server answered with steps: a plain reply is not a number")

    def test_a_reasoning_run_gives_records_and_stats_like_assemble_on_the_same_replies(self):
        res = self.run_stage(reasoning=True, max_len=4096)
        results = {t["id"]: {"outs": [[traces.reply_text(reasoning_reply(t, i), "stop"), False] for i in range(3)]} for t in TASKS}
        want, stats = records.assemble(TASKS, results, "m", lambda t: ARITH.messages(t, "reasoning"), True, False, domain=ARITH)
        self.assertEqual(res["sft"], want)
        self.assertEqual({k: v for k, v in res["stats"].items() if k not in ("perf", "sampling", "over_length")}, stats)
        self.assertEqual(res["stats"]["sampling"], config.sampling_record(False, {"temperature": 0.7}, 768, seed_base=1234, slot_context=8192, parallel=4))
        self.assertTrue((self.root / "data" / "sft-m-seed7-reasoning.partial.jsonl").exists())
        self.assertIn("over_length", res["stats"])

    def test_a_thinking_run_keeps_the_servers_reasoning_and_samples_like_qwen3(self):
        res = self.run_stage(reasoning=True, thinking=True, max_len=8192)
        self.assertEqual({k: self.bodies[0][k] for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens")}, {**config.THINKING_SAMPLING, "max_tokens": 4096})
        self.assertTrue(all(r["messages"][-1]["content"].startswith("<think>\nReasoning") for r in res["sft"] if r["target_source"] == "teacher-reasoning"))
        self.assertEqual(res["stats"]["reasoning"]["mode"], "thinking")

    def test_the_stage_refuses_before_asking_anything(self):
        with self.assertRaisesRegex(SystemExit, "--max-len"):
            self.run_stage(reasoning=True)
        with self.assertRaisesRegex(SystemExit, "never greedy"):
            self.run_stage(reasoning=True, thinking=True, max_len=8192, sampling={"temperature": 0})
        with self.assertRaisesRegex(SystemExit, "selects no task"):
            self.run_stage(reasoning=True, max_len=4096, quota="nothing=1")
        self.assertEqual(self.bodies, [])

    def test_quota_and_resume_work_for_any_subject(self):
        res = self.run_stage(reasoning=True, max_len=4096, quota="add=2,sub=1")
        self.assertEqual(res["stats"]["tasks"], 3)
        self.assertEqual(len(self.bodies), 9)
        self.bodies.clear()
        self.run_stage(reasoning=True, max_len=4096, quota="add=2,sub=1", resume=True)
        self.assertEqual(self.bodies, [], "a complete checkpoint asks nothing")


class Contract(unittest.TestCase):
    """What every Domain must guarantee. The toy subject and REA's frozen tasks both pass."""

    def check(self, domain, tasks, heldout_tasks):
        ids = [t["id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)), "ids are unique")
        self.assertTrue(domain.name)
        heldout = domain.heldout_ids()
        self.assertFalse([i for i in ids if i in heldout], "a training task is never in the held-out set")
        self.assertTrue(all(t["id"] in heldout for t in heldout_tasks), "the held-out set holds the held-out tasks")
        for t in tasks:
            self.assertIn("kind", t)
            for mode in MODES:
                msgs = domain.messages(t, mode)
                self.assertEqual([m["role"] for m in msgs], ["system", "user"], (t["id"], mode))
                self.assertTrue(all(isinstance(m["content"], str) and m["content"] for m in msgs))
            truth = domain.truth_target(t)
            if truth is not None:
                self.assertTrue(domain.verify(truth, t), f"verify accepts the truth of {t['id']}")
            self.assertFalse(domain.verify("", t), "an empty answer is wrong")
            self.assertIsInstance(domain.needs_agreement(t), bool)
            domain.check_task(t)
        self.assertEqual(mode_of(False, False), "plain")
        self.assertEqual(mode_of(True, False), "reasoning")
        self.assertEqual(mode_of(True, True), "thinking")

    def test_the_toy_subject(self):
        self.check(ARITH, TASKS, make_tasks(HELDOUT_SEED, 12))
        self.assertTrue(ARITH.verify(" 12 ", {**TASKS[0], "answer": 12}))
        self.assertFalse(ARITH.verify("12 apples", {**TASKS[0], "answer": 12}), "the whole answer is the number")
        self.assertFalse(ARITH.verify("13", {**TASKS[0], "answer": 12}))

    def test_rea(self):
        sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))
        rea = load("rea")
        tasks = fx.load_tasks()
        self.check(rea, tasks, [{"id": "crc-1001-0003"}, {"id": "xor-key-1001-0000"}])
        self.assertIsInstance(rea, Domain)
        self.assertEqual(rea.tasks_prefix, "decompiled-")

    def test_every_domain_is_a_domain_and_names_itself_as_its_package(self):
        for name in ("rea", "example_arith"):
            self.assertIsInstance(load(name), Domain)
        with self.assertRaisesRegex(SystemExit, "unknown domain"):
            load("nope")
        with self.assertRaisesRegex(SystemExit, "lower-case package name"):
            load("../x")

    def test_a_domain_without_the_required_members_cannot_be_made(self):
        class Half(Domain):
            name = "half"

            def load_tasks(self, source):
                return []
        with self.assertRaises(TypeError):
            Half()

    def test_a_domain_without_a_truth_target_stores_traces_only_and_a_plain_run_refuses(self):
        class NoTruth(type(ARITH)):
            def truth_target(self, task):
                return None
        d, t = NoTruth(), TASKS[0]
        kept, _ = records.assemble([t], {t["id"]: {"outs": [[reasoning_reply(t, 0), False], [reasoning_reply(t, 1), False]]}}, "m", lambda x: d.messages(x, "reasoning"), True, domain=d)
        self.assertEqual([r["target_source"] for r in kept], ["teacher-reasoning"] * 2)
        with self.assertRaisesRegex(ValueError, "no truth_target"):
            records.assemble([t], {t["id"]: {"outs": [[str(t["answer"]), False], [str(t["answer"]), False]]}}, "m", lambda x: [], domain=d)


class Config(unittest.TestCase):
    def test_the_shipped_configs_are_todays_runs(self):
        plain, reasoning, thinking = (config.RunConfig.load(ROOT / "distill" / "configs" / f"rea-{n}.json") for n in ("plain", "reasoning", "thinking"))
        self.assertEqual(plain.resolved(), {"mode": "plain", "sampling": {"temperature": 0.7}, "max_tokens": 300, "max_len": 2048, "slot_context": 8192, "report_windows": ()})
        self.assertEqual((plain.k, plain.teacher, plain.student), (6, "codeforge-27b-heretic-q4", "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"))
        self.assertEqual(reasoning.resolved(), {"mode": "reasoning", "sampling": {"temperature": 0.7}, "max_tokens": 768, "max_len": 3072, "slot_context": 8192, "report_windows": (2048, 3072, 4096)})
        self.assertEqual((reasoning.k, reasoning.max_traces, reasoning.quota.split(",")[0]), (4, 3, "structs=220"))
        self.assertEqual(thinking.resolved(), {"mode": "thinking", "sampling": config.THINKING_SAMPLING, "max_tokens": 4096, "max_len": 8192, "slot_context": 8192, "report_windows": (4096, 8192)})
        self.assertEqual(thinking.k, 4)
        for c in (plain, reasoning, thinking):
            self.assertEqual(c.domain, "rea")
            self.assertIn(c.teacher, [t.name for t in teachers.TEACHERS])
            self.assertEqual(config.RunConfig.load(self.write(c.dump())).dump(), c.dump(), "dump and load are inverse")

    def write(self, text):
        p = pathlib.Path(self.tmp.name) / "c.json"
        p.write_text(text)
        return p

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_a_config_that_is_wrong_is_refused_before_anything_is_rented(self):
        for text, why in (('{"mode": "thinking", "sampling": {"temperature": 0}}', "never greedy"), ('{"mode": "x"}', "mode"), ('{"k": 0}', "k 0"), ('{"typo": 1}', "unknown key"), ("[]", "not a run config"),
                          ('{"sampling": {"seed": 3}}', "only temperature"), ('{"max_len": -1}', "counts")):
            with self.assertRaisesRegex(SystemExit, why):
                config.RunConfig.load(self.write(text))

    def test_the_run_config_sets_what_the_flags_set_and_the_two_never_mix(self):
        flags = dict(config.FLAG_DEFAULTS)
        path = ROOT / "distill" / "configs" / "rea-thinking.json"
        got = config.merge_flags(path, flags, "rea")
        self.assertEqual((got["model"], got["k"], got["reasoning"], got["thinking"], got["max_tokens"], got["max_len"], got["top_k"]), ("codeforge-27b-heretic-q4", 4, True, True, 4096, 8192, 20))
        with self.assertRaisesRegex(SystemExit, "do not also give --k, --max-len"):
            config.merge_flags(path, {**flags, "k": 2, "max_len": 3072}, "rea")
        with self.assertRaisesRegex(SystemExit, "is for domain"):
            config.merge_flags(path, flags, "example_arith")
        self.assertEqual(config.merge_train_flags(path, dict(config.TRAIN_FLAG_DEFAULTS)), {"student": config.DEFAULT_STUDENT, "max_len": 8192, "mix": False})
        with self.assertRaisesRegex(SystemExit, "--student"):
            config.merge_train_flags(path, {**config.TRAIN_FLAG_DEFAULTS, "student": "x/y"})

    def test_the_flag_defaults_are_the_ones_of_the_entrypoints(self):
        """`FLAG_DEFAULTS` is what a flag left alone means: if an entrypoint's default moves, a `--run-config` run and a flag run would differ."""
        import inspect
        sys.path.insert(0, str(ROOT / "tools" / "modal"))
        fake = mock.MagicMock(name="modal")
        fake.App.return_value.function.side_effect = lambda *a, **k: (lambda f: f)
        fake.App.return_value.local_entrypoint.side_effect = lambda *a, **k: (lambda f: f)
        with mock.patch.dict(sys.modules, {"modal": fake}):
            for name in ("modal_rea", "modal_rea_data", "modal_qbench", "distill.backends.modal_train"):
                sys.modules.pop(name, None)
            try:
                import modal_rea_data
                import distill.backends.modal_train as modal_train
                got = {k: p.default for k, p in inspect.signature(modal_rea_data.main).parameters.items() if k in config.FLAG_DEFAULTS}
                self.assertEqual(got, config.FLAG_DEFAULTS)
                got = {k: p.default for k, p in inspect.signature(modal_train.main).parameters.items() if k in config.TRAIN_FLAG_DEFAULTS}
                self.assertEqual(got, config.TRAIN_FLAG_DEFAULTS)
            finally:
                for name in ("modal_rea", "modal_rea_data", "modal_qbench", "distill.backends.modal_train"):
                    sys.modules.pop(name, None)
                sys.path.remove(str(ROOT / "tools" / "modal"))

    def test_a_run_config_reproduces_the_stage_of_the_flags(self):
        """`RunConfig.sample_kwargs()` is the argument list of the stage that the flags of the same run make."""
        self.assertEqual(config.RunConfig(mode="reasoning", max_len=4096).sample_kwargs(),
                         {"reasoning": True, "thinking": False, "max_tokens": 0, "max_len": 4096, "quota": "", "max_traces": 3, "sampling": None})
        self.assertEqual(config.RunConfig(mode="thinking", max_len=8192, max_tokens=900, sampling={"top_k": 40}).sample_kwargs(),
                         {"reasoning": True, "thinking": True, "max_tokens": 900, "max_len": 8192, "quota": "", "max_traces": 3, "sampling": {"top_k": 40}})
        self.assertEqual(config.RunConfig().sample_kwargs()["reasoning"], False)

    def test_the_teacher_list_is_the_one_modal_and_kaggle_share(self):
        self.assertEqual(len(teachers.TEACHERS), 10)
        self.assertEqual(teachers.find("codeforge-27b-heretic-q4").file, "Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic.i1-Q4_K_M.gguf")
        self.assertEqual(teachers.TEACHERS[0][3], "L4")
        sys.path.insert(0, str(ROOT / "distill" / "backends" / "kaggle"))
        import run_teacher
        found = teachers.find("codeforge-27b-heretic-q4")
        self.assertEqual((run_teacher.DEFAULTS["model_repo"], run_teacher.DEFAULTS["model_file"]), (found.repo, found.file), "the kernel's own default is the same teacher")
        for t in teachers.TEACHERS:
            self.assertIn(t.gpu, ("L4", "L40S"))


if __name__ == "__main__":
    unittest.main()

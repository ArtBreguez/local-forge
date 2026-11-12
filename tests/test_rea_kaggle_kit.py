"""Round 13: the Kaggle kit (tools/kaggle): `build_tasks.py` -> `run_teacher.py` against a stub llama-server -> `kaggle-score`. Nothing here touches Kaggle, the network (the stub listens on 127.0.0.1) or a GPU.

  EndToEnd      build_tasks writes the dataset, the notebook asks a stub server with the exact messages, the thinking flag and the thinking sampling, stores `<think>...</think>` + content, and kaggle-score
                turns the replies file into records and stats
  Request       the notebook's request is the Modal stage's request; every copy of a constant is pinned to rea_bench
  StoredText    the text a thinking reply is stored as, read back by kaggle-score exactly as `reasoning.thinking_reply` builds it
  Checkpoint    resume asks only the missing and failed samples; the deadline; prompts that do not fit; failed and cut-off replies
  Server        llama.cpp for the T4: the libcuda stub fix, a prebuilt server, the command line
  Build         tasks by family with a quota, the config, the metadata, the refusals
  Score         the run.json and the prompt fingerprint in kaggle-score, the projected fit
  Readme        the commands of tools/kaggle/README.md
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import pathlib
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_tasks import right
from .test_rea_kaggle import out, write_replies, write_tasks
from .test_rea_modal import StubbedSampleStage
from .test_rea_reasoning import reply
from .test_rea_traces import think_body

KIT = ROOT / "distill" / "backends" / "kaggle"          # run_teacher.py (the kernel) lives here; tools/kaggle/build_tasks.py is the REA wrapper of build_tasks.py beside it
WRAPPERS = ROOT / "tools" / "kaggle"
sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(WRAPPERS))

import build_tasks  # noqa: E402
import run_teacher  # noqa: E402
from rea_bench import cli, families, kaggle_score  # noqa: E402
from .rea_bound import rsn, sft  # noqa: E402


def make_tasks(counts=(("crc", 4), ("structs", 4)), seed=9) -> list:
    """Decompiled task records (with pseudocode) of the given families; the task id is a comment in the pseudocode, which is how the stub server tells the tasks apart."""
    tasks = []
    for fam, n in counts:
        for t in families.generate(fam, n, seed):
            tasks.append({**t.meta(), "pseudocode": f"/* {t.id} */\nint f(int x) {{ return x + 1; }}"})
    return tasks


class StubServer:
    """A llama-server stand-in on 127.0.0.1: POST /v1/chat/completions is answered by `reply(body)` (a dict; None = HTTP 500), POST /tokenize counts words."""

    def __init__(self, reply_fn):
        self.bodies, self.lock, self.reply_fn = [], threading.Lock(), reply_fn
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    payload = {"tokens": [0] * len(body["content"].split())}
                else:
                    with outer.lock:
                        outer.bodies.append(body)
                    payload = outer.reply_fn(body)
                if payload is None:
                    self.send_response(500)
                    self.end_headers()
                    return
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def task_in(body: dict, tasks: dict) -> dict:
    return next(t for tid, t in tasks.items() if tid in body["messages"][-1]["content"])


def teacher_reply(tasks: dict, cfg: dict):
    """The stub's answer: the reasoning apart (`reasoning_content`), a visible reply with the right Answer line, a different text per task and sample."""
    def fn(body):
        t, i = task_in(body, tasks), body["seed"] - cfg["seed_base"]
        steps = ["The function reads a field at a fixed offset from the base.", "Following the loop gives the key material directly.", "Looking at the comparison the literal stands out clearly."][i % 3]
        return {"choices": [{"message": {"content": f"{steps}\nAnswer: {right(t)}", "reasoning_content": f"Let me think about {t['id']}, sample {i}. The code returns what is asked, so the value is clear."},
                             "finish_reason": "stop"}], "usage": {"completion_tokens": 40 + i, "prompt_tokens": 100}}
    return fn


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = make_tasks()
        self.decompiled = write_tasks(self.d, self.tasks, "decompiled-seed9-all.jsonl")
        self.cfg_built = build_tasks.build([self.decompiled], self.d / "kb", "someone", quota="crc=3,structs=2", k=2)
        self.data = self.d / "kb" / "data"

    def run_notebook(self, reply_fn=None, **cfg_over):
        """What the kernel does after the server is up: read the dataset, ask, checkpoint, write run.json. Returns (replies path, run.json path, stub server)."""
        user = json.loads((self.data / "config.json").read_text())
        cfg = run_teacher.resolve_config({**user, **cfg_over})
        tasks = [json.loads(line) for line in (self.data / "tasks.jsonl").read_text().splitlines()]
        by_id = {t["id"]: t for t in self.tasks}
        server = StubServer(reply_fn or teacher_reply(by_id, cfg))
        self.addCleanup(server.close)
        replies, run = self.d / "replies.jsonl", self.d / "run.json"
        summary = run_teacher.run_session(cfg, tasks, {}, replies, time.time() + 600, base_url=server.url, log_fn=lambda *a: None)
        run.write_text(json.dumps({**cfg, **summary}, indent=1))
        return replies, run, server, cfg, summary

    def test_build_tasks_writes_the_dataset_the_kernel_and_the_config(self):
        files = sorted(str(p.relative_to(self.d / "kb")) for p in (self.d / "kb").rglob("*") if p.is_file())
        self.assertEqual(files, ["data/config.json", "data/dataset-metadata.json", "data/tasks.jsonl", "kernel/kernel-metadata.json", "kernel/run.py"])
        lines = [json.loads(l) for l in (self.data / "tasks.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 5)
        by_id = {t["id"]: t for t in self.tasks}
        for line in lines:
            self.assertEqual(sorted(line), ["id", "kind", "messages"], "nothing of the task but its messages: the truth stays here")
            self.assertEqual(line["messages"], rsn.messages(by_id[line["id"]]), "the exact messages of reasoning.messages(task)")
            self.assertEqual(line["messages"][0], {"role": "system", "content": rsn.SYSTEM_REASONING})
        self.assertEqual((self.d / "kb" / "kernel" / "run.py").read_text(), (KIT / "run_teacher.py").read_text())
        cfg = json.loads((self.data / "config.json").read_text())
        self.assertEqual((cfg["thinking"], cfg["sampling"], cfg["max_tokens"], cfg["k"], cfg["ctx_per_slot"], cfg["slots"]), (True, rsn.THINKING_SAMPLING, 4096, 2, 8192, 4))
        self.assertEqual(cfg["tasks"]["by_kind"], {"crc": 3, "structs": 2})
        self.assertEqual(cfg["tasks"]["system_sha"], rsn.prompt_sha([{"role": "system", "content": rsn.SYSTEM_REASONING}]))

    def test_the_notebook_asks_the_exact_messages_with_thinking_on_and_the_thinking_sampling(self):
        replies, run, server, cfg, summary = self.run_notebook()
        self.assertEqual(len(server.bodies), 5 * 2)
        by_id = {t["id"]: t for t in self.tasks}
        for b in server.bodies:
            self.assertEqual(b["messages"], rsn.messages(task_in(b, by_id)))
            self.assertEqual({k: b[k] for k in ("temperature", "top_p", "top_k", "min_p")}, rsn.THINKING_SAMPLING)
            self.assertEqual((b["chat_template_kwargs"], b["max_tokens"], b["cache_prompt"], b["model"]), ({"enable_thinking": True}, 4096, False, "local"))
        self.assertEqual({b["seed"] for b in server.bodies}, {1234, 1235})
        self.assertEqual((summary["tasks_written"], summary["requests"], summary["cut_off"], summary["failed"], summary["skipped_too_long"], summary["deadline_hit"]), (5, 10, 0, 0, [], False))

    def test_a_reply_is_stored_as_think_block_plus_content_with_its_counts_and_the_prompt_fingerprint(self):
        replies, run, server, cfg, _ = self.run_notebook()
        lines = [json.loads(l) for l in replies.read_text().splitlines()]
        self.assertEqual(len(lines), 5)
        by_id = {t["id"]: t for t in self.tasks}
        for line in lines:
            self.assertEqual(sorted(line), ["id", "kind", "outs", "prompt_sha"])
            self.assertEqual(line["prompt_sha"], rsn.prompt_sha(rsn.messages(by_id[line["id"]])))
            for i, o in enumerate(line["outs"]):
                self.assertEqual(sorted(o), ["finish", "max_tokens", "prompt_sha", "prompt_tokens", "secs", "text", "tokens"])
                self.assertEqual((o["prompt_sha"], o["max_tokens"]), (line["prompt_sha"], 4096), "a reply says what it answered and the budget it was asked with")
                self.assertTrue(o["text"].startswith(f"<think>\nLet me think about {line['id']}, sample {i}."), o["text"])
                self.assertIn("\n</think>\n\n", o["text"])
                self.assertTrue(o["text"].endswith(f"Answer: {right(by_id[line['id']])}"))
                self.assertEqual((o["finish"], o["tokens"], o["prompt_tokens"]), ("stop", 40 + i, 100))

    def test_kaggle_score_turns_the_replies_into_records_and_stats(self):
        replies, run, server, cfg, summary = self.run_notebook()
        records, stats = kaggle_score.score(replies, self.decompiled, "teacher-x", thinking=True, run_info=kaggle_score.read_run_info(run))
        by_source = lambda s: [r for r in records if r["target_source"] == s]
        self.assertEqual((len(by_source("truth-template")), len(by_source("teacher-reasoning"))), (5, 10), "one truth record and both distinct traces of every task")
        by_id = {t["id"]: t for t in self.tasks}
        for r in records:
            self.assertEqual(r["messages"][:2], rsn.messages(by_id[r["task"]]))
            self.assertEqual(sorted(r), sorted(sft.RECORD_KEYS))
        for r in by_source("teacher-reasoning"):
            self.assertTrue(r["messages"][-1]["content"].startswith("<think>\nLet me think about"))
        self.assertEqual((stats["reasoning"]["mode"], stats["thinking"], stats["kept_tasks"]), ("thinking", True, 5))
        self.assertEqual(stats["sampling"], {"thinking": True, **rsn.THINKING_SAMPLING, "max_tokens": 4096, "seed_base": 1234, "slot_context": 8192, "parallel": 4})
        self.assertEqual(stats["prompts"], {"lines": 5, "checked": 5, "mismatch": 0, "mismatched_ids": [], "unchecked": 0, "unchecked_replies": 0, "unchecked_ids": [], "kept_unchecked": False})
        self.assertEqual(stats["projected_fit"]["windows"], {"4096": 10, "8192": 10})
        self.assertEqual((stats["projected_fit"]["records"], stats["projected_fit"]["measured"]), (10, 10))
        self.assertEqual(stats["run"]["requests"], 10)
        json.dumps(stats)

    def test_the_command_line_finds_the_run_json_beside_the_replies_and_prints_the_projected_fit(self):
        replies, run, server, cfg, _ = self.run_notebook()
        dest = self.d / "sft.jsonl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["kaggle-score", "--replies", str(replies), "--tasks", str(self.decompiled), "--model", "m", "--thinking", "--out", str(dest)])
        text = buf.getvalue()
        self.assertTrue(dest.exists(), "the subcommand wrote the records")
        stats = json.loads((self.d / "sft.stats.json").read_text())
        self.assertEqual(stats["sampling"]["temperature"], 0.6, "run.json beside replies.jsonl was read")
        self.assertIn("projected fit of the teacher-reasoning records", text)
        self.assertIn("4096 tokens: 10 of 10 (100%)", text)
        self.assertIn("8192 tokens: 10 of 10 (100%)", text)


class Request(StubbedSampleStage):
    def test_the_notebooks_request_is_the_modal_stages_request(self):
        self.reply = lambda tid, i, body: think_body(reply(self.tasks[tid], "Step.\nRead once."), "Reasoning.")
        self.run_stage(False, reasoning=True, thinking=True, max_len=8192)
        cfg = run_teacher.resolve_config({})
        for b in self.bodies:
            t = next(t for tid, t in self.tasks.items() if tid in b["messages"][-1]["content"])
            mine = run_teacher.request_body(cfg, rsn.messages(t), b["seed"] - 1234)
            self.assertEqual(mine, b)
            self.assertEqual(list(mine), list(b), "same keys in the same order")

    def test_the_visible_reasoning_request_too(self):
        self.reply = lambda tid, i, body: {"choices": [{"message": {"content": reply(self.tasks[tid])}, "finish_reason": "stop"}]}
        self.run_stage(False, reasoning=True, max_len=4096)
        cfg = run_teacher.resolve_config({"thinking": False})
        for b in self.bodies:
            t = next(t for tid, t in self.tasks.items() if tid in b["messages"][-1]["content"])
            self.assertEqual(run_teacher.request_body(cfg, rsn.messages(t), b["seed"] - 1234), b)


class Pins(unittest.TestCase):
    """The notebook has no copy of rea_bench on Kaggle: every constant and function it repeats is pinned to the original."""

    def test_the_constants(self):
        self.assertEqual(run_teacher.THINKING_SAMPLING, rsn.THINKING_SAMPLING)
        self.assertEqual(list(run_teacher.THINKING_SAMPLING), list(rsn.THINKING_SAMPLING))
        self.assertEqual(run_teacher.DEFAULT_SAMPLING, rsn.DEFAULT_SAMPLING)
        self.assertEqual(run_teacher.LENGTH_OVERHEAD, sft.LENGTH_OVERHEAD)
        self.assertEqual(run_teacher.SAMPLING_KEYS, rsn.SAMPLING_KEYS)
        self.assertEqual(run_teacher.DEFAULTS["ctx_per_slot"], rsn.slot_context(rsn.DEFAULT_THINKING_MAX_TOKENS))
        self.assertEqual(run_teacher.resolve_config({})["max_tokens"], rsn.DEFAULT_THINKING_MAX_TOKENS)
        self.assertEqual(run_teacher.resolve_config({"thinking": False})["max_tokens"], rsn.DEFAULT_MAX_TOKENS)
        self.assertEqual(run_teacher.DEFAULTS["seed_base"], 1234)

    def test_the_prompt_fingerprint(self):
        for t in make_tasks((("crc", 2),)):
            self.assertEqual(run_teacher.prompt_sha(rsn.messages(t)), rsn.prompt_sha(rsn.messages(t)))
        self.assertEqual(run_teacher.prompt_sha([]), rsn.prompt_sha([]))

    def test_the_default_model_is_the_teacher_of_the_plan(self):
        self.assertIn("CodeForge", run_teacher.DEFAULTS["model_repo"])
        self.assertTrue(run_teacher.DEFAULTS["model_file"].endswith("Q4_K_M.gguf"))

    def test_the_kernel_uses_nothing_but_the_standard_library(self):
        src = (KIT / "run_teacher.py").read_text()
        imports = set(re.findall(r"^(?:import|from) (\w+)", src, re.M)) - {"__future__"}
        self.assertLessEqual(imports, {"glob", "hashlib", "json", "os", "pathlib", "shutil", "subprocess", "sys", "threading", "time", "urllib", "concurrent"})
        self.assertNotIn("rea_bench", src.replace("rea_bench/reasoning.py", "").replace("rea_bench.reasoning", "").replace("rea_bench.sft", "").replace("rea_bench has", "").replace("copy of rea_bench", ""))


class StoredText(unittest.TestCase):
    CASES = [("Steps.\nAnswer: 8", "Let me look."), ("Steps.\nAnswer: 8", "  Wait.\n\nSo it is 8.  "), ("  Steps.\nAnswer: 8\n", "r"), ("", "only reasoning, cut"), ("Steps.\nAnswer: 8", ""), ("Steps.\nAnswer: 8", None),
             ("<think>inline</think>\n\nSteps.\nAnswer: 8", None), ("inline reasoning\n</think>\n\nSteps.\nAnswer: 8", ""), (None, "r")]

    def test_the_text_kaggle_score_builds_the_same_reply_from(self):
        for content, reasoning in self.CASES:
            stored = run_teacher.stored_text(content, reasoning, True)
            self.assertEqual(rsn.thinking_reply(stored, ""), rsn.thinking_reply(content, reasoning), (content, reasoning))
            self.assertEqual(rsn.reply_text(stored, "stop", "", True), rsn.reply_text(content, "stop", reasoning, True), (content, reasoning))
            self.assertEqual(kaggle_score.reply_of({"text": stored, "finish": "stop"}, True), rsn.reply_text(content, "stop", reasoning, True))

    def test_the_form_is_the_trace_form(self):
        self.assertEqual(run_teacher.stored_text("Steps.\nAnswer: 8", "Let me look.", True), "<think>\nLet me look.\n</think>\n\nSteps.\nAnswer: 8")
        self.assertEqual(run_teacher.stored_text("Steps.\nAnswer: 8", "Let me look.", True), rsn.compose_trace("Let me look.", "Steps.\nAnswer: 8"))
        self.assertEqual(run_teacher.stored_text("Steps.", "", True), "Steps.", "no reasoning from the server: the content as it is (kaggle-score drops the reply as no_think)")
        self.assertEqual(run_teacher.stored_text("Steps.\nAnswer: 8", "ignored", False), "Steps.\nAnswer: 8", "a run that does not think stores the content")

    def test_a_cut_off_thinking_reply_is_rejected_by_kaggle_score_whatever_was_stored(self):
        t = make_tasks((("crc", 1),))[0]
        for content in ("", f"Steps.\nAnswer: {right(t)}"):
            o = {"text": run_teacher.stored_text(content, "Still thinking about it", True), "finish": "length"}
            self.assertEqual(rsn.verdict(kaggle_score.reply_of(o, True), t), "truncated")


# ---- the checkpoint, the resume, the deadline -----------------------------------------------------------------------------------------------------

class FakeServer:
    """`urlopen` for ask_one / prompt_tokens without a socket, and a clock: every chat request takes `secs` seconds of the fake clock. `answers[(task id, sample)]` overrides the default reply."""

    def __init__(self, tasks, cfg, secs=1.0):
        self.tasks, self.cfg, self.secs, self.now, self.asked, self.answers = {t["id"]: t for t in tasks}, cfg, secs, 1000.0, [], {}
        self.lock = threading.Lock()

    def clock(self):
        return self.now

    def urlopen(self, req, timeout=None):
        body = json.loads(req.data)
        if req.full_url.endswith("/tokenize"):
            payload = {"tokens": [0] * len(body["content"].split())}
        else:
            t = next(t for tid, t in self.tasks.items() if tid in body["messages"][-1]["content"])
            i = body["seed"] - self.cfg["seed_base"]
            with self.lock:
                self.asked.append((t["id"], i))
                self.now += self.secs
            payload = self.answers.get((t["id"], i)) or teacher_reply(self.tasks, self.cfg)(body)
            if payload == "fail":
                raise OSError("connection refused")
        return types.SimpleNamespace(read=lambda *a: json.dumps(payload).encode())


class Checkpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = make_tasks((("crc", 3), ("structs", 2)))
        self.cfg = run_teacher.resolve_config({"k": 2, "slots": 1})
        self.jobs = [{"id": t["id"], "kind": t["kind"], "messages": rsn.messages(t)} for t in self.tasks]
        self.fake = FakeServer(self.tasks, self.cfg)
        patch = mock.patch.object(run_teacher, "time", types.SimpleNamespace(time=self.fake.clock, strftime=time.strftime, gmtime=time.gmtime))
        patch.start()
        self.addCleanup(patch.stop)

    def session(self, previous=None, deadline=1e9, jobs=None, cfg=None):
        ck = self.d / "replies.jsonl"
        summary = run_teacher.run_session(cfg or self.cfg, jobs or self.jobs, previous or {}, ck, deadline, urlopen=self.fake.urlopen, log_fn=lambda *a: None)
        return summary, run_teacher.load_previous([ck])

    def ok(self, tid, i, text=None):
        """An earlier reply as the current kernel writes it: with the fingerprint of the task's prompt and the budget it was asked with (`ask_one`)."""
        job = next(j for j in self.jobs if j["id"] == tid)
        return {"text": text or f"<think>\nearlier {tid} {i}\n</think>\n\nSteps {i}.\nAnswer: 1", "finish": "stop", "tokens": 5, "prompt_tokens": 50, "secs": 1.0,
                "prompt_sha": run_teacher.prompt_sha(job["messages"]), "max_tokens": self.cfg["max_tokens"]}

    def failed(self):
        return {"text": "[error OSError: connection refused]", "finish": "error", "tokens": 0, "prompt_tokens": 0, "secs": 0.1}

    def test_every_sample_of_every_task_is_asked_once_and_checkpointed(self):
        summary, got = self.session()
        self.assertEqual(sorted(self.fake.asked), sorted((j["id"], i) for j in self.jobs for i in range(2)))
        self.assertEqual((summary["asked"], summary["tasks_written"], summary["requests"]), (5, 5, 10))
        self.assertTrue(all(len(r["outs"]) == 2 for r in got.values()))

    def test_resume_asks_only_the_missing_and_the_failed_samples(self):
        a, b, c, d = (j["id"] for j in self.jobs[:4])
        previous = {a: {"id": a, "kind": "crc", "outs": [self.ok(a, 0), self.ok(a, 1)]},                    # complete: not asked
                    b: {"id": b, "kind": "crc", "outs": [self.ok(b, 0), self.failed()]},                    # sample 1 failed: asked again
                    c: {"id": c, "kind": "crc", "outs": [self.ok(c, 0)]},                                   # sample 1 missing
                    d: {"id": d, "kind": "structs", "outs": [{**self.ok(d, 0), "finish": "length", "text": "cut\n[truncated]"}, self.ok(d, 1)]}}      # cut off: an answer, not asked again
        summary, got = self.session(previous)
        self.assertEqual(sorted(self.fake.asked), sorted([(b, 1), (c, 1), (self.jobs[4]["id"], 0), (self.jobs[4]["id"], 1)]))
        self.assertEqual(got[a]["outs"], previous[a]["outs"], "the complete task is as it was")
        self.assertEqual(got[b]["outs"][0], previous[b]["outs"][0])
        self.assertTrue(got[b]["outs"][1]["text"].startswith("<think>\nLet me think about"), "the failed sample was answered this time")
        self.assertEqual(got[d]["outs"][0]["finish"], "length")
        self.assertEqual(summary["asked"], 3)

    def test_a_session_that_asks_nothing_loses_nothing(self):
        previous = {j["id"]: {"id": j["id"], "kind": j["kind"], "outs": [self.ok(j["id"], 0), self.ok(j["id"], 1)]} for j in self.jobs}
        summary, got = self.session(previous)
        self.assertEqual((self.fake.asked, summary["asked"], got), ([], 0, previous))

    def test_a_failed_request_is_an_error_entry_that_is_never_an_answer(self):
        tid = self.jobs[0]["id"]
        self.fake.answers[(tid, 0)] = "fail"
        summary, got = self.session()
        o = got[tid]["outs"][0]
        self.assertEqual((o["finish"], o["tokens"]), ("error", 0))
        self.assertTrue(o["text"].startswith("[error OSError"))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(run_teacher.samples_to_ask(got[tid]["outs"], 2), [0], "a resume asks it again")
        self.assertEqual(rsn.score_reasoning(kaggle_score.reply_of(o, True), self.tasks[0]), False)

    def test_a_cut_off_reply_keeps_its_finish_and_is_counted(self):
        tid = self.jobs[0]["id"]
        self.fake.answers[(tid, 1)] = {"choices": [{"message": {"content": "", "reasoning_content": "Thinking and thinking"}, "finish_reason": "length"}], "usage": {"completion_tokens": 4096, "prompt_tokens": 80}}
        summary, got = self.session()
        self.assertEqual((got[tid]["outs"][1]["finish"], got[tid]["outs"][1]["tokens"], summary["cut_off"]), ("length", 4096, 1))
        self.assertEqual(run_teacher.samples_to_ask(got[tid]["outs"], 2), [], "asking again with the same seed and budget gives the same cut")

    def test_the_deadline_stops_new_requests_and_the_samples_answered_so_far_are_kept(self):
        cfg = {**self.cfg, "k": 3}
        start = self.fake.now
        self.fake.secs = 10.0          # each request takes 10 s of the fake clock: the first starts at 0, the second at 10 (10 + the mean 10 <= 25), the third would end at 30 > 25
        summary, got = self.session(deadline=start + 25, cfg=cfg, jobs=self.jobs[:1])
        self.assertEqual(len(got[self.jobs[0]["id"]]["outs"]), 2)
        self.assertEqual((summary["deadline_hit"], summary["requests"]), (True, 2))
        self.assertEqual(run_teacher.samples_to_ask(got[self.jobs[0]["id"]]["outs"], 3), [2], "the next session asks the one that is left")

    def test_a_deadline_that_has_passed_asks_nothing_and_writes_no_line_for_it(self):
        summary, got = self.session(deadline=self.fake.now - 1)
        self.assertEqual((self.fake.asked, got, summary["deadline_hit"], summary["tasks_written"]), ([], {}, True, 0))

    def test_the_rule_of_the_deadline(self):
        self.assertTrue(run_teacher.can_start(100.0, 200.0, 0.0))
        self.assertTrue(run_teacher.can_start(100.0, 200.0, 100.0))
        self.assertFalse(run_teacher.can_start(100.0, 200.0, 100.1))
        self.assertFalse(run_teacher.can_start(201.0, 200.0, 0.0))

    def test_a_task_whose_prompt_leaves_no_room_for_max_tokens_in_its_slot_is_not_asked(self):
        big = {**self.jobs[0], "messages": [self.jobs[0]["messages"][0], {"role": "user", "content": "word " * 5000 + self.jobs[0]["id"]}]}
        summary, got = self.session(jobs=[big, *self.jobs[1:]])
        self.assertEqual(summary["skipped_too_long"], [big["id"]])
        self.assertNotIn(big["id"], {i for i, _ in self.fake.asked})
        self.assertEqual((summary["asked"], len(got)), (4, 4))
        summary, got = self.session(jobs=[big], cfg={**self.cfg, "max_tokens": 2000})
        self.assertEqual((summary["skipped_too_long"], summary["asked"]), ([], 1), "a smaller budget fits it")

    def test_the_session_counts_what_it_did_for_run_json(self):
        summary, _ = self.session()
        self.assertEqual(sorted(summary), sorted(["asked", "cut_off", "deadline_hit", "failed", "completion_tokens", "requests", "seconds", "skipped_too_long", "stale", "tasks", "tasks_written", "tok_per_s", "unfingerprinted"]))
        self.assertEqual((summary["tasks"], summary["completion_tokens"]), (5, 5 * (40 + 41)))

    def test_read_lines_skips_a_line_cut_off_by_a_timeout_and_the_last_line_of_an_id_wins(self):
        p = self.d / "r.jsonl"
        p.write_text('{"id": "a", "kind": "k", "outs": [1]}\nnot json\n{"id": "a", "kind": "k", "outs": [1, 2]}\n{"id": 5, "outs": []}\n{"id": "b"}\n{"id": "c", "kind": "k", "outs": [3]}\n{"id": "d", "kind": "k", "o')
        self.assertEqual([r["id"] for r in run_teacher.read_lines(p)], ["a", "a", "c"])
        q = self.d / "q.jsonl"
        q.write_text('{"id": "c", "kind": "k", "outs": [9, 9]}\n')
        got = run_teacher.load_previous([p, q])
        self.assertEqual((got["a"]["outs"], got["c"]["outs"]), ([1, 2], [9, 9]))

    def test_every_replies_file_of_an_earlier_session_is_found(self):
        (self.d / "in" / "kernel-a").mkdir(parents=True)
        (self.d / "in" / "data").mkdir()
        (self.d / "in" / "kernel-a" / "replies.jsonl").write_text("{}\n")
        (self.d / "in" / "data" / "tasks.jsonl").write_text("")
        self.assertEqual(run_teacher.find_previous(self.d / "in"), [self.d / "in" / "kernel-a" / "replies.jsonl"])
        self.assertEqual(run_teacher.find_previous(self.d / "absent"), [])


class Config(unittest.TestCase):
    def test_a_thinking_config_follows_the_thinking_defaults(self):
        c = run_teacher.resolve_config({})
        self.assertEqual((c["thinking"], c["sampling"], c["max_tokens"], c["k"]), (True, rsn.THINKING_SAMPLING, 4096, 4))
        self.assertIsNot(c["sampling"], run_teacher.THINKING_SAMPLING)

    def test_a_run_that_does_not_think_follows_phase_as_defaults(self):
        c = run_teacher.resolve_config({"thinking": False})
        self.assertEqual((c["thinking"], c["sampling"], c["max_tokens"]), (False, {"temperature": 0.7}, 768))
        self.assertEqual(run_teacher.request_body(c, [], 0)["chat_template_kwargs"], {"enable_thinking": False})

    def test_explicit_values_win_and_the_flat_keys_of_an_older_config_are_folded_in(self):
        c = run_teacher.resolve_config({"sampling": {"temperature": 0.8, "top_p": 0.9}, "max_tokens": 3000})
        self.assertEqual((c["sampling"], c["max_tokens"]), ({"temperature": 0.8, "top_p": 0.9}, 3000))
        c = run_teacher.resolve_config({"temperature": 0.5, "top_p": 0.9, "top_k": 10, "min_p": 0.1, "thinking": True})
        self.assertEqual(c["sampling"], {"temperature": 0.5, "top_p": 0.9, "top_k": 10, "min_p": 0.1})

    def test_a_thinking_run_is_never_greedy(self):
        for over in ({"sampling": {"temperature": 0}}, {"sampling": {"temperature": 0.6, "top_k": 1}}, {"temperature": 0.0}):
            with self.assertRaisesRegex(ValueError, "never greedy"):
                run_teacher.resolve_config(over)
        self.assertEqual(run_teacher.resolve_config({"thinking": False, "sampling": {"temperature": 0}})["sampling"], {"temperature": 0})


# ---- llama.cpp for the T4 ---------------------------------------------------------------------------------------------------------------------------

class Server(unittest.TestCase):
    def test_the_libcuda_stub_fix_finds_the_toolkits_stub_before_the_drivers_copy(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "stubs").mkdir()
            (d / "nvidia").mkdir()
            (d / "nvidia" / "libcuda.so.1").write_text("")
            pats = (str(d / "stubs" / "libcuda.so"), str(d / "nvidia" / "libcuda.so*"))
            self.assertEqual(run_teacher.find_libcuda(pats), str(d / "nvidia" / "libcuda.so.1"))
            (d / "stubs" / "libcuda.so").write_text("")
            self.assertEqual(run_teacher.find_libcuda(pats), str(d / "stubs" / "libcuda.so"), "the stub is first")
            self.assertIsNone(run_teacher.find_libcuda((str(d / "none" / "libcuda.so*"),)))
        self.assertIn("/usr/local/nvidia/lib64/libcuda.so*", run_teacher.LIBCUDA_PATTERNS, "where the log of the pilot run found it")

    def test_the_build_passes_the_driver_library_and_targets_the_t4(self):
        src = (KIT / "run_teacher.py").read_text()
        for part in ("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUDA_cuda_driver_LIBRARY=", "--target llama-server", "-DLLAMA_CURL=OFF"):
            self.assertIn(part, src)

    def test_a_prebuilt_server_from_an_attached_dataset_is_used_instead_of_a_build(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            self.assertIsNone(run_teacher.find_prebuilt(d))
            (d / "rea-kernel" / "llama-bin").mkdir(parents=True)
            (d / "rea-kernel" / "llama-bin" / "llama-server").write_text("")
            self.assertEqual(run_teacher.find_prebuilt(d), d / "rea-kernel" / "llama-bin" / "llama-server")
            self.assertIsNone(run_teacher.find_prebuilt(d / "absent"))

    def test_an_attached_server_is_copied_made_executable_and_tried_before_it_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src = d / "in" / "llama-bin"
            src.mkdir(parents=True)
            (src / "llama-server").write_text("#!/bin/sh\nexit 0\n")
            (src / "libggml-cuda.so").write_text("lib")
            (src / "llama-server").chmod(0o644)                                   # a mount that lost the executable bit
            new = run_teacher.prepare_prebuilt(src / "llama-server", d / "tmp" / "copy")
            self.assertEqual((new, (d / "tmp" / "copy" / "libggml-cuda.so").exists()), (d / "tmp" / "copy" / "llama-server", True))
            self.assertTrue(run_teacher.binary_works(new), "it starts")
            (src / "llama-server").write_text("#!/bin/sh\nexit 3\n")
            self.assertFalse(run_teacher.binary_works(run_teacher.prepare_prebuilt(src / "llama-server", d / "tmp" / "copy2")), "one that fails is not used")
            self.assertFalse(run_teacher.binary_works(d / "absent"), "one that is not there is not used")

    def test_the_command_line_has_jinja_the_reasoning_format_and_one_slot_per_request(self):
        cfg = run_teacher.resolve_config({"slots": 4, "ctx_per_slot": 8192})
        cmd = run_teacher.server_command("/x/llama-server", "/m.gguf", cfg)
        self.assertEqual(cmd[:3], ["/x/llama-server", "-m", "/m.gguf"])
        self.assertEqual(cmd[cmd.index("-c") + 1], "32768")
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "4")
        self.assertIn("--jinja", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "deepseek")
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "layer")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "999")

    def test_the_slot_holds_the_prompt_and_max_tokens(self):
        self.assertGreaterEqual(run_teacher.DEFAULTS["ctx_per_slot"], run_teacher.resolve_config({})["max_tokens"] + rsn.SLOT_PROMPT_ROOM)


# ---- build_tasks ------------------------------------------------------------------------------------------------------------------------------------

class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = make_tasks((("crc", 5), ("structs", 5), ("pktsum", 3)))
        self.file = write_tasks(self.d, self.tasks, "decompiled-seed9-all.jsonl")

    def ids(self, out):
        return [json.loads(l)["id"] for l in (pathlib.Path(out) / "data" / "tasks.jsonl").read_text().splitlines()]

    def test_the_quota_picks_the_first_tasks_of_each_family_in_the_order_of_the_modal_stage(self):
        build_tasks.build([self.file], self.d / "o", "me", quota="crc=3,structs=2")
        want = [t["id"] for t in rsn.apply_quota(sorted(self.tasks, key=lambda t: f"{t['id']}.json"), {"crc": 3, "structs": 2})]
        self.assertEqual(self.ids(self.d / "o"), want)
        self.assertEqual([i.split("-")[0] for i in want][:4], ["crc", "structs", "crc", "structs"], "round-robin over the families")
        self.assertNotIn("pktsum", " ".join(want), "no task of a family that is not listed")

    def test_without_a_quota_every_task_is_taken(self):
        build_tasks.build([self.file], self.d / "o", "me")
        self.assertEqual(sorted(self.ids(self.d / "o")), sorted(t["id"] for t in self.tasks))

    def test_the_config_follows_thinking(self):
        cfg = build_tasks.build([self.file], self.d / "o", "me", quota="crc=1", k=3, deadline_h=9.0)
        self.assertEqual((cfg["thinking"], cfg["sampling"], cfg["max_tokens"], cfg["k"], cfg["deadline_h"], cfg["ctx_per_slot"]), (True, rsn.THINKING_SAMPLING, 4096, 3, 9.0, 8192))
        cfg = build_tasks.build([self.file], self.d / "v", "me", quota="crc=1", thinking=False)
        self.assertEqual((cfg["thinking"], cfg["sampling"], cfg["max_tokens"]), (False, {"temperature": 0.7}, 768))
        cfg = build_tasks.build([self.file], self.d / "m", "me", quota="crc=1", max_tokens=6144, sampling_overrides={"temperature": 0.8, "top_k": None})
        self.assertEqual((cfg["max_tokens"], cfg["ctx_per_slot"], cfg["sampling"]["temperature"], cfg["sampling"]["top_k"]), (6144, rsn.slot_context(6144), 0.8, 20))
        for kw in ({"sampling_overrides": {"temperature": 0}}, {"sampling_overrides": {"top_k": 1}}):
            with self.assertRaisesRegex(SystemExit, "never greedy"):
                build_tasks.build([self.file], self.d / "g", "me", **kw)
        self.assertFalse((self.d / "g").exists(), "nothing is written for a refused config")

    def test_the_metadata_names_the_owner_the_dataset_and_the_earlier_kernel_to_resume_from(self):
        build_tasks.build([self.file], self.d / "o", "some-owner", quota="crc=1", dataset_slug="my-tasks", kernel_slug="my-kernel", resume_from="some-owner/old-run")
        ds = json.loads((self.d / "o" / "data" / "dataset-metadata.json").read_text())
        kn = json.loads((self.d / "o" / "kernel" / "kernel-metadata.json").read_text())
        self.assertEqual((ds["id"], ds["title"]), ("some-owner/my-tasks", "my-tasks"))
        self.assertEqual((kn["id"], kn["code_file"], kn["is_private"], kn["enable_gpu"], kn["enable_internet"]), ("some-owner/my-kernel", "run.py", True, True, True))
        self.assertEqual((kn["dataset_sources"], kn["kernel_sources"]), (["some-owner/my-tasks"], ["some-owner/old-run"]))
        build_tasks.build([self.file], self.d / "p", "some-owner", quota="crc=1")
        self.assertEqual(json.loads((self.d / "p" / "kernel" / "kernel-metadata.json").read_text())["kernel_sources"], [])

    def test_the_refusals(self):
        held = write_tasks(self.d, make_tasks((("crc", 2),), seed=1001), "decompiled-seed1001-crc.jsonl")
        with self.assertRaisesRegex(SystemExit, "held-out"):
            build_tasks.build([held], self.d / "h", "me")
        renamed = write_tasks(self.d, make_tasks((("crc", 2),), seed=1001), "decompiled-something.jsonl")
        with self.assertRaisesRegex(SystemExit, "held-out"):
            build_tasks.build([renamed], self.d / "h2", "me")                       # the task ids say it
        named = write_tasks(self.d, make_tasks((("crc", 2),), seed=9), "decompiled-seed1001-crc.jsonl")
        with self.assertRaisesRegex(SystemExit, "held-out"):
            build_tasks.build([named], self.d / "h3", "me")                         # the file name says it
        self.assertFalse(any((self.d / x).exists() for x in ("h", "h2", "h3")))
        for kw, why in ((dict(quota="nosuchkind=3"), "no task kind|not a task kind|known"), (dict(quota="crc=0"), "positive"), (dict(owner="Bad Owner"), "--owner"), (dict(k=0), "--k"),
                        (dict(resume_from="nonsense"), "--resume-from"), (dict(dataset_slug="X"), "--dataset-slug")):
            with self.assertRaisesRegex(SystemExit, why):
                build_tasks.build([self.file], self.d / "r", **{"owner": "me", **kw})
        with self.assertRaisesRegex(SystemExit, "no decompiled task"):
            build_tasks.build([write_tasks(self.d, [], "decompiled-seed9-none.jsonl")], self.d / "r", "me")

    def test_an_earlier_build_is_not_overwritten_without_force(self):
        build_tasks.build([self.file], self.d / "o", "me", quota="crc=1")
        before = (self.d / "o" / "data" / "tasks.jsonl").read_bytes()
        with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
            build_tasks.build([self.file], self.d / "o", "me", quota="crc=2")
        self.assertEqual((self.d / "o" / "data" / "tasks.jsonl").read_bytes(), before)
        build_tasks.build([self.file], self.d / "o", "me", quota="crc=2", force=True)
        self.assertEqual(len(self.ids(self.d / "o")), 2)

    def test_the_command_line(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_tasks.main(["--tasks", str(self.file), "--owner", "me", "--out", str(self.d / "c"), "--quota", "crc=2,structs=2", "--k", "3", "--temperature", "0.5"])
        cfg = json.loads((self.d / "c" / "data" / "config.json").read_text())
        self.assertEqual((cfg["k"], cfg["sampling"]["temperature"], cfg["tasks"]["n"]), (3, 0.5, 4))
        self.assertIn("kaggle datasets create -p", buf.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            build_tasks.main(["--tasks", str(self.file), "--owner", "me", "--out", str(self.d / "n"), "--no-thinking", "--quota", "crc=1"])
        self.assertEqual(json.loads((self.d / "n" / "data" / "config.json").read_text())["thinking"], False)


# ---- kaggle-score: the run.json, the fingerprint, the projection ------------------------------------------------------------------------------------

class Score(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = make_tasks((("structs", 3),))
        self.decompiled = write_tasks(self.d, self.tasks, "decompiled-seed9-all.jsonl")

    def line(self, t, think=True, sha=True, tokens=(500, 300), prompt_tokens=200, wrong=False):
        steps = ["The function reads a field at a fixed offset from the base.", "Following the loop gives the key material directly."]
        outs = []
        for i, n in enumerate(tokens):
            body = reply(t, steps[i], "0xbad" if wrong else None)
            text = run_teacher.stored_text(body, f"Reasoning {i} about {t['id']}: it reads the field.", True) if think else body
            outs.append(out(text, tokens=n, prompt_tokens=prompt_tokens))
        rec = {"id": t["id"], "kind": t["kind"], "outs": outs}
        if sha:
            rec["prompt_sha"] = rsn.prompt_sha(rsn.messages(t))
        return rec

    def score(self, lines, **kw):
        return kaggle_score.score(write_replies(self.d, lines), self.decompiled, "m", **{"thinking": True, **kw})

    def test_a_reply_to_another_prompt_is_skipped_and_counted_never_stored_under_this_one(self):
        good, bad = self.line(self.tasks[0]), {**self.line(self.tasks[1]), "prompt_sha": "0" * 16}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            records, stats = self.score([good, bad])
        self.assertEqual({r["task"] for r in records}, {self.tasks[0]["id"]})
        self.assertEqual(stats["prompts"], {**{"lines": 2, "checked": 2, "mismatch": 0, "mismatched_ids": [], "unchecked": 0, "unchecked_replies": 0, "unchecked_ids": [], "kept_unchecked": False}, "mismatch": 1, "mismatched_ids": [self.tasks[1]["id"]]})
        self.assertIn("answered a prompt other than", stderr.getvalue())
        self.assertEqual(stats["tasks"], 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "disagree on reasoning.messages"):
            self.score([bad])

    def test_a_file_from_an_older_notebook_has_no_fingerprint_and_is_skipped_unless_it_is_kept_on_purpose(self):
        """Round 12, M1: nothing shows that its replies answered this prompt. They are counted (`prompts.unchecked`), and skipped like a mismatch; `keep_unchecked` is the owner's word that they did."""
        lines = [self.line(t, sha=False) for t in self.tasks]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "3 line.s. have no fingerprint at all.*--keep-unchecked"):
            self.score(lines)
        self.assertIn("3 replies line(s) hold 6 repl(ies) with no prompt fingerprint", stderr.getvalue())
        records, stats = self.score(lines, keep_unchecked=True)
        self.assertEqual((stats["prompts"]["checked"], stats["prompts"]["mismatch"], stats["kept_tasks"]), (0, 0, 3))
        self.assertEqual((stats["prompts"]["unchecked"], stats["prompts"]["unchecked_replies"], stats["prompts"]["kept_unchecked"]), (3, 6, True), "kept, and still counted")

    def test_the_thinking_flag_must_agree_with_the_notebooks_own_record(self):
        lines = [self.line(t) for t in self.tasks]
        info = {"thinking": True, "sampling": rsn.THINKING_SAMPLING, "max_tokens": 4096}
        with self.assertRaisesRegex(SystemExit, "pass --thinking"):
            self.score(lines, thinking=False, run_info=info)
        with self.assertRaisesRegex(SystemExit, "leave --thinking out"):
            self.score(lines, thinking=True, run_info={**info, "thinking": False})
        records, stats = self.score(lines, thinking=True, run_info=info)
        self.assertEqual(stats["sampling"]["max_tokens"], 4096)
        records, stats = self.score(lines, thinking=True)
        self.assertNotIn("sampling", stats, "no run.json: nothing is claimed")

    def test_run_info_is_read_from_a_file_and_a_bad_one_is_refused(self):
        p = self.d / "run.json"
        self.assertIsNone(kaggle_score.read_run_info(p))
        p.write_text('{"thinking": true}')
        self.assertEqual(kaggle_score.read_run_info(p), {"thinking": True})
        p.write_text("[1]")
        with self.assertRaisesRegex(SystemExit, "run.json"):
            kaggle_score.read_run_info(p)
        p.write_text("{")
        with self.assertRaisesRegex(SystemExit, "run.json"):
            kaggle_score.read_run_info(p)

    def test_the_projected_fit_counts_prompt_plus_completion_tokens_per_record(self):
        # structs/linked_next-like tasks have chance < 0.05 here: both distinct replies are kept. tokens 500 + 200 prompt = 700; 7,900 + 200 = 8,100 fits 8192 only; 10,000 + 200 fits neither
        lines = [self.line(self.tasks[0], tokens=(500, 300)), self.line(self.tasks[1], tokens=(3900, 7900)), self.line(self.tasks[2], tokens=(10000, 300))]
        records, stats = self.score(lines)
        teacher = [r for r in records if r["target_source"] == "teacher-reasoning"]
        f = stats["projected_fit"]
        self.assertEqual((f["records"], f["measured"]), (len(teacher), len(teacher)))
        sizes = sorted(n + 200 for line in lines for n in (o["tokens"] for o in line["outs"]))
        want = lambda w: sum(n <= w for n in sizes)
        self.assertEqual(f["windows"], {"4096": want(4096), "8192": want(8192)})
        self.assertGreater(f["windows"]["8192"], f["windows"]["4096"])
        line = kaggle_score.projected_fit_line(stats)
        self.assertIn(f"4096 tokens: {want(4096)} of {len(teacher)}", line)
        self.assertIn(f"8192 tokens: {want(8192)} of {len(teacher)}", line)

    def test_a_reply_without_counts_is_not_counted_as_fitting_or_not(self):
        lines = [self.line(self.tasks[0])]
        for o in lines[0]["outs"]:
            del o["prompt_tokens"]
        _, stats = self.score(lines)
        self.assertEqual((stats["projected_fit"]["measured"], stats["projected_fit"]["windows"]), (0, {"4096": 0, "8192": 0}))
        self.assertIn("no token counts", kaggle_score.projected_fit_line(stats))

    def test_a_visible_reasoning_scoring_projects_to_its_own_windows(self):
        lines = [self.line(t, think=False, sha=True) for t in self.tasks]
        _, stats = kaggle_score.score(write_replies(self.d, lines), self.decompiled, "m", thinking=False)
        self.assertEqual(list(stats["projected_fit"]["windows"]), ["2048", "3072", "4096"])


# ---- the README of the kit --------------------------------------------------------------------------------------------------------------------------

class Readme(unittest.TestCase):
    def setUp(self):
        self.text = (WRAPPERS / "README.md").read_text()

    def test_it_has_the_commands_with_the_kaggle_cli(self):
        for part in ("build_tasks.py", "kaggle datasets create -p", "kaggle datasets version -p", "kaggle kernels push -p", "kaggle kernels status", "kaggle kernels output", "--resume-from",
                     "kaggle-score --replies", "--thinking", "run.json", "replies.jsonl", "~/.kaggle/access_token", "llama-bin", "never print", "--quota", "ctx_per_slot", "deadline"):
            self.assertIn(part.lower(), self.text.lower(), part)

    def test_it_never_holds_a_token(self):
        self.assertNotRegex(self.text, r"KGAT_[0-9a-f]{8,}|\"key\"\s*:\s*\"[0-9a-f]{20,}")
        self.assertNotRegex(self.text, r"[0-9a-f]{32}")

    def test_the_commands_name_files_that_exist(self):
        for name in ("run_teacher.py", "build_tasks.py"):
            self.assertTrue((KIT / name).exists())
            self.assertIn(name, self.text)


if __name__ == "__main__":
    unittest.main()

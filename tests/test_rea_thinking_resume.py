"""Round 15, M1 of the review of the thinking-pilot fixes: a resume must not launder the prompt fingerprint or keep a reply cut off at an old budget.

The line of a resumed task used to be stamped with the CURRENT prompt's fingerprint while its earlier replies (another session, maybe another prompt) were copied into it, and a reply cut off at 4096 was "an
answer" for a family whose budget had become 8192. Now each reply carries its own `prompt_sha` and `max_tokens`; a reply is asked again when it answered another prompt, or was cut off at a smaller budget than
the task's now; `kaggle-score` checks the fingerprint reply by reply.

  KernelResume    `run_teacher.run_session` against a stub server: what is asked again, what each reply carries (the reviewer's `repro_laundering.py` is the first two tests)
  OlderLines      replies of a kernel that wrote no fingerprint per reply, or none at all: asked again, unless `trust_unfingerprinted` (round 12, M1)
  Out2Shape       the thinking pilot's `out2`, in its REAL shape (no `prompt_sha` anywhere, no `max_tokens`, no `prompt_tokens`): resumed under the new prompt, every task is asked again (the
                  reviewer's `out2_resume.py`); with the frozen replies of the pilot, and with the real file when it is on this machine
  TrustFlag       `trust_unfingerprinted` from the run config / the flag to the kit's config.json and to the Modal stage's call
  ModalResume     the same rules in the Modal stage's checkpoint, and `outputs.samples_to_ask`
  PerReplyScore   `kaggle-score` skips and counts the replies of another prompt, not the lines that hold some; and the ones with no fingerprint (`prompts.unchecked`)
  OneRule         the kernel's copy and the package's copy of the rule give the same answer
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest import mock

from . import distill_fixtures as fx  # noqa: F401
from .helpers import ROOT
from .test_rea_kaggle import out as kit_out, write_replies, write_tasks
from .test_rea_kaggle_kit import make_tasks
from .test_rea_modal import ModalCase, StubbedSampleStage
from .test_rea_reasoning import reply
from .thinking_pilot import ROWS, TASKS as PILOT_TASKS, build_tasks, run_teacher

from distill import config, outputs, score_replies, traces  # noqa: E402
from distill.backends import modal_sample  # noqa: E402
from distill.domains.rea import DOMAIN  # noqa: E402
from rea_bench import kaggle_score, outputs as rea_outputs  # noqa: E402
from .rea_bound import rsn  # noqa: E402

OLD = [{"role": "system", "content": "OLD system prompt"}, {"role": "user", "content": "q"}]
NEW = [{"role": "system", "content": "NEW system prompt"}, {"role": "user", "content": "q"}]
SHA = {"old": run_teacher.prompt_sha(OLD), "new": run_teacher.prompt_sha(NEW)}


def out(text="step\nAnswer: 1", finish="stop", sha=None, max_tokens=None):
    """One reply as a kernel stores it; `sha` and `max_tokens` are left out when None (a reply of an older kernel)."""
    o = {"text": text, "finish": finish, "tokens": 3, "prompt_tokens": 9, "secs": 1}
    if sha:
        o["prompt_sha"] = sha
    if max_tokens:
        o["max_tokens"] = max_tokens
    return o


class Server:
    """`urlopen` for the kernel's two requests: /tokenize (5 tokens) and /v1/chat/completions (a reply that stops, or is cut off); `asked` holds the body of every chat request."""

    def __init__(self, finish="stop"):
        self.asked, self.finish = [], finish

    def urlopen(self, req, timeout=None):
        if req.full_url.endswith("/tokenize"):
            body = {"tokens": [0] * 5}
        else:
            self.asked.append(json.loads(req.data))
            body = {"choices": [{"message": {"content": "NEW-SESSION REPLY\nAnswer: 1"}, "finish_reason": self.finish}], "usage": {"completion_tokens": 3, "prompt_tokens": 9}}
        return io.BytesIO(json.dumps(body).encode())


class KernelResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ck = pathlib.Path(self.tmp.name) / "replies.jsonl"
        self.task = {"id": "a-1", "kind": "crc", "messages": NEW}

    def session(self, previous, cfg=None, server=None, deadline=None):
        cfg = run_teacher.resolve_config({"k": 2, "thinking": False, **(cfg or {})})
        self.server = server or Server()
        summary = run_teacher.run_session(cfg, [self.task], previous, self.ck, deadline or time.time() + 600, urlopen=self.server.urlopen, log_fn=lambda *a: None)
        return summary, [json.loads(line) for line in self.ck.read_text().splitlines()]

    def test_a_task_that_a_session_began_under_another_prompt_is_asked_again_not_completed(self):
        """The reviewer's repro: one reply of the old prompt, the new session used to ask only the second sample and stamp the line with the new prompt."""
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["old"], "outs": [out("OLD-PROMPT REPLY\nAnswer: 1")]}}
        summary, lines = self.session(previous)
        last = lines[-1]
        self.assertEqual(last["prompt_sha"], SHA["new"])
        self.assertEqual([o["text"].splitlines()[0] for o in last["outs"]], ["NEW-SESSION REPLY", "NEW-SESSION REPLY"], "the reply of the old prompt is asked again, not kept under the new one")
        self.assertEqual([o["prompt_sha"] for o in last["outs"]], [SHA["new"]] * 2)
        self.assertEqual((len(self.server.asked), summary["stale"]), (2, 1))

    def test_a_task_that_was_complete_under_another_prompt_is_asked_again_in_full(self):
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["old"], "outs": [out("OLD 0\nAnswer: 1"), out("OLD 1\nAnswer: 1")]}}
        summary, lines = self.session(previous)
        self.assertEqual(sorted(b["seed"] for b in self.server.asked), [1234, 1235], "both samples, with the seeds of their positions")
        self.assertEqual({o["prompt_sha"] for o in lines[-1]["outs"]}, {SHA["new"]})
        self.assertEqual((summary["asked"], summary["stale"]), (1, 2))

    def test_a_reply_that_could_not_be_asked_again_keeps_the_fingerprint_it_was_asked_under(self):
        """The deadline has passed: nothing is asked, the old reply stays in the line, and the line's own fingerprint (the task's prompt now) must not speak for it."""
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["old"], "outs": [out("OLD-PROMPT REPLY\nAnswer: 1")]}}
        summary, lines = self.session(previous, deadline=time.time() - 1)
        self.assertEqual(self.server.asked, [])
        self.assertEqual(lines[-1]["prompt_sha"], SHA["new"])
        self.assertEqual([o["prompt_sha"] for o in lines[-1]["outs"]], [SHA["old"]], "the reply says which prompt it answered")
        self.assertEqual(lines[-1]["outs"][0]["text"], "OLD-PROMPT REPLY\nAnswer: 1")

    def test_a_reply_cut_off_at_a_smaller_budget_than_its_familys_now_is_asked_again(self):
        """The reviewer's second repro: a family whose budget grew 4096 -> 8192, both replies cut off at 4096: nothing was asked."""
        cut = out("long reasoning...\n[truncated at max_tokens]", "length", SHA["new"], 4096)
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["new"], "outs": [cut, dict(cut)]}}
        summary, lines = self.session(previous, {"max_tokens": 4096, "max_tokens_by_family": {"crc": 8192}, "ctx_per_slot": 12288})        # the kernel starts the slot for the longest budget (`server_geometry`)
        self.assertEqual([b["max_tokens"] for b in self.server.asked], [8192, 8192])
        self.assertEqual([(o["finish"], o["max_tokens"]) for o in lines[-1]["outs"]], [("stop", 8192)] * 2, "each reply says the budget it was asked with")
        self.assertEqual(summary["stale"], 2)

    def test_a_cut_off_reply_asked_with_the_budget_it_has_now_is_an_answer(self):
        cut = out("cut\n[truncated at max_tokens]", "length", SHA["new"], 8192)
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["new"], "outs": [cut, dict(cut)]}}
        for cfg in ({"max_tokens": 4096, "max_tokens_by_family": {"crc": 8192}}, {"max_tokens": 4096}, {"max_tokens": 8192}, {"max_tokens": 8192, "max_tokens_by_family": {"crc": 4096}}):
            summary, lines = self.session(previous, cfg)
            self.assertEqual((self.server.asked, summary["asked"], summary["stale"]), ([], 0, 0), cfg)
            self.assertEqual(lines[-1]["outs"], [cut, cut])

    def test_a_reply_that_stopped_is_never_asked_again_for_its_budget(self):
        done = out("fine\nAnswer: 1", "stop", SHA["new"], 4096)
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["new"], "outs": [done, dict(done)]}}
        summary, _ = self.session(previous, {"max_tokens": 4096, "max_tokens_by_family": {"crc": 8192}})
        self.assertEqual((self.server.asked, summary["asked"]), ([], 0), "a reply that ended by itself does not depend on the budget")

    def test_a_failed_reply_is_still_asked_again_and_a_fresh_one_carries_the_two_fields(self):
        previous = {"a-1": {"id": "a-1", "kind": "crc", "prompt_sha": SHA["new"], "outs": [out(sha=SHA["new"], max_tokens=768), out("[error OSError: x]", "error")]}}
        summary, lines = self.session(previous)
        self.assertEqual([b["seed"] for b in self.server.asked], [1235])
        self.assertEqual((summary["asked"], summary["stale"]), (1, 0), "a failed reply is not stale: it is a failure")
        self.assertEqual([(o["prompt_sha"], o["max_tokens"]) for o in lines[-1]["outs"]], [(SHA["new"], 768), (SHA["new"], 768)])

    def test_a_failed_request_is_stamped_too(self):
        def urlopen(req, timeout=None):
            if req.full_url.endswith("/tokenize"):
                return io.BytesIO(b'{"tokens": [0]}')
            raise OSError("connection refused")
        o = run_teacher.ask_one(run_teacher.resolve_config({"thinking": False}), NEW, 0, urlopen=urlopen, max_tokens=300)
        self.assertEqual((o["finish"], o["prompt_sha"], o["max_tokens"]), ("error", SHA["new"], 300))

    def test_an_earlier_line_is_not_changed_when_nothing_of_it_is_asked(self):
        line = {"id": "a-1", "kind": "crc", "prompt_sha": SHA["new"], "outs": [out(sha=SHA["new"], max_tokens=4096), out(sha=SHA["new"], max_tokens=4096)]}
        _, lines = self.session({"a-1": line})
        self.assertEqual(lines, [line], "copied at the start, and the session writes nothing for it")


class OlderLines(unittest.TestCase):
    """Round 12, M1. A reply of a kernel that wrote one fingerprint per line and no budget takes its line's fingerprint; one with no fingerprint at all (no `prompt_sha` of its own, none on its line) answered a
    prompt that is not known, and a cut-off reply that does not say its budget was cut off at one that is not known: both are asked again, unless the run was told to trust such a file."""

    def cfg(self, **kw):
        return run_teacher.resolve_config({"k": 2, "thinking": False, **kw})

    def test_a_reply_without_its_own_fields_takes_its_lines_fingerprint(self):
        rec = {"prompt_sha": SHA["old"], "outs": [out(), out(sha=SHA["new"])]}
        self.assertEqual([o["prompt_sha"] for o in run_teacher.with_own_sha(rec)], [SHA["old"], SHA["new"]], "its own fingerprint wins")
        self.assertEqual(run_teacher.with_own_sha({"outs": [out()]}), [out()], "a line without one: nothing to take, nothing is made up")
        self.assertEqual(rec["outs"][0].get("prompt_sha"), None, "the earlier line is not changed in place")

    def test_what_a_run_that_trusts_the_file_takes_for_a_reply_that_lacks_it(self):
        assume = {"prompt_sha": SHA["new"], "max_tokens": 4096}
        self.assertEqual(run_teacher.with_own_sha({"outs": [out()]}, assume), [out(sha=SHA["new"], max_tokens=4096)])
        self.assertEqual(run_teacher.with_own_sha({"outs": [out(sha=SHA["old"], max_tokens=2048)]}, assume), [out(sha=SHA["old"], max_tokens=2048)], "what a reply says itself stands")
        self.assertEqual(run_teacher.with_own_sha({"prompt_sha": SHA["old"], "outs": [out()]}, assume)[0]["prompt_sha"], SHA["old"], "so does its line's fingerprint")
        self.assertEqual(run_teacher.with_own_sha({"outs": [{**out(), "max_tokens": "4096"}]}, assume)[0]["max_tokens"], 4096, "a budget that is not one is the file's missing one")
        task = {"id": "a-1", "kind": "crc", "messages": NEW}
        self.assertIsNone(run_teacher.assumed(self.cfg(), task))
        self.assertEqual(run_teacher.assumed(self.cfg(trust_unfingerprinted=True, max_tokens=2048), task), {"prompt_sha": SHA["new"], "max_tokens": 2048}, "the config's max_tokens: the one budget the old kernels had")

    def test_a_line_of_another_prompt_is_asked_in_full_and_a_line_with_no_fingerprint_too(self):
        task = {"id": "a-1", "kind": "crc", "messages": NEW}
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"prompt_sha": SHA["old"], "outs": [out(), out()]}), [0, 1])
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"prompt_sha": SHA["new"], "outs": [out(), out()]}), [], "the line says what its replies answered: the prompt now, and they stopped by themselves")
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"outs": [out(), out()]}), [0, 1], "a line from before the fingerprint: what its replies answered is not known")
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"outs": [out(), out(sha=SHA["new"])]}), [0], "one by one")
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"outs": [out(), out()]}), [0, 1])
        self.assertEqual(run_teacher.pending(self.cfg(trust_unfingerprinted=True), task, {"outs": [out(), out()]}), [], "unless the owner says it was asked under this prompt")
        self.assertEqual(run_teacher.pending(self.cfg(), task, None), [0, 1])

    def test_a_cut_off_reply_that_does_not_say_its_budget_is_asked_again_unless_the_file_is_trusted(self):
        cut = out("cut", "length")
        task = {"id": "a-1", "kind": "cipher", "messages": NEW}
        rec = {"prompt_sha": SHA["new"], "outs": [cut, out()]}
        self.assertEqual(run_teacher.pending(self.cfg(), task, rec), [0], "cut off at a budget the file does not say: asked again, whatever the budget is now")
        self.assertEqual(run_teacher.pending(self.cfg(max_tokens=8192), task, rec), [0], "a raised base budget (--max-tokens 8192) no longer keeps what was cut off at 4096")
        self.assertEqual(run_teacher.pending(self.cfg(trust_unfingerprinted=True), task, rec), [], "trusted: asked with the config's max_tokens, the budget now: the same cut")
        self.assertEqual(run_teacher.pending(self.cfg(trust_unfingerprinted=True, max_tokens_by_family={"cipher": 8192}), task, rec), [0], "the family's budget is longer than the one every reply had")
        self.assertEqual(run_teacher.pending(self.cfg(trust_unfingerprinted=True, max_tokens=8192, max_tokens_by_family={"cipher": 8192}), task, rec), [], "the config's own max_tokens is the old budget")
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"prompt_sha": SHA["new"], "outs": [out(), out()]}), [], "a reply that stopped by itself does not depend on a budget")

    def test_trust_unfingerprinted_is_true_or_false_and_is_in_the_config_json(self):
        self.assertIs(run_teacher.resolve_config({})["trust_unfingerprinted"], False)
        self.assertIs(run_teacher.resolve_config({"trust_unfingerprinted": True})["trust_unfingerprinted"], True)
        for bad in ("yes", 1, None):
            with self.assertRaisesRegex(ValueError, "true or false"):
                run_teacher.resolve_config({"trust_unfingerprinted": bad})


def out2_lines() -> dict:
    """{task id: line} of the thinking pilot's replies in the shape of `out2/replies.jsonl`, which is what the pilot's kernel wrote: {id, kind, outs: [{text, finish, tokens, secs}]}. No `prompt_sha` on the
    line or on a reply, no `max_tokens`, no `prompt_tokens`. The frozen rows (tests/fixtures/thinking-pilot) are the replies of 16 of its 57 tasks; the sample a row has is its position."""
    by = {}
    for r in ROWS:
        by.setdefault(r["id"], {})[r["sample"]] = {"text": r["text"], "finish": r["finish"], "tokens": r["tokens"], "secs": 100.0}
    return {tid: {"id": tid, "kind": PILOT_TASKS[tid]["kind"], "outs": [by[tid][i] for i in sorted(by[tid])]} for tid in by}


OUT2 = pathlib.Path(os.environ.get("OUT2_REPLIES", pathlib.Path.home() / "work" / "kaggle-rea" / "out2" / "replies.jsonl"))
SEED7 = ROOT / "results" / "phase6" / "rea-data" / "decompiled-seed7-all.jsonl"


class Out2Shape(unittest.TestCase):
    """Round 12, M1: the one real file the resume was written for. `out2` has no fingerprint anywhere, so a resume under the new prompt could not tell which of its replies answered which prompt: it kept 52 of
    57 tasks and stamped them with the new one. Now none is kept (the reviewer's `out2_resume.py`, as a test)."""

    GREW = {"cipher": 8192, "magic": 8192, "password": 8192}        # the shipped thinking config's map

    def resume(self, previous, tasks, **kw):
        cfg = run_teacher.resolve_config({"thinking": True, "k": 2, "max_tokens": 4096, "max_tokens_by_family": self.GREW, **kw})
        jobs = [{"id": t["id"], "kind": t["kind"], "messages": DOMAIN.messages(t, "thinking")} for t in tasks]
        cfg["slots"], cfg["ctx_per_slot"] = run_teacher.server_geometry(cfg, jobs)           # what the kernel starts the server with: 2 x 12288
        server = Server()
        with tempfile.TemporaryDirectory() as d:
            ck = pathlib.Path(d) / "replies.jsonl"
            logged = []
            summary = run_teacher.run_session(cfg, jobs, previous, ck, time.time() + 3600, urlopen=server.urlopen, log_fn=logged.append)
            lines = [json.loads(line) for line in ck.read_text().splitlines()]
        return cfg, jobs, server, summary, {r["id"]: r for r in lines}, logged

    def check_all_asked_again(self, previous, tasks):
        n, replies = len(tasks), sum(len(r["outs"]) for r in previous.values())
        cfg, jobs, server, summary, last, logged = self.resume(previous, tasks)
        self.assertEqual((summary["tasks"], summary["asked"], summary["requests"], summary["skipped_too_long"]), (n, n, 2 * n, []), "every task is asked again, both samples")
        self.assertEqual((summary["stale"], summary["unfingerprinted"]), (replies, replies), "every reply of the old file was a stale one, for want of a fingerprint")
        self.assertTrue(any("carry no fingerprint" in line for line in logged), logged)
        self.assertTrue(any(f"{replies} of them carry no fingerprint at all" in line and "trust_unfingerprinted" in line for line in logged), "and the log says how to keep them")
        for job in jobs:
            self.assertEqual([(o["prompt_sha"], o["max_tokens"]) for o in last[job["id"]]["outs"]], [(run_teacher.prompt_sha(job["messages"]), run_teacher.task_max_tokens(cfg, job["kind"]))] * 2)
            self.assertTrue(all(o["text"].startswith("NEW-SESSION REPLY") for o in last[job["id"]]["outs"]), "no reply of the old session is left in the line, under the new fingerprint")
        self.assertEqual({b["max_tokens"] for b in server.asked}, {4096, 8192})

    def test_the_pilots_replies_in_their_real_shape_are_all_asked_again_under_the_new_prompt(self):
        previous = out2_lines()
        for line in previous.values():
            self.assertEqual((sorted(line), {k for o in line["outs"] for k in o}), (["id", "kind", "outs"], {"text", "finish", "tokens", "secs"}), "the shape of out2: nothing that could be compared")
        self.check_all_asked_again(previous, [PILOT_TASKS[tid] for tid in previous])

    @unittest.skipUnless(OUT2.exists() and SEED7.exists(), "the pilot's real out2 and the seed-7 tasks are not on this machine (OUT2_REPLIES names the file)")
    def test_the_real_out2_is_all_asked_again_under_the_new_prompt(self):
        by_id = {t["id"]: t for t in map(json.loads, SEED7.read_text().splitlines())}
        previous = {r["id"]: r for r in map(json.loads, OUT2.read_text().splitlines())}
        self.assertEqual(len(previous), 57)
        self.assertFalse(any("prompt_sha" in r or any("prompt_sha" in o or "max_tokens" in o for o in r["outs"]) for r in previous.values()), "out2 has no fingerprint, anywhere")
        self.check_all_asked_again(previous, [by_id[tid] for tid in previous])

    def test_with_the_owners_word_only_what_the_budget_changed_is_asked_and_the_rest_is_stamped(self):
        """`trust_unfingerprinted`: the pilot ran every family at 4096 under a prompt the owner says is the one now. Its cut-off cipher and password replies are asked at 8192, the other replies stay, and each
        kept reply is written with the fingerprint and budget it was trusted for, so that nothing later reads the line's own fingerprint for it."""
        previous = out2_lines()
        tasks = [PILOT_TASKS[tid] for tid in previous]
        cfg, jobs, server, summary, last, logged = self.resume(previous, tasks, trust_unfingerprinted=True)
        # the 8 tasks with one reply (a missing second sample) are asked; two of them were also cut off at 4096 in a family that has 8192 now
        single = sorted(tid for tid, line in previous.items() if len(line["outs"]) == 1)
        self.assertEqual(single, ["cipher-7-0204", "crc-7-0083", "password-7-0291", "pktsum-7-0110", "rotate-7-0088", "structs-7-0079", "threshold-7-0069", "threshold-7-0149"])
        self.assertEqual((summary["asked"], summary["requests"], summary["stale"], summary["unfingerprinted"]), (8, 10, 2, 0))
        self.assertEqual(sorted((b["max_tokens"] for b in server.asked)), [4096] * 6 + [8192] * 4)
        for tid in ("magic-7-0062", "bounds-7-0072"):                     # complete and not cut off at a budget that grew: not asked, not rewritten (the line is the copy of the old one)
            self.assertEqual(last[tid], previous[tid])
        for tid in ("pktsum-7-0110", "rotate-7-0088"):                    # cut off at 4096, a family that is still at 4096: kept, stamped
            kept = last[tid]["outs"][0]
            self.assertEqual((kept["finish"], kept["max_tokens"], kept["prompt_sha"]), ("length", 4096, run_teacher.prompt_sha(next(j for j in jobs if j["id"] == tid)["messages"])))
            self.assertEqual(kept["text"], previous[tid]["outs"][0]["text"])
        self.assertEqual([o["max_tokens"] for o in last["cipher-7-0204"]["outs"]], [8192, 8192])

    def test_kaggle_score_skips_the_unfingerprinted_file_unless_told_to_keep_it(self):
        previous = out2_lines()
        tasks = [PILOT_TASKS[tid] for tid in previous]
        with tempfile.TemporaryDirectory() as d:
            tp = write_tasks(d, tasks, "decompiled-seed7-all.jsonl")
            rp = write_replies(d, list(previous.values()))
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaisesRegex(SystemExit, f"{len(previous)} line.s. have no fingerprint at all.*--keep-unchecked"):
                kaggle_score.score(rp, tp, "m", thinking=True)
            self.assertIn(f"{len(previous)} replies line(s) hold {sum(len(r['outs']) for r in previous.values())} repl(ies) with no prompt fingerprint", err.getvalue())
            _, stats = kaggle_score.score(rp, tp, "m", thinking=True, keep_unchecked=True)
        n = len(previous)
        self.assertEqual({k: stats["prompts"][k] for k in ("lines", "checked", "mismatch", "unchecked", "unchecked_replies", "kept_unchecked")}, {"lines": n, "checked": 0, "mismatch": 0, "unchecked": n, "unchecked_replies": 24, "kept_unchecked": True})
        self.assertEqual(sorted(stats["prompts"]["unchecked_ids"]), sorted(previous))
        self.assertEqual((stats["tasks"], stats["answers"]), (n, 24), "kept on request: scored as they are")


class TrustFlag(unittest.TestCase):
    """`trust_unfingerprinted` is the owner's word: it travels only by an explicit setting, from the run config or the kit's flag to `config.json`, and from the run config or the flag to the Modal stage."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.file = write_tasks(self.d, make_tasks((("crc", 2), ("structs", 2))), "decompiled-seed9-all.jsonl")

    def cfg_json(self, out):
        return json.loads((self.d / out / "data" / "config.json").read_text())

    def build(self, out, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            build_tasks.main(["--tasks", str(self.file), "--owner", "someone", "--out", str(self.d / out), *args])
        return self.cfg_json(out)

    def test_the_kit_writes_it_only_when_it_is_given_and_the_kernel_reads_it(self):
        self.assertNotIn("trust_unfingerprinted", self.build("a"))
        self.assertNotIn("trust_unfingerprinted", build_tasks.build([self.file], self.d / "b", "someone", trust_unfingerprinted=False))
        c = self.build("c", "--trust-unfingerprinted")
        self.assertIs(c["trust_unfingerprinted"], True)
        self.assertIs(run_teacher.resolve_config(c)["trust_unfingerprinted"], True)
        self.assertIs(run_teacher.resolve_config(self.cfg_json("a"))["trust_unfingerprinted"], False, "absent: the default, false")

    def test_a_run_config_sets_it_and_a_second_source_is_refused(self):
        run = self.d / "run.json"
        run.write_text(json.dumps({"mode": "thinking", "max_len": 8192, "trust_unfingerprinted": True}))
        self.assertIs(self.build("d", "--run-config", str(run))["trust_unfingerprinted"], True)
        run.write_text(json.dumps({"mode": "thinking", "max_len": 8192}))
        self.assertNotIn("trust_unfingerprinted", self.build("e", "--run-config", str(run), "--force"))
        for flag in ("--trust-unfingerprinted", "--no-trust-unfingerprinted"):
            with self.assertRaisesRegex(SystemExit, "do not also give --trust-unfingerprinted"):
                self.build("e", "--run-config", str(run), flag, "--force")

    def test_the_run_config_checks_it(self):
        self.assertFalse(config.RunConfig(mode="thinking", max_len=8192).trust_unfingerprinted)
        self.assertTrue(config.RunConfig(mode="reasoning", max_len=4096, trust_unfingerprinted=True).trust_unfingerprinted)
        for bad in ("yes", 1, None):
            with self.assertRaisesRegex(SystemExit, "trust_unfingerprinted .* true or false"):
                config.RunConfig(mode="thinking", max_len=8192, trust_unfingerprinted=bad)
        with self.assertRaisesRegex(SystemExit, "trust_unfingerprinted is for a reasoning or thinking run"):
            config.RunConfig(trust_unfingerprinted=True)
        self.assertNotIn("trust_unfingerprinted", config.RunConfig(mode="thinking", max_len=8192).sample_kwargs(), "the stage's call is what it was unless the flag is set")
        self.assertIs(config.RunConfig(mode="thinking", max_len=8192, trust_unfingerprinted=True).sample_kwargs()["trust_unfingerprinted"], True)
        run = self.d / "run.json"
        run.write_text(json.dumps({"mode": "thinking", "max_len": 8192, "trust_unfingerprinted": True}))
        self.assertIs(config.merge_flags(run, dict(config.FLAG_DEFAULTS), "rea")["trust_unfingerprinted"], True)
        with self.assertRaisesRegex(SystemExit, "do not also give --trust-unfingerprinted"):
            config.merge_flags(run, {**config.FLAG_DEFAULTS, "trust_unfingerprinted": True}, "rea")
        self.assertFalse(config.merge_flags(ROOT / "distill" / "configs" / "rea-thinking.json", dict(config.FLAG_DEFAULTS), "rea")["trust_unfingerprinted"], "no shipped config trusts anything")


class TrustFlagAtTheModalEntrypoint(ModalCase):
    def call(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            remote, _run = self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", **({} if "run_config" in kw else {"model": "codeforge-27b-heretic-q4"}), **kw)
        return remote.call_args

    def test_the_flag_and_the_run_config_reach_the_stage_and_nothing_else_does(self):
        self.assertIs(self.call(reasoning=True, thinking=True, max_len=8192, trust_unfingerprinted=True).kwargs["trust_unfingerprinted"], True)
        self.assertIs(self.call(reasoning=True, thinking=True, max_len=8192).kwargs["trust_unfingerprinted"], False)
        with tempfile.TemporaryDirectory() as d:
            run = pathlib.Path(d) / "run.json"
            run.write_text(json.dumps({"teacher": "codeforge-27b-heretic-q4", "mode": "thinking", "max_len": 8192, "trust_unfingerprinted": True}))
            self.assertIs(self.call(run_config=str(run)).kwargs["trust_unfingerprinted"], True)
            with self.assertRaisesRegex(SystemExit, "do not also give --trust-unfingerprinted"):
                self.call(run_config=str(run), trust_unfingerprinted=True)

    def test_a_default_run_has_nothing_to_trust(self):
        with self.assertRaisesRegex(SystemExit, "--trust-unfingerprinted is for --reasoning or --thinking"):
            self.call(trust_unfingerprinted=True)


class ModalResume(StubbedSampleStage):
    """The real `sample_answers` of a thinking run, then a resume."""

    ARGS = dict(reasoning=True, thinking=True, max_tokens=4096, max_len=8192)

    def setUp(self):
        super().setUp()
        self.finish = lambda tid, i: "stop"
        self.reply = lambda tid, i, body: {"choices": [{"message": {"content": "Steps\nAnswer: 1", "reasoning_content": "thinking"}, "finish_reason": self.finish(tid, i)}], "usage": {"completion_tokens": 7}}
        self.partial = self.root / "data" / f"{outputs.sft_stem('m', outputs.sample_tag(self.tag, True, True))}.partial.jsonl"

    def lines(self):
        return {rec["id"]: rec for rec in map(json.loads, self.partial.read_text().splitlines())}

    def run_modal(self, resume, **kw):
        return self.run_stage(resume, **{**self.ARGS, **kw})

    def test_each_reply_of_a_reasoning_run_carries_its_prompt_and_its_budget(self):
        self.run_modal(False, max_tokens_by_family={"password": 8192})
        for tid, rec in self.lines().items():
            want = {"prompt_sha": outputs.prompt_sha(DOMAIN.messages(self.tasks[tid], "thinking")), "max_tokens": 8192 if self.tasks[tid]["kind"] == "password" else 4096}
            self.assertEqual([outputs.out_meta(o) for o in rec["outs"]], [want] * self.k, tid)
            self.assertEqual(sorted(rec), ["id", "kind", "outs"])

    def test_a_default_run_keeps_the_two_element_checkpoint(self):
        """The default mode stores no teacher text (its record is rendered from the truth), and its checkpoint is byte for byte what it was (tests/golden/distill/expected/stage-plain.json)."""
        self.run_stage(False)
        partial = self.root / "data" / f"{outputs.sft_stem('m', self.tag)}.partial.jsonl"
        self.assertTrue(all(len(o) == 2 for rec in map(json.loads, partial.read_text().splitlines()) for o in rec["outs"]))

    def test_a_resume_asks_again_what_was_cut_off_at_a_smaller_budget_and_nothing_else(self):
        self.finish = lambda tid, i: "length" if "password" in tid and i == 0 else "stop"
        self.run_modal(False)
        before = self.lines()
        self.finish = lambda tid, i: "stop"
        self.run_modal(True, max_tokens_by_family={"password": 8192})
        self.assertEqual(sorted(self.requests), sorted((tid, 0) for tid in self.tasks if "password" in tid), "the cut-off replies of the family whose budget grew, and no other reply")
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {8192})
        after = self.lines()
        for tid in self.tasks:
            if "password" in tid:
                self.assertEqual((outputs.out_meta(after[tid]["outs"][0])["max_tokens"], traces.is_truncated(after[tid]["outs"][0][0])), (8192, False), tid)
                self.assertEqual(after[tid]["outs"][1:], before[tid]["outs"][1:])
            else:
                self.assertEqual(after[tid], before[tid], "a task that is not asked is as it was")
        self.assertIn("0 failed or missing and 2 stale samples of checkpointed tasks", self.printed)

    def test_a_resume_with_the_same_budget_asks_nothing_for_a_cut_off_reply(self):
        self.finish = lambda tid, i: "length"
        self.run_modal(False, max_tokens_by_family={"password": 8192})
        self.run_modal(True, max_tokens_by_family={"password": 8192})
        self.assertEqual(self.requests, [])

    def test_a_reply_to_another_prompt_is_asked_again_and_the_checkpoint_says_the_new_one(self):
        self.run_modal(False)
        lines = self.lines()
        lines["password-7-0001"]["outs"] = [[o[0], o[1], {**o[2], "prompt_sha": "0" * 16}] if i < 2 else o for i, o in enumerate(lines["password-7-0001"]["outs"])]
        self.partial.write_text("".join(json.dumps(r) + "\n" for r in lines.values()))
        self.run_modal(True)
        self.assertEqual(sorted(self.requests), [("password-7-0001", 0), ("password-7-0001", 1)])
        self.assertEqual({outputs.out_meta(o)["prompt_sha"] for o in self.lines()["password-7-0001"]["outs"]}, {outputs.prompt_sha(DOMAIN.messages(self.tasks["password-7-0001"], "thinking"))})

    def test_a_reply_to_another_prompt_of_a_task_that_cannot_be_asked_again_is_not_stored(self):
        self.run_modal(False)
        tid = "password-7-0001"
        rec = {**self.lines()[tid]}
        rec["outs"] = [[o[0], o[1], {**o[2], "prompt_sha": "0" * 16}] for o in rec["outs"]]
        self.partial.write_text("".join(json.dumps(r) + "\n" for r in (rec, *(r for i, r in self.lines().items() if i != tid))))
        p = self.root / "data" / rea_outputs.decompiled_stem(self.tag) / f"{tid}.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), "pseudocode": f"/* {tid} */ " + "x " * 9000}))        # no longer fits a slot: not asked
        res = self.run_modal(True)
        self.assertNotIn(tid, {t for t, _ in self.requests})
        self.assertEqual(res["stats"]["slot"]["prompt_too_long"], 1)
        self.assertFalse([r for r in res["sft"] if r["task"] == tid], "its replies answered another prompt: no record of it")

    def older_checkpoint(self):
        """A checkpoint of the code before the fingerprints: the replies are `[text, flag]`, with no `prompt_sha` and no `max_tokens`; the first reply of every task was cut off."""
        self.finish = lambda tid, i: "length" if i == 0 else "stop"
        self.run_modal(False)
        self.partial.write_text("".join(json.dumps({**r, "outs": [o[:2] for o in r["outs"]]}) + "\n" for r in self.lines().values()))
        self.finish = lambda tid, i: "stop"

    def test_an_older_checkpoint_has_no_fingerprint_and_every_reply_of_it_is_asked_again(self):
        """Round 12, M1: what its replies answered is not known, so a resume keeps none of them."""
        self.older_checkpoint()
        self.run_modal(True)
        self.assertEqual(sorted(self.requests), sorted((tid, i) for tid in self.tasks for i in range(self.k)), "every sample of every task, whatever its finish")
        self.assertIn(f"0 failed or missing and {len(self.tasks) * self.k} stale samples of checkpointed tasks", self.printed)
        self.assertIn("carries no fingerprint", self.printed)
        want = lambda tid: {"prompt_sha": outputs.prompt_sha(DOMAIN.messages(self.tasks[tid], "thinking")), "max_tokens": 4096}
        for tid, rec in self.lines().items():
            self.assertEqual([outputs.out_meta(o) for o in rec["outs"]], [want(tid)] * self.k, tid)
        self.run_modal(True)
        self.assertEqual(self.requests, [], "and then the checkpoint is complete")

    def test_with_trust_unfingerprinted_an_older_checkpoint_counts_as_asked_with_max_tokens(self):
        """The owner's word (run config `trust_unfingerprinted`): the old behaviour, written down. The cut-off replies were asked with `max_tokens`; they are asked again only where the budget grew."""
        self.older_checkpoint()
        self.run_modal(True, trust_unfingerprinted=True)
        self.assertEqual(self.requests, [], "no budget grew and no fingerprint can differ")
        self.run_modal(True, trust_unfingerprinted=True, max_tokens_by_family={"password": 8192})
        self.assertEqual(sorted(self.requests), sorted((tid, 0) for tid in self.tasks if "password" in tid), "the old checkpoint's budget is max_tokens: the family's grew")
        for tid, rec in self.lines().items():
            if "password" in tid:
                self.assertEqual([outputs.out_meta(o) for o in rec["outs"][1:]], [{"prompt_sha": outputs.prompt_sha(DOMAIN.messages(self.tasks[tid], "thinking")), "max_tokens": 4096}] * (self.k - 1),
                                 "the replies that were trusted are written with what they were trusted for")

    def test_trust_unfingerprinted_is_for_a_reasoning_run(self):
        with self.assertRaisesRegex(SystemExit, "trust_unfingerprinted is for a reasoning or thinking run"):
            self.run_stage(False, trust_unfingerprinted=True)


class SamplesToAsk(unittest.TestCase):
    CUT = ["cut\n" + traces.TRUNCATED, False]

    def test_the_rule(self):
        ask = outputs.samples_to_ask
        meta = lambda sha, n: {"prompt_sha": sha, "max_tokens": n}
        rec = {"outs": [["a", True, meta("s", 4096)], [*self.CUT, meta("s", 4096)], ["[error x]", False, meta("s", 4096)]]}
        self.assertEqual(ask(rec, 3), [2], "without the prompt and the budget only the failed one")
        self.assertEqual(ask(rec, 3, "s", 4096), [2])
        self.assertEqual(ask(rec, 3, "s", 8192), [1, 2], "the cut-off reply was asked with less than the budget now")
        self.assertEqual(ask(rec, 3, "t", 4096), [0, 1, 2], "another prompt: every reply that is not an answer to this one")
        self.assertEqual(ask({"outs": [["a", True, meta("s", 4096)]]}, 3, "s", 4096), [1, 2])
        self.assertEqual(ask({"outs": [["a", True, meta("s", 4096)]]}, 1, None, None), [])

    def test_a_reply_with_no_fingerprint_is_stale_when_the_prompt_is_known(self):
        """Round 12, M1. The two-element reply of an older checkpoint says nothing about the prompt it answered, nor about the budget it was cut off at."""
        ask = outputs.samples_to_ask
        self.assertEqual(ask({"outs": [["fine", True]]}, 1, "s", 8192), [0], "no fingerprint: asked again, though it stopped by itself")
        self.assertEqual(ask({"outs": [self.CUT]}, 1, "s", 8192), [0])
        self.assertEqual(ask({"outs": [["fine", True, {"max_tokens": 4096}]]}, 1, "s", 4096), [0], "a budget alone is not a fingerprint")
        self.assertEqual(ask({"outs": [["fine", True, {"prompt_sha": "s"}]]}, 1, "s", 4096), [], "a reply that stopped needs no budget")
        self.assertEqual(ask({"outs": [["fine", True]]}, 1, None, 8192), [], "the prompt is not known: nothing to compare")

    def test_a_cut_off_reply_that_does_not_say_its_budget_is_stale_when_the_budget_is_known(self):
        ask = outputs.samples_to_ask
        for meta in ({"prompt_sha": "s"}, {"prompt_sha": "s", "max_tokens": "4096"}, {"prompt_sha": "s", "max_tokens": 0}, {"prompt_sha": "s", "max_tokens": True}):
            self.assertEqual(ask({"outs": [[*self.CUT, meta]]}, 1, "s", 4096), [0], meta)
            self.assertEqual(ask({"outs": [[*self.CUT, meta]]}, 1, "s", None), [], "no budget to compare with")

    def test_assume_meta_is_the_owners_word_for_the_replies_that_lack_it(self):
        rec = {"id": "x", "outs": [["a", True], ["b", False, {"prompt_sha": "old"}], ["c", True, {"prompt_sha": "old", "max_tokens": 2048}], ["d", True, {"max_tokens": "n"}]]}
        got = outputs.assume_meta(rec, "s", 4096)
        self.assertEqual([outputs.out_meta(o) for o in got["outs"]], [{"prompt_sha": "s", "max_tokens": 4096}, {"prompt_sha": "old", "max_tokens": 4096}, {"prompt_sha": "old", "max_tokens": 2048},
                                                                    {"prompt_sha": "s", "max_tokens": 4096}])
        self.assertEqual([o[:2] for o in got["outs"]], [o[:2] for o in rec["outs"]], "the replies themselves are not touched")
        self.assertEqual(len(rec["outs"][0]), 2, "and the checkpoint's own record is not changed in place")

    def test_current_outs_keeps_only_the_replies_that_answered_this_prompt(self):
        """Round 12, M1: a reply of another prompt, and one with no fingerprint, cannot be stored under this prompt."""
        outs = [["a", True, {"prompt_sha": "s", "max_tokens": 4096}], ["b", True, {"prompt_sha": "t", "max_tokens": 4096}], ["c", True], ["d", True, {"max_tokens": 4096}], ["e", False, {"prompt_sha": "s"}]]
        self.assertEqual([o[0] for o in outputs.current_outs(outs, "s")], ["a", "e"])
        self.assertEqual(outputs.current_outs([], "s"), [])


class PerReplyScore(unittest.TestCase):
    """`kaggle-score` against a line that holds replies of several sessions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = make_tasks((("structs", 3),))
        self.decompiled = write_tasks(self.d, self.tasks, "decompiled-seed9-all.jsonl")
        self.want = lambda t: rsn.prompt_sha(rsn.messages(t))

    def line(self, t, shas, line_sha="own"):
        """A replies line of `t` with one reply per entry of `shas` (the reply's own fingerprint, None = it has none); `line_sha`: the line's ("own" = the prompt's, None = none)."""
        outs = []
        for i, sha in enumerate(shas):
            body = reply(t, ["The function reads a field at a fixed offset.", "Following the loop gives the key material."][i % 2] + f" Sample {i}.", None)
            outs.append(kit_out(run_teacher.stored_text(body, f"Reasoning {i} about {t['id']}: it reads the field.", True), tokens=400, prompt_tokens=200, **({"prompt_sha": sha} if sha else {})))
        rec = {"id": t["id"], "kind": t["kind"], "outs": outs}
        if line_sha:
            rec["prompt_sha"] = self.want(t) if line_sha == "own" else line_sha
        return rec

    def score(self, lines, **kw):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                kept, stats = kaggle_score.score(write_replies(self.d, lines), self.decompiled, "m", thinking=True, **kw)
            finally:
                self.err = err.getvalue()
        return kept, stats

    def test_only_the_replies_of_another_prompt_are_skipped_and_counted(self):
        a, b = self.tasks[:2]
        w = self.want
        kept, stats = self.score([self.line(a, ["0" * 16, w(a), w(a)]), self.line(b, [w(b)] * 3)])
        self.assertEqual((stats["answers"], stats["prompts"]["partial"], stats["prompts"]["mismatch"]), (5, {a["id"]: 1}, 0))
        self.assertNotIn(f"Reasoning 0 about {a['id']}", json.dumps(kept), "the reply of the other prompt is not stored")
        self.assertIn(f"Reasoning 1 about {a['id']}", json.dumps(kept))
        self.assertIn("1 repl", self.err)
        self.assertIn(a["id"], self.err)

    def test_the_lines_own_fingerprint_does_not_speak_for_a_reply_that_has_its_own(self):
        """The laundering: the line carries the current prompt's fingerprint, the replies it holds answered another one."""
        a, b = self.tasks[:2]
        laundered = self.line(a, ["0" * 16, "0" * 16])
        with self.assertRaisesRegex(SystemExit, "no replies line"):
            self.score([laundered])
        kept, stats = self.score([laundered, self.line(b, [None, None])])
        self.assertEqual((stats["prompts"]["mismatch"], stats["prompts"]["mismatched_ids"], stats["tasks"], stats["answers"]), (1, [a["id"]], 1, 2))
        self.assertEqual({r["task"] for r in kept}, {b["id"]})

    def test_a_reply_without_its_own_fingerprint_takes_its_lines(self):
        a = self.tasks[0]
        kept, stats = self.score([self.line(a, [None, self.want(a)], line_sha="0" * 16)])
        self.assertEqual((stats["answers"], stats["prompts"]["partial"], stats["prompts"]["mismatch"]), (1, {a["id"]: 1}, 0), "the first took the old line's fingerprint, the second has its own")

    def test_a_line_from_before_the_fingerprint_is_counted_and_skipped_unless_it_is_kept_on_request(self):
        """Round 12, M1: nothing shows that its replies answered this prompt (they were not compared; now they are not stored either, unless the owner says so)."""
        lines = [self.line(t, [None, None], line_sha=None) for t in self.tasks]
        with self.assertRaisesRegex(SystemExit, "3 line.s. have no fingerprint at all.*--keep-unchecked"):
            self.score(lines)
        self.assertIn("3 replies line(s) hold 6 repl(ies) with no prompt fingerprint", self.err)
        kept, stats = self.score(lines, keep_unchecked=True)
        self.assertEqual(stats["answers"], 6)
        self.assertEqual({k: v for k, v in stats["prompts"].items() if k != "unchecked_ids"}, {"lines": 3, "checked": 0, "mismatch": 0, "mismatched_ids": [], "unchecked": 3, "unchecked_replies": 6, "kept_unchecked": True})
        self.assertEqual(stats["prompts"]["unchecked_ids"], [t["id"] for t in self.tasks])

    def test_the_unfingerprinted_replies_of_a_line_that_has_others_are_the_ones_skipped(self):
        """A resumed line from an old file: the replies that carry a fingerprint stay, the ones that carry none go (or stay, with `keep_unchecked`), and only the first is counted as unchecked."""
        a, b = self.tasks[:2]
        w = self.want
        mixed = self.line(a, [None, w(a), w(a)], line_sha=None)
        kept, stats = self.score([mixed, self.line(b, [w(b)] * 2)])
        self.assertEqual((stats["answers"], stats["prompts"]["unchecked"], stats["prompts"]["unchecked_replies"], stats["prompts"]["unchecked_ids"], stats["prompts"].get("partial")), (4, 1, 1, [a["id"]], None))
        self.assertNotIn(f"Reasoning 0 about {a['id']}", json.dumps(kept), "the reply with no fingerprint is not stored")
        self.assertIn(f"Reasoning 1 about {a['id']}", json.dumps(kept))
        kept, stats = self.score([mixed, self.line(b, [w(b)] * 2)], keep_unchecked=True)
        self.assertEqual((stats["answers"], stats["prompts"]["unchecked_replies"]), (5, 1))
        self.assertIn(f"Reasoning 0 about {a['id']}", json.dumps(kept))

    def test_a_line_of_another_prompt_with_an_unfingerprinted_reply_is_one_mismatch_and_one_unchecked(self):
        a, b = self.tasks[:2]
        kept, stats = self.score([self.line(a, ["0" * 16, None], line_sha=None), self.line(b, [self.want(b)] * 2)])
        self.assertEqual((stats["prompts"]["mismatch"], stats["prompts"]["unchecked"], stats["prompts"]["checked"], stats["tasks"]), (1, 1, 2, 1))
        self.assertEqual({r["task"] for r in kept}, {b["id"]})

    def test_the_command_says_how_many_replies_were_skipped(self):
        a = self.tasks[0]
        write_replies(self.d, [self.line(a, ["0" * 16, self.want(a)])])
        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            kaggle_score.main(["--replies", str(self.d / "replies.jsonl"), "--tasks", str(self.decompiled), "--model", "m", "--thinking", "--out", str(self.d / "o.jsonl")])
        self.assertIn("prompts: 1 replies of 1 lines answered another prompt and were skipped (the rest of those lines was kept)", printed.getvalue())

    def test_the_command_says_how_many_lines_have_no_fingerprint_and_what_became_of_them(self):
        a, b = self.tasks[:2]
        write_replies(self.d, [self.line(a, [None, None], line_sha=None), self.line(b, [self.want(b)] * 2)])
        args = ["--replies", str(self.d / "replies.jsonl"), "--tasks", str(self.decompiled), "--model", "m", "--thinking"]
        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            kaggle_score.main([*args, "--out", str(self.d / "o.jsonl")])
        self.assertIn("prompts: 1 of 2 lines hold 2 replies with no fingerprint (an older notebook), so unchecked: skipped (--keep-unchecked keeps them)", printed.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            kaggle_score.main([*args, "--keep-unchecked", "--out", str(self.d / "k.jsonl")])
        self.assertIn("prompts: 1 of 2 lines hold 2 replies with no fingerprint (an older notebook), so unchecked: kept (--keep-unchecked), on your word that they answered this prompt", printed.getvalue())
        stats = json.loads((self.d / "k.stats.json").read_text())
        self.assertEqual((stats["prompts"]["unchecked"], stats["prompts"]["kept_unchecked"], stats["tasks"]), (1, True, 2))
        self.assertEqual(json.loads((self.d / "o.stats.json").read_text())["tasks"], 1)


class OneRule(unittest.TestCase):
    """The kernel (stdlib only, its own copy) and the package decide a stale reply alike, over the same replies."""

    def test_the_kernel_and_the_package_agree(self):
        cut = traces.TRUNCATED
        cases = []
        for finish, text in (("stop", "fine"), ("length", "cut\n" + cut)):
            for sha in (None, "s", "t"):
                for asked in (None, 2048, 4096, 8192, "4096", True, 0, -5):
                    cases.append((finish, text, sha, asked))
        for finish, text, sha, asked in cases:
            kernel = {**out(text, finish, sha), **({"max_tokens": asked} if asked is not None else {})}
            meta = {k: v for k, v in (("prompt_sha", sha), ("max_tokens", asked)) if v is not None}
            entry = [text, False, meta] if meta else [text, False]
            for prompt in (None, "s"):
                for budget in (None, 4096, 8192):
                    self.assertEqual(run_teacher.is_stale(kernel, prompt, budget), outputs.is_stale(entry, prompt, budget), (finish, sha, asked, prompt, budget))
        self.assertEqual(score_replies.reply_fingerprints({"prompt_sha": "x", "outs": [{"prompt_sha": "y"}, {}]}), ["y", "x"])


if __name__ == "__main__":
    unittest.main()

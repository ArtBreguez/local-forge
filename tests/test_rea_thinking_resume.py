"""Round 15, M1 of the review of the thinking-pilot fixes: a resume must not launder the prompt fingerprint or keep a reply cut off at an old budget.

The line of a resumed task used to be stamped with the CURRENT prompt's fingerprint while its earlier replies (another session, maybe another prompt) were copied into it, and a reply cut off at 4096 was "an
answer" for a family whose budget had become 8192. Now each reply carries its own `prompt_sha` and `max_tokens`; a reply is asked again when it answered another prompt, or was cut off at a smaller budget than
the task's now; `kaggle-score` checks the fingerprint reply by reply.

  KernelResume    `run_teacher.run_session` against a stub server: what is asked again, what each reply carries (the reviewer's `repro_laundering.py` is the first two tests)
  OlderLines      replies of a kernel that wrote no fingerprint per reply (the thinking pilot's `out2`, the first sessions of a run)
  ModalResume     the same rules in the Modal stage's checkpoint, and `outputs.samples_to_ask`
  PerReplyScore   `kaggle-score` skips and counts the replies of another prompt, not the lines that hold some
  OneRule         the kernel's copy and the package's copy of the rule give the same answer
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import time
import unittest
from unittest import mock

from . import distill_fixtures as fx  # noqa: F401
from .test_rea_kaggle import out as kit_out, write_replies, write_tasks
from .test_rea_kaggle_kit import make_tasks
from .test_rea_modal import StubbedSampleStage
from .test_rea_reasoning import reply
from .thinking_pilot import run_teacher

from distill import outputs, score_replies, traces  # noqa: E402
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
    """A reply of a kernel that wrote one fingerprint per line and no budget: its line's fingerprint, and the config's `max_tokens` (that kernel had one budget for all)."""

    def cfg(self, **kw):
        return run_teacher.resolve_config({"k": 2, "thinking": False, **kw})

    def test_a_reply_without_its_own_fields_takes_its_lines_fingerprint(self):
        rec = {"prompt_sha": SHA["old"], "outs": [out(), out(sha=SHA["new"])]}
        self.assertEqual([o["prompt_sha"] for o in run_teacher.with_own_sha(rec)], [SHA["old"], SHA["new"]], "its own fingerprint wins")
        self.assertEqual(run_teacher.with_own_sha({"outs": [out()]}), [out()], "a line without one: nothing to take, nothing to compare")
        self.assertEqual(rec["outs"][0].get("prompt_sha"), None, "the earlier line is not changed in place")

    def test_a_line_of_another_prompt_is_asked_in_full_a_line_of_this_one_is_kept(self):
        task = {"id": "a-1", "kind": "crc", "messages": NEW}
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"prompt_sha": SHA["old"], "outs": [out(), out()]}), [0, 1])
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"prompt_sha": SHA["new"], "outs": [out(), out()]}), [])
        self.assertEqual(run_teacher.pending(self.cfg(), task, {"outs": [out(), out()]}), [], "a line from before the fingerprint: not compared")
        self.assertEqual(run_teacher.pending(self.cfg(), task, None), [0, 1])

    def test_a_cut_off_reply_without_a_budget_counts_as_asked_with_the_configs_max_tokens(self):
        cut = out("cut", "length")
        task = {"id": "a-1", "kind": "cipher", "messages": NEW}
        rec = {"prompt_sha": SHA["new"], "outs": [cut, cut]}
        self.assertEqual(run_teacher.pending(self.cfg(), task, rec), [], "the family has the config's budget: the same cut")
        self.assertEqual(run_teacher.pending(self.cfg(max_tokens_by_family={"cipher": 8192}), task, rec), [0, 1], "the family's budget is longer than the one every reply had")
        self.assertEqual(run_teacher.pending(self.cfg(max_tokens=8192, max_tokens_by_family={"cipher": 8192}), task, rec), [], "the config's own max_tokens is the old budget")

    def test_the_thinking_pilots_replies_are_asked_again_where_the_budget_grew(self):
        """out2: the pilot ran every family at 4096 and cipher/magic/password were cut off; asked now with the shipped thinking config they are asked again, the rest is not."""
        cfg = run_teacher.resolve_config({"thinking": True, "k": 2, "max_tokens_by_family": {"cipher": 8192, "magic": 8192, "password": 8192}})
        pilot = lambda kind, finish: {"prompt_sha": SHA["new"], "outs": [out("t", finish), out("t", "stop")]}
        for kind, finish, want in (("cipher", "length", [0]), ("cipher", "stop", []), ("crc", "length", []), ("password", "length", [0]), ("password", "stop", [])):
            self.assertEqual(run_teacher.pending(cfg, {"id": "x", "kind": kind, "messages": NEW}, pilot(kind, finish)), want, (kind, finish))


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

    def test_an_older_checkpoint_has_no_fingerprint_and_counts_as_asked_with_max_tokens(self):
        self.finish = lambda tid, i: "length" if i == 0 else "stop"
        self.run_modal(False)
        lines = self.lines()
        self.partial.write_text("".join(json.dumps({**r, "outs": [o[:2] for o in r["outs"]]}) + "\n" for r in lines.values()))
        self.finish = lambda tid, i: "stop"
        self.run_modal(True)
        self.assertEqual(self.requests, [], "no budget grew and no fingerprint can differ")
        self.run_modal(True, max_tokens_by_family={"password": 8192})
        self.assertEqual(sorted(self.requests), sorted((tid, 0) for tid in self.tasks if "password" in tid), "the old checkpoint's budget is max_tokens: the family's grew")


class SamplesToAsk(unittest.TestCase):
    CUT = ["cut\n" + traces.TRUNCATED, False]

    def test_the_rule(self):
        ask = outputs.samples_to_ask
        meta = lambda sha, n: {"prompt_sha": sha, "max_tokens": n}
        rec = {"outs": [["a", True, meta("s", 4096)], [*self.CUT, meta("s", 4096)], ["[error x]", False, meta("s", 4096)]]}
        self.assertEqual(ask(rec, 3), [2], "without the prompt and the budget only the failed one")
        self.assertEqual(ask(rec, 3, "s", 4096, 4096), [2])
        self.assertEqual(ask(rec, 3, "s", 8192, 4096), [1, 2], "the cut-off reply was asked with less than the budget now")
        self.assertEqual(ask(rec, 3, "t", 4096, 4096), [0, 1, 2], "another prompt: every reply that is not an answer to this one")
        self.assertEqual(ask({"outs": [["a", True, meta("s", 4096)]]}, 3, "s", 4096, 4096), [1, 2])
        self.assertEqual(ask({"outs": [self.CUT]}, 1, "s", 8192, 8192), [], "an older reply: no fingerprint, and it was asked with the old budget, which is the budget now")
        self.assertEqual(ask({"outs": [self.CUT]}, 1, "s", 8192, 4096), [0])
        self.assertEqual(ask({"outs": [["fine", True]]}, 1, "s", 8192, 4096), [], "not cut off: the budget does not matter")
        self.assertEqual(ask({"outs": [["a", True, meta("s", 4096)]]}, 1, None, None, None), [])


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

    def score(self, lines):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            kept, stats = kaggle_score.score(write_replies(self.d, lines), self.decompiled, "m", thinking=True)
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

    def test_a_line_from_before_the_fingerprint_is_not_compared(self):
        kept, stats = self.score([self.line(t, [None, None], line_sha=None) for t in self.tasks])
        self.assertEqual((stats["answers"], stats["prompts"]), (6, {"checked": 0, "mismatch": 0, "mismatched_ids": []}))

    def test_the_command_says_how_many_replies_were_skipped(self):
        a = self.tasks[0]
        write_replies(self.d, [self.line(a, ["0" * 16, self.want(a)])])
        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            kaggle_score.main(["--replies", str(self.d / "replies.jsonl"), "--tasks", str(self.decompiled), "--model", "m", "--thinking", "--out", str(self.d / "o.jsonl")])
        self.assertIn("prompts: 1 replies of 1 lines answered another prompt and were skipped (the rest of those lines was kept)", printed.getvalue())


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
                    for old in (None, 4096):
                        self.assertEqual(run_teacher.is_stale(kernel, prompt, budget, old), outputs.is_stale(entry, prompt, budget, old), (finish, sha, asked, prompt, budget, old))
        self.assertEqual(score_replies.reply_fingerprints({"prompt_sha": "x", "outs": [{"prompt_sha": "y"}, {}]}), ["y", "x"])


if __name__ == "__main__":
    unittest.main()

"""Round 11: the teacher-reasoning mode of the sample stage (`--stage sample --reasoning`, rea_bench/reasoning.py). Modal and the model are stubbed; nothing here rents anything.

Owner's rule (rejection sampling, STaR / R1-distill style): a reply's FINAL ANSWER passes `score_strict` against the computed truth; the reasoning in front of it is kept as it is and
is not checked. A kept task gives the truth-rendered record and, since round 12, up to three distinct clean accepted replies verbatim (tests/test_rea_traces.py has the rules for those).
Default off = what phase A did, byte for byte.

  AnswerLine        the text after the last `Answer:` line is what is scored; no line, text after it, a think tag, a cut-off or a failed request is rejected
  ReasoningRecords  both sources per kept task, the teacher's text verbatim, the agreement rule, check_records
  OffIsUnchanged    the default run: same requests, same stats, same files as before (digest taken from the code of the commit before)
  ReasoningStage    the stubbed sampler: prompt, max_tokens, thinking off, tag/checkpoint/resume, length rule and report, quota, throughput
  ReasoningEntry    the entrypoint: --max-len required, the cost line before the GPU, output names, the recommended command of the README
  Sizing            the cost model, the quota
  TrainingWindow    modal_lora's window is an argument
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_tasks import meta as real_task, right
from .test_rea_modal import ModalCase, StubbedSampleStage, failing

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import outputs, scoring, targets  # noqa: E402
from distill import traces  # noqa: E402
from .rea_bound import rsn, sft  # noqa: E402

MSGS = lambda task: [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
STEPS = "The function loads the pointer at an offset.\nThe constant added to the base is the offset."
OLD_BLOCK = ("replies", "accepted", "wrong", "no_answer_line", "truncated", "error", "teacher_records")      # the keys of the `reasoning` block of round 11; round 12 added more (test_rea_traces.py)


def wrong_for(t: dict) -> str:
    return f"0x{int(right(t), 16) + 0x40:x}"


def reply(t: dict, steps: str = STEPS, answer: str = None) -> str:
    return f"{steps}\nAnswer: {right(t) if answer is None else answer}"


# ---- the Answer line ---------------------------------------------------------------------------------------------------------------------

class AnswerLine(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.k = right(self.t)
        self.bad = wrong_for(self.t)

    def verdict(self, text):
        return rsn.verdict(text, self.t)

    def test_a_final_answer_line_that_is_right_is_accepted_and_only_it_is_scored(self):
        self.assertEqual(self.verdict(reply(self.t)), "accepted")
        self.assertTrue(rsn.score_reasoning(reply(self.t), self.t))
        self.assertEqual(rsn.final_answer(reply(self.t)), self.k)
        self.assertTrue(rsn.score_reasoning(f"{STEPS}\nAnswer: {self.k}\n\n  \n", self.t), "blank lines after the answer are not text after it")
        self.assertEqual(self.verdict(reply(self.t, answer=self.bad)), "wrong")

    def test_a_reply_with_no_answer_line_is_rejected_even_when_its_headline_is_right(self):
        """The phase-A format (the answer on line 1) is a right reply for `score_strict`; in reasoning mode it has no `Answer:` line."""
        phase_a = f"{self.k}\nThe pointer is at that offset in the structure."
        self.assertTrue(scoring.score_strict(phase_a, self.t))
        self.assertEqual(self.verdict(phase_a), "no_answer_line")
        self.assertFalse(rsn.score_reasoning(phase_a, self.t))
        self.assertFalse(rsn.score_reasoning(f"{STEPS}\nSo the offset is {self.k}.", self.t))
        self.assertIsNone(rsn.final_answer(""))

    def test_text_after_the_last_answer_line_rejects_the_reply(self):
        for tail in ("Hope that helps.", "```", "\nBecause the loop reads it twice in a row.", "Answer", "x"):
            self.assertFalse(rsn.score_reasoning(f"{STEPS}\nAnswer: {self.k}\n{tail}", self.t), tail)
        self.assertEqual(self.verdict(f"{STEPS}\nAnswer: {self.k}\nHope that helps."), "no_answer_line")
        self.assertFalse(rsn.score_reasoning(f"{STEPS}\nAnswer:\n{self.k}", self.t), "the value on the next line is text after the Answer line")
        self.assertFalse(rsn.score_reasoning(f"{STEPS}\nAnswer: ", self.t), "an empty answer line")

    def test_two_answer_lines_the_last_one_wins_and_an_earlier_one_in_the_steps_counts_for_nothing(self):
        self.assertTrue(rsn.score_reasoning(f"Answer: {self.bad}\nWait, I misread the constant.\nAnswer: {self.k}", self.t), "a wrong first line and a right last one: right")
        self.assertFalse(rsn.score_reasoning(f"Answer: {self.k}\nWait, I misread the constant.\nAnswer: {self.bad}", self.t), "a right first line does not save a wrong last one")
        self.assertEqual(rsn.final_answer(f"Answer: 1\nAnswer: 2\nAnswer: {self.k}"), self.k)
        self.assertFalse(rsn.score_reasoning(f"Answer: {self.k}\nBut the loop also runs once more, so I am not sure.", self.t), "an Answer line earlier and prose after it: not a final line")
        self.assertTrue(rsn.score_reasoning(f"I think the Answer: {self.bad} is wrong.\n{STEPS}\nAnswer: {self.k}", self.t), "`Answer:` inside a line of the steps is not an answer line")
        self.assertFalse(rsn.score_reasoning(f"{STEPS}\nSo the Answer: {self.k}", self.t), "`Answer:` in the middle of the last line is not an answer line either")

    def test_the_grammar_is_the_strict_one_and_is_not_loosened(self):
        for answer in (f"at least {self.k}", f"not {self.k}", f"{self.k} or {self.bad}", f"{self.k}, {self.bad}", f"maybe {self.k}"):
            self.assertFalse(scoring.score_strict(answer, self.t), answer)
            self.assertFalse(rsn.score_reasoning(reply(self.t, answer=answer), self.t), answer)
        for answer in (self.k, f"`{self.k}`", f"The answer is {self.k}"):
            self.assertEqual(rsn.score_reasoning(reply(self.t, answer=answer), self.t), scoring.score_strict(answer, self.t), answer)

    def test_markdown_around_the_label_is_allowed_and_a_different_spelling_is_not(self):
        self.assertTrue(rsn.score_reasoning(f"{STEPS}\n**Answer:** {self.k}", self.t))
        self.assertTrue(rsn.score_reasoning(f"{STEPS}\n**Answer: {self.k}**", self.t))
        self.assertTrue(rsn.score_reasoning(f"{STEPS}\n  Answer: {self.k}", self.t))
        for label in ("answer:", "ANSWER:", "Final answer:", "Answer -", "The Answer:"):
            self.assertFalse(rsn.score_reasoning(f"{STEPS}\n{label} {self.k}", self.t), label)

    def test_think_blocks_are_stripped_and_a_think_tag_that_is_left_rejects(self):
        self.assertTrue(rsn.score_reasoning(f"<think>Answer: {self.bad}\nhmm</think>\n{STEPS}\nAnswer: {self.k}", self.t))
        self.assertEqual(rsn.clean_reply(f"<think>x\ny</think>\n  {STEPS}  \n"), STEPS)
        self.assertEqual(rsn.final_answer(f"<think>Answer: {self.bad}</think>\nAnswer: {self.k}"), self.k)
        self.assertFalse(rsn.score_reasoning(f"<think>hmm\n{STEPS}\nAnswer: {self.k}", self.t), "an unterminated block is not stripped: the reply is malformed")
        self.assertFalse(rsn.score_reasoning(f"{STEPS}</think>\nAnswer: {self.k}", self.t), "a closing tag with no block")

    def test_a_reply_cut_off_at_max_tokens_is_rejected_whatever_the_cut_left(self):
        cut = f"{STEPS}\nAnswer: {self.k}\n{rsn.TRUNCATED}"
        self.assertEqual(self.verdict(cut), "truncated")
        self.assertFalse(rsn.score_reasoning(cut, self.t), "the marker is text after the Answer line")
        self.assertFalse(rsn.score_reasoning(f"{STEPS}\n{rsn.TRUNCATED}", self.t))

    def test_the_text_of_a_failed_request_is_never_scored(self):
        for txt in (f"[error TimeoutError('x')]\nAnswer: {self.k}", f"[error URLError(ConnectionRefusedError(111, 'Connection refused'))]"):
            self.assertEqual(self.verdict(txt), "error")
            self.assertFalse(rsn.score_reasoning(txt, self.t))
            self.assertIsNone(rsn.teacher_target(txt, self.t))

    def test_the_teacher_text_is_the_reply_verbatim(self):
        steps = "Step 1:   reads *(p + 8).\n\n  Step 2: adds 0x8, twice.\n\tStep 3."
        r = rsn.teacher_target(f"  <think>x</think>\n\n{steps}\nAnswer: {self.k}  \n\n", self.t)
        self.assertEqual(r["text"], f"{steps}\nAnswer: {self.k}", "only think blocks and whitespace at both ends are removed: inner spaces, blank lines and tabs stay")
        self.assertEqual((r["target_source"], r["headline"]), ("teacher-reasoning", self.k))
        self.assertIsNone(rsn.teacher_target(reply(self.t, answer=self.bad), self.t))

    def test_the_label_of_the_system_message_is_the_label_that_is_parsed(self):
        self.assertIn("`Answer: <", rsn.SYSTEM_REASONING)
        self.assertIn("one last line", rsn.SYSTEM_REASONING)
        self.assertNotIn("Answer in one short line", rsn.SYSTEM_REASONING)
        self.assertEqual(rsn.SYSTEM_REASONING.count("Answer:"), 1)


# ---- the records --------------------------------------------------------------------------------------------------------------------------

class ReasoningRecords(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.k, self.bad = right(self.t), wrong_for(self.t)

    def assemble(self, outs, t=None, reasoning=True):
        t = t or self.t
        return sft.assemble([t], {t["id"]: {"outs": [[o, True] for o in outs]}}, "m", MSGS, reasoning)

    def test_a_kept_task_gives_the_truth_record_and_its_accepted_replies_verbatim_in_sample_order(self):
        first = f"<think>t</think>\n  Step one.\n\n  Step two   keeps its spaces.\nAnswer: {self.k}\n"
        kept, stats = self.assemble([reply(self.t, answer=self.bad), first, reply(self.t, "Another path, other words.")])
        self.assertEqual([r["target_source"] for r in kept], ["truth-template", "teacher-reasoning", "teacher-reasoning"])
        self.assertEqual(kept[0]["messages"][-1]["content"], targets.render(self.t)["text"])
        self.assertEqual(kept[1]["messages"][-1]["content"], f"Step one.\n\n  Step two   keeps its spaces.\nAnswer: {self.k}", "the first accepted reply comes first")
        self.assertEqual(kept[2]["messages"][-1]["content"], reply(self.t, "Another path, other words."), "then the next distinct one (round 12: several traces per task)")
        self.assertEqual(kept[0]["messages"][:2], kept[1]["messages"][:2], "the same system and user message")
        self.assertTrue(all(sorted(r) == sorted(sft.RECORD_KEYS) for r in kept))
        self.assertEqual((stats["tasks"], stats["kept_tasks"], stats["kept_records"], stats["correct"]), (1, 1, 3, 2))
        self.assertEqual({k: stats["reasoning"][k] for k in OLD_BLOCK}, {"replies": 3, "accepted": 2, "wrong": 1, "no_answer_line": 0, "truncated": 0, "error": 0, "teacher_records": 2})

    def test_the_default_run_still_scores_the_headline_and_gives_one_truth_record(self):
        phase_a = f"{self.k}\nIt is the offset of the pointer in the node."
        kept, stats = self.assemble([phase_a, phase_a], reasoning=False)
        self.assertEqual([r["target_source"] for r in kept], ["truth-template"])
        self.assertNotIn("reasoning", stats)
        kept, _ = self.assemble([reply(self.t)], reasoning=False)
        self.assertEqual(kept, [], "a reply with steps first has no headline for the default scorer: only the reasoning mode reads the last line")
        kept, stats = self.assemble([phase_a], reasoning=True)
        self.assertEqual((kept, stats["reasoning"]["no_answer_line"]), ([], 1), "a phase-A reply has no Answer line")

    def test_the_agreement_rule_of_a_high_chance_task_applies_to_the_final_answers(self):
        easy = real_task("crc", "bitwise_init", chance=0.55)
        one, two = reply(easy), reply(easy, "A second route to the same register value.")
        kept, stats = self.assemble([one, reply(easy, answer="0x7"), f"{right(easy)}\nIt is the register value."], easy)
        self.assertEqual((kept, stats["agreement"], stats["reasoning"]["accepted"]), ([], 1, 1), "a headline in the phase-A format is not an accepted reply here")
        kept, stats = self.assemble([one, two], easy)
        self.assertEqual([r["target_source"] for r in kept], ["truth-template", "teacher-reasoning", "teacher-reasoning"])
        self.assertEqual([r["messages"][-1]["content"] for r in kept[1:]], [one, two])

    def test_a_task_without_an_accepted_reply_gives_nothing_and_a_failed_request_is_counted(self):
        kept, stats = self.assemble([reply(self.t, answer=self.bad), "[error TimeoutError('x')]", f"{STEPS}\nAnswer: {self.k}\nDone.", f"{STEPS}\n{rsn.TRUNCATED}"])
        self.assertEqual((kept, stats["no_correct"], stats["errors"]), ([], 1, 1))
        self.assertEqual({k: stats["reasoning"][k] for k in OLD_BLOCK}, {"replies": 4, "accepted": 0, "wrong": 1, "no_answer_line": 1, "truncated": 1, "error": 1, "teacher_records": 0})

    def test_check_records_accepts_the_two_sources_and_refuses_any_other(self):
        good = {"messages": [], "task": "x", "kind": "k", "model": "m", "target_source": "truth-template"}
        sft.check_records([good, {**good, "target_source": "teacher-reasoning"}])
        for source in ("teacher-verified", "teacher", "", "Truth-Template", None):
            with self.assertRaisesRegex(ValueError, "target_source"):
                sft.check_records([{**good, "target_source": source}])
        with self.assertRaises(ValueError):
            sft.check_records([{**good, "target_source": "teacher-reasoning", "teacher_headline": "0x8"}])
        self.assertEqual(sft.TARGET_SOURCES, ("truth-template", "teacher-reasoning"))
        self.assertEqual((targets.TARGET_SOURCE, rsn.TARGET_SOURCE), sft.TARGET_SOURCES)

    def test_assemble_goes_through_check_records(self):
        """A record with another source never reaches the file: `assemble` ends in `check_records`."""
        real = traces.teacher_target
        bad = lambda o, t, thinking=False, *, domain: {**real(o, t, thinking, domain=domain), "target_source": "teacher-verified"}
        with mock.patch.object(traces, "teacher_target", bad):
            with self.assertRaisesRegex(ValueError, "target_source"):
                self.assemble([reply(self.t)])
        self.assertIn("check_records(kept)", inspect.getsource(sft.assemble))


# ---- the default run is unchanged -------------------------------------------------------------------------------------------------------

OLD_SYSTEM_RECIPE = ("You are a reverse engineer. You are given decompiled C pseudocode of one function from a stripped binary and a question. "
                     "Answer in one short line with the concrete value or name asked for, then one sentence of justification.")
# sha256 of the requests, the stats, the result keys, the record order and the record sources of the default sample stage on the stub below, taken from the code of the commit BEFORE the
# reasoning mode (023b71a): `modal run ... --stage sample` without --reasoning must do exactly what it did.
OFF_DIGEST = "381076ebb490c0a56a37853bf02304a7e3575c99b63ef3485cfb2e2b14e77aea"


class OffIsUnchanged(StubbedSampleStage):
    def digest(self, res):
        bodies = sorted(self.bodies, key=lambda b: (b["messages"][-1]["content"], b["seed"]))
        blob = json.dumps({"bodies": bodies, "stats": res["stats"], "keys": sorted(res), "tasks": [r["task"] for r in res["sft"]],
                           "sources": sorted({r["target_source"] for r in res["sft"]}), "n_requests": len(self.requests)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def test_the_default_run_is_what_it_was_byte_for_byte(self):
        res = self.run_stage(resume=False)
        self.assertEqual(self.digest(res), OFF_DIGEST)
        self.assertEqual(sorted(res), ["sft", "stats"])
        self.assertNotIn("reasoning", res["stats"])
        self.assertNotIn("perf", res["stats"])
        self.assertNotIn("fits", res["stats"]["over_length"])
        self.assertEqual(sorted(res["stats"]["over_length"]), ["by_kind", "dropped", "longest_tokens", "max_len", "near_limit"])

    def test_the_requests_and_the_records_of_the_default_run_are_written_out(self):
        """Independent of the digest: the body of every request and the messages of every record, as the phase-A sampler built them."""
        res = self.run_stage(resume=False)
        want = lambda t: [{"role": "system", "content": OLD_SYSTEM_RECIPE}, {"role": "user", "content": f"```c\n{t['pseudocode']}\n```\n\nQuestion: {t['question']}"}]
        for b in self.bodies:
            t = next(t for tid, t in self.tasks.items() if tid in b["messages"][-1]["content"])
            self.assertEqual(b, {"model": "local", "messages": want(t), "temperature": 0.7, "seed": b["seed"], "max_tokens": 300,
                                 "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False})
            self.assertEqual(list(b), ["model", "messages", "temperature", "seed", "max_tokens", "chat_template_kwargs", "cache_prompt"])
        self.assertEqual(self.data.SYSTEM_RECIPE, OLD_SYSTEM_RECIPE, "SYSTEM_RECIPE is untouched")
        self.assertEqual(len(res["sft"]), len(self.tasks))
        for r in res["sft"]:
            t = self.tasks[r["task"]]
            self.assertEqual(r["messages"], [*want(t), {"role": "assistant", "content": targets.render(t)["text"]}])
            self.assertEqual((r["target_source"], r["model"]), ("truth-template", "m"))
        self.assertEqual(self.data._messages(self.tasks[res["sft"][0]["task"]]), want(self.tasks[res["sft"][0]["task"]]))

    def test_the_default_checkpoint_and_output_names_have_no_suffix(self):
        self.run_stage(resume=False)
        names = sorted(p.name for p in (self.root / "data").glob("*.jsonl"))
        self.assertEqual(names, [f"{outputs.sft_stem('m', self.tag)}.partial.jsonl"])
        self.assertEqual(outputs.sample_tag(self.tag), self.tag)
        self.assertEqual(outputs.sample_tag(self.tag, False), self.tag)


# ---- the sampler in reasoning mode ---------------------------------------------------------------------------------------------------------

def answer_body(text, finish="stop", tokens=10):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}], "usage": {"completion_tokens": tokens}}


class ReasoningStage(StubbedSampleStage):
    def setUp(self):
        super().setUp()
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], f"Step for {tid}.\nThe constant is read once."))
        self.rtag = outputs.sample_tag(self.tag, True)

    def plain(self):
        """The fake server answers in the phase-A format (the answer on line 1)."""
        self.reply = lambda tid, i, body: {"choices": [{"message": {"content": right(self.tasks[tid])}}]}

    def partial(self, tag):
        return self.root / "data" / f"{outputs.sft_stem('m', tag)}.partial.jsonl"

    def kept(self, res, source):
        return [r for r in res["sft"] if r["target_source"] == source]

    def test_the_request_asks_for_the_steps_with_thinking_off_and_768_tokens(self):
        res = self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual(len(self.bodies), len(self.tasks) * self.k)
        for b in self.bodies:
            self.assertEqual(b["max_tokens"], 768)
            self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual((b["messages"][0], b["temperature"], b["cache_prompt"]), ({"role": "system", "content": rsn.SYSTEM_REASONING}, 0.7, False))
            self.assertEqual(b["messages"][1]["role"], "user")
            self.assertIn("Question:", b["messages"][1]["content"])
        self.assertNotEqual(self.data.SYSTEM_RECIPE, rsn.SYSTEM_REASONING)
        self.assertEqual(self.data.SYSTEM_RECIPE, OLD_SYSTEM_RECIPE, "SYSTEM_RECIPE itself is untouched")
        self.run_stage(False, reasoning=True, max_len=4096, max_tokens=400)
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {400}, "--max-tokens is configurable")
        self.plain()
        self.run_stage(False, max_tokens=250)
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {250})
        self.assertEqual({b["messages"][0]["content"] for b in self.bodies}, {OLD_SYSTEM_RECIPE}, "no --reasoning: the old system message, whatever --max-tokens says")
        self.assertEqual(len(res["sft"]), 2 * len(self.kept(res, "teacher-reasoning")))

    def test_both_records_of_a_kept_task_carry_the_reasoning_prompt_and_the_teacher_text(self):
        res = self.run_stage(False, reasoning=True, max_len=4096)
        truth, teacher = self.kept(res, "truth-template"), self.kept(res, "teacher-reasoning")
        self.assertEqual(len(truth), len(self.tasks))
        self.assertEqual([r["task"] for r in truth], [r["task"] for r in teacher])
        for a, b in zip(truth, teacher):
            self.assertEqual(a["messages"][:2], b["messages"][:2])
            self.assertEqual(a["messages"][0]["content"], rsn.SYSTEM_REASONING)
            t = self.tasks[a["task"]]
            self.assertEqual(a["messages"][-1]["content"], targets.render(t)["text"])
            self.assertEqual(b["messages"][-1]["content"], reply(t, f"Step for {t['id']}.\nThe constant is read once."))
            self.assertEqual(sorted(b), sorted(sft.RECORD_KEYS))
        st = res["stats"]
        self.assertEqual((st["kept_tasks"], st["kept_records"]), (len(self.tasks), 2 * len(self.tasks)))
        self.assertEqual(st["reasoning"]["teacher_records"], len(self.tasks))
        self.assertEqual(st["reasoning"]["accepted"], len(self.tasks) * self.k)

    def test_the_flag_in_the_checkpoint_is_the_verdict_of_the_reasoning_scorer(self):
        """The flag is only for reading a checkpoint (every reply is scored again from its text at the end), but it must say what that scorer says, not the phase-A scorer's."""
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid]) if i % 2 == 0 else right(self.tasks[tid]) + "\nIt is the offset of the field in the node.")
        self.run_stage(False, reasoning=True, max_len=2048)
        part = [json.loads(l) for l in self.partial(self.rtag).read_text().splitlines()]
        self.assertEqual(len(part), len(self.tasks))
        for r in part:
            self.assertEqual([ok for _o, ok in r["outs"]], [i % 2 == 0 for i in range(self.k)], r["id"])

    def test_a_failed_request_is_never_scored_in_reasoning_mode(self):
        self.fail = failing(lambda tid, i: True)
        st = self.run_stage(False, reasoning=True, max_len=2048)["stats"]
        n = len(self.tasks)
        self.assertEqual((st["errors"], st["correct"], st["kept_records"]), (n * self.k, 0, 0))
        self.assertEqual(st["reasoning"]["error"], n * self.k)
        self.assertEqual(st["reasoning"]["accepted"], 0)
        part = [json.loads(l) for l in self.partial(self.rtag).read_text().splitlines()]
        self.assertTrue(all(scoring.is_error_reply(o) and not ok for r in part for o, ok in r["outs"]))

    def test_a_reply_the_server_cut_off_is_rejected_and_counted(self):
        """finish_reason "length": the reply ends where max_tokens cut it; here the cut fell right after a correct Answer line."""
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid]), finish="length")
        res = self.run_stage(False, reasoning=True, max_len=2048)
        n = len(self.tasks)
        self.assertEqual((res["sft"], res["stats"]["kept_tasks"]), ([], 0))
        self.assertEqual(res["stats"]["reasoning"]["truncated"], n * self.k)
        part = [json.loads(l) for l in self.partial(self.rtag).read_text().splitlines()]
        self.assertTrue(all(o.endswith(rsn.TRUNCATED) for r in part for o, _ in r["outs"]), "the checkpoint says it was cut, so a re-score from the text rejects it too")
        self.assertIn("cut off at max_tokens", "\n".join(sft.reasoning_report(res["stats"])))
        self.assertTrue(any(l.startswith("WARNING") and "max_tokens" in l for l in sft.reasoning_report(res["stats"])))

    def test_a_reasoning_run_has_its_own_checkpoint_and_never_touches_a_phase_a_one(self):
        self.plain()
        self.run_stage(False)
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], f"Step for {tid}.\nThe constant is read once."))
        phase_a = self.partial(self.tag)
        before = phase_a.read_bytes()
        self.run_stage(False, reasoning=True, max_len=2048)
        self.assertEqual(phase_a.read_bytes(), before, "the phase-A checkpoint is neither written nor moved aside")
        self.assertEqual(list((self.root / "data").glob("*.old-*")), [])
        self.assertTrue(self.partial(self.rtag).exists())
        self.assertEqual(outputs.sft_stem("m", self.rtag), f"sft-m-{self.tag}-reasoning")
        self.assertTrue(all("Answer:" in o for r in map(json.loads, self.partial(self.rtag).read_text().splitlines()) for o, _ in r["outs"]))
        self.assertFalse(any("Answer:" in o for r in map(json.loads, phase_a.read_text().splitlines()) for o, _ in r["outs"]))

    def test_resume_of_a_reasoning_run_asks_only_what_failed_in_its_own_checkpoint(self):
        n = len(self.tasks)
        self.fail = failing(lambda tid, i: i == 1)
        first = self.run_stage(False, reasoning=True, max_len=2048)["stats"]
        self.assertEqual((first["answers"], first["errors"]), (n * self.k, n))
        self.fail = failing(lambda tid, i: False)
        res = self.run_stage(True, reasoning=True, max_len=2048)
        self.assertEqual(sorted(i for _t, i in self.requests), [1] * n, "sample 1 of every task, nothing else")
        self.assertEqual(res["stats"]["errors"], 0)
        self.assertEqual(len(self.kept(res, "teacher-reasoning")), n)
        self.assertEqual(self.run_stage(True, reasoning=True, max_len=2048)["stats"]["kept_records"], 2 * n)
        self.assertEqual(self.requests, [], "a complete reasoning checkpoint asks nothing")
        self.assertIn("to ask", self.printed.replace("0 to ask", "to ask"))

    def test_resume_does_not_read_the_phase_a_checkpoint_as_its_own(self):
        self.plain()
        self.run_stage(False)                                     # phase A: a complete checkpoint with the same tag
        self.run_stage(True, reasoning=True, max_len=2048)
        self.assertEqual(len(self.requests), len(self.tasks) * self.k, "the reasoning run has no checkpoint of its own: everything is asked")

    def test_a_reasoning_run_needs_the_training_window(self):
        with self.assertRaisesRegex(SystemExit, "--max-len"):
            self.run_stage(False, reasoning=True)
        self.assertEqual(self.requests, [], "nothing was asked")
        self.assertFalse(self.partial(self.rtag).exists())
        self.assertEqual(self.data._window(False, 0), 2048)
        self.assertEqual(self.data._window(False, 3072), 3072)
        self.assertEqual(self.data._window(True, 4096), 4096)
        with self.assertRaises(SystemExit):
            self.data._window(True, -1)

    def long_replies(self, words: dict):
        """A reply with `words[tid]` more words, every one different (a repeated word is a loop: the repetition filter of round 12 would drop the trace before the length rule sees it)."""
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], f"Step for {tid}.\n" + " ".join(f"w{j}" for j in range(words.get(tid, 0)))))

    def test_the_length_report_counts_what_fits_2048_3072_4096_and_the_window_drops_never_cuts(self):
        base = self.run_stage(False, reasoning=True, max_len=4096)
        ids = [r["task"] for r in self.kept(base, "teacher-reasoning")]
        n = len(ids)
        self.assertEqual(base["stats"]["over_length"]["fits"]["teacher-reasoning"], {"records": n, "2048": n, "3072": n, "4096": n})
        self.long_replies({ids[0]: 2500, ids[1]: 3500, ids[2]: 5000})        # the stub's tokenizer counts one token per word
        res = self.run_stage(False, reasoning=True, max_len=2048)
        o = res["stats"]["over_length"]
        self.assertEqual(o["fits"]["teacher-reasoning"], {"records": n, "2048": n - 3, "3072": n - 2, "4096": n - 1})
        self.assertEqual(o["fits"]["truth-template"]["records"], n)
        self.assertEqual(o["fits"]["truth-template"]["2048"], n, "the rendered record is short: it fits")
        self.assertEqual((o["max_len"], o["dropped"], o["by_source"]), (2048, 3, {"teacher-reasoning": 3}))
        self.assertEqual([r["task"] for r in self.kept(res, "teacher-reasoning")], [i for i in ids if i not in ids[:3]])
        self.assertEqual(len(self.kept(res, "truth-template")), n, "the long task keeps its truth-rendered record")
        st = res["stats"]
        self.assertEqual((st["kept_tasks"], st["kept_records"]), (n, 2 * n - 3), "a task is lost when ALL of its records are")
        for r in self.kept(res, "teacher-reasoning"):
            self.assertNotIn(rsn.TRUNCATED, r["messages"][-1]["content"])
            self.assertTrue(r["messages"][-1]["content"].endswith(f"Answer: {right(self.tasks[r['task']])}"), "a kept record is whole")
        res3 = self.run_stage(False, reasoning=True, max_len=3072)
        self.assertEqual((res3["stats"]["over_length"]["dropped"], len(self.kept(res3, "teacher-reasoning"))), (2, n - 2))
        res4 = self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual((res4["stats"]["over_length"]["dropped"], len(self.kept(res4, "teacher-reasoning"))), (1, n - 1))
        kept = {r["task"]: r["messages"][-1]["content"] for r in self.kept(res4, "teacher-reasoning")}
        self.assertEqual(kept[ids[1]], reply(self.tasks[ids[1]], f"Step for {ids[1]}.\n" + " ".join(f"w{j}" for j in range(3500))), "the 3,500-word reply is stored whole, byte for byte")
        line = sft.fits_line(res["stats"])
        self.assertIn(f"2048 tokens: {n - 3} of {n}", line)
        self.assertIn(f"3072 tokens: {n - 2} of {n}", line)
        self.assertIn(f"4096 tokens: {n - 1} of {n}", line)
        self.assertIn("--max-len 2048", line)

    def test_a_task_whose_prompt_alone_is_over_the_window_is_lost_with_both_records(self):
        base = self.run_stage(False, reasoning=True, max_len=4096)
        ids = [r["task"] for r in self.kept(base, "teacher-reasoning")]
        n, tid = len(ids), ids[1]
        p = self.root / "data" / outputs.decompiled_stem(self.tag) / f"{tid}.json"
        m = json.loads(p.read_text())
        m["pseudocode"] = f"/* {tid} */ void f(void) {{ /* " + "word " * 2100 + "*/ }"
        p.write_text(json.dumps(m))
        self.long_replies({ids[0]: 2500})
        res = self.run_stage(False, reasoning=True, max_len=2048)
        st = res["stats"]
        self.assertEqual(st["over_length"]["dropped"], 3, "the teacher record of ids[0], both of the task with the long prompt")
        self.assertEqual((st["kept_tasks"], st["kept_records"]), (n - 1, 2 * n - 3))
        kind = self.tasks[tid]["kind"]
        self.assertEqual(st["by_kind"][kind]["kept_tasks"], sum(self.tasks[i]["kind"] == kind for i in ids) - 1)
        self.assertNotIn(tid, [r["task"] for r in res["sft"]])

    def test_the_default_run_does_not_report_what_fits(self):
        self.plain()
        res = self.run_stage(False)
        self.assertNotIn("fits", res["stats"]["over_length"])
        self.assertIsNone(sft.fits_line(res["stats"]))
        self.assertEqual(sft.reasoning_report(res["stats"]), [])

    def test_the_run_reports_its_own_throughput(self):
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid]), tokens=40)
        res = self.run_stage(False, reasoning=True, max_len=2048)
        p = res["stats"]["perf"]
        self.assertEqual((p["requests"], p["completion_tokens"]), (len(self.tasks) * self.k, len(self.tasks) * self.k * 40))
        self.assertGreater(p["tok_per_s"], 0)
        self.assertIn("tokens/s", "\n".join(sft.reasoning_report(res["stats"])))
        self.assertEqual(json.loads(json.dumps(res["stats"]))["perf"]["completion_tokens"], p["completion_tokens"])

    def test_quota_samples_the_listed_kinds_up_to_the_cap_and_round_robin(self):
        self.plain()
        res = self.run_stage(False, quota="xor-key=1,password=2")
        self.assertEqual([r["task"] for r in res["sft"]], ["xor-key-7-0000", "password-7-0001", "password-7-0011"], "capped per kind, first tasks, round-robin over the kinds")
        self.assertEqual(sorted({tid for tid, _ in self.requests}), ["password-7-0001", "password-7-0011", "xor-key-7-0000"])
        self.assertEqual(len(self.requests), 3 * self.k)
        self.assertIn("--quota xor-key=1,password=2: 3 tasks to sample: xor-key 1/1, password 2/2", self.printed)
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid]))
        res = self.run_stage(False, reasoning=True, max_len=2048, quota="password=5,hash=1")
        self.assertEqual(sorted({r["task"] for r in res["sft"]}), ["hash-7-0004", "password-7-0001", "password-7-0011"])
        self.assertEqual(len(res["sft"]), 6, "two records per task")
        self.assertIn("password 2/5", self.printed)
        with self.assertRaisesRegex(SystemExit, "selects no task"):
            self.run_stage(False, quota="crc=4")

    def test_a_quota_run_is_balanced_when_it_stops_early(self):
        order = rsn.apply_quota(list(self.tasks.values()), {"password": 2, "xor-key": 2})
        self.assertEqual([t["id"] for t in order], ["password-7-0001", "xor-key-7-0000", "password-7-0011", "xor-key-7-0010"])


# ---- the entrypoint ---------------------------------------------------------------------------------------------------------------------

class ReasoningEntry(ModalCase):
    def sample(self, root, **kw):
        return self.run_main(root, stage="sample", model=self.data.MODELS[0][0], seed=7, family=kw.pop("family", "crc"), **kw)

    def rd(self, root):
        return pathlib.Path(root) / "results" / "phase6" / "rea-data"

    def test_a_reasoning_run_without_max_len_stops_before_anything_is_rented_or_written(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, "--max-len"):
                self.sample(pathlib.Path(d), reasoning=True, remote_side_effect=AssertionError("a GPU was rented"))
            self.assertEqual(list(self.rd(d).glob("sft-*")), [])
            self.sample(pathlib.Path(d), reasoning=False, remote_side_effect=None)            # the default run needs no window

    def test_the_cost_line_is_printed_before_the_remote_call_and_the_remote_call_carries_the_flags(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d), reasoning=True, max_len=3072, k=4, max_tokens=500, quota="crc=100,structs=50",
                                    remote_side_effect=lambda *a, **k: seen.append(sys.stdout.getvalue()) or {"sft": [], "stats": {"tasks": 0}})
        self.assertEqual(len(seen), 1)
        self.assertIn("projected cost: 150 tasks x k=4 = 600 replies", seen[0], "printed before the GPU is rented")
        self.assertIn("the sum of --quota", seen[0])
        self.assertIn("--max-len 3072; max_tokens 500; up to 3 traces per kept task; checkpoint and output tag seed7-crc-reasoning", seen[0])
        kw = remote.call_args.kwargs
        self.assertEqual((kw["reasoning"], kw["max_tokens"], kw["max_len"], kw["quota"], kw["resume"]), (True, 500, 3072, "crc=100,structs=50", False))
        self.assertEqual(remote.call_args.args[-1], "seed7-crc", "the decompiled tasks are read by the decompile tag")

    def test_the_default_run_prints_no_cost_line_and_asks_for_nothing_new(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d))
            self.assertNotIn("projected cost", self.printed)
            self.assertNotIn("max-len", self.printed)
            kw = remote.call_args.kwargs
            self.assertEqual((kw["reasoning"], kw["max_tokens"], kw["max_len"], kw["quota"]), (False, 0, 0, ""))

    def test_the_cost_line_counts_the_local_decompiled_tasks_after_the_quota_and_warns_over_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.rd(root).mkdir(parents=True)
            rows = [{"id": f"{kind}-7-{i:04d}", "kind": kind, "pseudocode": "x"} for kind in ("crc", "structs") for i in range(30)] + [{"id": "crc-7-9999", "kind": "crc", "error": "x"}]
            (self.rd(root) / "decompiled-seed7-crc+structs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.sample(root, family="crc,structs", reasoning=True, max_len=2048, k=4, quota="crc=10,structs=100")
            self.assertIn("projected cost: 40 tasks x k=4 = 160 replies", self.printed, "10 crc (capped) + 30 structs; the error record is not a task")
            self.assertIn("60 decompiled tasks in decompiled-seed7-crc+structs.jsonl", self.printed)
            self.assertNotIn("WARNING: the expected cost", self.printed)
            self.sample(root, family="crc,structs", reasoning=True, max_len=2048, k=4, budget_usd=0.5, force=True)
            self.assertIn("projected cost: 60 tasks", self.printed)
            self.assertIn("WARNING: the expected cost US$", self.printed)
            self.assertIn("is over the budget US$0.50", self.printed)

    def test_the_output_has_its_own_name_and_a_phase_a_output_does_not_block_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.sample(root)                                                        # phase A: sft-<model>-seed7-crc.jsonl
            name = self.data.MODELS[0][0]
            self.assertTrue((self.rd(root) / f"sft-{name}-seed7-crc.jsonl").exists())
            self.sample(root, reasoning=True, max_len=2048)                          # no --force needed
            self.assertTrue((self.rd(root) / f"sft-{name}-seed7-crc-reasoning.jsonl").exists())
            self.assertTrue((self.rd(root) / f"sft-{name}-seed7-crc-reasoning.stats.json").exists())
            self.assertIn(f"{name} seed7-crc-reasoning", self.printed)
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite.*crc-reasoning"):
                self.sample(root, reasoning=True, max_len=2048, remote_side_effect=AssertionError("a GPU was rented"))
            self.sample(root, reasoning=True, max_len=2048, force=True)

    def test_a_bad_quota_is_refused_before_a_gpu_is_rented(self):
        for quota, why in (("crc", "bad --quota entry"), ("crc=0", "positive"), ("crc=x", "positive"), ("=5", "bad --quota entry"), ("crc=5,crc=6", "twice"),
                           ("nosuch=5", "no task kind"), (",", "empty")):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(SystemExit, why):
                    self.sample(pathlib.Path(d), reasoning=True, max_len=2048, quota=quota, remote_side_effect=AssertionError("a GPU was rented"))
        self.assertEqual(rsn.parse_quota("crc=250, structs=100", rsn.known_kinds()), {"crc": 250, "structs": 100})
        self.assertIn("xor-key", rsn.known_kinds())
        self.assertIn("antidebug", rsn.known_kinds())
        self.assertNotIn("legacy", rsn.known_kinds())

    def test_the_decompile_stage_says_it_ignores_the_sampling_flags(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_main(pathlib.Path(d), stage="decompile", n=4, seed=7, family="crc", quota="crc=2")
            self.assertIn("belong to --stage sample", self.printed)
            self.run_main(pathlib.Path(d), stage="decompile", n=4, seed=8, family="crc")
            self.assertNotIn("belong to --stage sample", self.printed)

    def test_the_recommended_command_of_the_readme_is_one_the_entrypoint_takes_and_fits_the_budget(self):
        readme = (ROOT / "tools" / "modal" / "README.md").read_text()
        section = readme[readme.index("## Next run"):]
        cmds = [c.replace("\\\n", " ") for c in re.findall(r"modal run tools/modal/modal_rea_data\.py ((?:[^\n`\\]|\\\n)*)", section)]
        sample = [c for c in cmds if "--stage sample" in c and "--reasoning" in c]
        self.assertTrue(sample, "the README names the sample command of the reasoning run")
        params = {f"--{p.replace('_', '-')}" for p in inspect.signature(self.data.main).parameters}
        for c in cmds:
            flags = set(re.findall(r"(?<!\w)--[a-z][a-z0-9-]*", c))
            self.assertLessEqual(flags - {"--no-resume"}, params | {f"--no-{p[2:]}" for p in params}, c)
        cmd = max(sample, key=lambda c: sum(rsn.parse_quota(re.search(r"--quota (\S+)", c).group(1)).values()))       # the run, not the pilot
        val = lambda flag: re.search(rf"{flag} (\S+)", cmd).group(1)
        self.assertEqual(val("--model"), "codeforge-27b-heretic-q4", "only the best teacher")
        quota = rsn.parse_quota(val("--quota"), rsn.known_kinds())
        k, tokens = int(val("--k")), int(val("--max-tokens")) if "--max-tokens" in cmd else rsn.DEFAULT_MAX_TOKENS
        plan = rsn.project(sum(quota.values()), k, tokens)
        decompile = sum(quota.values()) / 1200 * 1.5            # US$ 1 per 1,200 tasks (README step 3), with a margin of half
        self.assertEqual(k, 4)
        self.assertLessEqual(plan["high"]["usd"] + decompile, 30.0, f"the high estimate of the recommended run, {plan['high']['usd']} + decompile {decompile:.2f}, is over US$ 30")
        self.assertGreater(plan["expected"]["usd"], 10, "the recommended run uses the budget")
        self.assertIn("--max-len", cmd)
        self.assertIn(f"US${plan['expected']['usd']:.0f}", section, "the README quotes the projected cost of this command")


# ---- sizing -------------------------------------------------------------------------------------------------------------------------------

class Sizing(unittest.TestCase):
    PER = 5.5 * 3600 / (3187 * 6)           # the owner's phase-A figure, written out here independently of reasoning.py

    def test_the_projection_is_the_phase_a_time_per_reply_scaled_by_the_reply_length(self):
        p = rsn.project(1000, 4, 768)
        self.assertEqual((p["tasks"], p["k"], p["replies"]), (1000, 4, 4000))
        for name, factor in (("low", 5), ("expected", 6.5), ("high", 8)):
            s = self.PER * factor
            self.assertAlmostEqual(p[name]["s_per_reply"], s, 2)
            self.assertAlmostEqual(p[name]["gpu_hours"], 4000 * s / 3600, 1)
            self.assertAlmostEqual(p[name]["usd"], 4000 * s / 3600 * 1.95, 1)
        self.assertAlmostEqual(p["worst"]["s_per_reply"], self.PER * 768 / 50, 1, msg="every reply runs to max_tokens")
        self.assertAlmostEqual(p["expected"]["usd"] / p["expected"]["gpu_hours"], 1.95, 1)

    def test_it_scales_with_tasks_k_and_the_price_and_is_cut_by_max_tokens(self):
        a, b = rsn.project(1000, 4, 768), rsn.project(2000, 4, 768)
        self.assertAlmostEqual(b["expected"]["gpu_hours"], 2 * a["expected"]["gpu_hours"], 1)
        self.assertAlmostEqual(rsn.project(1000, 8, 768)["expected"]["usd"], 2 * a["expected"]["usd"], 1)
        self.assertAlmostEqual(rsn.project(1000, 4, 768, usd_per_h=3.9)["expected"]["usd"], 2 * a["expected"]["usd"], 0)
        short = rsn.project(1000, 4, 100)
        self.assertEqual(short["low"], short["expected"])
        self.assertEqual(short["expected"], short["high"])
        self.assertAlmostEqual(short["expected"]["s_per_reply"], self.PER * 2, 2, msg="a cap of 100 tokens is twice a 50-token reply")
        self.assertLess(short["high"]["usd"], a["low"]["usd"])
        self.assertEqual(rsn.project(100, 4, 768, call_hours=5)["calls"], 1)
        self.assertEqual(rsn.project(1000, 4, 768, call_hours=5)["calls"], 2, "about 7.5 GPU-h: two calls of 5 h")
        self.assertEqual(rsn.project(2000, 4, 768, call_hours=5)["calls"], 3, "about 15 GPU-h: three calls")
        self.assertEqual(rsn.project(2000, 4, 768, call_hours=24)["calls"], 1)

    def test_the_line_says_what_it_assumes_and_the_budget_warning_has_two_levels(self):
        p = rsn.project(1450, 4, 768)
        line = rsn.cost_line(p, "1450 decompiled tasks")
        for part in ("projected cost: 1,450 tasks x k=4 = 5,800 replies", "GPU-h on one L40S", "US$", "5.5 GPU-h for 19,122 replies", "an overestimate", "worst case",
                     "times out after 5 h", "--resume", "Tasks: 1450 decompiled tasks."):
            self.assertIn(part, line)
        self.assertIsNone(rsn.budget_warning(p, 1000.0))
        self.assertIn("high estimate", rsn.budget_warning(p, (p["expected"]["usd"] + p["high"]["usd"]) / 2))
        self.assertIn("expected cost US$", rsn.budget_warning(p, p["expected"]["usd"] / 2))

    def test_local_task_count_reads_the_decompiled_file_then_the_quota_then_n(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d, "decompiled-seed7-all.jsonl")
            self.assertEqual(rsn.local_task_count(f, {}, 300)[0], 300)
            self.assertEqual(rsn.local_task_count(f, {"crc": 40, "fsm": 10}, 300)[0], 50)
            f.write_text("\n".join(json.dumps(r) for r in [{"id": "a", "kind": "crc", "pseudocode": "x"}, {"id": "b", "kind": "crc", "pseudocode": "x"},
                                                              {"id": "c", "kind": "fsm", "pseudocode": "x"}, {"id": "d", "kind": "fsm", "error": "e"}]) + "\n")
            self.assertEqual(rsn.local_task_count(f, {}, 300)[0], 3)
            self.assertEqual(rsn.local_task_count(f, {"crc": 1}, 300)[0], 1)
            self.assertEqual(rsn.local_task_count(f, {"crc": 5, "fsm": 5}, 300)[0], 3)

    def test_apply_quota(self):
        tasks = [{"id": f"{k}-{i}", "kind": k} for k in ("a", "b", "c") for i in range(3)]
        self.assertEqual(rsn.apply_quota(tasks, {}), tasks)
        got = [t["id"] for t in rsn.apply_quota(tasks, {"c": 2, "a": 3, "z": 4})]
        self.assertEqual(got, ["c-0", "a-0", "c-1", "a-1", "a-2"], "listed kinds only, capped, the kinds in turn in the order of the quota")


# ---- the training window ----------------------------------------------------------------------------------------------------------------

class TrainingWindow(ModalCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.modules.pop("distill.backends.modal_train", None)
        import distill.backends.modal_train as modal_lora
        cls.lora = modal_lora

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("distill.backends.modal_train", None)
        super().tearDownClass()

    def test_the_window_is_an_argument_with_the_old_default(self):
        self.assertEqual(self.lora.MAX_LEN, 2048)
        for fn in (self.lora.train, self.lora.main):
            p = inspect.signature(fn).parameters["max_len"]
            self.assertEqual(p.default, 2048, fn.__name__)
        src = inspect.getsource(self.lora.train)
        self.assertIn('encode_record(tok, ex["messages"], max_len,', src, "round 13: train encodes through encode_record, with the window")
        self.assertIn("encode_ids(p_ids, f_ids, max_len)", inspect.getsource(self.lora.encode_record))
        self.assertNotIn("longer than {MAX_LEN}", src)
        self.assertIn("train.remote(smoke, epochs, lr, rank, max_len, config, tag, mix)", inspect.getsource(self.lora.main))

    def test_a_longer_window_keeps_a_longer_pair_and_still_drops_never_cuts(self):
        enc = self.lora.encode_ids
        p, a = list(range(3000)), [7, 8, 9, 2]
        self.assertIsNone(enc(p, p + a, 2048))
        row = enc(p, p + a, 4096)
        self.assertEqual(row["input_ids"], p + a)
        self.assertEqual(row["labels"][-4:], a)
        self.assertIsNone(enc(list(range(4093)), list(range(4093)) + a, 4096), "one token over the window")
        self.assertIsNotNone(enc(list(range(4092)), list(range(4092)) + a, 4096), "exactly the window")

    def test_the_batch_keeps_the_tokens_per_micro_batch_and_the_effective_batch(self):
        self.assertEqual(self.lora.batch_plan(2048), (4, 4), "the default is what it was")
        self.assertEqual(self.lora.batch_plan(3072), (2, 8))
        self.assertEqual(self.lora.batch_plan(4096), (2, 8))
        self.assertEqual(self.lora.batch_plan(8192), (1, 16))
        self.assertEqual(self.lora.batch_plan(1024), (4, 4), "a shorter window does not raise the batch beyond what is known to fit")
        self.assertIn("batch_plan(max_len)", inspect.getsource(self.lora.train))


if __name__ == "__main__":
    unittest.main()

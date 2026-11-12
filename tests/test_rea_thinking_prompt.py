"""Round 14, fix 1 of the Kaggle thinking pilot (57 tasks x 2 thinking replies, 4096 tokens: 89 accepted, 10 wrong, 15 cut off, 13 dropped as a repetition): antidebug failed 6 of 6 because the `Answer:` line
held the number and not the technique the question asks for as well. The prompt now asks for every part, in the question's order; `score_strict` is not loosened. The replies are the pilot's (tests/fixtures/thinking-pilot).

  MultiPartAnswers  the prompt asks for every part on the last line, in the question's order; the fingerprint moved, the plain prompt did not; kaggle-score refuses what was asked under the old prompt
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import tempfile
import unittest

from . import distill_fixtures as fx
from .rea_bound import rsn
from .test_rea_kaggle import write_replies, write_tasks
from .thinking_pilot import ROWS, TASKS, omitted_parts, reply_of, rows  # noqa: F401

from distill import outputs, traces  # noqa: E402
from distill.domains.rea import DOMAIN, SYSTEM_RECIPE  # noqa: E402
from rea_bench import kaggle_score, reasoning, scoring  # noqa: E402

import json  # noqa: E402

# The prompt of the pilot (phase A / refactor/distill), kept here so that the fingerprint test has something to compare with.
OLD_PROMPT = ("You are a reverse engineer. You are given decompiled C pseudocode of one function from a stripped binary and a question. "
              "First write a short step-by-step analysis, at most eight short lines: what the code reads, which constants it uses and why they give the answer. "
              "Then write one last line of the form `Answer: <the concrete value or name asked for, in the form the question asks>` and nothing after it.")
OLD_SYSTEM_SHA = "bd6b48eaa3a0131b"       # outputs.prompt_sha([system message]) with OLD_PROMPT: the `system_sha` of the pilot's config.json
NEW_SYSTEM_SHA = "88584e3550d4407f"       # the same with SYSTEM_REASONING now
RECIPE_SHA256 = "962523c291037ec3"        # first 16 hex digits of sha256(SYSTEM_RECIPE): the plain prompt, which no fix touches


def system_sha(text: str) -> str:
    return outputs.prompt_sha([{"role": "system", "content": text}])


class MultiPartAnswers(unittest.TestCase):
    def test_the_prompt_asks_for_every_part_in_the_order_of_the_question_separated_by_commas(self):
        p = rsn.SYSTEM_REASONING
        self.assertTrue(p.startswith(OLD_PROMPT), "the instruction of the pilot is kept as it was; the new one follows it")
        for part in ("If the question asks for several things", "that last line must give every one of them", "in the order the question asks for them", "separated by commas as the question shows",
                     "applies to that value only", "never lets you leave another part out"):
            self.assertIn(part, p)
        self.assertEqual(p.count("Answer:"), 1, "the label that is parsed is named once")

    def test_the_fingerprint_of_the_prompt_moved_and_the_plain_prompt_did_not(self):
        self.assertEqual(system_sha(OLD_PROMPT), OLD_SYSTEM_SHA, "the phase-A prompt is what the pilot's config.json fingerprinted")
        self.assertEqual(system_sha(rsn.SYSTEM_REASONING), NEW_SYSTEM_SHA, "a new prompt is a new fingerprint: update this number on purpose, and read what kaggle-score then refuses")
        self.assertNotEqual(OLD_SYSTEM_SHA, NEW_SYSTEM_SHA)
        self.assertEqual(hashlib.sha256(SYSTEM_RECIPE.encode()).hexdigest()[:16], RECIPE_SHA256, "SYSTEM_RECIPE, the plain prompt, is byte for byte what it was")
        t = fx.load_tasks()[0]
        self.assertEqual(DOMAIN.messages(t, "plain")[0], {"role": "system", "content": SYSTEM_RECIPE})
        for mode in ("reasoning", "thinking"):
            self.assertEqual(DOMAIN.messages(t, mode)[0], {"role": "system", "content": rsn.SYSTEM_REASONING})

    def test_kaggle_score_refuses_replies_that_were_asked_under_the_old_prompt(self):
        tasks = fx.load_tasks()[:3]
        old_messages = lambda t: [{"role": "system", "content": OLD_PROMPT}, {"role": "user", "content": reasoning.user_message(t)}]
        with tempfile.TemporaryDirectory() as d:
            tp = write_tasks(d, tasks)
            lines = [{"id": t["id"], "kind": t["kind"], "prompt_sha": outputs.prompt_sha(old_messages(t) if i == 0 else DOMAIN.messages(t, "reasoning")),
                      "outs": [{"text": fx.reply_text(t, fx.STEPS[0]), "finish": "stop", "tokens": 10, "secs": 1.0}]} for i, t in enumerate(tasks)]
            with contextlib.redirect_stderr(io.StringIO()):
                _kept, stats = kaggle_score.score(write_replies(d, lines), tp, "m")
        self.assertEqual((stats["prompts"]["checked"], stats["prompts"]["mismatch"], stats["prompts"]["mismatched_ids"]), (3, 1, [tasks[0]["id"]]),
                         "the reply asked under the old prompt is skipped and counted, never stored under the new one")
        self.assertEqual(stats["tasks"], 2)

    def test_the_pilots_antidebug_replies_gave_the_number_and_not_the_technique(self):
        got = rows("antidebug")
        self.assertEqual(sorted({r["id"] for r in got}), ["antidebug-7-0135", "antidebug-7-0153", "antidebug-7-0255"])
        self.assertEqual(len(got), 6)
        for r in got:
            t, reply = TASKS[r["id"]], reply_of(r)
            final = traces.final_answer(reply)
            self.assertEqual(rsn.verdict(reply, t), "wrong")
            self.assertRegex(final, r"^[0-9]+$", "the line holds the number alone")
            self.assertEqual(omitted_parts(final, t), 1, "and the question asks for the technique too")
            self.assertTrue(any(ch in final for ch in "0123456789") and "timing" in t["question"] + json.dumps(t["accept"]))
            # score_strict is not loosened: the number alone stays wrong, the two parts in the question's order are right, in the other order they are not
            self.assertFalse(scoring.score_strict(final, t))
            self.assertTrue(scoring.score_strict(f"timing, {final}", t))
            self.assertFalse(scoring.score_strict(f"{final}, timing", t))

    def test_no_accepted_reply_of_the_pilot_leaves_a_part_out(self):
        for r in rows("quotes"):
            self.assertEqual(omitted_parts(traces.final_answer(reply_of(r)), TASKS[r["id"]]), 0, r["id"])


if __name__ == "__main__":
    unittest.main()

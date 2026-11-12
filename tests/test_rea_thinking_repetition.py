"""Round 14, fix 2 of the Kaggle thinking pilot: the repetition filter dropped 13 of the 89 accepted replies, 12 of which only quoted the question (an 8-word window such as `(one of: ptrace, timing, env,
tracerpid), and above` 4 to 12 times). What is in the task's own prompt is no longer counted; the thresholds are the same, and the replies that really loop are still dropped. The replies are the pilot's
(tests/fixtures/thinking-pilot).

  QuotedQuestion  the accepted replies the filter dropped for quoting the question are kept now
  RealLoops       the replies that really loop are still dropped
  Repetition      the rule on small texts: what is not counted (the task's own words) and what still is
"""
from __future__ import annotations

import unittest
from unittest import mock

from .rea_bound import rsn
from .thinking_pilot import TASKS, reply_of, rows

from distill import traces  # noqa: E402
from distill.domains.rea import DOMAIN  # noqa: E402


class QuotedQuestion(unittest.TestCase):
    def test_a_reply_that_only_quotes_the_question_is_not_dropped_as_a_repetition(self):
        got = rows("quotes")
        self.assertEqual(len(got), 12)
        for r in got:
            t, reply = TASKS[r["id"]], reply_of(r)
            text, prompt = traces.trace_text(reply, True), traces.prompt_text(DOMAIN, t, True)
            self.assertEqual(rsn.verdict(reply, t), "accepted", r["id"])
            self.assertTrue(traces.repetition(text), f"{r['id']} s{r['sample']}: the rule without the prompt dropped it")
            self.assertFalse(traces.repetition(text, prompt), f"{r['id']} s{r['sample']}")
            self.assertIsNone(rsn.quality_drop(reply, t, True), f"{r['id']} s{r['sample']}")

    def test_the_limit_for_the_prompts_own_words_is_the_lowest_that_keeps_the_pilots_quotes(self):
        """Round 15, m4: a window or a line of the prompt is a loop beyond PROMPT_REPEAT_MAX. At 8 four of the 12 replies that only quote the question (9, 10, 11 and 11 times) would be dropped, so it is 12."""
        got = rows("quotes")

        def dropped(limit):
            with mock.patch.object(traces, "PROMPT_REPEAT_MAX", limit):
                return [(r["id"], r["sample"]) for r in got if traces.repetition(traces.trace_text(reply_of(r), True), traces.prompt_text(DOMAIN, TASKS[r["id"]], True))]

        self.assertEqual(traces.PROMPT_REPEAT_MAX, 12)
        self.assertEqual(dropped(8), [("bounds-7-0072", 1), ("bounds-7-0294", 1), ("magic-7-0172", 0), ("magic-7-0062", 0)])
        self.assertEqual(dropped(10), [("bounds-7-0294", 1), ("magic-7-0172", 0)])
        self.assertEqual(dropped(12), [], "all 12 stay")

    def test_the_prompt_is_the_system_and_the_user_message_of_the_run(self):
        t = TASKS["bounds-7-0072"]
        want = "\n".join(m["content"] for m in DOMAIN.messages(t, "thinking"))
        self.assertEqual(traces.prompt_text(DOMAIN, t, True), want)
        self.assertEqual(traces.prompt_text(DOMAIN, t, False), want)
        self.assertIn(rsn.SYSTEM_REASONING, want)
        self.assertIn(t["question"], want)
        self.assertEqual(traces.prompt_text(DOMAIN, None, True), "")
        bare = {k: v for k, v in t.items() if k != "pseudocode"}
        self.assertEqual(traces.prompt_text(DOMAIN, bare, True), "", "a task record without its code has no prompt to quote: the strict rule applies, nothing raises")
        r = next(x for x in rows("quotes") if x["id"] == t["id"])
        self.assertEqual(rsn.quality_drop(reply_of(r), bare, True), "repetition")


class RealLoops(unittest.TestCase):
    def test_the_replies_that_really_loop_are_still_dropped(self):
        got = rows("loop")
        self.assertEqual(len(got), 6)
        for r in got:
            t, reply = TASKS[r["id"]], reply_of(r)
            self.assertTrue(traces.repetition(traces.trace_text(reply, True), traces.prompt_text(DOMAIN, t, True)), f"{r['id']} s{r['sample']} ({r['tokens']} tokens) is a loop")

    def test_the_accepted_one_is_dropped_by_the_filter_under_its_name(self):
        r = next(x for x in rows("loop") if x["finish"] == "stop")
        self.assertEqual((r["id"], r["tokens"] > 3000), ("magic-7-0062", True))
        self.assertEqual(rsn.verdict(reply_of(r), TASKS[r["id"]]), "accepted")
        self.assertEqual(rsn.quality_drop(reply_of(r), TASKS[r["id"]], True), "repetition", "it repeats its own computation and its Answer line more than three times")

    def test_the_cut_off_loops_are_loops_by_the_filter_too_and_are_rejected_before_it_by_the_cut(self):
        for r in (x for x in rows("loop") if x["finish"] == "length"):
            self.assertEqual(r["tokens"], 4096)
            self.assertEqual(rsn.verdict(reply_of(r), TASKS[r["id"]]), "truncated")

    def test_the_pilot_in_numbers(self):
        """Of the 13 accepted replies the old rule dropped, the 12 that only quote the question pass and the one that repeats itself is dropped."""
        accepted = [r for r in rows("quotes") + [x for x in rows("loop") if x["finish"] == "stop"]]
        self.assertEqual(len(accepted), 13)
        why = [rsn.quality_drop(reply_of(r), TASKS[r["id"]], True) for r in accepted]
        self.assertEqual((why.count(None), why.count("repetition")), (12, 1))
        self.assertTrue(all(traces.repetition(traces.trace_text(reply_of(r), True)) for r in accepted), "the rule without the prompt dropped all 13")

    def test_the_thresholds_are_the_ones_of_the_pilot(self):
        self.assertEqual((traces.LINE_REPEAT_MAX, traces.LINE_MIN_CHARS, traces.NGRAM, traces.NGRAM_REPEAT_MAX), (3, 12, 8, 3))


class Repetition(unittest.TestCase):
    PROMPT = ("Question: Which anti-debugging technique does proc implement (one of: ptrace, timing, env, tracerpid), and above how many nanoseconds between the two clock readings does it report a debugger?\n"
              "```c\nint proc(void) {\n  if (elapsed > 0x1234) return 1;\n  return 0;\n}\n```")
    QUOTE = "which anti-debugging technique does proc implement (one of: ptrace, timing, env, tracerpid), and above how many nanoseconds"
    LEADS = ("Hmm,", "Right,", "Okay,", "So,", "Now,", "Next,", "Again,", "Still,", "Finally,")
    TAILS = ("ptrace?", "timing!", "env.", "tracerpid;", "clock", "readings", "debugger", "report", "above")

    def test_quoting_the_question_many_times_is_not_a_loop_and_looping_in_ones_own_words_is(self):
        text = "\n".join(f"{lead} {self.QUOTE} {tail}" for lead, tail in zip(self.LEADS, self.TAILS))
        self.assertTrue(traces.repetition(text), "without the prompt, nine quotes of the question are a loop")
        self.assertFalse(traces.repetition(text, self.PROMPT), "with it they are the question")
        loop = "\n".join(f"{lead} wait let me recheck the clock difference against the threshold again {tail}" for lead, tail in zip(self.LEADS[:5], self.TAILS))
        self.assertTrue(traces.repetition(loop, self.PROMPT), "what the model says itself still counts")

    def test_a_line_of_the_prompt_is_not_a_repeated_line_and_any_other_line_is(self):
        said = lambda line, n: "\n".join(f"{line}\n{lead} {tail}" for lead, tail in zip(self.LEADS[:n], self.TAILS))        # the line quoted n times, a different remark between the quotes
        code = said("if (elapsed > 0x1234) return 1;", 5)
        self.assertTrue(traces.repetition(code))
        self.assertFalse(traces.repetition(code, self.PROMPT), "the pseudocode's own line, as the model quotes it")
        other = said("if (elapsed > 0x9999) return 1;", 5)
        self.assertTrue(traces.repetition(other, self.PROMPT), "a line that is not in the prompt")
        bullets = "\n".join(f"{b} if (elapsed > 0x1234) return 1;\n{lead} {tail}" for b, lead, tail in zip(("-", "*", "1.", "2)", "> "), self.LEADS, self.TAILS))
        self.assertTrue(traces.repetition(bullets))
        self.assertFalse(traces.repetition(bullets, self.PROMPT), "a bullet or a list number does not make it another line")

    def test_a_window_that_only_partly_overlaps_the_prompt_still_counts(self):
        prompt = "alpha beta gamma delta epsilon zeta eta theta"
        text = " ".join(["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 4)
        self.assertTrue(traces.repetition(text, prompt), "the windows that run on past the prompt's words are the model's own")
        quoted = " ".join(f"{lead} alpha beta gamma delta epsilon zeta eta theta {tail}" for lead, tail in zip(self.LEADS, self.TAILS))
        self.assertFalse(traces.repetition(quoted, prompt))
        self.assertTrue(traces.repetition(quoted), "the same text, no prompt")

    def test_three_repeats_are_still_allowed_and_four_are_not(self):
        for n, want in ((3, False), (4, True)):
            text = "\n".join(f"{lead} first the code subtracts the base then compares the width {tail}" for lead, tail in zip(self.LEADS[:n], self.TAILS))
            self.assertEqual(traces.repetition(text, self.PROMPT), want, n)

    def test_the_prompts_own_words_are_counted_against_a_higher_limit_not_ignored(self):
        """Round 15, m4: `repetition` used to skip every window and line of the prompt, so any number of restatings passed. Now more than PROMPT_REPEAT_MAX (12) is a loop, the strict limit for the rest."""
        self.assertEqual(traces.PROMPT_REPEAT_MAX, 12)
        for rounds, want in ((12, False), (13, True)):
            windows = "\n".join(f"{lead} {self.QUOTE} {tail}" for lead, tail in zip((self.LEADS * 2)[:rounds], (self.TAILS * 2)[:rounds]))     # the same prompt window, `rounds` times
            self.assertEqual(traces.repetition(windows, self.PROMPT), want, f"window x {rounds}")
            said = lambda line: "\n".join(f"{line}\n{lead} {tail}" for lead, tail in zip((self.LEADS * 2)[:rounds], (self.TAILS * 2)[:rounds]))
            self.assertEqual(traces.repetition(said("if (elapsed > 0x1234) return 1;"), self.PROMPT), want, f"a prompt line (5 words, so no window of its own) x {rounds}")
            labelled = "\n".join(f"check {i}: {self.QUOTE} {i}" for i in range(rounds))
            self.assertEqual(traces.repetition(labelled, self.PROMPT), want, f"the question's sentence, a number before and after it in each round, x {rounds}: the windows inside it are the prompt's and count")
        self.assertTrue(traces.repetition("\n".join(f"{lead} {self.QUOTE} {tail}" for lead, tail in zip(self.LEADS, self.TAILS))), "no prompt: the strict limit")

    def test_what_still_passes_is_the_documented_hole_and_nothing_wider(self):
        label = lambda line, rounds: "\n".join(f"check {i}: {line}" for i in range(rounds))
        self.assertFalse(traces.repetition(label("if (elapsed > 0x1234) return 1;", 40), self.PROMPT), "a short line restated with a numbered label: no window repeats, and the line is not the prompt's")
        self.assertFalse(traces.repetition(label("if (elapsed > 0x1234) return 1;", 40)), "... with no prompt as well: that is why the docstring and the READMEs say so")
        seven = "if (elapsed > 0x1234) return 1 now ok"
        self.assertEqual(len(traces._gram_words(seven)), 7)
        self.assertTrue(traces.repetition(label(seven, 6), self.PROMPT.replace("return 1;", "return 1 now ok;")), "7 words and the label's word are a window that repeats: not in the hole")
        six = "if (elapsed > 0x1234) return 1 now"
        self.assertFalse(traces.repetition(label(six, 40), self.PROMPT.replace("return 1;", "return 1 now;")), "6 words: in it (NGRAM - 2)")
        copied = self.PROMPT.replace("\n", " ") + " "
        self.assertTrue(traces.repetition(copied * 6, self.PROMPT), "a block of the question copied over and over: the windows across the copies are not in the prompt")
        doc = traces.repetition.__doc__
        for part in ("THE HOLE", "PROMPT_REPEAT_MAX", "12", "SHORT line", "NGRAM - 2 = 6 words", "numbered" if False else "a number which changes each round", "length rule"):
            self.assertIn(part, doc)

    def test_a_row_of_numbers_is_not_a_window_whatever_the_prompt(self):
        self.assertFalse(traces.repetition("0x00 " * 40, ""))
        self.assertFalse(traces.repetition("a b c d e f g h i j k l"))

    def test_quality_drop_asks_the_domain_for_the_prompt_of_the_run(self):
        seen = []

        class Spy(type(DOMAIN)):
            def messages(self, task, mode):
                seen.append(mode)
                return super().messages(task, mode)
        t = TASKS["structs-7-0079"]
        r = next(x for x in rows("quotes") if x["id"] == t["id"])
        self.assertIsNone(traces.quality_drop(reply_of(r), t, True, domain=Spy()))
        self.assertEqual(seen, ["thinking"])
        seen.clear()
        traces.quality_drop(traces.reply_text(r["text"], "stop", "", False), t, False, domain=Spy())
        self.assertEqual(seen, ["reasoning"])


if __name__ == "__main__":
    unittest.main()

"""Round 12: several distinct correct traces per task (RFT), the quality filters of the traces (DeepSeek-R1 SFT-data style), the thinking pilot (`--thinking`) and the output path of
modal_lora.py. Modal and the model are stubbed; nothing here rents anything. tests/test_rea_reasoning.py keeps the round-11 rules (the Answer line, the checkpoint, the length rule, the default run
byte for byte), tests/test_rea_kaggle.py the standalone scorer.

  Traces            up to `--max-traces` distinct accepted replies per kept task, in sample order; one truth record; the cap; the stats
  Duplicates        identical normalized text, token-set Jaccard >= 0.9, in sample order, against the traces kept
  MixedLanguage     non-ASCII letters outside code and quotes: dropped; hex dumps, quoted strings, identifiers of the task, symbols: not
  Repetition        the same line or the same 8-gram more than three times: dropped; rows of numbers, short lines: not
  Unreadable        no analysis in front of the Answer line, or the answer restated: dropped
  Contradiction     the last explicit statement of a final value names another number than the answer: dropped; self-corrections, hedges, anything unclear: kept
  Accounting        every reply is a trace or in exactly one drop reason, per family; agreement counts answers
  Thinking          `reasoning_content` or inline <think>, only the visible Answer line is scored, the request, the tag, the stats, the entrypoint
  MaxTraces         the flag in the stage and the entrypoint
  TrainingOutputs   modal_lora's output path and dataset config
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_tasks import meta as real_task, right
from .test_rea_modal import ModalCase, StubbedSampleStage
from .test_rea_reasoning import MSGS, STEPS, answer_body, reply, wrong_for

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import outputs, reasoning as rsn, sft, targets  # noqa: E402


def distinct(t: dict, n: int = 5) -> list:
    """n accepted replies with different words (token sets far apart), same final answer."""
    lines = ["The function reads a field at a fixed offset from the base.\nThe constant is what the question asks for.",
             "Following the loop gives the key material directly.\nOnly one value of it is needed here.",
             "Looking at the comparison the literal stands out clearly.\nThat literal answers the question.",
             "First the argument is cast, then a table entry is fetched.\nThe entry holds the number that is asked.",
             "Nothing is computed at run time in this routine.\nThe immediate operand carries the result."]
    return [reply(t, lines[i]) for i in range(n)]


# ---- the traces a task keeps -----------------------------------------------------------------------------------------------------------------

class Traces(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.k, self.bad = right(self.t), wrong_for(self.t)

    def assemble(self, outs, t=None, **kw):
        t = t or self.t
        return sft.assemble([t], {t["id"]: {"outs": [[o, True] for o in outs]}}, "m", MSGS, True, **kw)

    def sources(self, kept):
        return [r["target_source"] for r in kept]

    def test_up_to_three_distinct_accepted_replies_are_kept_in_sample_order_and_the_cap_counts_the_rest(self):
        outs = distinct(self.t, 5)
        kept, stats = self.assemble(outs)
        self.assertEqual(rsn.DEFAULT_MAX_TRACES, 3)
        self.assertEqual(self.sources(kept), ["truth-template", "teacher-reasoning", "teacher-reasoning", "teacher-reasoning"])
        self.assertEqual([r["messages"][-1]["content"] for r in kept[1:]], outs[:3], "the first three, verbatim, in the order they were sampled")
        self.assertEqual(stats["reasoning"]["dropped"]["over_cap"], 2)
        self.assertEqual((stats["kept_tasks"], stats["kept_records"], stats["reasoning"]["teacher_records"], stats["reasoning"]["accepted"]), (1, 4, 3, 5))

    def test_max_traces_is_the_cap_and_one_gives_the_old_single_record(self):
        outs = distinct(self.t, 5)
        for cap in (1, 2, 4, 5):
            kept, stats = self.assemble(outs, max_traces=cap)
            self.assertEqual(self.sources(kept).count("teacher-reasoning"), cap, cap)
            self.assertEqual(stats["reasoning"]["dropped"]["over_cap"], 5 - cap)
            self.assertEqual(stats["reasoning"]["max_traces"], cap)
        self.assertEqual(self.assemble(outs, max_traces=1)[0][1]["messages"][-1]["content"], outs[0])
        self.assertEqual(len(self.assemble(outs[:2])[0]), 3, "fewer distinct replies than the cap: all of them")

    def test_the_truth_template_record_stays_one_per_task(self):
        kept, _ = self.assemble(distinct(self.t, 5))
        truth = [r for r in kept if r["target_source"] == "truth-template"]
        self.assertEqual(len(truth), 1)
        self.assertEqual(truth[0]["messages"][-1]["content"], targets.render(self.t)["text"])
        self.assertEqual(kept[0], truth[0], "and it comes first")
        self.assertEqual(len({json.dumps(r["messages"][:2]) for r in kept}), 1, "every record of the task has the same prompt")

    def test_a_rejected_reply_is_never_a_trace(self):
        kept, stats = self.assemble([reply(self.t, answer=self.bad), *distinct(self.t, 2), f"{STEPS}\nAnswer: {self.k}\nDone."])
        self.assertEqual(self.sources(kept).count("teacher-reasoning"), 2)
        self.assertEqual((stats["reasoning"]["wrong"], stats["reasoning"]["no_answer_line"], stats["reasoning"]["accepted"]), (1, 1, 2))

    def test_a_task_whose_traces_are_all_filtered_keeps_its_truth_record(self):
        loop = reply(self.t, "\n".join(["Wait, let me recheck the code again."] * 5))
        kept, stats = self.assemble([loop, loop, loop])
        self.assertEqual(self.sources(kept), ["truth-template"])
        r = stats["reasoning"]
        self.assertEqual((stats["kept_tasks"], r["teacher_records"], r["tasks_without_trace"], r["dropped"]["repetition"]), (1, 0, 1, 3))

    def test_agreement_counts_the_answers_not_the_quality_of_the_traces(self):
        easy = real_task("crc", "bitwise_init", chance=0.55)
        cjk = reply(easy, "The function folds the register once per byte. 这个函数读取指针并加上偏移量。")
        clean = reply(easy, "A second route: the table entry is the value.")
        kept, stats = self.assemble([cjk, clean], easy)
        self.assertEqual(self.sources(kept), ["truth-template", "teacher-reasoning"], "two right answers: the task is kept; only the clean trace is stored")
        self.assertEqual(stats["reasoning"]["dropped"]["mixed_language"], 1)
        kept, stats = self.assemble([cjk, reply(easy, answer="0x7")], easy)
        self.assertEqual((kept, stats["agreement"]), ([], 1), "one right answer: not kept, whatever its trace")
        self.assertEqual(stats["reasoning"]["in_lost_tasks"], 1, "its accepted reply is counted as in a task that was not kept")
        self.assertEqual(stats["reasoning"]["dropped"]["mixed_language"], 0, "and not examined")

    def test_the_default_run_ignores_the_trace_options(self):
        phase_a = f"{self.k}\nIt is the offset of the pointer in the node."
        kept, stats = sft.assemble([self.t], {self.t["id"]: {"outs": [[phase_a, True], [f"{self.k}\nThe field sits there in the node.", True]]}}, "m", MSGS, False, max_traces=5)
        self.assertEqual(self.sources(kept), ["truth-template"])
        self.assertNotIn("reasoning", stats)


# ---- duplicates ---------------------------------------------------------------------------------------------------------------------------

class Duplicates(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)

    def test_the_normal_form_is_lower_case_with_one_space_and_the_digits_kept(self):
        self.assertEqual(rsn.normalized("  The CONSTANT,\n\tis   0x20 (32)!  "), "the constant, is 0x20 (32)!")
        self.assertEqual(rsn.normalized("<think>\nA b\n</think>\n\nC"), "a b c")
        self.assertNotEqual(rsn.normalized("offset 0x20"), rsn.normalized("offset 0x24"), "digits are kept")
        self.assertEqual(rsn.token_set("The CONSTANT, is 0x20 (32)! FUN_00101234 a-b"), {"the", "constant", "is", "0x20", "32", "fun_00101234", "a", "b"})

    def test_identical_normalized_text_is_a_duplicate_whatever_the_case_and_the_spacing(self):
        a = reply(self.t, "The function loads the pointer at an offset.\nThe constant is the offset.")
        b = a.upper().replace("\n", "\n\n").replace(" ", "  ")
        self.assertEqual(rsn.normalized(a), rsn.normalized(b))
        self.assertTrue(rsn.is_duplicate(a, b))
        self.assertTrue(rsn.is_duplicate(a, a.replace(".", " .").replace(":", " -")), "punctuation only: the same set of words")
        self.assertFalse(rsn.is_duplicate(a, reply(self.t, "A table lookup gives it directly.\nNothing is computed.")))
        self.assertTrue(rsn.is_duplicate("", " ... "), "two texts without a word are the same")

    def test_the_digits_tell_two_otherwise_identical_replies_apart(self):
        a = "the stride is 0x20 bytes so the record size is 0x20"
        b = "the stride is 0x24 bytes so the record size is 0x24"
        self.assertEqual(rsn.jaccard(rsn.token_set(a), rsn.token_set(b)), 7 / 9, "the two numbers are two words that differ")
        self.assertFalse(rsn.is_duplicate(a, b))
        self.assertFalse(rsn.is_duplicate("size 0x20", "size 0x24"), "short texts: one digit changes half of the words")

    def test_near_duplicates_are_token_set_jaccard_at_least_nine_tenths(self):
        words = [f"w{i}" for i in range(19)]
        a = " ".join(words)
        one_changed = " ".join(words[:18] + ["x"])               # 18 shared of 20 in the union = 0.90
        two_changed = " ".join(words[:17] + ["x", "y"])          # 17 shared of 21 = 0.81
        self.assertEqual(rsn.jaccard(set(a.split()), set(one_changed.split())), 18 / 20)
        self.assertEqual(rsn.JACCARD_DUPLICATE, 0.9)
        self.assertTrue(rsn.is_duplicate(a, one_changed), "exactly 0.9 is a duplicate (>=)")
        self.assertFalse(rsn.is_duplicate(a, two_changed))
        self.assertTrue(rsn.is_duplicate(a, " ".join(reversed(words))), "the SET of words: the same words in another order are one trace")
        self.assertTrue(rsn.is_duplicate(a, a + " w3 w4 w5"), "repeated words do not change a set")

    def test_in_sample_order_the_first_of_two_duplicates_is_kept_and_the_second_counted(self):
        first = reply(self.t, "The function loads the pointer at an offset.\nThe constant added to the base is the offset.")
        again = first.replace("The constant", "the constant")                  # same normal form
        other = distinct(self.t, 1)[0]
        kept, dropped = rsn.pick_traces([first, again, other], self.t)
        self.assertEqual([k["text"] for k in kept], [first, other])
        self.assertEqual(dict(dropped), {"duplicate": 1})
        kept, dropped = rsn.pick_traces([again, first, other], self.t)
        self.assertEqual([k["text"] for k in kept], [again, other], "order = sample order: the one sampled first stays")

    def test_a_duplicate_is_compared_with_the_traces_kept_not_with_the_ones_dropped(self):
        cjk = reply(self.t, "The function loads the pointer at an offset. 这个函数读取指针。\nThe constant added to the base is the offset.")
        clean = reply(self.t, "The function loads the pointer at an offset.\nThe constant added to the base is the offset.")
        kept, dropped = rsn.pick_traces([cjk, clean], self.t)
        self.assertEqual([k["text"] for k in kept], [clean], "the near-copy of a reply that was filtered is not a duplicate of anything stored")
        self.assertEqual(dict(dropped), {"mixed_language": 1})

    def test_the_stage_counts_duplicates_per_family(self):
        t2 = real_task("pktsum", "sum8", chance=0.01)
        same = reply(self.t, "The function loads the pointer at an offset.\nThe constant is the offset.")
        r2 = reply(t2, "A loop adds every byte of the packet.\nThe checksum sits right after the last byte.")
        kept, stats = sft.assemble([self.t, t2], {self.t["id"]: {"outs": [[same, True]] * 3}, t2["id"]: {"outs": [[r2, True]] * 2}}, "m", MSGS, True)
        self.assertEqual(stats["reasoning"]["by_kind"]["structs"]["dropped"]["duplicate"], 2)
        self.assertEqual(stats["reasoning"]["by_kind"]["pktsum"]["dropped"]["duplicate"], 1)
        self.assertEqual(stats["reasoning"]["dropped"]["duplicate"], 3)
        self.assertEqual(sum(r["target_source"] == "teacher-reasoning" for r in kept), 2, "one trace per task")


# ---- the quality filters ---------------------------------------------------------------------------------------------------------------------

class Filters(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.k = right(self.t)                 # 0x10

    def drop(self, steps, answer=None, t=None, thinking=False):
        t = t or self.t
        return rsn.quality_drop(reply(t, steps, answer), t, thinking)


class MixedLanguage(Filters):
    CJK = "这个函数读取指针并加上偏移量。"
    CYR = "Функция читает указатель."

    def test_cjk_and_cyrillic_prose_is_dropped(self):
        for text in (self.CJK, self.CYR, "مرحبا بالعالم"):
            self.assertEqual(self.drop(f"{STEPS}\n{text}"), "mixed_language", text)
            self.assertEqual(self.drop(f"{STEPS} {text} and then {STEPS}"), "mixed_language", "in the middle of a line too")

    def test_hex_dumps_are_not_language(self):
        dump = "e4 b8 ad e6 96 87 e5 ad 97 e7 ac a6 00 00 00 00 ff fe \\xe4\\xb8\\xad 0xe4b8ad \\u4e2d"
        self.assertIsNone(self.drop(f"{STEPS}\nThe table holds these bytes: {dump}"))
        self.assertFalse(rsn.mixed_language(dump))

    def test_quoted_strings_and_code_are_not_prose(self):
        for text in (f'The string "{self.CYR}" is compared with the input.', f"The literal `{self.CJK}` is compared with the input.",
                     f"The pseudocode:\n```c\nputs(\"{self.CJK}\");\n```", f"It prints '{self.CYR}' once.", f"It prints “{self.CJK}” once."):
            self.assertIsNone(self.drop(f"{STEPS}\n{text}"), text)
        self.assertEqual(self.drop(f'{STEPS}\nThe string "x" is {self.CYR}'), "mixed_language", "outside the quotes it counts again")
        self.assertEqual(self.drop(f"{STEPS}\nIt isn't the key's value {self.CYR} here"), "mixed_language", "an apostrophe in `isn't` or `key's` opens no quoted string")

    def test_a_non_ascii_identifier_of_the_task_is_not_language_mixing_where_the_task_does_not_have_it(self):
        t = {**self.t, "pseudocode": "int ключ = 0x10; return p->next + ключ;"}
        steps = f"{STEPS}\nThe variable ключ holds the offset that is added to the pointer."
        self.assertIsNone(self.drop(steps, t=t), "the word is in the pseudocode of the task")
        self.assertEqual(self.drop(steps), "mixed_language", "the same reply for a task whose code has no such word")
        t2 = {**self.t, "refs": "table: ключ", "pseudocode": "x"}
        self.assertIsNone(self.drop(steps, t=t2), "or in the data the task shows")

    def test_symbols_digits_and_punctuation_are_not_letters(self):
        text = f"{STEPS}\n→ p + 8 ≤ end × 2 — “stride” ≠ 0 ✓ ± 1 … base • ok"
        self.assertIsNone(self.drop(text))
        self.assertFalse(rsn.mixed_language(text))

    def test_the_threshold_is_half_a_percent_of_the_letters(self):
        self.assertEqual(rsn.MIXED_LANGUAGE_MAX, 0.005)
        base = "abcdefghij " * 20                           # 200 ASCII letters
        self.assertFalse(rsn.mixed_language(base + "é"), "1 of 201 letters is 0.498%: not more than 0.5%")
        self.assertTrue(rsn.mixed_language(base + "éé"), "2 of 202 is 0.99%")
        self.assertFalse(rsn.mixed_language(""), "no letters at all")
        self.assertFalse(rsn.mixed_language("0x10 0x20 12345"))

    def test_it_is_counted_per_family_and_the_reply_is_not_stored(self):
        bad = reply(self.t, f"{STEPS}\n{self.CJK}")
        kept, stats = sft.assemble([self.t], {self.t["id"]: {"outs": [[bad, True], [reply(self.t, "A table entry gives it.\nNothing else is needed."), True]]}}, "m", MSGS, True)
        self.assertEqual(stats["reasoning"]["by_kind"]["structs"]["dropped"]["mixed_language"], 1)
        self.assertFalse(any(self.CJK in r["messages"][-1]["content"] for r in kept))

    def test_a_thinking_trace_is_checked_whole(self):
        r = f"<think>\nHmm, {self.CJK} the pointer.\n</think>\n\n{STEPS}\nAnswer: {self.k}"
        self.assertEqual(rsn.quality_drop(r, self.t, True), "mixed_language", "the foreign words are in the reasoning, not in the visible text")


class Repetition(Filters):
    def test_the_same_line_more_than_three_times_is_a_loop(self):
        line = "Wait, let me recheck the offset again."
        self.assertEqual(self.drop("\n".join([STEPS] + [line] * 4)), "repetition")
        self.assertEqual(self.drop("\n".join([line.upper(), f"  - {line}", line.lower(), f"1. {line}", STEPS])), "repetition", "case, bullets and numbering do not hide it")
        self.assertIsNone(self.drop("\n".join([STEPS] + [line] * 3)), "three times is the limit")
        self.assertEqual((rsn.LINE_REPEAT_MAX, rsn.NGRAM, rsn.NGRAM_REPEAT_MAX), (3, 8, 3))

    def test_the_same_window_of_eight_words_more_than_three_times_is_a_loop(self):
        phrase = "The loop reads the same byte and then writes it back to the same place."
        self.assertEqual(self.drop(" Hmm. ".join([phrase] * 4)), "repetition")
        self.assertIsNone(self.drop(" Hmm. ".join([phrase] * 3)))
        self.assertEqual(self.drop(" ".join(["wait"] * 12) + "\n" + STEPS), "repetition", "one word over and over")

    def test_rows_of_numbers_and_short_lines_are_not_a_loop(self):
        rows = "\n".join(["00 00 00 00 00 00 00 00 00 00 00 00"] * 6)
        self.assertIsNone(self.drop(f"{STEPS}\nThe table in memory:\n{rows}"))
        self.assertIsNone(self.drop(f"{STEPS}\n" + "\n".join(["return 0;"] * 6)), "a line shorter than 12 characters")
        self.assertIsNone(self.drop(f"{STEPS}\n" + "0x10 0x10 0x10 0x10 0x10 0x10 0x10 0x10 0x10 0x10"), "a window of numbers only")
        self.assertFalse(rsn.repetition("a b c d e f g h i j k l"))

    def test_a_self_correction_that_says_new_things_is_not_a_loop(self):
        text = f"{STEPS}\nWait, the struct has a pad.\nWait, no: the pad is before the field.\nWait, let me recount the bytes.\nWait, the first field is 8 bytes."
        self.assertIsNone(self.drop(text))

    def test_it_is_counted_per_family(self):
        loop = reply(self.t, "\n".join(["Wait, let me recheck the code again."] * 4))
        _, stats = sft.assemble([self.t], {self.t["id"]: {"outs": [[loop, True], [loop, True]]}}, "m", MSGS, True)
        self.assertEqual(stats["reasoning"]["by_kind"]["structs"]["dropped"]["repetition"], 2)


class Unreadable(Filters):
    def test_no_analysis_in_front_of_the_answer_line_is_dropped(self):
        self.assertEqual(self.drop(""), "unreadable")
        self.assertEqual(rsn.quality_drop(f"Answer: {self.k}", self.t), "unreadable")
        self.assertEqual(rsn.quality_drop(f"\n\n  \nAnswer: {self.k}", self.t), "unreadable")
        self.assertEqual(self.drop("**Analysis:**"), "unreadable", "a heading with nothing under it")

    def test_an_analysis_that_only_restates_the_answer_is_dropped(self):
        for steps in (self.k, f"The answer is {self.k}.", "So the answer is 16.", "Therefore: 0x10", f"`{self.k}`", "The final result is the value 0x10"):
            self.assertEqual(self.drop(steps), "unreadable", steps)

    def test_any_real_reasoning_in_front_of_the_answer_is_readable(self):
        for steps in ("It reads the pointer.", "The pointer sits at offset 16.", f"The code adds {self.k} to the base pointer.", STEPS):
            self.assertIsNone(self.drop(steps), steps)

    def test_in_a_thinking_run_the_reasoning_block_is_the_analysis(self):
        r = f"<think>\nThe node struct has the next pointer after one field.\n</think>\n\nAnswer: {self.k}"
        self.assertIsNone(rsn.quality_drop(r, self.t, True))
        self.assertEqual(rsn.quality_drop(f"<think>\nIt is {self.k}.\n</think>\n\nAnswer: {self.k}", self.t, True), "unreadable")

    def test_it_is_counted_per_family(self):
        _, stats = sft.assemble([self.t], {self.t["id"]: {"outs": [[f"Answer: {self.k}", True], [reply(self.t, self.k), True], [reply(self.t), True]]}}, "m", MSGS, True)
        self.assertEqual(stats["reasoning"]["by_kind"]["structs"]["dropped"]["unreadable"], 2)


class Contradiction(Filters):
    def test_the_last_explicit_statement_of_another_number_is_dropped(self):
        for steps in ("It loads the pointer.\nSo the offset is 0x20.", "It loads the pointer at some place.\nThe answer is 0x20.", "It loads the pointer.\nFinal answer: 32",
                      "It loads the pointer.\nTherefore 0x8 + 0x18 = 0x20.", "It loads it.\nSo the answer is 32 bytes.", "It loads the pointer.\n**The answer is `0x20`**"):
            self.assertEqual(self.drop(steps), "contradiction", steps)

    def test_the_task_accept_groups_read_the_values_so_other_spellings_are_the_same_number(self):
        for steps in ("It loads the pointer.\nSo the offset is 16.", "It loads the pointer.\nThe answer is 0x10 (16).", "It loads the pointer.\nSo the answer is sixteen.",
                      "It loads the pointer.\nThe answer is 0x10", "It loads the pointer.\nresult = 0x0010"):
            self.assertIsNone(self.drop(steps), steps)
        t = real_task("bounds", "range_chain", chance=0.01)               # two numeric groups: lo and hi
        lo, hi = right(t).replace(",", "").split()[:2]
        self.assertIsNone(self.drop(f"It tests a chain.\nSo the answer is {lo}, {hi}.", right(t), t))
        self.assertEqual(self.drop("It tests a chain.\nSo the answer is 1, 2.", right(t), t), "contradiction")

    def test_a_self_correction_is_kept_like_r1_does(self):
        self.assertIsNone(self.drop("It loads the pointer.\nSo the offset is 0x20.\nWait, the struct has padding, let me recheck."), "no new statement after the wrong one")
        self.assertIsNone(self.drop("It loads the pointer.\nThe answer is 0x20.\nActually, the first field is 8 bytes."))
        self.assertIsNone(self.drop("The answer is 0x20. Hmm, that does not look right."))
        self.assertIsNone(self.drop("It loads the pointer.\nSo the offset is 0x20.\nWait, the struct has a field before it.\nSo the offset is 0x10."), "wrong, then corrected: the LAST statement agrees")
        self.assertEqual(self.drop("It loads the pointer.\nSo the offset is 0x10.\nWait, no, I mean the padding.\nSo the offset is 0x20."), "contradiction", "right, then wrong as the last word")
        self.assertEqual(self.drop("Wait, let me start again.\nThe struct is read.\nSo the offset is 0x20."), "contradiction", "the correction came before the statement")

    def test_anything_unclear_is_kept(self):
        for steps in ("It loads the pointer.\nThe answer is probably around 0x20.",              # a word that is not a value
                      "It loads the pointer.\nThe answer is 0x20 or 0x10.",                      # a hedge: one number is right
                      "It loads the pointer.\nThe answer is 0x20, 0x10.",
                      "It loads the pointer.\nSo the stride is 8 and the offset is 0x10.",       # more than a value
                      "It loads the pointer.\nSo the size is 0x20 bytes, which is more than a page.",
                      "It computes 16 / 2 = 8.\nThe pointer is at +0x10.",                       # a calculation is not a conclusion
                      "It loads the pointer.\nThe size is 0x20.",                                # no conclusion marker: may name a part
                      "It loads the pointer.\nThe answer is x = 0x20.", "It loads the pointer.\nThe answer is 3.5."):
            self.assertIsNone(self.drop(steps), steps)

    def test_names_and_strings_are_not_looked_at(self):
        enc = real_task("encoding", "base64_encode", chance=0.01)
        self.assertIsNone(self.drop("It maps six bits to a letter.\nSo the answer is base32 decode.", right(enc), enc), "a name group: the Answer line is the grammar's job")
        pw = real_task("strobf", "stack_string", chance=0.0)
        self.assertIsNone(self.drop("It writes bytes one by one.\nSo the answer is 42.", right(pw), pw))

    def test_it_is_counted_per_family_and_does_not_touch_the_task(self):
        r = reply(self.t, "It loads the pointer.\nSo the offset is 0x20.")
        kept, stats = sft.assemble([self.t], {self.t["id"]: {"outs": [[r, True], [reply(self.t, "A table gives the value.\nNothing else."), True]]}}, "m", MSGS, True)
        self.assertEqual(stats["reasoning"]["by_kind"]["structs"]["dropped"]["contradiction"], 1)
        self.assertEqual(sum(x["target_source"] == "teacher-reasoning" for x in kept), 1)

    def test_a_contradiction_in_a_thinking_trace_counts_the_whole_trace(self):
        r = f"<think>\nIt loads the pointer. So the offset is 0x20.\n</think>\n\n{STEPS}\nAnswer: {self.k}"
        self.assertEqual(rsn.quality_drop(r, self.t, True), "contradiction")


class Accounting(unittest.TestCase):
    def test_every_reply_is_a_trace_or_in_exactly_one_drop_reason_per_family(self):
        a, b = real_task("structs", "linked_next", chance=0.01), real_task("pktsum", "sum8", chance=0.01)
        high = real_task("crc", "bitwise_init", chance=0.55)
        cjk = "The function loads it. 这个函数读取指针。"
        outs_a = [*distinct(a, 4), reply(a, answer=wrong_for(a)), reply(a, "\n".join(["Wait, let me recheck the code again."] * 4)), "[error x]", f"{STEPS}\n{rsn.TRUNCATED}",
                  reply(a, cjk), reply(a, "It loads the pointer.\nSo the offset is 0x20."), f"Answer: {right(a)}", f"{STEPS}\nAnswer: {right(a)}\nbye"]
        outs_b = [reply(b, "A loop adds every byte.\nThe sum is stored after the last one.")] * 2
        outs_h = [reply(high, "The register is folded once per byte.")]                     # one right answer for a chance 0.55 task: lost to the agreement rule
        tasks = [a, b, high]
        res = {a["id"]: {"outs": [[o, False] for o in outs_a]}, b["id"]: {"outs": [[o, False] for o in outs_b]}, high["id"]: {"outs": [[o, False] for o in outs_h]}}
        _, stats = sft.assemble(tasks, res, "m", MSGS, True)
        blocks = {"ALL": stats["reasoning"], **stats["reasoning"]["by_kind"]}
        self.assertEqual(sorted(blocks), ["ALL", "crc", "pktsum", "structs"])
        for name, r in blocks.items():
            drops = sum(r["dropped"].values())
            self.assertEqual(r["replies"], r["accepted"] + r["wrong"] + r["no_answer_line"] + r["truncated"] + r["error"], name)
            self.assertEqual(r["accepted"], r["teacher_records"] + drops + r["in_lost_tasks"], name)
        s = stats["reasoning"]["by_kind"]["structs"]
        self.assertEqual({k: v for k, v in s["dropped"].items() if v}, {"repetition": 1, "mixed_language": 1, "contradiction": 1, "unreadable": 1, "over_cap": 1})
        self.assertEqual((s["teacher_records"], s["wrong"], s["error"], s["truncated"], s["no_answer_line"]), (3, 1, 1, 1, 1))
        self.assertEqual(stats["reasoning"]["by_kind"]["crc"]["in_lost_tasks"], 1)
        self.assertEqual(sum(stats["reasoning"]["dropped"].values()), sum(r["dropped"] and sum(r["dropped"].values()) for r in stats["reasoning"]["by_kind"].values()))
        self.assertEqual(sorted(stats["reasoning"]["dropped"]), sorted(rsn.TRACE_DROPS))

    def test_the_report_lists_the_reasons(self):
        t = real_task("structs", "linked_next", chance=0.01)
        _, stats = sft.assemble([t], {t["id"]: {"outs": [[reply(t, "\n".join(["Wait, let me recheck the code again."] * 4)), False]] + [[o, False] for o in distinct(t, 4)]}}, "m", MSGS, True)
        text = "\n".join(sft.reasoning_report(stats))
        for part in ("accepted replies not stored:", "repetition 1", "over_cap 1", "mixed_language 0", "up to 3 per kept task", "reasoning replies: 5"):
            self.assertIn(part, text)


# ---- thinking -----------------------------------------------------------------------------------------------------------------------------

class ThinkingReplies(unittest.TestCase):
    def setUp(self):
        self.t = real_task("structs", "linked_next", chance=0.01)
        self.k = right(self.t)
        self.visible = f"{STEPS}\nAnswer: {self.k}"
        self.think = "Let me read the node. The next pointer follows the first field."
        self.stored = f"<think>\n{self.think}\n</think>\n\n{self.visible}"

    def test_reasoning_content_becomes_the_think_block_in_front_of_the_visible_text(self):
        self.assertEqual(rsn.thinking_reply(self.visible, self.think), self.stored)
        self.assertEqual(rsn.thinking_reply(f"\n{self.visible}\n", f"  {self.think}\n"), self.stored, "whitespace at the ends is trimmed")
        self.assertEqual(rsn.reply_text(self.visible, "stop", self.think, True), self.stored)

    def test_an_inline_think_block_is_kept_the_same_way(self):
        self.assertEqual(rsn.thinking_reply(f"<think>{self.think}</think>\n\n{self.visible}"), self.stored)
        self.assertEqual(rsn.thinking_reply(f"<think>\n{self.think}\n</think>{self.visible}", ""), self.stored)
        two = rsn.thinking_reply(f"<think>a</think>\n<think>b</think>\n{self.visible}")
        self.assertEqual(two, f"<think>\na\n\nb\n</think>\n\n{self.visible}")

    def test_a_closing_tag_with_no_opening_one_ends_a_reasoning_whose_opening_tag_was_in_the_prompt(self):
        self.assertEqual(rsn.thinking_reply(f"{self.think}\n</think>\n\n{self.visible}"), self.stored)
        self.assertEqual(rsn.thinking_reply(f"{self.visible}</think>x", self.think), f"<think>\n{self.think}\n</think>\n\n{self.visible}</think>x", "with reasoning_content a stray tag stays and the reply is rejected")
        self.assertIsNone(rsn.final_answer(rsn.thinking_reply(f"{self.visible}</think>x", self.think)))

    def test_without_any_reasoning_the_reply_is_the_visible_text_and_a_thinking_run_drops_it(self):
        self.assertEqual(rsn.thinking_reply(self.visible, ""), self.visible)
        self.assertEqual(rsn.thinking_reply(self.visible, None), self.visible)
        self.assertEqual(rsn.thinking_reply("<think>  </think>\n" + self.visible, ""), self.visible, "an empty block is no reasoning")
        self.assertEqual(rsn.quality_drop(self.visible, self.t, True), "no_think")
        self.assertIsNone(rsn.quality_drop(self.visible, self.t, False), "a visible-reasoning run does not need it")

    def test_only_the_visible_texts_last_answer_line_is_scored(self):
        bad = wrong_for(self.t)
        mk = lambda think, visible: rsn.thinking_reply(visible, think)
        self.assertEqual(rsn.verdict(mk(f"So.\nAnswer: {bad}\nHmm wait.", self.visible), self.t), "accepted", "a wrong Answer line in the reasoning counts for nothing")
        self.assertEqual(rsn.verdict(mk(f"I have it.\nAnswer: {self.k}", f"{STEPS}\nAnswer: {bad}"), self.t), "wrong", "a right one in the reasoning does not save a wrong visible one")
        self.assertEqual(rsn.verdict(mk(f"Answer: {self.k}", STEPS), self.t), "no_answer_line", "the visible text has no Answer line")
        self.assertEqual(rsn.verdict(mk(f"Answer: {self.k}", f"{STEPS}\nAnswer: {self.k}\nBye."), self.t), "no_answer_line")
        self.assertEqual(rsn.verdict(mk(self.think, ""), self.t), "no_answer_line")
        self.assertEqual(rsn.verdict(mk(f"I have it.\nAnswer: {self.k}", ""), self.t), "no_answer_line", "the answer line is the last line of the REASONING and there is no visible text: not scored")
        self.assertTrue(rsn.score_reasoning(self.stored, self.t))

    def test_the_record_text_is_the_trace_in_a_thinking_run_and_the_visible_text_otherwise(self):
        self.assertEqual(rsn.teacher_target(self.stored, self.t, True)["text"], self.stored)
        self.assertEqual(rsn.teacher_target(self.stored, self.t)["text"], self.visible, "round 11: think blocks stripped")
        self.assertEqual(rsn.teacher_target(self.stored, self.t, True)["target_source"], "teacher-reasoning")
        self.assertEqual(rsn.teacher_target(self.stored, self.t, True)["headline"], self.k)
        self.assertEqual(rsn.trace_text("<think>\n x \n</think>\n\n  v\nAnswer: 1  ", True), "<think>\nx\n</think>\n\nv\nAnswer: 1")

    def test_a_cut_off_thinking_reply_is_rejected(self):
        for content, rc in ("", self.think), (STEPS, self.think), (self.visible, self.think):
            text = rsn.reply_text(content, "length", rc, True)
            self.assertTrue(text.endswith(rsn.TRUNCATED))
            self.assertEqual(rsn.verdict(text, self.t), "truncated")

    def test_a_thinking_trace_goes_through_the_filters_and_check_records(self):
        kept, stats = sft.assemble([self.t], {self.t["id"]: {"outs": [[self.stored, True], [rsn.thinking_reply(self.visible, ""), True]]}}, "m", MSGS, True, True)
        self.assertEqual([r["target_source"] for r in kept], ["truth-template", "teacher-reasoning"])
        self.assertTrue(kept[1]["messages"][-1]["content"].startswith("<think>\n"))
        self.assertEqual(stats["reasoning"]["dropped"]["no_think"], 1)
        self.assertEqual(stats["reasoning"]["mode"], "thinking")
        self.assertEqual(sft.RECORD_KEYS, ("messages", "task", "kind", "model", "target_source"), "no new field")
        self.assertEqual(sft.TARGET_SOURCES, ("truth-template", "teacher-reasoning"))
        self.assertEqual(sft.assemble([self.t], {self.t["id"]: {"outs": [[reply(self.t), True]]}}, "m", MSGS, True)[1]["reasoning"]["mode"], "reasoning")


def think_body(visible, think, finish="stop", tokens=10):
    msg = {"content": visible}
    if think is not None:
        msg["reasoning_content"] = think
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": {"completion_tokens": tokens}}


class ThinkingStage(StubbedSampleStage):
    KW = dict(reasoning=True, thinking=True, max_tokens=2000, max_len=4096)

    def setUp(self):
        super().setUp()
        self.reply = lambda tid, i, body: think_body(reply(self.tasks[tid], f"Step for {tid}.\nThe constant is read once."), f"Reasoning about {tid}: the code reads it, then returns it.")
        self.ttag = outputs.sample_tag(self.tag, True, True)

    def partial(self, tag):
        return self.root / "data" / f"{outputs.sft_stem('m', tag)}.partial.jsonl"

    def test_the_request_turns_thinking_on_and_carries_the_token_budget_and_the_thinking_sampling(self):
        self.run_stage(False, **self.KW)
        self.assertEqual(len(self.bodies), len(self.tasks) * self.k)
        for b in self.bodies:
            self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual((b["max_tokens"], b["temperature"], b["cache_prompt"]), (2000, 0.6, False), "round 13: Qwen3's thinking temperature (tests/test_rea_thinking.py has the rest)")
            self.assertEqual(b["messages"][0], {"role": "system", "content": rsn.SYSTEM_REASONING}, "the same prompt as the visible-reasoning run")
        self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual({json.dumps(b["chat_template_kwargs"]) for b in self.bodies}, {'{"enable_thinking": false}'}, "without --thinking it is off, as in round 11")

    def test_thinking_needs_reasoning_and_a_sane_max_tokens_and_nothing_is_asked_before_that(self):
        for kw, why in ((dict(reasoning=True, thinking=True, max_tokens=-1, max_len=4096), "--max-tokens"), (dict(thinking=True, max_tokens=500), "--reasoning")):          # round 13: no max_tokens = 4096
            with self.assertRaisesRegex(SystemExit, why):
                self.run_stage(False, **kw)
            self.assertEqual(self.requests, [], "nothing was asked")
        self.assertFalse(self.partial(self.ttag).exists())

    def test_the_record_is_the_trace_with_its_own_source_tag_and_stats_that_say_thinking(self):
        res = self.run_stage(False, **self.KW)
        teacher = [r for r in res["sft"] if r["target_source"] == "teacher-reasoning"]
        self.assertEqual(len(teacher), len(self.tasks), "the k samples of a task are the same here: one trace")
        for r in teacher:
            t = self.tasks[r["task"]]
            self.assertEqual(r["messages"][-1]["content"], f"<think>\nReasoning about {t['id']}: the code reads it, then returns it.\n</think>\n\n" + reply(t, f"Step for {t['id']}.\nThe constant is read once."))
            self.assertEqual(sorted(r), sorted(sft.RECORD_KEYS))
        self.assertEqual(len([r for r in res["sft"] if r["target_source"] == "truth-template"]), len(self.tasks))
        self.assertEqual(res["stats"]["reasoning"]["mode"], "thinking")
        self.assertIn("thinking replies:", "\n".join(sft.reasoning_report(res["stats"])))
        self.assertEqual(self.run_stage(False, reasoning=True, max_len=4096)["stats"]["reasoning"]["mode"], "reasoning")

    def test_a_thinking_run_has_its_own_checkpoint_and_never_mixes_with_the_other_runs(self):
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], f"Step for {tid}.\nThe constant is read once."))
        self.run_stage(False, reasoning=True, max_len=4096)
        visible = self.partial(outputs.sample_tag(self.tag, True))
        before = visible.read_bytes()
        self.setUp()
        self.run_stage(False, **self.KW)
        self.assertEqual(outputs.sample_tag(self.tag, True, True), f"{self.tag}-reasoning-think")
        self.assertNotEqual(outputs.sample_tag(self.tag, True, True), outputs.sample_tag(self.tag, True))
        self.assertTrue(self.partial(self.ttag).exists())
        self.assertTrue(all("<think>" in o for r in map(json.loads, self.partial(self.ttag).read_text().splitlines()) for o, _ in r["outs"]))
        self.assertFalse(any("<think>" in o for r in map(json.loads, visible.read_text().splitlines()) for o, _ in r["outs"]))
        self.assertEqual(visible.read_bytes(), before, "the visible-reasoning checkpoint is untouched")
        with self.assertRaises(ValueError):
            outputs.sample_tag(self.tag, False, True)

    def test_inline_think_blocks_and_a_reply_without_reasoning(self):
        self.reply = lambda tid, i, body: answer_body(f"<think>Inline reasoning for {tid}.</think>\n\n" + reply(self.tasks[tid], "Step.\nRead once."))
        res = self.run_stage(False, **self.KW)
        self.assertTrue(all(r["messages"][-1]["content"].startswith("<think>\nInline reasoning for") for r in res["sft"] if r["target_source"] == "teacher-reasoning"))
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], "Step.\nRead once."))             # the server returned no reasoning
        res = self.run_stage(False, **self.KW)
        n = len(self.tasks)
        self.assertEqual([r for r in res["sft"] if r["target_source"] == "teacher-reasoning"], [])
        self.assertEqual((res["stats"]["reasoning"]["dropped"]["no_think"], res["stats"]["reasoning"]["tasks_without_trace"]), (n * self.k, n))
        self.assertTrue(any(l.startswith("WARNING") and "without reasoning" in l for l in sft.reasoning_report(res["stats"])))

    def test_a_reply_cut_off_in_the_reasoning_is_rejected_and_counted(self):
        self.reply = lambda tid, i, body: think_body("", "Still thinking about it and it goes on", finish="length")
        res = self.run_stage(False, **self.KW)
        self.assertEqual((res["stats"]["reasoning"]["truncated"], res["stats"]["kept_tasks"]), (len(self.tasks) * self.k, 0))
        self.assertTrue(all(o.endswith(rsn.TRUNCATED) for r in map(json.loads, self.partial(self.ttag).read_text().splitlines()) for o, _ in r["outs"]))

    def test_the_flag_in_the_checkpoint_is_the_verdict_on_the_visible_text(self):
        self.reply = lambda tid, i, body: think_body(reply(self.tasks[tid], "x"), "Answer: 0x1\nthen more") if i == 0 else think_body(reply(self.tasks[tid]), "ok")
        self.run_stage(False, **self.KW)
        for r in map(json.loads, self.partial(self.ttag).read_text().splitlines()):
            self.assertEqual([ok for _o, ok in r["outs"]], [True] * self.k, r["id"])


class ThinkingEntry(ModalCase):
    def sample(self, root, **kw):
        return self.run_main(root, stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", **kw)

    def test_thinking_without_reasoning_stops_before_a_gpu_is_rented(self):
        for kw, why in ((dict(thinking=True, max_len=2048, max_tokens=500), "--reasoning"), (dict(thinking=True, reasoning=True, max_len=2048, max_tokens=-1), "--max-tokens")):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(SystemExit, why):
                    self.sample(pathlib.Path(d), remote_side_effect=AssertionError("a GPU was rented"), **kw)
                self.assertEqual(list((pathlib.Path(d) / "results" / "phase6" / "rea-data").glob("sft-*")), [])

    def test_the_tag_the_output_the_cost_line_and_the_remote_call_say_thinking(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d), reasoning=True, thinking=True, max_tokens=4096, max_len=4096, k=4, quota="crc=10", max_traces=2,
                                    remote_side_effect=lambda *a, **k: seen.append(sys.stdout.getvalue()) or {"sft": [], "stats": {"tasks": 0}})
            name = self.data.MODELS[0][0]
            rd = pathlib.Path(d) / "results" / "phase6" / "rea-data"
            self.assertTrue((rd / f"sft-{name}-seed7-crc-reasoning-think.jsonl").exists())
            self.assertTrue((rd / f"sft-{name}-seed7-crc-reasoning-think.stats.json").exists())
            self.assertFalse((rd / f"sft-{name}-seed7-crc-reasoning.jsonl").exists())
        self.assertIn("projected cost (thinking pilot): 10 tasks x k=4 = 40 replies", seen[0])
        self.assertIn("No estimate", seen[0])
        self.assertIn("checkpoint and output tag seed7-crc-reasoning-think", seen[0])
        self.assertNotIn("replies 5x to 8x as long", seen[0])
        kw = remote.call_args.kwargs
        self.assertEqual((kw["reasoning"], kw["thinking"], kw["max_tokens"], kw["max_traces"]), (True, True, 4096, 2))

    def test_a_pilot_over_budget_is_warned_about_by_its_worst_case(self):
        with tempfile.TemporaryDirectory() as d:
            self.sample(pathlib.Path(d), reasoning=True, thinking=True, max_tokens=4096, max_len=4096, k=4, quota="crc=400", budget_usd=10.0)
        self.assertIn("WARNING: the worst case US$", self.printed)

    def test_a_visible_reasoning_run_asks_for_no_thinking(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d), reasoning=True, max_len=2048)
        kw = remote.call_args.kwargs
        self.assertEqual((kw["thinking"], kw["max_traces"]), (False, 3))
        self.assertIn("projected cost: ", self.printed)

    def test_the_decompile_stage_says_it_ignores_the_new_flags(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_main(pathlib.Path(d), stage="decompile", n=4, seed=7, family="crc", thinking=True)
            self.assertIn("belong to --stage sample", self.printed)
            self.run_main(pathlib.Path(d), stage="decompile", n=4, seed=8, family="crc", max_traces=5)
            self.assertIn("belong to --stage sample", self.printed)


# ---- --max-traces in the stage ------------------------------------------------------------------------------------------------------------

class MaxTraces(StubbedSampleStage):
    def setUp(self):
        super().setUp()
        self.reply = lambda tid, i, body: answer_body(distinct(self.tasks[tid], 3)[i])

    def teacher(self, res):
        return [r for r in res["sft"] if r["target_source"] == "teacher-reasoning"]

    def test_the_stage_keeps_up_to_max_traces_distinct_replies_per_task(self):
        n = len(self.tasks)
        res = self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual((len(self.teacher(res)), len(res["sft"])), (3 * n, 4 * n), "k = 3 distinct replies, default cap 3")
        res = self.run_stage(False, reasoning=True, max_len=4096, max_traces=2)
        self.assertEqual(len(self.teacher(res)), 2 * n)
        self.assertEqual(res["stats"]["reasoning"]["dropped"]["over_cap"], n)
        res = self.run_stage(False, reasoning=True, max_len=4096, max_traces=1)
        self.assertEqual((len(self.teacher(res)), len(res["sft"])), (n, 2 * n))
        self.assertEqual([r["messages"][-1]["content"] for r in self.teacher(res)][:1], [distinct(self.tasks[self.teacher(res)[0]["task"]], 3)[0]], "the first sample")
        with self.assertRaisesRegex(SystemExit, "--max-traces"):
            self.run_stage(False, reasoning=True, max_len=4096, max_traces=0)

    def test_the_default_run_has_no_traces_whatever_max_traces_says(self):
        self.reply = lambda tid, i, body: {"choices": [{"message": {"content": right(self.tasks[tid])}}]}
        res = self.run_stage(False, max_traces=5)
        self.assertEqual({r["target_source"] for r in res["sft"]}, {"truth-template"})
        self.assertEqual(len(res["sft"]), len(self.tasks))

    def test_the_defaults_are_three_traces_and_thinking_off(self):
        self.assertEqual(inspect.signature(self.data.main).parameters["max_traces"].default, 3)
        self.assertEqual(inspect.signature(self.data.sample_answers).parameters["max_traces"].default, 3)
        self.assertEqual(inspect.signature(self.data.main).parameters["thinking"].default, False)
        self.assertEqual(inspect.signature(self.data.sample_answers).parameters["thinking"].default, False)


class MaxTracesEntry(ModalCase):
    def test_the_flag_reaches_the_remote_call_and_zero_is_refused_before_a_gpu_is_rented(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", reasoning=True, max_len=2048, max_traces=5)
            self.assertEqual(remote.call_args.kwargs["max_traces"], 5)
            self.assertIn("up to 5 traces per kept task", self.printed)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, "--max-traces"):
                self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", reasoning=True, max_len=2048, max_traces=0,
                              remote_side_effect=AssertionError("a GPU was rented"))


# ---- modal_lora: its own output path -------------------------------------------------------------------------------------------------------

class TrainingOutputs(ModalCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.modules.pop("modal_lora", None)
        import modal_lora
        cls.lora = modal_lora

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("modal_lora", None)
        super().tearDownClass()

    def test_the_default_run_writes_where_it_always_did(self):
        self.assertEqual(self.lora.DEFAULT_CONFIG, "rea_sft")
        self.assertEqual((self.lora.run_tag(), self.lora.run_tag(False), self.lora.run_tag(True)), ("full", "full", "smoke"))
        self.assertEqual((self.lora.run_tag(False, "rea_sft"), self.lora.run_tag(True, "rea_sft", "")), ("full", "smoke"))
        for fn in (self.lora.train, self.lora.main):
            self.assertEqual(inspect.signature(fn).parameters["config"].default, "rea_sft", fn.__name__)
        src = inspect.getsource(self.lora.train)
        self.assertIn("tag = out_tag or run_tag(smoke, config)", src, "the name comes from the tag main computed, or from run_tag: never a literal")
        self.assertNotIn('else "full"', src)
        self.assertIn('f"/out/{tag}/adapter"', src)
        self.assertIn('f"/out/{tag}/merged"', src)
        self.assertIn("load_dataset(DATASET, config,", src)
        self.assertEqual(self.lora.DATASET, "HiddenTerps/local-forge-agent-sft")

    def go(self, root, remote, gmap=None, **kw):
        """`modal_lora.main` with the remote functions stubbed and `root` as the repository; its prints are swallowed."""
        gmap = gmap or mock.MagicMock(return_value=[])
        with mock.patch.object(self.lora, "ROOT", pathlib.Path(root)), mock.patch.object(self.lora, "train", types.SimpleNamespace(remote=remote)), \
                mock.patch.object(self.lora, "to_gguf", types.SimpleNamespace(map=gmap)), contextlib.redirect_stdout(io.StringIO()):
            self.lora.main(**kw)
        return sorted(p.name for p in (pathlib.Path(root) / "results" / "phase6" / "lora").glob("*"))

    def test_the_default_main_trains_the_default_config_into_full_and_writes_the_old_file_names(self):
        remote, gmap = mock.MagicMock(return_value={"tag": "full"}), mock.MagicMock(return_value=[{"which": "lora"}, {"which": "base"}])
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.go(d, remote, gmap), ["full-gguf.json", "full-train.json"])
        remote.assert_called_once_with(False, 2.0, 1.5e-4, 32, 2048, "rea_sft", "full", False)
        gmap.assert_called_once_with(["full", "full"], ["lora", "base"])

    def test_a_run_on_a_reasoning_file_has_its_own_output_and_never_the_phase_a_one(self):
        remote, gmap = mock.MagicMock(return_value={"tag": "x"}), mock.MagicMock(return_value=[])
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.go(d, remote, gmap, config="rea_reasoning", max_len=3072), ["rea_reasoning-gguf.json", "rea_reasoning-train.json"])
        remote.assert_called_once_with(False, 2.0, 1.5e-4, 32, 3072, "rea_reasoning", "rea_reasoning", False)
        gmap.assert_called_once_with(["rea_reasoning", "rea_reasoning"], ["lora", "base"])
        remote.reset_mock()
        with tempfile.TemporaryDirectory() as d:
            self.go(d, remote, smoke=True, config="rea_reasoning", tag="r1")
        remote.assert_called_once_with(True, 2.0, 1.5e-4, 32, 2048, "rea_reasoning", "r1-smoke", False)

    def test_the_tag_rule(self):
        rt = self.lora.run_tag
        self.assertEqual(rt(False, "rea_reasoning"), "rea_reasoning")
        self.assertEqual(rt(True, "rea_reasoning"), "rea_reasoning-smoke")
        self.assertEqual(rt(False, "rea_sft", "mine"), "mine", "a tag on the default config is its own directory too")
        self.assertEqual(rt(False, "rea_reasoning", "v1.2-a"), "v1.2-a")
        self.assertEqual(rt(False, "rea reasoning/3"), "rea-reasoning-3", "a config name is made a path-safe name")
        self.assertNotIn(rt(False, "rea_reasoning"), ("full", "smoke"))
        self.assertNotEqual(rt(True, "rea_reasoning"), rt(False, "rea_reasoning"))
        for config, tag in (("rea_reasoning", "full"), ("rea_reasoning", "smoke"), ("rea_sft", "full"), ("full", ""), ("smoke", ""), ("rea_reasoning", "../full"), ("rea_reasoning", "a/b"),
                            ("rea_reasoning", ".hidden"), ("///", ""), ("", "x y")):
            with self.assertRaises(SystemExit, msg=(config, tag)):
                rt(False, config, tag)
        with self.assertRaises(SystemExit):
            rt(True, "rea_reasoning", "smoke")

    def test_main_refuses_a_reserved_tag_before_anything_is_rented(self):
        remote = mock.MagicMock()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, "reserved"):
                self.go(d, remote, config="rea_reasoning", tag="full")
            self.assertFalse((pathlib.Path(d) / "results").exists(), "nothing is created either")
        remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()

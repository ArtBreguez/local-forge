"""The content keys of a task, for the check that an evaluation shares nothing with a training set: `Domain.task_keys` and `Domain.normalize_prompt`, what REA and the arithmetic fixture say about the content of
a task, and the owner's pair (structs-7-0292 and structs-1001-0028: one function, one question, one answer, under two seeds)."""
from __future__ import annotations

import unittest

from distill.domain import Domain

from .test_kaggle_lora import ARITH, HELDOUT, PAIR, SHARED, TRAIN


class ContentKeys(unittest.TestCase):
    """`Domain.task_keys` and `Domain.normalize_prompt`: what REA and the arithmetic fixture say about the content of a task."""

    @classmethod
    def setUpClass(cls):
        from distill.domains.rea import DOMAIN
        cls.rea, (cls.a, cls.b) = DOMAIN, PAIR

    def test_the_owners_pair_is_one_task_under_both_rules_and_two_ids(self):
        self.assertEqual((self.a["id"], self.b["id"]), ("structs-7-0292", "structs-1001-0028"))
        self.assertEqual(self.rea.task_keys(self.a), self.rea.task_keys(self.b))
        self.assertEqual(sorted(self.rea.task_keys(self.a)), ["answer", "code"])
        self.assertEqual(self.rea.messages(self.a, "thinking"), self.rea.messages(self.b, "thinking"), "the held-out arm would be asked a training prompt")

    def test_two_decompilations_of_one_function_differ_in_their_addresses_and_spacing_only(self):
        again = {**self.b, "pseudocode": self.a["pseudocode"].replace("FUN_00101140", "FUN_0010a2c0").replace("{\n", "{   \n")}
        self.assertNotEqual(again["pseudocode"], self.a["pseudocode"])
        self.assertEqual(self.rea.task_keys(again)["code"], self.rea.task_keys(self.a)["code"])

    def test_the_normalisation_collapses_ghidras_address_names_and_whitespace_and_nothing_else(self):
        n = self.rea.normalize_prompt
        self.assertEqual(n("FUN_00101140(DAT_00104010, &LAB_0010116a);\n\tSUB_0010ab;"), "FUN_ADDR(DAT_ADDR, &LAB_ADDR); SUB_ADDR;")
        self.assertEqual(n("PTR_DAT_00104010 PTR_FUN_00104010"), "PTR_DAT_ADDR PTR_FUN_ADDR", "the name inside the name is the one that collapses")
        self.assertEqual(n("PTR_00104010 *PTR_0010a2c0"), "PTR_ADDR *PTR_ADDR", "a pointer name of its own")
        self.assertEqual(n("FUN_0010A2C0 DAT_0010Ab00"), "FUN_ADDR DAT_ADDR", "hex of either case")
        for kept in ("local_10", "FUN_zzz", "xFUN_00101140", "param_1", "FUN_", "DAT_00104010_x"):
            self.assertEqual(n(kept), kept, kept)
        self.assertNotEqual(n("FUN_00101140"), n("DAT_00101140"), "the kind of name stays")
        self.assertEqual(n("a  b\n\n\tc "), "a b c")

    def test_another_question_or_other_code_is_another_problem(self):
        base = self.rea.task_keys(self.a)
        self.assertNotEqual(self.rea.task_keys({**self.a, "question": self.a["question"] + " Think of the fields."})["code"], base["code"])
        self.assertNotEqual(self.rea.task_keys({**self.a, "pseudocode": "int FUN_00101140(void)\n{ return 8; }"})["code"], base["code"])

    def test_the_two_rules_read_different_things(self):
        base = self.rea.task_keys(self.a)
        other_code = self.rea.task_keys({**self.b, "pseudocode": "int FUN_1(void) { return 8; }"})
        self.assertEqual((other_code["answer"], other_code["code"] != base["code"]), (base["answer"], True), "same question and answer, other code: the answer rule only")
        other_answer = self.rea.task_keys({**self.b, "accept": [["0x10"]]})
        self.assertEqual((other_answer["code"], other_answer["answer"] != base["answer"]), (base["code"], True), "same code and question, other answer: the code rule only")
        self.assertNotEqual(self.rea.task_keys({**self.b, "kind": "crc"})["answer"], base["answer"], "the kind is part of the answer key")

    def test_an_id_never_enters_a_key(self):
        self.assertEqual(self.rea.task_keys({**self.a, "id": "anything-at-all-1"}), self.rea.task_keys(self.a))

    def test_a_task_that_cannot_give_a_rule_leaves_it_out(self):
        no_code = {k: v for k, v in self.a.items() if k != "pseudocode"}
        self.assertEqual(sorted(self.rea.task_keys(no_code)), ["answer"])
        self.assertEqual(sorted(self.rea.task_keys({k: v for k, v in self.a.items() if k != "accept"})), ["code"])
        self.assertEqual(self.rea.task_keys({k: v for k, v in self.a.items() if k != "question"}), {})

    def test_the_default_says_nothing_and_collapses_whitespace_only(self):
        self.assertEqual(Domain.task_keys(ARITH, HELDOUT[0]), {})
        self.assertEqual(Domain.normalize_prompt(ARITH, "a  b\n c"), "a b c")
        self.assertEqual(Domain.normalize_prompt(ARITH, "FUN_00101140 FUN_0010a2c0"), "FUN_00101140 FUN_0010a2c0", "the Ghidra rule is REA's")

    def test_the_fixture_domain_keys_the_problem_and_the_problem_with_its_answer(self):
        k = ARITH.task_keys(HELDOUT[3])
        self.assertEqual(sorted(k), ["answer", "code"])
        self.assertEqual(k, ARITH.task_keys({**HELDOUT[3], "id": "x"}))
        self.assertNotEqual(k["answer"], ARITH.task_keys({**HELDOUT[3], "answer": 99})["answer"])

    def test_the_fixtures_own_overlap_is_what_the_other_tests_assume(self):
        questions = {t["question"]: t["id"] for t in TRAIN}
        self.assertEqual({t["id"] for t in HELDOUT if t["question"] in questions}, SHARED)


if __name__ == "__main__":
    unittest.main()

"""The content separation of the Kaggle LoRA kit's held-out tasks (PR 34 review, B1) and the records it may train on (M2): `Domain.task_keys`, the exclusion, the floors, the overlap report, and the owner's pair
(structs-7-0292 and structs-1001-0028) through the REA builder.
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from distill import teachers
from distill.domain import Domain
from distill.domains.example_arith import Arith

from .test_kaggle_lora import ARITH, HELDOUT, PAIR, SHARED, TEACHER, THINK, TRAIN, TRAIN_BY_ID, Kit, build_lora, make_tasks, row, sft_rows, write_jsonl


class CodeArith(Arith):
    """Arithmetic read the way REA is: a prompt is a piece of `code` and a question, and the two keys can disagree: "code" = (code, question), "answer" = (kind, question, answer)."""

    def messages(self, task, mode):
        msgs = super().messages(task, mode)
        return [msgs[0], {"role": "user", "content": f"```c\n{task['code']}\n```\n\nQuestion: {task['question']}"}] if "code" in task else msgs

    def task_keys(self, task):
        q, h = self.normalize_prompt(task["question"]), lambda *parts: json.dumps(parts)
        return {"code": h("code", self.normalize_prompt(task.get("code", "")), q), "answer": h("answer", task["kind"], q, task["answer"])}


CODE = CodeArith()


class Plain(Arith):
    """A subject that did not implement `task_keys`."""

    def task_keys(self, task):
        return Domain.task_keys(self, task)


def hold(n: int, **fields) -> dict:
    """The held-out task `arith-9999-<n>` of the fixture with `fields` changed."""
    return {**HELDOUT[n], **fields}


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


class Separation(Kit):
    """A held-out task that shares its content with a training task is out of the evaluation, whatever its id (PR 34 review, B1)."""

    def info(self, **kw):
        return self.build(**kw)["built_from"]["eval"]["excluded"]

    def pick(self, held, train, **kw):
        path = write_jsonl(self.dir / "pool.jsonl", held)
        return build_lora.pick_eval([path], kw.pop("quota", ""), kw.pop("per_family", 10), kw.pop("families", ["add"]), domain=CODE, train_tasks={t["id"]: t for t in train}, min_family=kw.pop("min_family", 1), **kw)

    POOL_TRAIN = [{"id": "arith-7-900", "kind": "add", "question": "Q one", "answer": 5, "code": "A"}]

    def pool(self):
        """four held-out tasks against POOL_TRAIN: 000 shares code and question only, 003 question and answer only, 006 both, 009 nothing."""
        return [hold(0, question="Q one", answer=6, code="A"), hold(3, question="Q one", answer=5, code="B"), hold(6, question="Q one", answer=5, code="A"), hold(9, question="Q two", answer=7, code="C")]

    def test_a_task_with_a_training_twin_is_excluded_and_its_id_is_not_what_decided(self):
        info = self.info()
        self.assertEqual(set(info["pairs"]), SHARED)
        twin = {t["id"] for t in HELDOUT if t["id"] in SHARED}
        self.assertTrue(all(set(v) == {"code", "answer"} and TRAIN_BY_ID[v["code"]]["question"] == next(h for h in HELDOUT if h["id"] == k)["question"] for k, v in info["pairs"].items()), twin)
        lines = [json.loads(line) for line in (self.dir / "out" / "data" / "eval_tasks.jsonl").read_text().splitlines()]
        self.assertFalse(SHARED & {line["id"] for line in lines})
        trained = {r["messages"][1]["content"] for r in sft_rows()}
        self.assertFalse(trained & {line["messages"][1]["content"] for line in lines}, "no evaluation prompt is a training prompt")

    def test_the_exclusion_comes_before_the_quota_so_the_quota_fills_from_what_remains(self):
        self.build()
        ids = [json.loads(line)["id"] for line in (self.dir / "out" / "data" / "eval_tasks.jsonl").read_text().splitlines()]
        adds = [i for i in ids if HELDOUT[int(i[-3:])]["kind"] == "add"]
        self.assertEqual(sorted(adds), ["arith-9999-000", "arith-9999-006", "arith-9999-009"], "003 is the second add by id: asked for 3, the quota takes 000, 006, 009")

    def test_the_counts_per_family_and_the_rule_are_recorded(self):
        info = self.info()
        self.assertEqual(info["rules"], ["code", "answer"])
        self.assertEqual(info["by_family"]["add"], {"heldout": 4, "code": 1, "answer": 1, "union": 1, "excluded": 1, "left": 3, "chosen": 3})
        self.assertEqual(info["by_family"]["sub"], {"heldout": 4, "code": 1, "answer": 1, "union": 1, "excluded": 1, "left": 3, "chosen": 3})
        self.assertEqual(info["by_family"]["mul"], {"heldout": 4, "code": 0, "answer": 0, "union": 0, "excluded": 0, "left": 4, "chosen": 3})
        self.assertEqual(info["total"], {"heldout": 12, "code": 2, "answer": 2, "union": 2, "excluded": 2, "chosen": 9})
        self.assertEqual((info["train_tasks"], info["train_task_files"], info["left_out"], info["unkeyed"]), (30, ["train-tasks.jsonl"], {}, []))

    def test_the_two_rules_work_apart(self):
        both, _ = (self.pick(self.pool(), self.POOL_TRAIN)[0], None)
        self.assertEqual([t["id"] for t in both], ["arith-9999-009"], "code+question OR question+answer: only the task with neither survives")
        for rules, kept in ((("code",), ["arith-9999-003", "arith-9999-009"]), (("answer",), ["arith-9999-000", "arith-9999-009"]), ((), ["arith-9999-000", "arith-9999-003", "arith-9999-006", "arith-9999-009"])):
            chosen, info = self.pick(self.pool(), self.POOL_TRAIN, rules=rules)
            self.assertEqual([t["id"] for t in chosen], kept, rules)
            self.assertEqual(sorted(info["pairs"]), sorted({"arith-9999-000", "arith-9999-003", "arith-9999-006"} - set(kept)), rules)
        _, info = self.pick(self.pool(), self.POOL_TRAIN)
        self.assertEqual(info["by_family"]["add"], {"heldout": 4, "code": 2, "answer": 2, "union": 3, "excluded": 3, "left": 1, "chosen": 1})
        self.assertEqual(info["pairs"], {"arith-9999-000": {"code": "arith-7-900"}, "arith-9999-003": {"answer": "arith-7-900"}, "arith-9999-006": {"code": "arith-7-900", "answer": "arith-7-900"}})

    def test_when_two_training_tasks_share_a_key_the_pair_names_the_first_by_id(self):
        twins = [{"id": "arith-7-900", "kind": "add", "question": "Q one", "answer": 5, "code": "A"}, {"id": "arith-7-800", "kind": "add", "question": "Q one", "answer": 5, "code": "A"}]
        _, info = self.pick([hold(6, question="Q one", answer=5, code="A"), hold(9, question="Q two", answer=7, code="C")], twins)
        self.assertEqual(info["pairs"], {"arith-9999-006": {"code": "arith-7-800", "answer": "arith-7-800"}})

    def test_it_compares_with_all_the_training_tasks_not_only_the_ones_that_have_a_record(self):
        question = HELDOUT[3]["question"]
        twin = next(t for t in TRAIN if t["question"] == question)["id"]
        records = [r for r in sft_rows() if r["task"] != twin]
        self.assertEqual(len(records), 58)
        info = self.info(sft=write_jsonl(self.dir / "fewer.jsonl", records))
        self.assertIn("arith-9999-003", info["pairs"], "the training task has no record in the SFT file, and still excludes")
        self.assertEqual(info["pairs"]["arith-9999-003"]["code"], twin)

    def test_a_family_that_is_left_under_the_floor_is_left_out_with_a_note(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            info = self.info(min_family=4)
        self.assertEqual(sorted(info["left_out"]), ["add", "sub"], "touched by the exclusion and 3 left, under 4; mul lost nothing")
        self.assertIn("3 of 4 held-out tasks remain after excluding 1", info["left_out"]["add"])
        self.assertIn("note: family add is left out of the evaluation", buf.getvalue())
        self.assertEqual(sorted(info["by_family"]), ["add", "mul", "sub"])
        self.assertEqual((info["by_family"]["add"]["chosen"], info["by_family"]["mul"]["chosen"]), (0, 3))
        self.assertEqual(info["total"]["chosen"], 3)

    def test_a_family_that_loses_every_task_is_left_out_and_the_others_stay(self):
        train = [{"id": f"arith-7-9{i:02d}", "kind": "add", "question": q, "answer": a, "code": c} for i, (q, a, c) in enumerate((("Q one", 5, "A"), ("Q two", 7, "C")))]
        held = [hold(0, question="Q one", answer=5, code="A"), hold(3, question="Q two", answer=7, code="C"), hold(1, question="Q three", answer=8, code="D")]
        held[2]["kind"] = "sub"
        chosen, info = self.pick(held, train, families=["add", "sub"])
        self.assertEqual([t["id"] for t in chosen], ["arith-9999-001"])
        self.assertEqual(list(info["left_out"]), ["add"])
        self.assertIn("0 of 2 held-out tasks remain after excluding 2", info["left_out"]["add"])

    def test_strict_families_refuses_instead(self):
        with self.assertRaisesRegex(SystemExit, "--strict-families: add: 3 of 4 held-out tasks remain after excluding 1"):
            self.build(min_family=4, strict_families=True)
        self.assertFalse((self.dir / "out").exists())
        self.build(min_family=3, strict_families=True)

    def test_a_family_that_was_small_before_the_exclusion_is_not_left_out_by_the_floor(self):
        info = self.info(quota="mul=3", min_family=5)
        self.assertEqual((info["left_out"], info["by_family"]["mul"]["chosen"]), ({}, 3), "four mul tasks, none excluded: the floor is about what the exclusion took")

    def test_when_every_family_is_left_out_nothing_is_built(self):
        with self.assertRaisesRegex(SystemExit, "no held-out task of add, sub .* after excluding 2"):
            self.build(quota="add=3,sub=3", min_family=4)
        self.assertFalse((self.dir / "out").exists())

    def test_a_subject_that_gives_no_content_key_is_refused_for_the_check_would_find_nothing(self):
        with self.assertRaisesRegex(SystemExit, "gives no content key"):
            build_lora.pick_eval([self.heldout], "", 3, ["add"], domain=Plain(), train_tasks=TRAIN_BY_ID)

    def test_the_training_tasks_are_checked_and_are_required(self):
        leaky = write_jsonl(self.dir / "leaky.jsonl", [*TRAIN, HELDOUT[0]])
        with self.assertRaisesRegex(SystemExit, "--train-tasks holds held-out tasks.*arith-9999-000"):
            self.build(train_tasks=[leaky])
        with self.assertRaisesRegex(SystemExit, "no task in --train-tasks"):
            self.build(train_tasks=[write_jsonl(self.dir / "empty.jsonl", [])])
        short = write_jsonl(self.dir / "short.jsonl", TRAIN[:10])
        with self.assertRaisesRegex(SystemExit, r"20 training record task\(s\) are not in --train-tasks"):
            self.build(train_tasks=[short])
        self.assertFalse((self.dir / "out").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_lora.main(["--domain", "example_arith", "--sft", str(self.sft), "--heldout", str(self.heldout), "--owner", "someone", "--out", str(self.dir / "cli")])

    def test_the_training_tasks_of_several_files_are_all_compared(self):
        first, second = write_jsonl(self.dir / "a.jsonl", TRAIN[:15]), write_jsonl(self.dir / "b.jsonl", TRAIN[15:])
        self.assertEqual(set(self.info(train_tasks=[first, second])["pairs"]), SHARED)

    def test_a_training_record_whose_prompt_is_an_evaluation_prompt_is_refused_without_the_training_tasks(self):
        mine = hold(1, question="Zed has 4 plums and buys 5 more. How many plums does he have?")
        twin = {**TRAIN[0], "id": "arith-7-777", "question": mine["question"]}
        rows = [*sft_rows(4), row(twin)]
        with self.assertRaisesRegex(SystemExit, "1 training record.*user message is an evaluation task's.*arith-7-777 = arith-9999-001"):
            build_lora.check_training(rows, [mine], domain=ARITH)
        spaced = [{**r, "messages": [r["messages"][0], {**r["messages"][1], "content": "  " + mine["question"].replace(" ", "  ") + "\n"}, r["messages"][2]]} for r in [row(twin)]]
        with self.assertRaisesRegex(SystemExit, "user message is an evaluation task's"):
            build_lora.check_training(sft_rows(4) + spaced, [mine], domain=ARITH)
        build_lora.check_training(sft_rows(4), [mine], domain=ARITH)

    def test_the_prompt_is_compared_as_the_subject_normalises_it(self):
        mine = {**HELDOUT[1], "code": "int FUN_00101140(void);", "question": "Q"}
        for spelling in ("int FUN_0010ffff(void);", "int  FUN_00102a30(void);"):
            twin = {**HELDOUT[1], "id": "arith-7-777", "code": spelling, "question": "Q"}
            rows = [{**row(twin), "messages": CODE.messages(twin, "thinking") + [{"role": "assistant", "content": THINK.format("x" * 40, 1)}]}]
            build_lora.check_training(rows, [mine], domain=CODE)         # CODE has no Ghidra rule: these are other prompts to it
        from distill.domains.rea import DOMAIN
        a, b = PAIR
        moved = {**b, "pseudocode": a["pseudocode"].replace("FUN_00101140", "FUN_0010a2c0")}
        rows = [{"messages": [*DOMAIN.messages(a, "thinking"), {"role": "assistant", "content": THINK.format("x" * 40, "0x8")}], "task": a["id"], "kind": "structs", "model": TEACHER, "target_source": "teacher-reasoning"}]
        with self.assertRaisesRegex(SystemExit, "user message is an evaluation task's"):
            build_lora.check_training(rows, [moved], domain=DOMAIN)


class OverlapReport(Kit):
    def test_the_report_counts_per_family_and_rule(self):
        held = [hold(0, question="Q one", answer=6, code="A"), hold(3, question="Q one", answer=5, code="B"), hold(6, question="Q one", answer=5, code="A"), hold(9, question="Q two", answer=7, code="C"), hold(1, question="Q two", answer=7, code="C")]
        held[4]["kind"] = "sub"
        rep = build_lora.overlap({"arith-7-900": {"id": "arith-7-900", "kind": "add", "question": "Q one", "answer": 5, "code": "A"}}, held, domain=CODE)
        self.assertEqual(rep["by_family"], {"add": {"heldout": 4, "code": 2, "answer": 2, "union": 3}, "sub": {"heldout": 1, "code": 0, "answer": 0, "union": 0}})
        self.assertEqual(rep["total"], {"heldout": 5, "code": 2, "answer": 2, "union": 3})
        self.assertEqual(rep["matches"]["arith-9999-006"], {"code": "arith-7-900", "answer": "arith-7-900"})
        self.assertEqual((rep["train_tasks"], rep["unkeyed"]), (1, []))

    def test_the_table_and_the_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_lora.main(["overlap", "--domain", "example_arith", "--train-tasks", str(self.train_tasks), "--heldout", str(self.heldout)])
        lines = buf.getvalue().splitlines()
        self.assertEqual(lines[0].split(), ["family", "held-out", "code+question", "question+answer", "either", "left"])
        self.assertEqual([line.split() for line in lines[1:5]], [["add", "4", "1", "1", "1", "3"], ["mul", "4", "0", "0", "0", "4"], ["sub", "4", "1", "1", "1", "3"], ["TOTAL", "12", "2", "2", "2", "10"]])
        self.assertIn("30 training tasks compared", lines[-1])
        self.assertFalse((self.dir / "out").exists(), "nothing is built")

    def test_the_command_refuses_a_file_that_is_not_the_held_out_split_and_a_training_file_that_is(self):
        with self.assertRaisesRegex(SystemExit, "not held-out ids"):
            build_lora.main(["overlap", "--domain", "example_arith", "--train-tasks", str(self.train_tasks), "--heldout", str(self.train_tasks)])
        with self.assertRaisesRegex(SystemExit, "--train-tasks holds held-out tasks"):
            build_lora.main(["overlap", "--domain", "example_arith", "--train-tasks", str(self.heldout), "--heldout", str(self.heldout)])

    def test_a_held_out_task_with_no_key_is_listed_not_compared(self):
        class Half(Arith):
            def task_keys(self, task):
                return {} if task["id"] == "arith-9999-001" else Arith.task_keys(self, task)
        rep = build_lora.overlap(TRAIN_BY_ID, HELDOUT, domain=Half())
        self.assertEqual(rep["unkeyed"], ["arith-9999-001"])
        self.assertIn("1 held-out task(s) have no key at all", build_lora.overlap_lines(rep)[-1])


class ReaSeparation(unittest.TestCase):
    """The owner's pair, on the real records: structs-7-0292 (a training task) and structs-1001-0028 (held out) are one function, one question, one answer."""

    @classmethod
    def setUpClass(cls):
        from distill.domains.rea import DOMAIN
        from .distill_fixtures import load_tasks
        cls.rea, cls.golden = DOMAIN, [t for t in load_tasks() if t["kind"] in ("xor-key", "password")]
        cls.train = [PAIR[0], *cls.golden]
        cls.other = [{**t, "id": t["id"].replace("-7-", "-1001-"), "question": t["question"] + " (held out)", "pseudocode": t["pseudocode"] + "\n/* other */"} for t in cls.golden]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        self.train_file = write_jsonl(self.dir / "decompiled-seed7-x.jsonl", self.train)
        rows = []
        for t in self.train:
            rows.append({"messages": [*self.rea.messages(t, "thinking"), {"role": "assistant", "content": THINK.format("looking at the code " * 8, "0x8")}], "task": t["id"], "kind": t["kind"], "model": TEACHER,
                         "target_source": "teacher-reasoning"})
            rows.append({"messages": [*self.rea.messages(t, "plain"), {"role": "assistant", "content": self.rea.truth_target(t)}], "task": t["id"], "kind": t["kind"], "model": TEACHER, "target_source": "truth-template"})
        self.sft = write_jsonl(self.dir / "sft.jsonl", rows)

    def build(self, held, **kw):
        heldout = write_jsonl(self.dir / "decompiled-seed1001-x.jsonl", held)
        return build_lora.build(self.sft, [heldout], kw.pop("out", self.dir / "out"), "someone", domain=self.rea, train_tasks=[self.train_file], per_family=1, min_family=kw.pop("min_family", 1), **kw)

    def eval_lines(self, out="out"):
        return [json.loads(line) for line in (self.dir / out / "data" / "eval_tasks.jsonl").read_text().splitlines()]

    def test_the_pair_goes_through_the_builder_no_more(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cfg = self.build([PAIR[1], *self.other])
        lines = self.eval_lines()
        self.assertNotIn("structs-1001-0028", [line["id"] for line in lines])
        self.assertEqual(sorted(line["kind"] for line in lines), ["password", "xor-key"])
        info = cfg["built_from"]["eval"]["excluded"]
        self.assertEqual(info["pairs"], {"structs-1001-0028": {"code": "structs-7-0292", "answer": "structs-7-0292"}})
        self.assertEqual(info["by_family"]["structs"], {"heldout": 1, "code": 1, "answer": 1, "union": 1, "excluded": 1, "left": 0, "chosen": 0})
        self.assertIn("structs", info["left_out"])
        self.assertIn("note: family structs is left out of the evaluation: 0 of 1 held-out tasks remain", buf.getvalue())
        trained = {json.loads(line)["messages"][1]["content"] for line in (self.dir / "out" / "data" / "train.jsonl").read_text().splitlines()}
        self.assertFalse(trained & {line["messages"][1]["content"] for line in lines}, "the review's repro: eval['messages'] == train['messages'][:2]")

    def test_the_same_function_at_other_addresses_is_still_the_pair_by_the_code_rule_alone(self):
        again = {**PAIR[1], "pseudocode": PAIR[0]["pseudocode"].replace("FUN_00101140", "FUN_0010a2c0").replace("{\n", "{  \n"), "accept": [["0x9"]]}
        self.assertNotEqual(again["pseudocode"], PAIR[0]["pseudocode"], "raw equality would not see it")
        cfg = self.build([again, *self.other])
        self.assertEqual(cfg["built_from"]["eval"]["excluded"]["pairs"], {"structs-1001-0028": {"code": "structs-7-0292"}})
        self.assertNotIn("structs-1001-0028", [line["id"] for line in self.eval_lines()])

    def test_strict_families_refuses_the_family_the_pair_empties(self):
        with self.assertRaisesRegex(SystemExit, "--strict-families: structs: 0 of 1 held-out tasks remain after excluding 1"):
            self.build([PAIR[1], *self.other], strict_families=True)
        self.assertFalse((self.dir / "out").exists())

    def test_a_task_that_is_not_the_pair_stays(self):
        cfg = self.build([*self.other])
        self.assertEqual(cfg["built_from"]["eval"]["excluded"]["pairs"], {})
        self.assertEqual(cfg["built_from"]["eval"]["excluded"]["total"]["excluded"], 0)

    def test_the_prompt_check_stands_without_the_training_tasks(self):
        with self.assertRaisesRegex(SystemExit, "user message is an evaluation task's.*structs-7-0292 = structs-1001-0028"):
            build_lora.check_training(build_lora.read_sft(self.sft), [PAIR[1]], domain=self.rea)

    def test_the_overlap_command_shows_the_pair(self):
        buf = io.StringIO()
        held = write_jsonl(self.dir / "decompiled-seed1001-y.jsonl", [PAIR[1], *self.other])
        with contextlib.redirect_stdout(buf):
            build_lora.main(["overlap", "--train-tasks", str(self.train_file), "--heldout", str(held)])
        rows = [line.split() for line in buf.getvalue().splitlines()]
        self.assertIn(["structs", "1", "1", "1", "1", "0"], rows)
        self.assertIn(["TOTAL", "5", "1", "1", "1", "4"], rows)


class TruthRecords(unittest.TestCase):
    """What a record may hold: a truth-template record is the truth and nothing of a model (PR 34 review, M2)."""

    def test_a_truth_template_record_with_think_text_is_refused_whatever_its_label(self):
        """PR 34 review, M2: a gpt-oss trace re-tagged `truth-template` used to be trained without a check firing."""
        for label in ("gpt-oss-120b", TEACHER):
            rows = [r for t in make_tasks(7, 4) for r in (row(t), {**row(t, "teacher-reasoning", label=label), "target_source": "truth-template"})]
            with self.assertRaisesRegex(SystemExit, "may only FILTER|stores the teacher's reply|truth-template record.* hold think text", msg=label):
                build_lora.check_training(rows, [], domain=ARITH)
        forty = [{**row(t, label="gpt-oss-120b"), "target_source": "truth-template"} for t in make_tasks(7, 30)] + [row(t) for t in make_tasks(7, 4)]
        with self.assertRaisesRegex(SystemExit, "may only FILTER|stores the teacher's reply"):
            build_lora.check_training(forty, [], domain=ARITH)

    def test_the_label_of_a_refused_teacher_is_named_before_the_shape_of_the_record(self):
        rows = [row(t, "truth-template", text=THINK.format("x" * 30, 1), label="claude-opus-5-5") for t in make_tasks(7, 2)] + [row(make_tasks(7, 3)[2])]
        with self.assertRaisesRegex(SystemExit, "claude-opus-5-5"):
            build_lora.check_training(rows, [], domain=ARITH)

    def test_a_truth_template_record_with_think_text_of_an_allowed_label_is_refused_for_what_it_is(self):
        for text in (THINK.format("work it out " * 5, 3), "<think>never closed", "a stray </think> in a truth text"):
            rows = [*sft_rows(3), row(make_tasks(7, 4)[3], "truth-template", text=text)]
            with self.assertRaisesRegex(SystemExit, r"1 truth-template record\(s\) hold think text \(arith-7-003\)", msg=text):
                build_lora.check_training(rows, [], domain=ARITH)

    def test_the_teacher_rule_covers_every_record_that_holds_think_text_and_every_teacher_reasoning_record(self):
        with mock.patch.object(teachers, "require_reasoning", wraps=teachers.require_reasoning) as rule:
            build_lora.check_training(sft_rows(3), [], domain=ARITH)
            think_only = [row(t) for t in make_tasks(7, 3)]
            build_lora.check_training(think_only, [], domain=ARITH)
        self.assertEqual([c.args for c in rule.call_args_list], [(TEACHER,), (TEACHER,)])
        plain_teacher = [row(t, label="gpt-oss-120b", text="Step 1: add. Step 2: done.") for t in make_tasks(7, 2)]
        with self.assertRaisesRegex(SystemExit, "may only FILTER|stores the teacher's reply"):
            build_lora.check_training([*sft_rows(2), *plain_teacher], [], domain=ARITH)          # a teacher-reasoning record is a teacher's text whatever its form
        self.assertEqual(build_lora.check_training(sft_rows(3), [], domain=ARITH)["labels"], [TEACHER], "a truth-template record's label never counts: it stores nothing of the model")

    def test_a_truth_template_text_is_the_truth_target_of_its_task(self):
        build_lora.check_training(sft_rows(4), [], domain=ARITH, train_tasks=TRAIN_BY_ID)
        for text in ("999", "12 apples", " "):
            rows = [*sft_rows(3), row(make_tasks(7, 4)[3], "truth-template", text=text)]
            with self.assertRaisesRegex(SystemExit, r"1 truth-template record\(s\) whose text is not `domain.truth_target` of their task \(arith-7-003\)", msg=text):
                build_lora.check_training(rows, [], domain=ARITH, train_tasks=TRAIN_BY_ID)
        build_lora.check_training([*sft_rows(3), row(make_tasks(7, 4)[3], "truth-template", text="999")], [], domain=ARITH)          # nothing to compare with: not checked

    def test_a_record_of_a_task_the_training_tasks_do_not_have_is_refused(self):
        with self.assertRaisesRegex(SystemExit, r"1 training record task\(s\) are not in --train-tasks \(arith-7-003\)"):
            build_lora.check_training(sft_rows(4), [], domain=ARITH, train_tasks={k: v for k, v in TRAIN_BY_ID.items() if k != "arith-7-003"})


if __name__ == "__main__":
    unittest.main()

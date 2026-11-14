"""What is scored is the build's evaluation set and nothing else (PR 34 second review, M1), the fit verdict against the kernel's own deadline (m2), the teacher labels that are in no catalogue entry (m3)
and the test gaps the surviving mutants named (m5): `--measured` through the builder, the scorer on a REA held-out file, the k > 1 note.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import tempfile
import unittest
import unittest.mock

from distill import outputs

from .test_kaggle_lora import ARITH, HELDOUT, PAIR, SHA, SHARED, TEACHER, THINK, TRAIN, Kit, build_lora, build_of, lora_eval, make_tasks, row, run_teacher, score_lora, sft_rows, write_jsonl

EXCLUDED = "arith-9999-003"          # the held-out add task that seed 7 draws too: every build of the fixture excludes it


def reply(task: dict, right: bool, sha: str, domain=ARITH, answer=None) -> dict:
    text = THINK.format("thinking", answer if answer is not None else task["answer"] if right else task["answer"] + 1)
    return {"text": text, "finish": "stop", "tokens": 100, "prompt_tokens": 50, "secs": 1.0, "prompt_sha": outputs.prompt_sha(domain.messages(task, "thinking")), "max_tokens": 4096, "config_sha": sha}


def line(task: dict, outs: list, domain=ARITH) -> dict:
    return {"id": task["id"], "kind": task["kind"], "prompt_sha": outputs.prompt_sha(domain.messages(task, "thinking")), "outs": outs}


class EvalSet(Kit):
    """The reviewer's repro: an attached earlier output holds the replies of a task this build excludes."""

    def setUp(self):
        super().setUp()
        self.cfg = self.build()
        self.data = self.dir / "out" / "data"
        self.sha = self.cfg["config_sha"]
        self.excluded = next(t for t in HELDOUT if t["id"] == EXCLUDED)
        self.asked = [json.loads(x) for x in (self.data / "eval_tasks.jsonl").read_text().splitlines()]
        self.kept = next(t for t in HELDOUT if t["id"] == self.asked[0]["id"])

    def earlier(self, where: pathlib.Path, sha=None) -> None:
        """An earlier kernel's output under `where`: both arms answered the excluded task (right in the adapter arm only) and one task that this build asks."""
        where.mkdir(parents=True, exist_ok=True)
        for arm in lora_eval.ARMS:
            write_jsonl(where / f"replies-{arm}.jsonl", [line(self.excluded, [reply(self.excluded, arm == "lora", sha or self.sha)]), line(self.kept, [reply(self.kept, True, sha or self.sha)])])

    def test_the_config_names_the_evaluation_set_and_it_is_the_tasks_the_kernel_is_given(self):
        ids = self.cfg["built_from"]["eval"]["ids"]
        self.assertEqual(ids, sorted(t["id"] for t in self.asked))
        self.assertEqual((len(ids), len(set(ids)), EXCLUDED in ids), (9, 9, False))
        self.assertEqual(json.loads((self.data / "lora-config.json").read_text())["built_from"]["eval"]["ids"], ids)
        self.assertEqual(lora_eval.eval_ids(self.data), set(ids))
        self.assertEqual(score_lora.read_build(self.data / "lora-config.json"), {"ids": set(ids), "config_sha": self.sha})

    def test_the_excluded_task_is_in_the_earlier_output_under_this_very_config_sha(self):
        """What made the first fix round's check blind: the reply of the excluded task carries the build's own `config_sha`."""
        self.earlier(self.dir / "in" / "earlier")
        got = run_teacher.load_previous(lora_eval.previous_files("lora", self.dir / "work", self.dir / "in"))
        self.assertEqual(got[EXCLUDED]["outs"][0]["config_sha"], self.sha)
        self.assertNotIn(EXCLUDED, lora_eval.eval_ids(self.data))

    def test_the_merge_never_brings_back_a_task_outside_the_evaluation_set(self):
        work, inp, logs = self.dir / "work", self.dir / "in", []
        work.mkdir()
        self.earlier(inp / "earlier")
        write_jsonl(lora_eval.shard_path(work, "base", 0), [line(self.excluded, [reply(self.excluded, True, self.sha)])])          # even a shard of this very session
        with unittest.mock.patch.object(lora_eval, "log", lambda *a: logs.append(" ".join(map(str, a)))):
            counts = {arm: lora_eval.merge_shards(work, arm, lora_eval.eval_ids(self.data), input_root=inp) for arm in lora_eval.ARMS}
        self.assertEqual(counts, {"base": 1, "lora": 1}, "the review counted 2 tasks per arm here")
        for arm in lora_eval.ARMS:
            merged = [json.loads(x)["id"] for x in (work / f"replies-{arm}.jsonl").read_text().splitlines()]
            self.assertEqual(merged, [self.kept["id"]], arm)
        self.assertEqual(len(logs), 2)
        self.assertIn(f"1 task(s) of earlier replies are not in this build's evaluation set and are left out: {EXCLUDED}", logs[0])

    def test_a_merge_of_tasks_that_are_all_asked_says_nothing_and_keeps_them_all(self):
        work, inp, logs = self.dir / "work", self.dir / "in", []
        work.mkdir()
        self.earlier(inp / "earlier")
        with unittest.mock.patch.object(lora_eval, "log", lambda *a: logs.append(a)):
            self.assertEqual(lora_eval.merge_shards(work, "base", {EXCLUDED, self.kept["id"], "another"}, input_root=inp), 2)
        self.assertEqual(logs, [])

    def test_the_kernel_merges_with_the_evaluation_set_of_its_own_dataset(self):
        import ast
        src = (pathlib.Path(lora_eval.__file__)).read_text()
        main = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
        self.assertIn("merge_shards(OUT, arm, eval_ids(data_dir), prefix)", main)
        self.assertEqual(src.count("merge_shards("), 2, "its definition and the one call")

    def test_the_scorer_scores_the_evaluation_set_only_and_says_what_it_left_out(self):
        """The review's second half: given replies files that DO hold the excluded task (an output merged by an older kernel), the report does not count it."""
        self.earlier(self.dir / "old")
        build = score_lora.read_build(self.data / "lora-config.json")
        r = score_lora.score(self.dir / "old" / "replies-base.jsonl", self.dir / "old" / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=build)
        self.assertEqual((r["paired"]["overall"]["n"], r["paired"]["overall"]["lora_only"], sorted(r["paired"]["by_kind"])), (1, 0, [self.kept["kind"]]), "the review got 2 paired replies, the excluded task's among them")
        self.assertEqual((r["arms"]["lora"]["overall"]["replies"], r["arms"]["lora"]["overall"]["accepted"]), (1, 1))
        self.assertEqual(r["outside"], {"replies": 2, "by_arm": {"base": 1, "lora": 1}, "tasks": [EXCLUDED]})
        self.assertEqual((r["tasks_answered"], r["tasks_in_set"], r["config_sha"]), (1, 9, self.sha))
        lines = score_lora.lines_of(r)
        self.assertIn(f"2 replies of tasks outside the evaluation set, not scored (base 1, adapter 1; tasks: {EXCLUDED})", lines)
        self.assertIn("tasks answered 1 of 9 in the evaluation set of the build", lines[-2])

    def test_what_is_counted_outside_is_replies_not_lines_and_not_empty_entries(self):
        ex = self.excluded
        got = score_lora.read_arm(write_jsonl(self.dir / "k.jsonl", [line(ex, [reply(ex, True, self.sha), None, reply(ex, False, self.sha)]), line(self.kept, [reply(self.kept, True, self.sha)])]),
                                  {self.kept["id"]: self.kept}, domain=ARITH, config_sha=self.sha)
        self.assertEqual((got["info"]["outside"], got["outside"], sorted(got["rows"])), (2, [EXCLUDED], [(self.kept["id"], 0)]), "two replies of one task, the entry that was never answered is none")

    def test_a_clean_run_says_that_nothing_was_outside(self):
        for arm in lora_eval.ARMS:
            write_jsonl(self.dir / f"{arm}.jsonl", [line(t, [reply(t, True, self.sha)]) for t in HELDOUT if t["id"] in self.cfg["built_from"]["eval"]["ids"]])
        r = score_lora.score(self.dir / "base.jsonl", self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        lines = score_lora.lines_of(r)
        self.assertEqual(lines[-1], "0 replies of tasks outside the evaluation set, not scored")
        self.assertEqual((r["paired"]["overall"]["n"], r["outside"]["replies"], r["other_build"]["replies"]), (9, 0, 0))
        self.assertFalse(any("another build" in x for x in lines))

    def test_the_command_takes_the_builds_config_and_prints_the_count(self):
        self.earlier(self.dir / "old")
        args = ["--domain", "example_arith", "--heldout", str(self.heldout), "--base", str(self.dir / "old" / "replies-base.jsonl"), "--lora", str(self.dir / "old" / "replies-lora.jsonl")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score_lora.main([*args, "--config", str(self.data / "lora-config.json"), "--out", str(self.dir / "report.json")])
        self.assertIn("2 replies of tasks outside the evaluation set, not scored", buf.getvalue())
        self.assertIn("paired replies 1, unpaired 0", buf.getvalue())
        self.assertEqual(json.loads((self.dir / "report.json").read_text())["outside"]["tasks"], [EXCLUDED])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            score_lora.main(args)          # no --config: nothing says what may be scored

    def test_a_reply_written_under_another_build_is_counted_and_not_scored(self):
        """A reply that a resume could not ask again before the deadline stays in the file under the `config_sha` it was written with (another adapter, or another exclusion)."""
        a, b = (next(t for t in HELDOUT if t["id"] == x["id"]) for x in self.asked[:2])
        for arm in lora_eval.ARMS:
            write_jsonl(self.dir / f"{arm}.jsonl", [line(a, [reply(a, True, "0000000000000000")]), line(b, [reply(b, True, self.sha), {k: v for k, v in reply(b, True, self.sha).items() if k != "config_sha"}])])
        r = score_lora.score(self.dir / "base.jsonl", self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        self.assertEqual((r["paired"]["overall"]["n"], r["tasks_answered"]), (1, 1))
        self.assertEqual(r["other_build"], {"replies": 4, "by_arm": {"base": 2, "lora": 2}}, "the reply of another build and the one that does not say")
        self.assertIn(f"4 replies written under another build than {self.sha} (another adapter or another exclusion), not scored (base 2, adapter 2)", score_lora.lines_of(r))

    def test_a_reply_of_another_build_is_left_out_before_its_prompt_is_looked_at(self):
        a = next(t for t in HELDOUT if t["id"] == self.asked[0]["id"])
        stale = {**reply(a, True, "0000000000000000"), "prompt_sha": "deadbeefdeadbeef"}
        got = score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", [line(a, [stale])]), {a["id"]: a}, domain=ARITH, config_sha=self.sha)
        self.assertEqual((got["rows"], got["info"]["other_build"]), ({}, 1))
        with self.assertRaisesRegex(SystemExit, "another prompt"):
            score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", [line(a, [{**stale, "config_sha": self.sha}])]), {a["id"]: a}, domain=ARITH, config_sha=self.sha)

    def test_the_held_out_files_must_hold_the_whole_evaluation_set(self):
        self.earlier(self.dir / "old")
        short = write_jsonl(self.dir / "short.jsonl", [t for t in HELDOUT if t["id"] != self.asked[1]["id"]])
        with self.assertRaisesRegex(SystemExit, rf"1 task\(s\) of the build's evaluation set are not in --heldout \({self.asked[1]['id']}\)"):
            score_lora.score(self.dir / "old" / "replies-base.jsonl", self.dir / "old" / "replies-lora.jsonl", [short], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))

    def test_a_file_that_is_not_a_builds_config_is_refused(self):
        good = json.loads((self.data / "lora-config.json").read_text())
        path = self.dir / "c.json"
        for bad in ("not json", "[]", "{}", json.dumps({"config_sha": "x"}), json.dumps({**good, "config_sha": ""}), json.dumps({k: v for k, v in good.items() if k != "config_sha"}),
                    json.dumps({**good, "built_from": {**good["built_from"], "eval": {k: v for k, v in good["built_from"]["eval"].items() if k != "ids"}}}),
                    json.dumps({**good, "built_from": {**good["built_from"], "eval": {**good["built_from"]["eval"], "ids": []}}}), json.dumps({**good, "built_from": {**good["built_from"], "eval": {**good["built_from"]["eval"], "ids": [1]}}}),
                    json.dumps({**good, "built_from": "x"}), json.dumps({**good, "built_from": {"eval": "x"}})):
            path.write_text(bad)
            with self.assertRaisesRegex(SystemExit, "is not the lora-config.json of a build_lora build", msg=bad[:60]):
                score_lora.read_build(path)
        with self.assertRaisesRegex(SystemExit, "is not the lora-config.json"):
            score_lora.read_build(self.dir / "absent.json")

    def test_a_run_json_of_another_build_is_refused(self):
        build = score_lora.read_build(self.data / "lora-config.json")
        with self.assertRaisesRegex(SystemExit, f"is the run of build 0123456789abcdef, and --config is build {self.sha}"):
            score_lora.check_run({"config": {"config_sha": "0123456789abcdef"}}, build)
        for fine in (None, {}, {"config": {}}, {"config": "x"}, {"config": {"config_sha": self.sha}}, {"train": {"status": "done"}}):
            score_lora.check_run(fine, build)
        self.earlier(self.dir / "old")
        (self.dir / "run.json").write_text(json.dumps({"config": {"config_sha": "0123456789abcdef"}, "train": {}}))
        with self.assertRaisesRegex(SystemExit, "is the run of build 0123456789abcdef"), contextlib.redirect_stdout(io.StringIO()):
            score_lora.main(["--domain", "example_arith", "--config", str(self.data / "lora-config.json"), "--heldout", str(self.heldout), "--base", str(self.dir / "old" / "replies-base.jsonl"),
                             "--lora", str(self.dir / "old" / "replies-lora.jsonl"), "--run-info", str(self.dir / "run.json")])


class ExclusionInTheSha(Kit):
    """`config_sha` holds the content exclusion: a build whose `--train-tasks` exclude other held-out tasks is another build (PR 34 second review, M1: both builds of the repro had one sha)."""

    def narrow(self):
        """The SFT file and the training tasks without the training twin of arith-9999-003: a narrower `--train-tasks`, which excludes one held-out task fewer."""
        twin = next(t for t in TRAIN if t["question"] == HELDOUT[3]["question"])["id"]
        return {"train_tasks": [write_jsonl(self.dir / "narrow-train.jsonl", [t for t in TRAIN if t["id"] != twin])]}, twin

    def test_other_train_tasks_that_exclude_other_tasks_are_another_build(self):
        rows = [r for r in sft_rows() if r["task"] != self.narrow()[1]]
        sft = write_jsonl(self.dir / "s.jsonl", rows)
        wide, narrow = self.build(sft=sft, out=self.dir / "wide"), self.build(sft=sft, out=self.dir / "narrow", **self.narrow()[0])
        self.assertEqual(wide["built_from"]["train"]["sha256"], narrow["built_from"]["train"]["sha256"], "the same records and the same settings")
        self.assertEqual((sorted(wide["built_from"]["eval"]["excluded"]["pairs"]), sorted(narrow["built_from"]["eval"]["excluded"]["pairs"])), (sorted(SHARED), ["arith-9999-004"]))
        self.assertIn(EXCLUDED, narrow["built_from"]["eval"]["ids"])
        self.assertNotIn(EXCLUDED, wide["built_from"]["eval"]["ids"])
        self.assertNotEqual(wide["config_sha"], narrow["config_sha"], "the review: de5e30dcd0d81042 in both")
        self.assertNotEqual(wide["built_from"]["eval"]["excluded"]["sha"], narrow["built_from"]["eval"]["excluded"]["sha"])

    def test_the_sha_of_the_build_is_the_records_the_settings_and_the_exclusion(self):
        cfg = self.build()
        ex = cfg["built_from"]["eval"]["excluded"]
        self.assertEqual(ex["sha"], build_lora.excluded_sha(ex))
        self.assertEqual(cfg["config_sha"], lora_eval.config_sha(cfg["built_from"]["train"]["sha256"], cfg, ex["sha"]))
        self.assertRegex(ex["sha"], r"^[0-9a-f]{16}$")

    def test_the_exclusion_hash_reads_the_rules_and_the_excluded_tasks_and_nothing_else(self):
        ex = self.build()["built_from"]["eval"]["excluded"]
        base = build_lora.excluded_sha(ex)
        pairs = ex["pairs"]
        first = sorted(pairs)[0]
        for name, other in (("a task fewer", {**ex, "pairs": {k: v for k, v in pairs.items() if k != first}}), ("a task more", {**ex, "pairs": {**pairs, "arith-9999-000": {"code": "arith-7-000"}}}),
                            ("another task", {**{**ex, "pairs": {k: v for k, v in pairs.items() if k != first}}, "pairs": {**{k: v for k, v in pairs.items() if k != first}, "arith-9999-000": pairs[first]}}),
                            ("the rules", {**ex, "rules": ["code"]})):
            self.assertNotEqual(build_lora.excluded_sha(other), base, name)
        for name, same in (("the order of the pairs", {**ex, "pairs": dict(reversed(list(pairs.items())))}), ("the training task a pair names", {**ex, "pairs": {**pairs, first: {"code": "arith-7-999"}}}), ("the floor", {**ex, "min_family": 9}), ("the families left out", {**ex, "left_out": {"add": "x"}}),
                           ("the counts", {**ex, "total": {}, "by_family": {}}), ("the files", {**ex, "train_task_files": ["x"]})):
            self.assertEqual(build_lora.excluded_sha(same), base, name)

    def test_the_same_exclusion_from_other_files_or_another_subset_is_the_same_build(self):
        base = self.build()["config_sha"]
        halves = [write_jsonl(self.dir / "a.jsonl", TRAIN[15:]), write_jsonl(self.dir / "b.jsonl", TRAIN[:15])]
        more = write_jsonl(self.dir / "more.jsonl", [*TRAIN, {"id": "arith-7-900", "kind": "add", "question": "A question no held-out task asks?", "answer": 3}])
        for name, kw in (("two files", {"train_tasks": halves}), ("a training task more that matches nothing", {"train_tasks": [more]}), ("the floor", {"min_family": 2}), ("per family", {"per_family": 1})):
            self.assertEqual(self.build(out=self.dir / name.replace(" ", "-"), **kw)["config_sha"], base, name)

    def test_an_adapter_and_a_reply_of_the_narrower_build_are_not_continued_by_the_wider_one(self):
        """The trigger of the review: build with a narrower `--train-tasks`, run it, build again with the full file and `--resume-from`."""
        rows = [r for r in sft_rows() if r["task"] != self.narrow()[1]]
        sft = write_jsonl(self.dir / "s.jsonl", rows)
        wide, narrow = self.build(sft=sft, out=self.dir / "wide"), self.build(sft=sft, out=self.dir / "narrow", **self.narrow()[0])
        earlier = self.dir / "in" / "earlier" / "adapter"
        earlier.mkdir(parents=True)
        (earlier / "adapter_model.safetensors").write_text("x")
        (earlier / "train.json").write_text(json.dumps({"status": "done", "config_sha": narrow["config_sha"]}))
        with self.assertRaisesRegex(SystemExit, "--train-tasks that exclude other held-out tasks"):
            lora_eval.find_adapter([self.dir / "in"], wide["config_sha"])
        self.assertEqual(lora_eval.find_adapter([self.dir / "in"], narrow["config_sha"]), earlier)
        ckpt = self.dir / "in2" / "ckpt" / "step-10"
        ckpt.mkdir(parents=True)
        (ckpt / "DONE").write_text(json.dumps({"step": 10, "config_sha": narrow["config_sha"]}))
        with self.assertRaisesRegex(SystemExit, "--train-tasks that exclude other held-out tasks"):
            lora_eval.latest_checkpoint([self.dir / "in2"], wide["config_sha"])
        task = json.loads((self.dir / "wide" / "data" / "eval_tasks.jsonl").read_text().splitlines()[0])
        held = next(t for t in HELDOUT if t["id"] == task["id"])
        previous = {arm: {task["id"]: line(held, [reply(held, True, narrow["config_sha"])])} for arm in lora_eval.ARMS}
        self.assertEqual(lora_eval.work_items([task], previous, lora_eval.resolve_config({k: v for k, v in wide.items() if k not in ("built_from", "plan_lines")})), [(task, 0)], "asked again under the wider build")


class ReaEvalSet(unittest.TestCase):
    """The reviewer's own example on REA records: `structs-1001-0028`, which the build excludes (it is `structs-7-0292`), in the replies of an attached earlier output. Also the scorer on a held-out file
    named as the decompile stage names it (`decompiled-seed1001-*.jsonl`), which `load_tasks` refuses by its name and `load_heldout` reads (m5 (c))."""

    @classmethod
    def setUpClass(cls):
        from distill.domains.rea import DOMAIN
        from .distill_fixtures import load_tasks, right
        cls.rea, cls.right = DOMAIN, staticmethod(right)
        golden = [t for t in load_tasks() if t["kind"] in ("xor-key", "password")]
        cls.train = [PAIR[0], *golden]
        cls.other = [{**t, "id": t["id"].replace("-7-", "-1001-"), "question": t["question"] + " (held out)", "pseudocode": t["pseudocode"] + "\n/* other */"} for t in golden]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        rows = []
        for t in self.train:
            rows.append({"messages": [*self.rea.messages(t, "thinking"), {"role": "assistant", "content": THINK.format("looking at the code " * 8, "0x8")}], "task": t["id"], "kind": t["kind"], "model": TEACHER, "target_source": "teacher-reasoning"})
        self.heldout = write_jsonl(self.dir / "decompiled-seed1001-x.jsonl", [PAIR[1], *self.other])
        with contextlib.redirect_stdout(io.StringIO()):
            self.cfg = build_lora.build(write_jsonl(self.dir / "sft.jsonl", rows), [self.heldout], self.dir / "out", "someone", domain=self.rea, train_tasks=[write_jsonl(self.dir / "decompiled-seed7-x.jsonl", self.train)],
                                        per_family=1, min_family=1)
        self.config = self.dir / "out" / "data" / "lora-config.json"

    def test_the_excluded_structs_task_of_an_attached_output_is_not_in_the_report(self):
        sha, asked = self.cfg["config_sha"], self.cfg["built_from"]["eval"]["ids"]
        self.assertNotIn("structs-1001-0028", asked)
        kept = next(t for t in self.other if t["id"] == asked[0])
        for arm in lora_eval.ARMS:
            write_jsonl(self.dir / f"replies-{arm}.jsonl", [line(PAIR[1], [reply(PAIR[1], True, sha, self.rea, answer="0x8")], self.rea), line(kept, [reply(kept, True, sha, self.rea, answer=self.right(kept))], self.rea)])
        with self.assertRaises(SystemExit, msg="the file name is the held-out seed's: the training loader refuses it"):
            self.rea.load_tasks([self.heldout])
        r = score_lora.score(self.dir / "replies-base.jsonl", self.dir / "replies-lora.jsonl", [self.heldout], domain=self.rea, build=score_lora.read_build(self.config))
        self.assertEqual((r["paired"]["overall"]["n"], r["paired"]["overall"]["both"], sorted(r["paired"]["by_kind"])), (1, 1, [kept["kind"]]), "the review: 2 paired replies, by kind {'antidebug': 1, 'structs': 1}")
        self.assertEqual(r["outside"], {"replies": 2, "by_arm": {"base": 1, "lora": 1}, "tasks": ["structs-1001-0028"]})
        self.assertIn("2 replies of tasks outside the evaluation set, not scored (base 1, adapter 1; tasks: structs-1001-0028)", score_lora.lines_of(r))
        all_held = build_of([PAIR[1], *self.other], sha)
        leaky = score_lora.score(self.dir / "replies-base.jsonl", self.dir / "replies-lora.jsonl", [self.heldout], domain=self.rea, build=all_held)
        self.assertEqual(leaky["paired"]["by_kind"]["structs"]["both"], 1, "the reply itself is a right answer: only the evaluation set keeps it out")

    def test_the_merge_of_the_kernel_drops_it_too(self):
        inp, work = self.dir / "in" / "earlier", self.dir / "work"
        inp.mkdir(parents=True), work.mkdir()
        sha = self.cfg["config_sha"]
        kept = next(t for t in self.other if t["id"] == self.cfg["built_from"]["eval"]["ids"][0])
        for arm in lora_eval.ARMS:
            write_jsonl(inp / f"replies-{arm}.jsonl", [line(PAIR[1], [reply(PAIR[1], True, sha, self.rea, answer="0x8")], self.rea), line(kept, [reply(kept, True, sha, self.rea, answer=self.right(kept))], self.rea)])
        with unittest.mock.patch.object(lora_eval, "log", lambda *a: None):
            self.assertEqual({arm: lora_eval.merge_shards(work, arm, lora_eval.eval_ids(self.dir / "out" / "data"), input_root=self.dir / "in") for arm in lora_eval.ARMS}, {"base": 1, "lora": 1})
        self.assertNotIn("structs-1001-0028", (work / "replies-base.jsonl").read_text() + (work / "replies-lora.jsonl").read_text())


class KNote(unittest.TestCase):
    def rows(self, spec):
        return {(tid, j): {"kind": "add", "verdict": "accepted" if v == "a" else "wrong", "finish": "stop", "tokens": 10} for tid, vs in spec.items() for j, v in enumerate(vs)}

    def test_per_task_is_the_most_replies_one_task_has(self):
        """PR 34 second review, m5 (c): tasks with 1 and with 3 paired replies: the k > 1 note is about the most, not the fewest."""
        base, lora = self.rows({"t1": "a", "t2": "waw", "t3": "aa"}), self.rows({"t1": "w", "t2": "aaa", "t3": "aa"})
        p = score_lora.pair(base, lora)
        self.assertEqual((p["overall"]["per_task"], p["by_kind"]["add"]["per_task"], p["overall"]["tasks"]), (3, 3, 3))
        self.assertEqual(score_lora.pair(self.rows({"t1": "a"}), self.rows({"t1": "w"}))["overall"]["per_task"], 1)
        self.assertEqual(score_lora.pair({}, {})["overall"]["per_task"], 0)
        report = {"arms": {a: {"overall": {"replies": 6, "accepted": 3, "accuracy": 0.5, "ci95": [0.2, 0.8], "cut_off": 0.0, "mean_tokens": 10.0}, "by_kind": {}} for a in ("base", "lora")}, "paired": p, "tasks_answered": 3,
                  "tasks_in_set": 3, "config_sha": SHA, "outside": {"replies": 0, "by_arm": {"base": 0, "lora": 0}, "tasks": []}, "other_build": {"replies": 0, "by_arm": {"base": 0, "lora": 0}}}
        self.assertTrue(any("k > 1 (up to 3 replies per task)" in x for x in score_lora.lines_of(report)))


class Fit(Kit):
    """`it fits` is the quota AND the kernel's own deadline (PR 34 second review, m2)."""

    def lines(self, name, **kw):
        cfg = self.build(out=self.dir / name, **kw)
        return cfg, cfg["plan_lines"]

    def test_a_plan_inside_the_quota_that_the_kernels_deadline_would_cut_does_not_fit(self):
        cfg, _ = self.lines("first")
        est = cfg["built_from"]["estimate"]
        work, smoke = est["total_upper_s"] / 3600, est["smoke_s"] / 3600
        for name, quota, fits in (("inside", work + smoke + 0.10, False), ("edge", work + smoke + 0.24, False), ("room", work + smoke + 0.27, True), ("over", work + smoke - 0.01, None)):
            if fits is None:
                continue
            cfg, lines = self.lines(name, gpu_hours=quota)
            self.assertLessEqual(est["total_upper_s"] + est["smoke_s"], quota * 3600, "the total is inside the quota in every case")
            self.assertEqual(cfg["deadline_h"], math.floor((quota - smoke - build_lora.DEADLINE_MARGIN_H) * 100) / 100)
            self.assertEqual(cfg["deadline_h"] * 3600 >= est["total_upper_s"], fits, name)
            self.assertEqual("it fits" in lines[-2], fits, lines[-2])
            self.assertEqual("IT DOES NOT FIT (inside the quota, but this run needs" in lines[-2], not fits, lines[-2])
            if not fits:
                late = math.ceil((est["total_upper_s"] - cfg["deadline_h"] * 3600) / 60)
                self.assertIn(f"{late} min past the {cfg['deadline_h']:g} h after which the kernel starts nothing: it would stop training early or leave the last batches unasked", lines[-2])
                self.assertGreaterEqual(late, 1)

    def test_the_verdict_is_the_kernels_arithmetic(self):
        """The kernel starts nothing `deadline_h` after its start and stops training `eval_reserve_s` before that: the plan fits exactly when its own work ends by then."""
        p = build_lora.plan([3000] * 1200, [2000] * 70, epochs=2.0, gpus=2, global_batch=16, train_hours=3.0, k=1, max_new_tokens=4096, eval_batch=8, export=False, rates=build_lora.RATES)
        work = p["total_upper_s"]
        self.assertEqual(build_lora.fit_verdict(p, 8.0, work / 3600), "")
        self.assertIn("1 min past the", build_lora.fit_verdict(p, 8.0, (work - 1) / 3600))
        self.assertIn("after which the kernel starts nothing", build_lora.fit_verdict(p, 8.0, (work - 1) / 3600))
        self.assertIn(" 2 min past the", build_lora.fit_verdict(p, 8.0, (work - 61) / 3600))
        self.assertEqual(build_lora.fit_verdict(p, work / 3600, 7.5), "")
        self.assertEqual(build_lora.fit_verdict(p, (work - 1) / 3600, 7.5), "more than the quota")
        with_smoke = {**p, "smoke_s": 1800}
        self.assertEqual(build_lora.fit_verdict(with_smoke, (work + 1800) / 3600, work / 3600), "", "the smoke run is out of the quota, not out of this kernel's deadline")
        self.assertEqual(build_lora.fit_verdict(with_smoke, (work + 1799) / 3600, work / 3600), "more than the quota")
        self.assertIn("after which the kernel starts nothing", build_lora.fit_verdict(with_smoke, 99.0, (work - 1) / 3600))
        # the kernel's own clock, with the plan's numbers: training that starts after the setup and runs its steps ends before `train_deadline`, exactly when the plan fits
        r = p["rates"]
        for deadline_s, fits in ((work, True), (work - 1, False)):
            stop = lora_eval.train_deadline(r["setup_s"], deadline_s, 3.0, p["eval_reserve_s"])
            self.assertEqual(r["setup_s"] + r["load_s"] + p["train_s"] <= stop, fits)
            self.assertEqual(build_lora.fit_verdict(p, 99.0, deadline_s / 3600) == "", fits)

    def test_the_readme_case_fits_and_the_reviews_case_does_not(self):
        p = build_lora.plan([3000] * 1200, [2000] * 70, epochs=2.0, gpus=2, global_batch=16, train_hours=3.0, k=1, max_new_tokens=4096, eval_batch=8, export=False, rates=build_lora.RATES)
        smoke = build_lora.smoke_plan(4000, gpus=2, steps=4, max_new_tokens=256, eval_batch=8, rates=build_lora.RATES)["total_upper_s"]
        deadline = lambda quota: math.floor((quota - smoke / 3600 - build_lora.DEADLINE_MARGIN_H) * 100) / 100
        self.assertIn("it fits", build_lora.describe({**p, "smoke_s": smoke}, 8.0, deadline(8.0))[-2])
        tight = (p["total_upper_s"] + smoke) / 3600 + 0.1          # the review's `--gpu-hours 5.7` for a 4.98 h run and a 0.6 h smoke run
        line = build_lora.describe({**p, "smoke_s": smoke}, tight, deadline(tight))[-2]
        self.assertIn("IT DOES NOT FIT (inside the quota", line)
        self.assertNotIn("it fits", line)

    def test_a_smoke_plan_is_judged_the_same_way(self):
        s = build_lora.smoke_plan(4000, gpus=2, steps=4, max_new_tokens=256, eval_batch=8, rates=build_lora.RATES)
        self.assertIn("it fits", build_lora.describe(s, 8.0, 7.5)[-2])
        self.assertIn("IT DOES NOT FIT (inside the quota", build_lora.describe(s, 8.0, 0.5)[-2])


class Measured(Kit):
    """`--measured` reaches the plan through the builder (PR 34 second review, m5 (b))."""

    def test_the_smoke_runs_numbers_are_the_ones_the_build_plans_with(self):
        path = self.dir / "run.json"
        path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": 75.0, "gen_step_s": 0.6}}))
        assumed, measured = self.build(out=self.dir / "a"), self.build(out=self.dir / "m", measured=path)
        self.assertEqual(assumed["built_from"]["rates"], build_lora.RATES)
        self.assertEqual((measured["built_from"]["rates"]["train_tok_per_s_per_gpu"], measured["built_from"]["rates"]["gen_step_s"]), (75.0, 0.6))
        a, m = assumed["built_from"]["estimate"], measured["built_from"]["estimate"]
        self.assertAlmostEqual(m["step_s"], 2 * a["step_s"], delta=0.11)
        self.assertGreater(m["eval_upper_s"], 1.9 * a["eval_upper_s"])
        self.assertEqual(json.loads((self.dir / "m" / "data" / "lora-config.json").read_text())["eval_reserve_s"], m["eval_reserve_s"], "the kernel reserves the measured evaluation")
        self.assertIn("assumed, with the numbers of --measured", measured["plan_lines"][0])
        self.assertNotIn("--measured", assumed["plan_lines"][0])

    def test_the_command_passes_it_on(self):
        path = self.dir / "run.json"
        path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": 75.0, "gen_step_s": 0.6}}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_lora.main(["--domain", "example_arith", "--sft", str(self.sft), "--train-tasks", str(self.train_tasks), "--heldout", str(self.heldout), "--owner", "someone", "--out", str(self.dir / "cli"),
                             "--per-family", "3", "--min-family", "1", "--measured", str(path)])
        self.assertIn("at 75 tok/s per GPU (assumed, with the numbers of --measured)", buf.getvalue())
        self.assertIn("the smoke run is done (--measured)", buf.getvalue())


class Labels(Kit):
    """A teacher label that is in no entry of `distill/teachers.py` is refused (PR 34 second review, m3): the kit fails closed."""

    def test_a_label_that_is_in_no_catalogue_entry_is_refused(self):
        for label in ("o3-mini", "openai-o4", "some-new-teacher", "gemini-3-pro", "gpt-3.5-turbo", "codex-davinci", "", "Codeforge-27B-heretic-q4"):
            rows = [r for t in make_tasks(7, 4) for r in (row(t, label=label), row(t, "truth-template"))]
            with self.assertRaisesRegex(SystemExit, "is in no entry of distill/teachers.py|may only FILTER", msg=label):
                build_lora.check_training(rows, [], domain=ARITH)
        with self.assertRaisesRegex(SystemExit, "o3-mini is in no entry of distill/teachers.py"):
            build_lora.check_training([row(t, label="o3-mini") for t in make_tasks(7, 4)], [], domain=ARITH)

    def test_one_such_record_among_good_ones_stops_the_build_before_anything_is_written(self):
        rows = [*sft_rows(), {**row(TRAIN[0]), "model": "openai-o4"}]
        with self.assertRaisesRegex(SystemExit, "openai-o4 is in no entry of distill/teachers.py"):
            self.build(sft=write_jsonl(self.dir / "mixed.jsonl", rows))
        self.assertFalse((self.dir / "out").exists())

    def test_every_catalogue_teacher_that_may_store_text_is_still_accepted_and_a_truth_label_is_not_judged(self):
        from distill import teachers
        for t in teachers.TEACHERS:
            if t.reasoning_ok and not teachers.marked_filter_only(t):
                self.assertEqual(build_lora.check_training(sft_rows(3, label=t.name), [], domain=ARITH)["labels"], [t.name])
        rows = [r for t in make_tasks(7, 4) for r in (row(t), row(t, "truth-template", label="some-new-teacher"))]
        self.assertEqual(build_lora.check_training(rows, [], domain=ARITH)["labels"], [TEACHER], "a truth-template record stores nothing of the model")


if __name__ == "__main__":
    unittest.main()

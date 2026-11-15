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

from .test_kaggle_lora import ARITH, HELDOUT, KEYS, PAIR, SHARED, TEACHER, THINK, TRAIN, Kit, build_lora, build_of, lora_eval, make_tasks, row, run_teacher, score_lora, sft_rows, write_jsonl

EXCLUDED = "arith-9999-003"          # the held-out add task that seed 7 draws too: every build of the fixture excludes it


def keys_of(cfg: dict) -> dict:
    """The two keys of a build's config: what its kernel writes into every reply."""
    return {"adapter_sha": cfg["adapter_sha"], "eval_sha": cfg["eval_sha"]}


def reply(task: dict, right: bool, keys: dict, domain=ARITH, answer=None) -> dict:
    """A reply written under `keys` ({"adapter_sha", "eval_sha"}, or fewer: a reply that does not say)."""
    text = THINK.format("thinking", answer if answer is not None else task["answer"] if right else task["answer"] + 1)
    return {"text": text, "finish": "stop", "tokens": 100, "prompt_tokens": 50, "secs": 1.0, "prompt_sha": outputs.prompt_sha(domain.messages(task, "thinking")), "max_tokens": 4096, **keys}


def line(task: dict, outs: list, domain=ARITH) -> dict:
    return {"id": task["id"], "kind": task["kind"], "prompt_sha": outputs.prompt_sha(domain.messages(task, "thinking")), "outs": outs}


class EvalSet(Kit):
    """The reviewer's repro: an attached earlier output holds the replies of a task this build excludes."""

    def setUp(self):
        super().setUp()
        self.cfg = self.build()
        self.data = self.dir / "out" / "data"
        self.sha = keys_of(self.cfg)
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
        self.assertEqual(score_lora.read_build(self.data / "lora-config.json"), {"ids": set(ids), **self.sha, "smoke": False})
        self.assertEqual(self.cfg["eval_sha"], lora_eval.eval_key(self.cfg["adapter_sha"], self.cfg["built_from"]["eval"]["excluded"]["sha"], lora_eval.file_sha(self.data / "eval_tasks.jsonl"), lora_eval.resolve_config(self.cfg)), "the evaluation key is of exactly the tasks the kernel is given")

    def test_the_excluded_task_is_in_the_earlier_output_under_this_very_builds_keys(self):
        """What made the first fix round's check blind: the reply of the excluded task carries the build's own keys."""
        self.earlier(self.dir / "in" / "earlier")
        got = run_teacher.load_previous(lora_eval.previous_files("lora", self.dir / "work", self.dir / "in"))
        self.assertEqual(keys_of(got[EXCLUDED]["outs"][0]), self.sha)
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
        self.assertIn("merge_shards(OUT, arm, eval_ids(data_dir), prefix, replies_root(cfg))", main)
        self.assertEqual(src.count("merge_shards("), 2, "its definition and the one call")

    def test_the_scorer_scores_the_evaluation_set_only_and_says_what_it_left_out(self):
        """The review's second half: given replies files that DO hold the excluded task (an output merged by an older kernel), the report does not count it."""
        self.earlier(self.dir / "old")
        build = score_lora.read_build(self.data / "lora-config.json")
        r = score_lora.score(self.dir / "old" / "replies-base.jsonl", self.dir / "old" / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=build)
        self.assertEqual((r["paired"]["overall"]["n"], r["paired"]["overall"]["lora_only"], sorted(r["paired"]["by_kind"])), (1, 0, [self.kept["kind"]]), "the review got 2 paired replies, the excluded task's among them")
        self.assertEqual((r["arms"]["lora"]["overall"]["replies"], r["arms"]["lora"]["overall"]["accepted"]), (1, 1))
        self.assertEqual(r["outside"], {"replies": 2, "by_arm": {"base": 1, "lora": 1}, "tasks": [EXCLUDED]})
        self.assertEqual((r["tasks_answered"], r["tasks_in_set"], keys_of(r)), (1, 9, self.sha))
        lines = score_lora.lines_of(r)
        self.assertIn(f"2 replies of tasks outside the evaluation set, not scored (base 1, adapter 1; tasks: {EXCLUDED})", lines)
        self.assertIn("tasks answered 1 of 9 in the evaluation set of the build", lines[-3])

    def test_what_is_counted_outside_is_replies_not_lines_and_not_empty_entries(self):
        ex = self.excluded
        got = score_lora.read_arm(write_jsonl(self.dir / "k.jsonl", [line(ex, [reply(ex, True, self.sha), None, reply(ex, False, self.sha)]), line(self.kept, [reply(self.kept, True, self.sha)])]),
                                  {self.kept["id"]: self.kept}, domain=ARITH, build=self.sha)
        self.assertEqual((got["info"]["outside"], got["outside"], sorted(got["rows"])), (2, [EXCLUDED], [(self.kept["id"], 0)]), "two replies of one task, the entry that was never answered is none")

    def test_a_clean_run_says_that_nothing_was_outside(self):
        for arm in lora_eval.ARMS:
            write_jsonl(self.dir / f"{arm}.jsonl", [line(t, [reply(t, True, self.sha)]) for t in HELDOUT if t["id"] in self.cfg["built_from"]["eval"]["ids"]])
        r = score_lora.score(self.dir / "base.jsonl", self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        lines = score_lora.lines_of(r)
        self.assertEqual(lines[-1], "0 replies of tasks outside the evaluation set, not scored")
        self.assertEqual(lines[-2], f"adapter key {self.cfg['adapter_sha']} (the training records and the training settings), evaluation key {self.cfg['eval_sha']} (that adapter, the exclusion, the tasks asked, the reply budget and the sampling): only the replies that carry both are scored")
        self.assertEqual((r["paired"]["overall"]["n"], r["outside"]["replies"], r["other_adapter"]["replies"], r["other_eval"]["replies"]), (9, 0, 0, 0))
        self.assertFalse(any("another adapter" in x or "another evaluation key" in x for x in lines), "the two lines are printed only when there is such a reply")

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

    def test_a_reply_of_another_adapter_or_another_evaluation_is_counted_and_not_scored(self):
        """A reply that a resume could not ask again before the deadline stays in the file under the keys it was written with. Six replies of six tasks, each RIGHT in the adapter arm and wrong in the base
        arm, so that any one of them that reached a figure would show: this build's (scored), this adapter under another evaluation key, another adapter under its own evaluation key, another adapter that
        names this evaluation key, one with no evaluation key, one with no key at all."""
        mine, tasks = self.sha, [next(t for t in HELDOUT if t["id"] == x["id"]) for x in self.asked[:6]]
        cases = [mine, {**mine, "eval_sha": "0000000000000000"}, {"adapter_sha": "1111111111111111", "eval_sha": "2222222222222222"}, {**mine, "adapter_sha": "1111111111111111"}, {"adapter_sha": mine["adapter_sha"]}, {}]
        for arm in lora_eval.ARMS:
            rows = [line(t, [reply(t, arm == "lora", keys)]) for t, keys in zip(tasks, cases)]
            if arm == "lora":
                rows.append(line(tasks[0], [rows[0]["outs"][0], reply(tasks[0], True, {**mine, "eval_sha": "0000000000000000"})]))          # the adapter arm alone has a second reply of the first task, of another evaluation
            write_jsonl(self.dir / f"{arm}.jsonl", rows)
        r = score_lora.score(self.dir / "base.jsonl", self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        self.assertEqual((r["paired"]["overall"]["n"], r["paired"]["overall"]["lora_only"], r["tasks_answered"], r["paired"]["unpaired"]), (1, 1, 1, 0), "the one reply that carries both keys of the build")
        self.assertEqual(r["other_adapter"], {"replies": 6, "by_arm": {"base": 3, "lora": 3}}, "another adapter, another adapter under this evaluation key, and no key at all")
        self.assertEqual(r["other_eval"], {"replies": 5, "by_arm": {"base": 2, "lora": 3}}, "this adapter under another evaluation key, and under none")
        lines = score_lora.lines_of(r)
        self.assertIn(f"6 replies of another adapter than {mine['adapter_sha']} (another --sft or training setting, or a smoke run's), not scored (base 3, adapter 3)", lines)
        self.assertIn(f"5 replies of this adapter written under another evaluation key than {mine['eval_sha']} (another exclusion, set of tasks, prompt, reply budget, sampling or k: a resume asks them again), not scored (base 2, adapter 3)", lines)

    def test_each_key_alone_keeps_a_reply_out(self):
        a, mine = next(t for t in HELDOUT if t["id"] == self.asked[0]["id"]), self.sha
        read = lambda keys, build=mine: score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", [line(a, [reply(a, True, keys)])]), {a["id"]: a}, domain=ARITH, build=build)
        self.assertEqual(sorted(read(mine)["rows"]), [(a["id"], 0)])
        for name, keys, counter in (("another evaluation key", {**mine, "eval_sha": "x"}, "other_eval"), ("no evaluation key", {"adapter_sha": mine["adapter_sha"]}, "other_eval"), ("another adapter key", {**mine, "adapter_sha": "x"}, "other_adapter"),
                                    ("no adapter key", {"eval_sha": mine["eval_sha"]}, "other_adapter"), ("the key of a kernel before the two keys", {"config_sha": mine["eval_sha"]}, "other_adapter")):
            got = read(keys)
            self.assertEqual((got["rows"], {k: got["info"].get(k, 0) for k in ("other_adapter", "other_eval")}), ({}, {"other_adapter": 0, "other_eval": 0, counter: 1}), name)
        self.assertEqual(read({"adapter_sha": mine["eval_sha"], "eval_sha": mine["adapter_sha"]})["rows"], {}, "the two keys are not interchangeable")

    def test_a_reply_of_another_build_is_left_out_before_its_prompt_is_looked_at(self):
        a = next(t for t in HELDOUT if t["id"] == self.asked[0]["id"])
        for counter, keys in (("other_adapter", {**self.sha, "adapter_sha": "0000000000000000"}), ("other_eval", {**self.sha, "eval_sha": "0000000000000000"})):
            stale = {**reply(a, True, keys), "prompt_sha": "deadbeefdeadbeef"}
            got = score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", [line(a, [stale])]), {a["id"]: a}, domain=ARITH, build=self.sha)
            self.assertEqual((got["rows"], got["info"][counter]), ({}, 1))
            unchecked = score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", [{**line(a, [{**stale, "prompt_sha": None}]), "prompt_sha": None}]), {a["id"]: a}, domain=ARITH, build=self.sha)
            self.assertEqual((unchecked["info"][counter], unchecked["info"].get("unchecked", 0)), (1, 0), "and before its missing fingerprint is counted")
        with self.assertRaisesRegex(SystemExit, "another prompt"):
            score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", [line(a, [{**stale, **self.sha}])]), {a["id"]: a}, domain=ARITH, build=self.sha)

    def test_the_line_of_what_was_not_scored_names_each_arms_count_and_the_tasks_of_either_arm(self):
        """PR 34 third review, m2: the outside replies DIFFER by arm (base: two replies of one task; adapter: one reply of another), so the counts cannot be swapped and the ids cannot come from one arm."""
        other = next(t for t in HELDOUT if t["id"] not in self.cfg["built_from"]["eval"]["ids"] and t["id"] != EXCLUDED)
        write_jsonl(self.dir / "base.jsonl", [line(self.excluded, [reply(self.excluded, True, self.sha), reply(self.excluded, True, self.sha)]), line(self.kept, [reply(self.kept, True, self.sha)])])
        write_jsonl(self.dir / "lora.jsonl", [line(other, [reply(other, True, self.sha)]), line(self.kept, [reply(self.kept, True, self.sha)])])
        r = score_lora.score(self.dir / "base.jsonl", self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        self.assertEqual(r["outside"], {"replies": 3, "by_arm": {"base": 2, "lora": 1}, "tasks": sorted([EXCLUDED, other["id"]])})
        self.assertIn(f"3 replies of tasks outside the evaluation set, not scored (base 2, adapter 1; tasks: {', '.join(sorted([EXCLUDED, other['id']]))})", score_lora.lines_of(r))
        self.assertEqual((r["paired"]["overall"]["n"], r["tasks_answered"]), (1, 1))
        swapped = score_lora.score(self.dir / "lora.jsonl", self.dir / "base.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        self.assertEqual(swapped["outside"], {"replies": 3, "by_arm": {"base": 1, "lora": 2}, "tasks": sorted([EXCLUDED, other["id"]])})
        self.assertIn(f"3 replies of tasks outside the evaluation set, not scored (base 1, adapter 2; tasks: {', '.join(sorted([EXCLUDED, other['id']]))})", score_lora.lines_of(swapped))
        only_lora = score_lora.score(write_jsonl(self.dir / "b2.jsonl", [line(self.kept, [reply(self.kept, True, self.sha)])]), self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))
        self.assertEqual(only_lora["outside"], {"replies": 1, "by_arm": {"base": 0, "lora": 1}, "tasks": [other["id"]]}, "a task that only the adapter arm's file holds is named too")
        self.assertIn(f"1 replies of tasks outside the evaluation set, not scored (base 0, adapter 1; tasks: {other['id']})", score_lora.lines_of(only_lora))

    def test_the_held_out_files_must_hold_the_whole_evaluation_set(self):
        self.earlier(self.dir / "old")
        short = write_jsonl(self.dir / "short.jsonl", [t for t in HELDOUT if t["id"] != self.asked[1]["id"]])
        with self.assertRaisesRegex(SystemExit, rf"1 task\(s\) of the build's evaluation set are not in --heldout \({self.asked[1]['id']}\)"):
            score_lora.score(self.dir / "old" / "replies-base.jsonl", self.dir / "old" / "replies-lora.jsonl", [short], domain=ARITH, build=score_lora.read_build(self.data / "lora-config.json"))

    def test_a_file_that_is_not_a_builds_config_is_refused(self):
        good = json.loads((self.data / "lora-config.json").read_text())
        path = self.dir / "c.json"
        for bad in ("not json", "[]", "{}", json.dumps({"adapter_sha": "x", "eval_sha": "y"}), json.dumps({**good, "adapter_sha": ""}), json.dumps({**good, "eval_sha": ""}), json.dumps({**good, "eval_sha": 7}),
                    json.dumps({k: v for k, v in good.items() if k != "adapter_sha"}), json.dumps({k: v for k, v in good.items() if k != "eval_sha"}),
                    json.dumps({**{k: v for k, v in good.items() if k not in ("adapter_sha", "eval_sha")}, "config_sha": good["eval_sha"]}),
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
        with self.assertRaisesRegex(SystemExit, f"is the run of adapter key 0123456789abcdef, and --config has adapter key {self.sha['adapter_sha']}"):
            score_lora.check_run({"config": {**self.sha, "adapter_sha": "0123456789abcdef"}}, build)
        with self.assertRaisesRegex(SystemExit, f"is the run of evaluation key 0123456789abcdef, and --config has evaluation key {self.sha['eval_sha']}"):
            score_lora.check_run({"config": {**self.sha, "eval_sha": "0123456789abcdef"}}, build)
        with self.assertRaisesRegex(SystemExit, "is the run of evaluation key 0123456789abcdef"):
            score_lora.check_run({"config": {"eval_sha": "0123456789abcdef"}}, build)
        for fine in (None, {}, {"config": {}}, {"config": "x"}, {"config": self.sha}, {"config": {"adapter_sha": self.sha["adapter_sha"]}}, {"config": {"eval_sha": self.sha["eval_sha"]}}, {"train": {"status": "done"}}):
            score_lora.check_run(fine, build)
        self.earlier(self.dir / "old")
        (self.dir / "run.json").write_text(json.dumps({"config": {**self.sha, "adapter_sha": "0123456789abcdef"}, "train": {}}))
        with self.assertRaisesRegex(SystemExit, "is the run of adapter key 0123456789abcdef"), contextlib.redirect_stdout(io.StringIO()):
            score_lora.main(["--domain", "example_arith", "--config", str(self.data / "lora-config.json"), "--heldout", str(self.heldout), "--base", str(self.dir / "old" / "replies-base.jsonl"),
                             "--lora", str(self.dir / "old" / "replies-lora.jsonl"), "--run-info", str(self.dir / "run.json")])


class TwoKeys(Kit):
    """The training identity and the evaluation identity are separate (PR 34 third review, the owner's decision). The second review's M1 made the exclusion part of ONE hash, which refused a trained adapter
    when only `--train-tasks` changed; now the exclusion and the tasks asked are in the EVALUATION key and the adapter key holds what the weights are made of."""

    def narrow(self):
        """The SFT file and the training tasks without the training twin of arith-9999-003: a narrower `--train-tasks`, which excludes one held-out task fewer."""
        twin = next(t for t in TRAIN if t["question"] == HELDOUT[3]["question"])["id"]
        return {"train_tasks": [write_jsonl(self.dir / "narrow-train.jsonl", [t for t in TRAIN if t["id"] != twin])]}, twin

    def both(self):
        """(the wider build, the narrower one): the same SFT file and settings, `--train-tasks` with and without one training task."""
        rows = [r for r in sft_rows() if r["task"] != self.narrow()[1]]
        sft = write_jsonl(self.dir / "s.jsonl", rows)
        return self.build(sft=sft, out=self.dir / "wide"), self.build(sft=sft, out=self.dir / "narrow", **self.narrow()[0])

    def kernel_cfg(self, cfg):
        return lora_eval.resolve_config({k: v for k, v in cfg.items() if k not in ("built_from", "plan_lines")})

    def test_other_train_tasks_that_exclude_other_tasks_are_another_evaluation_of_the_same_adapter(self):
        wide, narrow = self.both()
        self.assertEqual(wide["built_from"]["train"]["sha256"], narrow["built_from"]["train"]["sha256"], "the same records and the same settings")
        self.assertEqual((sorted(wide["built_from"]["eval"]["excluded"]["pairs"]), sorted(narrow["built_from"]["eval"]["excluded"]["pairs"])), (sorted(SHARED), ["arith-9999-004"]))
        self.assertIn(EXCLUDED, narrow["built_from"]["eval"]["ids"])
        self.assertNotIn(EXCLUDED, wide["built_from"]["eval"]["ids"])
        self.assertEqual(wide["adapter_sha"], narrow["adapter_sha"], "the adapter is what the SFT file and the training settings make it")
        self.assertNotEqual(wide["eval_sha"], narrow["eval_sha"], "the second review: one sha, de5e30dcd0d81042, for both evaluations")
        self.assertNotEqual(wide["built_from"]["eval"]["excluded"]["sha"], narrow["built_from"]["eval"]["excluded"]["sha"])

    def test_train_tasks_decide_the_exclusion_and_never_a_training_record(self):
        """What the adapter key rests on: `--train-tasks` changes which held-out tasks are asked, and not one byte of what the kernel trains on."""
        wide, narrow = self.both()
        data = lambda name, f: (self.dir / name / "data" / f).read_bytes()
        self.assertEqual(data("wide", "train.jsonl"), data("narrow", "train.jsonl"), "the training records are the SFT file's, whatever the exclusion")
        self.assertNotEqual(data("wide", "eval_tasks.jsonl"), data("narrow", "eval_tasks.jsonl"))
        self.assertEqual({k: self.kernel_cfg(wide)[k] for k in lora_eval.TRAIN_KEYS}, {k: self.kernel_cfg(narrow)[k] for k in lora_eval.TRAIN_KEYS})
        self.assertEqual((wide["max_steps"], wide["built_from"]["train"]), (narrow["max_steps"], narrow["built_from"]["train"]))

    def test_the_keys_of_the_build_are_the_kernels_functions_of_what_it_was_built_from(self):
        cfg = self.build()
        ex = cfg["built_from"]["eval"]["excluded"]
        self.assertEqual(ex["sha"], build_lora.excluded_sha(ex))
        train = self.dir / "out" / "data" / "train.jsonl"
        self.assertEqual(cfg["built_from"]["train"]["train_sha256"], lora_eval.file_sha(train), "the records as the kernel reads them")
        self.assertEqual(cfg["built_from"]["train"]["sha256"], lora_eval.file_sha(self.sft), "and the SFT file they were read from, for the record")
        self.assertEqual(cfg["adapter_sha"], lora_eval.adapter_key(lora_eval.file_sha(train), self.kernel_cfg(cfg)))
        self.assertEqual(cfg["eval_sha"], lora_eval.eval_key(cfg["adapter_sha"], ex["sha"], cfg["built_from"]["eval"]["sha256"], self.kernel_cfg(cfg)))
        self.assertRegex(ex["sha"], r"^[0-9a-f]{16}$")
        self.assertEqual(build_lora.keys_line(cfg), f"adapter key {cfg['adapter_sha']} (the training records and the training settings, the cap on the steps among them: a resume keeps an adapter or a checkpoint of this key and refuses any other), "
                         f"evaluation key {cfg['eval_sha']} (the adapter key, the exclusion, the tasks asked, the reply budget, the sampling and k: a reply of another evaluation key is asked again and never scored)")
        smoke = self.build(out=self.dir / "smoke", smoke=True)
        self.assertEqual(build_lora.keys_line(smoke), build_lora.keys_line({**smoke, "smoke": False}) + " [SMOKE build: keys of its own, never a full run's]")

    def test_the_command_prints_both_keys(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_lora.main(["--domain", "example_arith", "--sft", str(self.sft), "--train-tasks", str(self.train_tasks), "--heldout", str(self.heldout), "--owner", "someone", "--out", str(self.dir / "cli"),
                             "--per-family", "3", "--min-family", "1"])
        cfg = json.loads((self.dir / "cli" / "data" / "lora-config.json").read_text())
        self.assertIn(build_lora.keys_line(cfg), buf.getvalue().splitlines())

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

    def test_the_same_evaluation_from_other_files_is_the_same_keys_and_another_subset_is_another_evaluation(self):
        base = keys_of(self.build())
        halves = [write_jsonl(self.dir / "a.jsonl", TRAIN[15:]), write_jsonl(self.dir / "b.jsonl", TRAIN[:15])]
        more = write_jsonl(self.dir / "more.jsonl", [*TRAIN, {"id": "arith-7-900", "kind": "add", "question": "A question no held-out task asks?", "answer": 3}])
        for name, kw in (("two files", {"train_tasks": halves}), ("a training task more that matches nothing", {"train_tasks": [more]}), ("a floor that leaves the same families", {"min_family": 2})):
            self.assertEqual(keys_of(self.build(out=self.dir / name.replace(" ", "-"), **kw)), base, name)
        fewer = self.build(out=self.dir / "fewer", per_family=1)
        self.assertEqual(fewer["adapter_sha"], base["adapter_sha"])
        self.assertNotEqual(fewer["eval_sha"], base["eval_sha"], "other ids are asked: another evaluation of the same adapter")
        self.assertEqual(fewer["built_from"]["eval"]["excluded"]["sha"], self.build(out=self.dir / "again")["built_from"]["eval"]["excluded"]["sha"], "the exclusion is the same: the ids alone make it another evaluation")

    def attach(self, cfg, where: pathlib.Path, ids=None) -> pathlib.Path:
        """The output of a session of build `cfg` under `where`: a finished adapter, and both arms' replies to the tasks `ids` (default: all it asks), the adapter arm right and the base arm wrong."""
        adapter = where / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_text("weights of " + cfg["adapter_sha"])
        (adapter / "train.json").write_text(json.dumps({"status": "done", "adapter_sha": cfg["adapter_sha"], "steps_done": 5, "steps_planned": 5}))
        for arm in lora_eval.ARMS:
            write_jsonl(where / f"replies-{arm}.jsonl", [line(t, [reply(t, arm == "lora", keys_of(cfg))]) for t in HELDOUT if t["id"] in (ids if ids is not None else cfg["built_from"]["eval"]["ids"])])
        return adapter

    def test_a_wider_resume_keeps_the_adapter_and_asks_the_evaluation_again(self):
        """The trigger of the second review, under the owner's decision: build with a narrower `--train-tasks`, run it, build again with the full file and `--resume-from`. The adapter is KEPT (the second
        review's fix refused it: 3 h of training lost), every task of the new evaluation is asked again, and nothing the earlier evaluation wrote is scored."""
        wide, narrow = self.both()
        inp, work = self.dir / "in", self.dir / "work"
        work.mkdir()
        adapter = self.attach(narrow, inp / "earlier")
        self.assertEqual(lora_eval.find_adapter([work, inp], wide["adapter_sha"]), adapter, "the adapter of the narrower build IS the wider build's adapter")
        ckpt = self.dir / "in2" / "ckpt" / "step-10"
        ckpt.mkdir(parents=True)
        (ckpt / "DONE").write_text(json.dumps({"step": 10, "adapter_sha": narrow["adapter_sha"]}))
        self.assertEqual(lora_eval.latest_checkpoint([self.dir / "in2"], wide["adapter_sha"]), ckpt, "and so is its checkpoint")
        tasks = [json.loads(x) for x in (self.dir / "wide" / "data" / "eval_tasks.jsonl").read_text().splitlines()]
        previous = {arm: run_teacher.load_previous(lora_eval.previous_files(arm, work, inp)) for arm in lora_eval.ARMS}
        self.assertIn(EXCLUDED, previous["lora"], "the earlier evaluation asked the task the wider build excludes")
        items = lora_eval.work_items(tasks, previous, self.kernel_cfg(wide))
        self.assertEqual(sorted(t["id"] for t, _ in items), wide["built_from"]["eval"]["ids"], "every task of the wider evaluation is asked again: its replies carry the narrower evaluation key")
        self.assertEqual(lora_eval.work_items([json.loads(x) for x in (self.dir / "narrow" / "data" / "eval_tasks.jsonl").read_text().splitlines()], previous, self.kernel_cfg(narrow)), [], "the narrower build itself has nothing left to ask")
        # the deadline comes before a single batch: the merged files hold the earlier replies of the tasks still asked, under their own keys, and the scorer scores none of them
        with unittest.mock.patch.object(lora_eval, "log", lambda *a: None):
            merged = {arm: lora_eval.merge_shards(work, arm, lora_eval.eval_ids(self.dir / "wide" / "data"), input_root=inp) for arm in lora_eval.ARMS}
        both_ask = sorted(set(wide["built_from"]["eval"]["ids"]) & set(narrow["built_from"]["eval"]["ids"]))
        self.assertEqual(merged, {"base": len(both_ask), "lora": len(both_ask)})
        self.assertNotIn(EXCLUDED, (work / "replies-lora.jsonl").read_text())
        r = score_lora.score(work / "replies-base.jsonl", work / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / "wide" / "data" / "lora-config.json"))
        self.assertEqual((r["paired"]["overall"]["n"], r["tasks_answered"], r["other_adapter"]["replies"]), (0, 0, 0))
        self.assertEqual(r["other_eval"], {"replies": 2 * len(both_ask), "by_arm": {"base": len(both_ask), "lora": len(both_ask)}}, "counted, printed, in no figure")
        # the session asks half of them again before the deadline: only those are scored
        again = both_ask[: len(both_ask) // 2]
        self.assertTrue(0 < len(again) < len(both_ask))
        for arm in lora_eval.ARMS:
            write_jsonl(lora_eval.shard_path(work, arm, 0), [line(t, [reply(t, arm == "lora", keys_of(wide))]) for t in HELDOUT if t["id"] in again])
        with unittest.mock.patch.object(lora_eval, "log", lambda *a: None):
            for arm in lora_eval.ARMS:
                lora_eval.merge_shards(work, arm, lora_eval.eval_ids(self.dir / "wide" / "data"), input_root=inp)
        r = score_lora.score(work / "replies-base.jsonl", work / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / "wide" / "data" / "lora-config.json"))
        self.assertEqual((r["paired"]["overall"]["n"], r["paired"]["overall"]["lora_only"], r["tasks_answered"]), (len(again), len(again), len(again)))
        self.assertEqual(r["other_eval"]["replies"], 2 * (len(both_ask) - len(again)))
        left = lora_eval.work_items(tasks, {arm: run_teacher.load_previous(lora_eval.previous_files(arm, work, inp)) for arm in lora_eval.ARMS}, self.kernel_cfg(wide))
        self.assertEqual(sorted(t["id"] for t, _ in left), sorted(set(wide["built_from"]["eval"]["ids"]) - set(again)), "the next resume asks what is still of the earlier evaluation, and the tasks never asked")

    def test_an_adapter_of_other_records_or_settings_is_still_refused(self):
        wide, _ = self.both()
        other_sft = self.build(sft=write_jsonl(self.dir / "other.jsonl", sft_rows(20)), out=self.dir / "other-sft")
        other_rank = self.build(sft=self.dir / "s.jsonl", out=self.dir / "other-rank", rank=8)
        for name, other in (("sft", other_sft), ("rank", other_rank)):
            self.assertEqual(other["built_from"]["eval"]["ids"], wide["built_from"]["eval"]["ids"], f"{name}: the very same evaluation set")
            self.assertEqual(other["built_from"]["eval"]["excluded"]["sha"], wide["built_from"]["eval"]["excluded"]["sha"])
            self.assertNotEqual(other["eval_sha"], wide["eval_sha"], f"{name}: the same set asked of another adapter is another evaluation")
            inp = self.dir / f"in-{name}"
            self.attach(other, inp / "earlier")
            with self.assertRaisesRegex(SystemExit, f"trained under adapter key '{other['adapter_sha']}', this build's is '{wide['adapter_sha']}'"):
                lora_eval.find_adapter([inp], wide["adapter_sha"])
            ckpt = inp / "ckpt" / "step-10"
            ckpt.mkdir(parents=True)
            (ckpt / "DONE").write_text(json.dumps({"step": 10, "adapter_sha": other["adapter_sha"]}))
            with self.assertRaisesRegex(SystemExit, "written under adapter key"):
                lora_eval.latest_checkpoint([inp / "ckpt"], wide["adapter_sha"])
            r = score_lora.score(inp / "earlier" / "replies-base.jsonl", inp / "earlier" / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / "wide" / "data" / "lora-config.json"))
            self.assertEqual((r["paired"]["overall"]["n"], r["other_adapter"]["replies"], r["other_eval"]["replies"]), (0, 2 * len(wide["built_from"]["eval"]["ids"]), 0), f"{name}: its replies are another adapter's")

    def test_an_adapter_that_is_kept_goes_into_this_sessions_output(self):
        """The next resume attaches THIS session's output: an adapter left in the attached one would not be there, and the kernel would train again."""
        wide, narrow = self.both()
        inp, work = self.dir / "in", self.dir / "work"
        work.mkdir()
        found = self.attach(narrow, inp / "earlier")
        (work / "adapter.tmp").mkdir()
        (work / "adapter.tmp" / "junk").write_text("a copy that was killed")
        kept = lora_eval.carry_adapter(found, work)
        self.assertEqual(kept, work / "adapter")
        self.assertEqual(sorted(p.name for p in kept.iterdir()), ["adapter_model.safetensors", "train.json"])
        self.assertEqual(((kept / "adapter_model.safetensors").read_text(), json.loads((kept / "train.json").read_text())), ((found / "adapter_model.safetensors").read_text(), json.loads((found / "train.json").read_text())))
        self.assertFalse((work / "adapter.tmp").exists())
        self.assertTrue(found.exists(), "the attached output is read-only: nothing is moved out of it")
        self.assertEqual(lora_eval.find_adapter([work, inp], wide["adapter_sha"]), kept, "this session's output comes first")
        self.assertEqual(lora_eval.find_adapter([work], wide["adapter_sha"]), kept, "and the next session, which has only this output attached, finds it")
        (kept / "train.json").write_text(json.dumps({"status": "done", "adapter_sha": wide["adapter_sha"], "mark": 1}))
        self.assertEqual(lora_eval.carry_adapter(kept, work), kept)
        self.assertEqual(json.loads((kept / "train.json").read_text())["mark"], 1, "an adapter that is in the output already is left as it is")


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
        sha, asked = keys_of(self.cfg), self.cfg["built_from"]["eval"]["ids"]
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
        sha = keys_of(self.cfg)
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
                  "tasks_in_set": 3, **KEYS, "outside": {"replies": 0, "by_arm": {"base": 0, "lora": 0}, "tasks": []}, "other_adapter": {"replies": 0, "by_arm": {"base": 0, "lora": 0}}, "other_eval": {"replies": 0, "by_arm": {"base": 0, "lora": 0}}}
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


class ExportReserve(unittest.TestCase):
    """With `--export-gguf` the kernel ends the evaluation `export_reserve_s` before the session deadline, so what training leaves has to hold the export too (PR 34 third review, m1). The reviewer's clock
    simulation: the kit's own `train_deadline`, `step_can_start` and `can_start` on the plan's figures, every batch at its upper bound and the optimizer steps `slow` times as slow as assumed, which is
    what the reserve is for."""

    DEADLINE_H = 7.2           # a deadline both plans fit (the review's build: 8 h of quota, less the smoke run, less the margin)

    def plan(self, export, n=360, prompts=85):
        return build_lora.plan([2900] * n, [2000] * prompts, epochs=2.0, gpus=2, global_batch=16, train_hours=3.0, k=1, max_new_tokens=4096, eval_batch=8, export=export, rates=build_lora.RATES)

    def clock(self, p: dict, export: bool, slow: float, reserve=None) -> dict:
        """What `lora_eval.main` and its two workers do with the time, in seconds from the kernel's start."""
        r, deadline = p["rates"], self.DEADLINE_H * 3600
        t = r["setup_s"]
        stop = lora_eval.train_deadline(t, deadline, 3.0, p["eval_reserve_s"] if reserve is None else reserve)          # main: taken before the training worker loads the student
        t += r["load_s"]
        secs = []
        for _ in range(p["steps"]):
            if not lora_eval.step_can_start(t, stop, secs):
                break
            secs.append(slow * p["step_s"])
            t += secs[-1]
        eval_deadline = lora_eval.eval_deadline({"export_gguf": export, "export_reserve_s": p["export_reserve_s"]}, deadline)          # main's own function for LORA_DEADLINE (run in test_kaggle_lora_keys.KernelMain)
        t += r["load_s"]
        batch, asked, mean = p["eval_upper_s"] / (2 * p["eval_batches_per_process"]), 0, 0.0          # one arm's batch at the budget
        for _ in range(p["eval_batches_per_process"]):
            if not run_teacher.can_start(t, eval_deadline, 2 * mean):
                break
            t, asked, mean = t + 2 * batch, asked + 1, batch
        return {"steps": len(secs), "train_stop": stop, "batches": asked, "eval_end": t, "eval_deadline": eval_deadline, "export_fits": (not export) or t + p["export_reserve_s"] <= deadline}

    def test_the_reserve_holds_the_export_and_the_window_is_the_evaluations(self):
        plain, export = self.plan(False), self.plan(True)
        load = build_lora.RATES["load_s"]
        self.assertEqual((plain["export_reserve_s"], export["export_reserve_s"]), (0, round(build_lora.RATES["export_s"])))
        self.assertEqual(plain["eval_reserve_s"], plain["eval_window_s"])
        self.assertEqual(export["eval_reserve_s"], export["eval_window_s"] + export["export_reserve_s"], "the review: `eval_upper + load_s`, no export")
        self.assertEqual(export["eval_window_s"], plain["eval_window_s"])
        self.assertGreaterEqual(plain["eval_window_s"], plain["eval_upper_s"] + load)
        self.assertLess(plain["eval_window_s"], plain["eval_upper_s"] + load + 2, "the upper bound and the load, rounded up and not padded")
        exact = 2 * 6 * (8 * 2000 / build_lora.RATES["prefill_tok_per_s_per_gpu"] + 4096 * build_lora.RATES["gen_step_s"]) + load          # six batches per process and arm at the budget, and the load: 15472.27 s
        self.assertEqual((plain["eval_window_s"], round(exact)), (math.ceil(exact), math.ceil(exact) - 1), "rounded UP: a reserve rounded down would be a fraction of a second short of its own upper bound")
        for p, on in ((plain, False), (export, True)):
            deadline = self.DEADLINE_H * 3600
            stop = lora_eval.train_deadline(0.0, deadline, 99.0, p["eval_reserve_s"])
            self.assertGreaterEqual(lora_eval.eval_deadline({"export_gguf": on, "export_reserve_s": p["export_reserve_s"]}, deadline) - stop, p["eval_upper_s"] + load, "between the training stop and the evaluation's OWN deadline: its upper bound and its load")

    def test_steps_twice_as_slow_stop_training_early_enough_for_every_batch(self):
        for export in (False, True):
            p = self.plan(export)
            self.assertEqual((p["steps"], p["eval_batches_per_process"]), (45, 6))
            self.assertEqual(build_lora.fit_verdict(p, 8.0, self.DEADLINE_H), "", "the plan says `it fits`")
            on_time = self.clock(p, export, 1.0)
            self.assertEqual((on_time["steps"], on_time["batches"]), (45, 6), f"export {export}: at the assumed rates everything runs")
            slow = self.clock(p, export, 2.0)
            self.assertEqual(slow["batches"], 6, f"export {export}: every batch of the evaluation is asked (85 of 85 tasks; the review got 80 of 85 with the export)")
            self.assertLess(slow["steps"], 45, "the training is what gives way: a `partial` adapter, which the report says")
            self.assertGreater(slow["steps"], 0)
            self.assertLessEqual(slow["eval_end"], slow["eval_deadline"])
            self.assertTrue(slow["export_fits"], "and the export still has its reserve")

    def test_without_the_export_in_the_reserve_the_last_batch_is_lost(self):
        """The review's finding, kept as the control: the same clock with the reserve that forgot the export."""
        p = self.plan(True)
        old = self.clock(p, True, 2.0, reserve=p["eval_window_s"])
        self.assertEqual(old["batches"], 5, "80 of 85 tasks")
        self.assertEqual(old["train_stop"] - self.clock(p, True, 2.0)["train_stop"], p["export_reserve_s"], "training stopped the export's reserve too late")

    def test_a_step_is_not_started_when_the_mean_step_would_end_after_the_training_stop(self):
        self.assertTrue(lora_eval.step_can_start(99.0, 100.0, []), "the first step starts while the stop has not come")
        self.assertTrue(lora_eval.step_can_start(100.0, 100.0, []))
        self.assertFalse(lora_eval.step_can_start(100.1, 100.0, []))
        self.assertTrue(lora_eval.step_can_start(60.0, 100.0, [30.0, 50.0]), "the mean of the finished steps, 40 s, ends at the stop")
        self.assertFalse(lora_eval.step_can_start(60.1, 100.0, [30.0, 50.0]))
        self.assertFalse(lora_eval.step_can_start(60.0, 100.0, [30.0, 52.0]))
        import ast
        src = pathlib.Path(lora_eval.__file__).read_text()
        worker = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "train_worker"))
        self.assertIn("int(not step_can_start(time.time(), deadline, step_secs))", worker, "the training worker decides each step by it")
        self.assertNotIn("time.time() >= deadline", worker)

    def test_the_builder_says_what_the_evaluation_and_the_export_are_left(self):
        h = lambda s: f"{s / 3600:.1f} h"
        plain, export = self.plan(False), self.plan(True)
        self.assertEqual(build_lora.describe(plain, 8.0, 7.5)[-1], f"the kernel starts nothing after 7.5 h, and training stops early enough to leave {h(plain['eval_window_s'])} for the evaluation")
        self.assertEqual(build_lora.describe(export, 8.5, 7.5)[-1], f"the kernel starts nothing after 7.5 h, and training stops early enough to leave {h(export['eval_window_s'])} for the evaluation and 0.5 h after it for the GGUF export")


class ExportReserveBuilt(Kit):
    def test_the_kernel_is_given_the_reserve_with_the_export(self):
        plain, export = self.build(out=self.dir / "plain"), self.build(out=self.dir / "export", export_gguf=True)
        on_disk = json.loads((self.dir / "export" / "data" / "lora-config.json").read_text())
        self.assertEqual((on_disk["export_gguf"], on_disk["export_reserve_s"]), (True, round(build_lora.RATES["export_s"])))
        self.assertEqual(on_disk["eval_reserve_s"], plain["eval_reserve_s"] + on_disk["export_reserve_s"])
        self.assertEqual(on_disk["eval_reserve_s"], export["built_from"]["estimate"]["eval_reserve_s"])
        self.assertEqual((plain["adapter_sha"], plain["eval_sha"]), (export["adapter_sha"], export["eval_sha"]), "the export changes neither the adapter nor the evaluation")

    def test_training_hours_that_end_before_the_student_is_loaded_are_refused(self):
        for hours in (0.01, build_lora.RATES["load_s"] / 3600):
            with self.assertRaisesRegex(SystemExit, "the kernel's training clock starts before the student is loaded"):
                self.build(out=self.dir / "bad", train_hours=hours)
        self.assertFalse((self.dir / "bad").exists())
        self.build(out=self.dir / "ok", train_hours=(build_lora.RATES["load_s"] + 60) / 3600)


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

"""The REA runs across the `distill/` refactor, byte for byte.

`tests/golden/distill/expected/*.json` was written by the code of origin/dataset/phase-a (commit 9afd8aebd97fb6c80941c5db232720e009752491), before anything moved: the procedures are those of tests/distill_fixtures.py,
run by tests/golden_baseline.py in a scratch worktree of that commit (its docstring has the commands). `provenance.json` records the sha256 of every expected file and of what the baseline writes, and `Frozen` checks
them, so an expected file cannot be regenerated with the new code unnoticed. Here the
same procedures run on the new layout and the output must be the same text, in the same key order:

  Stage      the Modal sample stage on a scripted fake llama-server, plain / reasoning / thinking / quota, first against a server that fails a fifth of the second samples, then with --resume: the requests it
             sent (a digest of every body), the checkpoint it wrote (a digest), the records, the stats, what it printed
  Scorer     `kaggle-score` on the replies of the same server: records, stats, the command's output, a digest of its files
  Traces     every filter of the teacher's reasoning on crafted replies for every task, and the traces a task keeps under each cap
  Plan       sampling, token budget, slot, quota and cost line for a few inputs
  Training   `run_tag`, `batch_plan` and the record parser of the training script on a grid
  Kit        the files the Kaggle kit writes for the same tasks

THREE DELIBERATE CHANGES since phase A, the first two in the thinking-pilot fix round (#33), the third in its review round 12:
  1. `SYSTEM_REASONING` now tells the model that the last line gives every part of a multi-part question (the thinking pilot's antidebug replies had the number and not the technique: 6 of 6 rejected). It moved
     kit, score-reasoning, score-thinking, traces and the three stage files: with the phase-A prompt put back, every test below passes byte for byte (plain mode, whose prompt is `SYSTEM_RECIPE`, did not move at all:
     `stage-plain.json`, `plan.json` and `lora.json` are the phase-A files).
  2. A reasoning or thinking run's checkpoint carries `{prompt_sha, max_tokens}` with each reply (round 15, M1: a resume asks again a reply to another prompt, or cut off at a smaller budget). It moved the two
     `partial_sha256` of `stage-reasoning`, `stage-reasoning-quota` and `stage-thinking` and nothing else: the stats, the records, the requests and what was printed are the same, and the checkpoint with the third
     element of every reply removed has the digest it had before (checked when the change was made). `stage-plain.json` did not move: a default run keeps its two-element replies.
  3. `kaggle-score` skips the replies that carry no prompt fingerprint unless it is told `--keep-unchecked` (round 12, M1: nothing shows which prompt they answered), and counts them: `prompts` of the stats
     gained `lines`, `unchecked`, `unchecked_replies`, `unchecked_ids` and `kept_unchecked`, and the command prints a line about them. It moved `score-reasoning` and `score-thinking` and nothing else: the lines of
     their fixture are those of an older notebook, which the new code scores with `keep_unchecked` (`fx.score_artifacts(..., keep_unchecked=True)`; the baseline's call is the one without it); the records and
     the per-family tables are the same text.
After a deliberate change, `DISTILL_GOLDEN_UPDATE=1 python3 -m unittest tests.test_distill_golden` rewrites the expected files; read `git diff --stat tests/golden` before committing them.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

from . import distill_fixtures as fx
from .helpers import ROOT
from .test_rea_modal import ModalCase, StubbedSampleStage

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))
sys.path.insert(0, str(ROOT / "tools" / "kaggle"))

from rea_bench import kaggle_score  # noqa: E402
from .rea_bound import rsn  # noqa: E402


def expected(name: str) -> str:
    return (fx.GOLDEN / "expected" / name).read_text()


class Same(unittest.TestCase):
    def same(self, name: str, got):
        text = fx.dumps(got)
        if os.environ.get("DISTILL_GOLDEN_UPDATE"):
            (fx.GOLDEN / "expected" / name).write_text(text)
        want = expected(name)
        self.assertEqual(json.loads(text), json.loads(want), f"{name}: the output is not the one of the code before the refactor")
        self.assertEqual(text, want, f"{name}: the same values in another order or form")


class Stage(StubbedSampleStage, Same):
    def runTest(self):                  # the case is driven by `test_*` methods of the subclasses below
        pass

    def check(self, mode: str):
        self.same(f"stage-{mode}.json", fx.stage_artifacts(self, mode))


def stage_test(mode):
    return lambda self: self.check(mode)


for _mode in fx.MODES:
    setattr(Stage, f"test_{_mode.replace('-', '_')}", stage_test(_mode))
del Stage.runTest


class Scorer(Same):
    def test_reasoning(self):
        self.same("score-reasoning.json", fx.score_artifacts(kaggle_score, "reasoning", keep_unchecked=True))

    def test_thinking(self):
        self.same("score-thinking.json", fx.score_artifacts(kaggle_score, "thinking", keep_unchecked=True))


class Traces(Same):
    def test_every_filter_and_the_traces_a_task_keeps(self):
        self.same("traces.json", fx.trace_artifacts(rsn))


class Plan(Same):
    def test_sampling_budget_slot_quota_and_cost_line(self):
        self.same("plan.json", fx.plan_artifacts(rsn))


class Training(ModalCase, Same):
    def test_the_pure_functions_of_the_training_script(self):
        import distill.backends.modal_train as modal_train
        sys.modules.pop("distill.backends.modal_train", None)
        import distill.backends.modal_train as modal_train  # noqa: F811  (under the stub of `modal`, like every test of the training script)
        self.same("lora.json", fx.lora_artifacts(modal_train))

    def test_the_wrapper_has_the_names_modal_run_uses(self):
        sys.modules.pop("distill.backends.modal_train", None)
        sys.modules.pop("modal_lora", None)
        import distill.backends.modal_train as modal_train
        import modal_lora
        for name in ("app", "train", "to_gguf", "main", "run_tag", "batch_plan", "encode_record", "BASE", "DATASET", "MAX_LEN"):
            self.assertIs(getattr(modal_lora, name), getattr(modal_train, name), name)


class RunConfigAtTheEntrypoint(ModalCase):
    """`--run-config` of the sample stage makes the call to the GPU function that the flags of the same run make: the shipped configs are today's runs."""

    def stage_call(self, **kw):
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            remote, _run = self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", **kw)
        return remote.call_args

    def test_each_shipped_config_gives_the_call_of_its_flags(self):
        cfg = ROOT / "distill" / "configs"
        flags = {"plain": dict(model="codeforge-27b-heretic-q4"),
                 "reasoning": dict(model="codeforge-27b-heretic-q4", k=4, reasoning=True, max_len=3072, quota="structs=220,crc=220,antidebug=220,bounds=220,fsm=220,pktsum=80,encoding=80"),
                 "thinking": dict(model="codeforge-27b-heretic-q4", k=4, reasoning=True, thinking=True, max_tokens=4096, max_len=12288, quota="structs=10,crc=10",
                                  max_tokens_by_family="cipher=8192,magic=8192,password=8192")}
        for name, kw in flags.items():
            from_file, from_flags = self.stage_call(run_config=str(cfg / f"rea-{name}.json")), self.stage_call(**kw)
            self.assertIsNotNone(from_file, "the stage was called")
            self.assertEqual(from_file, from_flags, name)
        call = self.stage_call(run_config=str(cfg / "rea-thinking.json"))
        self.assertEqual((call.args[0], call.args[3], call.kwargs["thinking"], call.kwargs["max_tokens"], call.kwargs["max_len"], call.kwargs["sampling"]),
                         ("codeforge-27b-heretic-q4", 4, True, 4096, 12288, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}))
        self.assertEqual(call.kwargs["max_tokens_by_family"], {"cipher": 8192, "magic": 8192, "password": 8192}, "the pilot's cut-off families have their own budget")

    def test_a_flag_that_the_file_also_sets_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "do not also give --k"):
            self.stage_call(run_config=str(ROOT / "distill" / "configs" / "rea-thinking.json"), k=2)
        with self.assertRaisesRegex(SystemExit, "--run-config is for --stage sample"):
            import pathlib
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                self.run_main(pathlib.Path(d), stage="decompile", n=6, seed=7, family="crc", run_config=str(ROOT / "distill" / "configs" / "rea-plain.json"))


class Kit(Same):
    def test_the_files_the_kaggle_kit_writes(self):
        # Load the REA wrapper by path: a bare `import build_tasks` resolves to whichever kaggle dir another test put first on sys.path
        # (test_distill_core inserts distill/backends/kaggle, whose build() needs domain=).
        import importlib.util
        spec = importlib.util.spec_from_file_location("rea_kaggle_build_tasks", ROOT / "tools" / "kaggle" / "build_tasks.py")
        build_tasks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_tasks)
        self.same("kit.json", fx.kit_artifacts(build_tasks))


def provenance_problems(prov: dict, expected_dir) -> list:
    """What is wrong between provenance.json and the expected files of `expected_dir` (a list of sentences, empty when all is as recorded): the files, their sha256, and for each one that it is either the
    baseline's (`changed` null: same sha256 as what tests/golden_baseline.py writes against the old code) or changed on purpose with a reason."""
    expected_dir, files, bad = pathlib.Path(expected_dir), prov["files"], []
    present = sorted(p.name for p in expected_dir.iterdir())
    for name in sorted(set(present) - set(files)):
        bad.append(f"{name} is in expected/ and not in provenance.json")
    for name in sorted(set(files) - set(present)):
        bad.append(f"{name} is in provenance.json and not in expected/")
    for name, row in sorted(files.items()):
        if name not in present:
            continue
        now = hashlib.sha256((expected_dir / name).read_bytes()).hexdigest()
        if now != row["sha256"]:
            bad.append(f"{name} is not the recorded file: regenerated? Record the new hash and the reason in provenance.json, in the same commit")
        elif row["changed"] is None and row["sha256"] != row["phase_a_sha256"]:
            bad.append(f"{name} says it is unchanged since the baseline, and it is not")
        elif row["changed"] is not None and (row["sha256"] == row["phase_a_sha256"] or len(row["changed"]) <= 60):
            bad.append(f"{name} says it was changed on purpose, and it equals the baseline's, or does not say why (say why it changed)")
    return bad


class Frozen(unittest.TestCase):
    PROVENANCE = fx.GOLDEN / "provenance.json"

    def test_the_expected_files_come_from_the_code_before_the_refactor(self):
        """The inputs are frozen in the repository: the generator is not what the expected files are compared against."""
        tasks = fx.load_tasks()
        self.assertEqual((len(tasks), len({t["id"] for t in tasks})), (22, 22))
        self.assertEqual(sorted(p.name for p in (fx.GOLDEN / "expected").iterdir()),
                         ["kit.json", "lora.json", "plan.json", "score-reasoning.json", "score-thinking.json", "stage-plain.json", "stage-reasoning-quota.json", "stage-reasoning.json", "stage-thinking.json", "traces.json"])

    def test_the_expected_files_are_the_recorded_ones_and_the_unchanged_ones_are_the_baseline_s(self):
        """Round 15, m6: `DISTILL_GOLDEN_UPDATE=1` rewrites an expected file with the CURRENT code, so a test that compares the code with files it can regenerate proves nothing. provenance.json records, per file,
        its sha256 and the sha256 of what tests/golden_baseline.py writes when run against the code of the commit before the refactor. A file is either the baseline's (`changed` null) or changed on
        purpose, with the reason; regenerating one means editing that list in the same commit, where the diff shows it."""
        from . import golden_baseline
        prov = json.loads(self.PROVENANCE.read_text())
        self.assertRegex(prov["baseline"]["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(prov["baseline"]["commit"], golden_baseline.COMMIT)
        self.assertEqual(hashlib.sha256((fx.GOLDEN / "tasks.jsonl").read_bytes()).hexdigest(), prov["tasks_sha256"], "the frozen tasks are the ones the baseline ran on")
        self.assertEqual(provenance_problems(prov, fx.GOLDEN / "expected"), [])
        self.assertEqual(sorted(prov["files"]), sorted(golden_baseline.FILES), "the baseline writes every expected file, and nothing else")

    def test_a_regenerated_expected_file_is_caught(self):
        """What the check is for: a file rewritten with the new code, with or without its record being edited to match, and a changed file that claims to be the baseline's."""
        import shutil
        import tempfile
        prov = json.loads(self.PROVENANCE.read_text())
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            shutil.copytree(fx.GOLDEN / "expected", root / "expected")
            self.assertEqual(provenance_problems(prov, root / "expected"), [])
            (root / "expected" / "stage-plain.json").write_text((root / "expected" / "stage-plain.json").read_text().replace("m", "n", 1))      # `DISTILL_GOLDEN_UPDATE=1` after a change of the code
            got = provenance_problems(prov, root / "expected")
            self.assertEqual(len(got), 1)
            self.assertIn("stage-plain.json is not the recorded file", got[0])
            edited = json.loads(json.dumps(prov))
            edited["files"]["stage-plain.json"]["sha256"] = hashlib.sha256((root / "expected" / "stage-plain.json").read_bytes()).hexdigest()       # and the record edited to match, but not the claim
            got = provenance_problems(edited, root / "expected")
            self.assertEqual(len(got), 1)
            self.assertIn("stage-plain.json says it is unchanged since the baseline, and it is not", got[0])
            edited["files"]["stage-plain.json"]["changed"] = "I regenerated it"
            self.assertIn("say why it changed", provenance_problems(edited, root / "expected")[0], "a reason is a sentence, not a flag")
            (root / "expected" / "extra.json").write_text("{}")
            self.assertTrue(any("extra.json" in x for x in provenance_problems(prov, root / "expected")), "a file nobody recorded")
            (root / "expected" / "extra.json").unlink()
            (root / "expected" / "plan.json").unlink()
            self.assertTrue(any("plan.json" in x for x in provenance_problems(prov, root / "expected")), "a recorded file that is gone")

    def test_the_baseline_driver_and_the_fixtures_are_the_ones_the_docs_name(self):
        from . import golden_baseline
        src = (ROOT / "tests" / "golden_baseline.py").read_text()
        for part in (golden_baseline.COMMIT, "git worktree add --detach", "REGEN_OUT", "provenance.json", "tests/distill_fixtures.py"):
            self.assertIn(part, src)
        self.assertIn("golden_baseline.py", (ROOT / "tests" / "distill_fixtures.py").read_text())
        with self.assertRaisesRegex(SystemExit, "REGEN_OUT must name an existing directory"):
            with mock.patch.dict(os.environ, {"REGEN_OUT": "/nonexistent-golden-dir"}):
                golden_baseline.out_dir()


if __name__ == "__main__":
    unittest.main()

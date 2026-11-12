"""The REA runs across the `distill/` refactor, byte for byte.

`tests/golden/distill/expected/*.json` was written by the code of origin/dataset/phase-a, before anything moved (the procedures are those of tests/distill_fixtures.py, run in a scratch worktree of that commit). Here the
same procedures run on the new layout and the output must be the same text, in the same key order:

  Stage      the Modal sample stage on a scripted fake llama-server, plain / reasoning / thinking / quota, first against a server that fails a fifth of the second samples, then with --resume: the requests it
             sent (a digest of every body), the checkpoint it wrote (a digest), the records, the stats, what it printed
  Scorer     `kaggle-score` on the replies of the same server: records, stats, the command's output, a digest of its files
  Traces     every filter of the teacher's reasoning on crafted replies for every task, and the traces a task keeps under each cap
  Plan       sampling, token budget, slot, quota and cost line for a few inputs
  Training   `run_tag`, `batch_plan` and the record parser of the training script on a grid
  Kit        the files the Kaggle kit writes for the same tasks

TWO DELIBERATE CHANGES since phase A, both in the thinking-pilot fix round (#33):
  1. `SYSTEM_REASONING` now tells the model that the last line gives every part of a multi-part question (the thinking pilot's antidebug replies had the number and not the technique: 6 of 6 rejected). It moved
     kit, score-reasoning, score-thinking, traces and the three stage files: with the phase-A prompt put back, every test below passes byte for byte (plain mode, whose prompt is `SYSTEM_RECIPE`, did not move at all:
     `stage-plain.json`, `plan.json` and `lora.json` are the phase-A files).
  2. A reasoning or thinking run's checkpoint carries `{prompt_sha, max_tokens}` with each reply (round 15, M1: a resume asks again a reply to another prompt, or cut off at a smaller budget). It moved the two
     `partial_sha256` of `stage-reasoning`, `stage-reasoning-quota` and `stage-thinking` and nothing else: the stats, the records, the requests and what was printed are the same, and the checkpoint with the third
     element of every reply removed has the digest it had before (checked when the change was made). `stage-plain.json` did not move: a default run keeps its two-element replies.
After a deliberate change, `DISTILL_GOLDEN_UPDATE=1 python3 -m unittest tests.test_distill_golden` rewrites the expected files; read `git diff --stat tests/golden` before committing them.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

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
        self.same("score-reasoning.json", fx.score_artifacts(kaggle_score, "reasoning"))

    def test_thinking(self):
        self.same("score-thinking.json", fx.score_artifacts(kaggle_score, "thinking"))


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
                 "thinking": dict(model="codeforge-27b-heretic-q4", k=4, reasoning=True, thinking=True, max_tokens=4096, max_len=8192, quota="structs=10,crc=10",
                                  max_tokens_by_family="cipher=8192,magic=8192,password=8192")}
        for name, kw in flags.items():
            from_file, from_flags = self.stage_call(run_config=str(cfg / f"rea-{name}.json")), self.stage_call(**kw)
            self.assertIsNotNone(from_file, "the stage was called")
            self.assertEqual(from_file, from_flags, name)
        call = self.stage_call(run_config=str(cfg / "rea-thinking.json"))
        self.assertEqual((call.args[0], call.args[3], call.kwargs["thinking"], call.kwargs["max_tokens"], call.kwargs["max_len"], call.kwargs["sampling"]),
                         ("codeforge-27b-heretic-q4", 4, True, 4096, 8192, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}))
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


class Frozen(unittest.TestCase):
    def test_the_expected_files_come_from_the_code_before_the_refactor(self):
        """The inputs are frozen in the repository: the generator is not what the expected files are compared against."""
        tasks = fx.load_tasks()
        self.assertEqual((len(tasks), len({t["id"] for t in tasks})), (22, 22))
        self.assertEqual(sorted(p.name for p in (fx.GOLDEN / "expected").iterdir()),
                         ["kit.json", "lora.json", "plan.json", "score-reasoning.json", "score-thinking.json", "stage-plain.json", "stage-reasoning-quota.json", "stage-reasoning.json", "stage-thinking.json", "traces.json"])


if __name__ == "__main__":
    unittest.main()

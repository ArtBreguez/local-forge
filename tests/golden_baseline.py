"""The baseline of tests/test_distill_golden.py: the procedures of tests/distill_fixtures.py run against the code from BEFORE the `distill/` refactor, which is what wrote the expected files.

    commit  9afd8aebd97fb6c80941c5db232720e009752491   (origin/dataset/phase-a, the tip of PR #31: the Modal sample stage in `tools/modal/modal_rea_data.py`, `rea_bench.reasoning`, `rea_bench.kaggle_score`,
                                                         `modal_lora.py`, `tools/kaggle/build_tasks.py`; the merge-base of PR #32's branch, which moved all of it into `distill/`)

It is not part of the suite (its name does not match `test*.py`) and it cannot run in this tree: it needs the OLD modules. Run it in a scratch worktree of that commit:

    git worktree add --detach /tmp/phase-a 9afd8aebd97fb6c80941c5db232720e009752491
    cp tests/distill_fixtures.py tests/golden_baseline.py /tmp/phase-a/tests/
    mkdir -p /tmp/phase-a/tests/golden/distill /tmp/old-out && cp tests/golden/distill/tasks.jsonl /tmp/phase-a/tests/golden/distill/
    (cd /tmp/phase-a && REGEN_OUT=/tmp/old-out python3 -m unittest tests.golden_baseline)       # 7 tests, about 1 s; prints the sha256 of the 10 files it wrote
    git worktree remove --force /tmp/phase-a

`tests/golden/distill/provenance.json` records, per expected file, the sha256 this driver gives (`phase_a_sha256`) and the sha256 of the file in the repository (`sha256`); the two are equal for
every file the code has not deliberately changed since (`changed` is null), and `Frozen` in tests/test_distill_golden.py checks the list, so regenerating an expected file with the new code
(`DISTILL_GOLDEN_UPDATE=1`) cannot pass unnoticed: the claim "compared with the code before the move" would be "compared with itself". Reproduced by hand in review round 11: all 10 files were
byte-identical to PR #32's expected files.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import sys
import unittest

from . import distill_fixtures as fx
from .helpers import ROOT
from .test_rea_modal import ModalCase, StubbedSampleStage

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))
sys.path.insert(0, str(ROOT / "tools" / "kaggle"))
from rea_bench import kaggle_score, reasoning as rsn  # noqa: E402

COMMIT = "9afd8aebd97fb6c80941c5db232720e009752491"
FILES = ("kit.json", "lora.json", "plan.json", "score-reasoning.json", "score-thinking.json", "stage-plain.json", "stage-reasoning-quota.json", "stage-reasoning.json", "stage-thinking.json", "traces.json")


def out_dir() -> pathlib.Path:
    """Where the files go: $REGEN_OUT, which must exist (SystemExit with the usage when it does not: the driver writes nowhere by default)."""
    path = os.environ.get("REGEN_OUT")
    if not path or not pathlib.Path(path).is_dir():
        raise SystemExit(f"REGEN_OUT must name an existing directory for the files; see the docstring of {__file__} (it runs in a worktree of {COMMIT}, not here)")
    return pathlib.Path(path)


def write(name: str, got) -> None:
    (out_dir() / name).write_text(fx.dumps(got))


class Stage(StubbedSampleStage):
    """The Modal sample stage of the old layout (`modal_rea_data.sample_answers`), one test per mode."""

    def runTest(self):
        pass


def stage_writer(mode):
    return lambda self: write(f"stage-{mode}.json", fx.stage_artifacts(self, mode))


for _m in fx.MODES:
    setattr(Stage, f"test_{_m.replace('-', '_')}", stage_writer(_m))
del Stage.runTest


class Scorer(unittest.TestCase):
    """`rea_bench.kaggle_score`, `rea_bench.reasoning` (the filters, the plan): the old homes of what `distill.score_replies`, `distill.traces` and `distill.config` are now."""

    def test_all(self):
        write("score-reasoning.json", fx.score_artifacts(kaggle_score, "reasoning"))
        write("score-thinking.json", fx.score_artifacts(kaggle_score, "thinking"))
        write("traces.json", fx.trace_artifacts(rsn))
        write("plan.json", fx.plan_artifacts(rsn))


class Training(ModalCase):
    def test_lora(self):
        sys.modules.pop("modal_lora", None)
        import modal_lora
        write("lora.json", fx.lora_artifacts(modal_lora))


class Kit(unittest.TestCase):
    def test_kit(self):
        spec = importlib.util.spec_from_file_location("rea_kaggle_build_tasks", ROOT / "tools" / "kaggle" / "build_tasks.py")
        build_tasks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_tasks)
        write("kit.json", fx.kit_artifacts(build_tasks))


def tearDownModule() -> None:
    """The ten files, and their sha256 for provenance.json: a file missing here means a procedure of tests/distill_fixtures.py was not run."""
    out = out_dir()
    names = sorted(p.name for p in out.iterdir())
    for name in names:
        print(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")
    if names != sorted(FILES):
        raise SystemExit(f"expected exactly {sorted(FILES)}, wrote {names}")

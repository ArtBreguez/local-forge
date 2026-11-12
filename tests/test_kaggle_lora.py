"""The Kaggle LoRA + held-out evaluation kit (`distill/backends/kaggle/`: build_lora.py, lora_eval.py, score_lora.py) and the `distill/sft.py` it shares with the Modal trainer.

Nothing here touches a GPU, torch, transformers or Kaggle: the kernel's decisions (steps, resume, deadline, sharding, parsing of a generation) are pure functions, and the builder and the scorer are run on the
arithmetic fixture domain, whose held-out ids are `arith-9999-*`. The REA domain is exercised where it differs (its held-out loader, its scorer)."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT

KIT = ROOT / "distill" / "backends" / "kaggle"
RB = ROOT / "tools" / "rea-bench"
for p in (RB, KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from distill import config, outputs, records, score_replies, sft, teachers, traces  # noqa: E402
from distill.backends.kaggle import build_lora, score_lora  # noqa: E402
from distill.domains.example_arith import DOMAIN as ARITH, HELDOUT_SEED, make_tasks  # noqa: E402
import lora_eval  # noqa: E402
import run_teacher  # noqa: E402

THINK = "<think>\n{}\n</think>\n\nAnswer: {}"
HELDOUT = make_tasks(HELDOUT_SEED)         # 12 tasks, kinds add / sub / mul four times each
TRAIN = make_tasks(7, 30)                  # the training tasks of the fixture (the SFT records are made from these)
SHARED = {"arith-9999-003", "arith-9999-004"}      # the two held-out problems that seed 7 draws too (same question, same answer): the fixture's own overlap, found by content, never by id
KEPT = [t for t in HELDOUT if t["id"] not in SHARED]
TRAIN_BY_ID = {t["id"]: t for t in TRAIN}
PAIR = [json.loads(line) for line in (ROOT / "tests" / "golden" / "kaggle_lora" / "structs-pair.jsonl").read_text().splitlines()]       # structs-7-0292 and structs-1001-0028: the same function, question and answer under two seeds
TEACHER = "codeforge-27b-heretic-q4"


def row(task: dict, source: str = "teacher-reasoning", label: str = TEACHER, text: str = None) -> dict:
    if source == "teacher-reasoning":
        msgs = ARITH.messages(task, "thinking") + [{"role": "assistant", "content": text or THINK.format("work it out " * 20, task["answer"])}]
    else:
        msgs = ARITH.messages(task, "plain") + [{"role": "assistant", "content": text or str(task["answer"])}]
    return {"messages": msgs, "task": task["id"], "kind": task["kind"], "model": label, "target_source": source}


def sft_rows(n: int = 30, **kw) -> list:
    return [r for t in make_tasks(7, n) for r in (row(t, **kw), row(t, "truth-template"))]


def write_jsonl(path, rows) -> pathlib.Path:
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return pathlib.Path(path)


class Kit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        self.sft = write_jsonl(self.dir / "sft.jsonl", sft_rows())
        self.heldout = write_jsonl(self.dir / "heldout.jsonl", HELDOUT)
        self.train_tasks = write_jsonl(self.dir / "train-tasks.jsonl", TRAIN)

    def build(self, **kw):
        """`min_family` 1: the fixture has four held-out tasks per family, under the floor of 5 that a real file is judged by (the tests of the floor say so)."""
        return build_lora.build(kw.pop("sft", self.sft), kw.pop("heldout", [self.heldout]), kw.pop("out", self.dir / "out"), kw.pop("owner", "someone"), domain=ARITH,
                                **{"train_tasks": [self.train_tasks], "per_family": 3, "min_family": 1, **kw})


class Builder(Kit):
    def test_it_writes_the_dataset_and_the_kernel(self):
        cfg = self.build()
        data, kernel = self.dir / "out" / "data", self.dir / "out" / "kernel"
        self.assertEqual(sorted(p.name for p in data.iterdir()), sorted(["train.jsonl", "eval_tasks.jsonl", "lora-config.json", "dataset-metadata.json", *build_lora.kernel_files()]))
        self.assertFalse([p for p in data.iterdir() if p.is_dir()], "the Kaggle CLI skips the sub-folders of a dataset: the files are flat")
        meta = json.loads((kernel / "kernel-metadata.json").read_text())
        self.assertEqual((meta["id"], meta["code_file"], meta["kernel_type"], meta["enable_gpu"], meta["enable_internet"], meta["is_private"]), ("someone/example-arith-lora-eval", "run.py", "script", True, True, True))
        self.assertEqual((meta["dataset_sources"], meta["kernel_sources"]), (["someone/example-arith-lora-data"], []))
        self.assertEqual(json.loads((data / "dataset-metadata.json").read_text())["id"], "someone/example-arith-lora-data")
        self.assertEqual((kernel / "run.py").read_text(), build_lora.LAUNCHER)
        self.assertEqual(json.loads((data / "lora-config.json").read_text())["student"], config.DEFAULT_STUDENT)
        self.assertEqual(cfg["student"], "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic")

    def test_the_kernel_runs_the_repositorys_files_byte_for_byte(self):
        self.build()
        data = self.dir / "out" / "data"
        for name, src in build_lora.kernel_files().items():
            self.assertEqual((data / name).read_bytes(), src.read_bytes(), name)
        self.assertEqual(build_lora.kernel_files()["distill.sft.py"], ROOT / "distill" / "sft.py", "the Modal trainer's record code, not a copy of it")
        self.assertEqual(build_lora.kernel_files()["run_teacher.py"], KIT / "run_teacher.py")

    def test_the_launcher_imports_the_kernel_by_the_dataset_not_by_a_path_of_its_own(self):
        ast.parse(build_lora.LAUNCHER)
        self.assertIn('rglob("lora_eval.py")', build_lora.LAUNCHER)
        self.assertIn("lora_eval.main()", build_lora.LAUNCHER)

    def test_the_evaluation_tasks_carry_the_messages_and_never_the_truth(self):
        self.build()
        lines = [json.loads(line) for line in (self.dir / "out" / "data" / "eval_tasks.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 9)
        for line, t in zip(lines, config.apply_quota(KEPT, {"add": 3, "mul": 3, "sub": 3})):
            self.assertEqual(sorted(line), ["id", "kind", "messages"])
            self.assertEqual(line["messages"], ARITH.messages(t, "thinking"), "the prompt of the training records")
        self.assertNotIn("answer", (self.dir / "out" / "data" / "eval_tasks.jsonl").read_text().replace("Answer:", ""))
        self.assertTrue(all(line["id"] in ARITH.heldout_ids() for line in lines))

    def test_the_default_subset_is_the_families_of_the_training_records_per_family(self):
        only_add = [r for r in sft_rows() if r["kind"] == "add"]
        cfg = self.build(sft=write_jsonl(self.dir / "add.jsonl", only_add), per_family=2)
        self.assertEqual(cfg["built_from"]["eval"]["by_kind"], {"add": 2})
        cfg = self.build(quota="sub=1,mul=2", out=self.dir / "q")
        self.assertEqual(cfg["built_from"]["eval"]["by_kind"], {"mul": 2, "sub": 1})

    def test_the_config_is_one_the_kernel_accepts(self):
        cfg = self.build(k=2, max_len=3072, rank=8, epochs=1.5, mix=True, export_gguf=True, train_hours=2.0)
        got = lora_eval.resolve_config({k: v for k, v in json.loads((self.dir / "out" / "data" / "lora-config.json").read_text()).items() if k != "built_from"})
        self.assertEqual((got["k"], got["max_len"], got["rank"], got["epochs"], got["mix"], got["export_gguf"], got["train_hours"]), (2, 3072, 8, 1.5, True, True, 2.0))
        self.assertEqual(got["sampling"], config.THINKING_SAMPLING)
        self.assertEqual(cfg["built_from"]["train"]["labels"], [TEACHER])
        self.assertEqual(cfg["built_from"]["train"]["by_source"], {"teacher-reasoning": 30, "truth-template": 30})

    def test_a_build_does_not_overwrite_an_earlier_one_without_force(self):
        self.build()
        with self.assertRaises(SystemExit):
            self.build()
        self.build(force=True)

    def test_slugs_and_resume_from_are_checked_and_attached(self):
        for bad in ({"owner": "Some One"}, {"dataset_slug": "UP"}, {"kernel_slug": "a b"}, {"resume_from": "no-slash"}):
            with self.assertRaises(SystemExit, msg=str(bad)):
                self.build(out=self.dir / "bad", **bad)
        self.build(resume_from="someone/earlier-run", out=self.dir / "r")
        self.assertEqual(json.loads((self.dir / "r" / "kernel" / "kernel-metadata.json").read_text())["kernel_sources"], ["someone/earlier-run"])

    def test_the_command_prints_the_plan_and_what_comes_next(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_lora.main(["--domain", "example_arith", "--sft", str(self.sft), "--train-tasks", str(self.train_tasks), "--heldout", str(self.heldout), "--owner", "someone", "--out", str(self.dir / "cli"),
                             "--per-family", "3", "--min-family", "1"])
        text = buf.getvalue()
        for needle in ("60 training records", "9 held-out tasks", "excluded from the evaluation: 2 of 12", "training:", "eval:", "overhead:", "total:", "it fits", "kaggle datasets create", "kaggle kernels push", "GPU T4 x2"):
            self.assertIn(needle, text)

    def test_a_smoke_build_is_marked_and_measures_instead_of_capping(self):
        cfg = self.build(smoke=True)
        self.assertTrue(cfg["smoke"])
        self.assertEqual(cfg["max_steps"], 0)
        self.assertTrue(cfg["built_from"]["estimate"]["smoke"])
        self.assertIn("smoke run:", cfg["plan_lines"][0])


class Refusals(Kit):
    def test_a_training_record_of_a_held_out_task_is_refused(self):
        leak = sft_rows() + [row(HELDOUT[0])]
        with self.assertRaisesRegex(SystemExit, "held-out tasks .* are in the training records.*arith-9999-000"):
            self.build(sft=write_jsonl(self.dir / "leak.jsonl", leak))
        self.assertFalse((self.dir / "out").exists(), "nothing is written before the checks pass")

    def test_a_training_record_of_a_task_of_the_evaluation_subset_is_refused(self):
        eval_tasks = make_tasks(7, 30)[:2]
        with self.assertRaisesRegex(SystemExit, "2 training record task.*evaluation subset"):
            build_lora.check_training(sft_rows(), eval_tasks, domain=ARITH)
        build_lora.check_training(sft_rows(), [HELDOUT[0]], domain=ARITH)

    def test_a_held_out_file_that_holds_tasks_that_are_not_held_out_is_refused(self):
        not_held_out = write_jsonl(self.dir / "t.jsonl", make_tasks(7, 6))
        with self.assertRaisesRegex(SystemExit, "not held-out ids"):
            build_lora.pick_eval([not_held_out], "", 3, ["add"], domain=ARITH, train_tasks=TRAIN_BY_ID)

    def test_a_teacher_that_may_only_filter_is_refused_whatever_the_spelling(self):
        for label in ("gpt-oss-120b", "gpt_oss_20b", "ArliAI-gpt-oss-120b-Derestricted", "claude-opus-5-5", "Claude Sonnet", "gpt-4o"):
            with self.assertRaisesRegex(SystemExit, "may only FILTER|stores the teacher's reply", msg=label):
                build_lora.check_training([r for t in make_tasks(7, 4) for r in (row(t, label=label), row(t, "truth-template"))], [], domain=ARITH)

    def test_the_rule_is_the_one_of_teachers_py(self):
        with mock.patch.object(teachers, "require_reasoning", wraps=teachers.require_reasoning) as rule:
            build_lora.check_training(sft_rows(4), [], domain=ARITH)
        self.assertEqual([c.args for c in rule.call_args_list], [(TEACHER,)])

    def test_a_truth_template_record_may_come_from_a_filter_only_label_for_it_stores_nothing_of_the_model(self):
        rows = [r for t in make_tasks(7, 4) for r in (row(t), row(t, "truth-template", label="gpt-oss-120b"))]
        self.assertEqual(build_lora.check_training(rows, [], domain=ARITH)["labels"], [TEACHER])

    def test_a_file_with_no_thinking_record_is_refused(self):
        plain = [row(t, "truth-template") for t in make_tasks(7, 6)]
        with self.assertRaisesRegex(SystemExit, "no thinking record"):
            build_lora.check_training(plain, [], domain=ARITH)

    def test_a_file_that_is_not_an_sft_file_is_refused(self):
        bad = [{**sft_rows(1)[0], "teacher_headline": "x"}]
        for rows in (bad, [{**sft_rows(1)[0], "target_source": "magic"}], [{**sft_rows(1)[0], "messages": [{"role": "user", "content": "x"}]}], []):
            with self.assertRaises(SystemExit):
                build_lora.read_sft(write_jsonl(self.dir / "bad.jsonl", rows))
        (self.dir / "junk.jsonl").write_text("{not json\n")
        with self.assertRaisesRegex(SystemExit, "not JSON"):
            build_lora.read_sft(self.dir / "junk.jsonl")

    def test_the_fates_follow_mix(self):
        self.assertEqual(build_lora.check_training(sft_rows(5), [], domain=ARITH)["fates"], {"think": 5, "skipped_plain": 5})
        self.assertEqual(build_lora.check_training(sft_rows(5), [], domain=ARITH, mix=True)["fates"], {"think": 5, "plain": 5})
        mal = sft_rows(2) + [row(make_tasks(7, 3)[2], text="<think>never closed")]
        self.assertEqual(build_lora.check_training(mal, [], domain=ARITH)["fates"]["malformed"], 1)

    def test_a_quota_that_names_no_family_of_the_subject_or_selects_nothing_is_refused(self):
        with self.assertRaises(SystemExit):
            build_lora.pick_eval([self.heldout], "add=0", 3, [], domain=ARITH, train_tasks=TRAIN_BY_ID)
        none = write_jsonl(self.dir / "mul.jsonl", [t for t in HELDOUT if t["kind"] == "mul"])
        with self.assertRaisesRegex(SystemExit, "no held-out task"):
            build_lora.pick_eval([none], "add=3", 3, [], domain=ARITH, train_tasks=TRAIN_BY_ID)


class Estimate(unittest.TestCase):
    R = build_lora.RATES
    PLAN = dict(epochs=2.0, gpus=2, global_batch=16, train_hours=3.0, k=1, max_new_tokens=4096, eval_batch=8, export=False, rates=build_lora.RATES)

    def plan(self, n=1200, tokens=3000, prompts=70, **kw):
        return build_lora.plan([tokens] * n, [2000] * prompts, **{**self.PLAN, **kw})

    def test_the_training_budget_caps_the_steps(self):
        p = self.plan()
        step_s = 16 * 3000 / (2 * 220)
        self.assertEqual((p["steps_per_epoch"], p["steps_full"]), (75, 150))
        self.assertAlmostEqual(p["step_s"], step_s, delta=0.1)
        self.assertEqual((p["steps"], p["max_steps"]), (int(10800 // step_s), int(10800 // step_s)))
        self.assertLessEqual(p["train_s"], 3 * 3600)
        self.assertLess(p["epochs_effective"], 2.0)

    def test_a_small_set_is_trained_for_every_epoch_and_the_budget_does_not_bind(self):
        p = self.plan(n=300, tokens=1500)
        self.assertEqual((p["steps"], p["steps_full"], p["max_steps"]), (38, 38, 0))
        self.assertEqual(p["epochs_effective"], 2.0)

    def test_the_evaluation_is_batches_of_the_budget_upper_bound(self):
        p = self.plan()
        self.assertEqual((p["eval_items_per_arm"], p["eval_batches_per_process"]), (70, 5))
        batch = 8 * 2000 / 660 + 4096 * 0.12
        self.assertAlmostEqual(p["eval_upper_s"], 2 * 5 * batch, delta=2)
        self.assertLess(p["eval_typical_s"], p["eval_upper_s"])
        self.assertGreater(self.plan(k=2)["eval_upper_s"], 1.9 * p["eval_upper_s"] - 2 * batch * 2)

    def test_the_default_sizes_fit_three_hours_of_training_and_the_quota_that_is_left(self):
        p = self.plan()
        self.assertLessEqual(p["train_s"], 3 * 3600)
        self.assertLess(p["total_upper_s"], 8 * 3600, "the assumed figures: ~5 h of the 8 h")
        self.assertEqual(p["overhead_s"], round(self.R["setup_s"] + 2 * self.R["load_s"]))
        self.assertEqual(p["eval_reserve_s"], round(p["eval_upper_s"] + self.R["load_s"]))
        self.assertGreater(self.plan(export=True)["total_upper_s"], p["total_upper_s"])
        self.assertEqual(self.plan(export=True)["export_reserve_s"], round(self.R["export_s"]))

    def test_the_verdict_says_whether_it_fits(self):
        p = self.plan()
        self.assertIn("it fits", build_lora.describe(p, 8.0, 7.5)[-2])
        self.assertIn("IT DOES NOT FIT", build_lora.describe(p, 2.0, 7.5)[-2])
        self.assertIn("starts nothing after 7.5 h", build_lora.describe(p, 8.0, 7.5)[-1])
        self.assertIn("assumed", build_lora.describe(p, 8.0, 7.5)[0])

    def test_a_measured_run_replaces_the_assumed_rates(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "run.json"
            path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": 400, "gen_step_s": 0.08, "gen_peak_gib": 9}}))
            rates = build_lora.rates_from(path)
            self.assertEqual((rates["train_tok_per_s_per_gpu"], rates["gen_step_s"], rates["setup_s"]), (400.0, 0.08, self.R["setup_s"]))
            faster = build_lora.plan([3000] * 1200, [2000] * 70, **{**self.PLAN, "rates": rates})
            self.assertLess(faster["step_s"], self.plan()["step_s"])
            self.assertIn("--measured", build_lora.describe(faster, 8.0, 7.5)[0])
            path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": -1}}))
            self.assertEqual(build_lora.rates_from(path)["train_tok_per_s_per_gpu"], self.R["train_tok_per_s_per_gpu"], "a number that is not positive is not a measurement")
            for bad in ("[]", "{}", "not json"):
                path.write_text(bad)
                with self.assertRaises(SystemExit):
                    build_lora.rates_from(path)

    def test_records_over_the_window_do_not_count(self):
        short, long = row(make_tasks(7, 1)[0]), row(make_tasks(7, 1)[0], text=THINK.format("x " * 20000, 1))
        self.assertEqual(len(build_lora.trainable_lengths([short, long], 4096, False)), 1)
        self.assertEqual(len(build_lora.trainable_lengths([short, long], 40000, False)), 2)
        truth = row(make_tasks(7, 1)[0], "truth-template")
        self.assertEqual(len(build_lora.trainable_lengths([short, truth], 4096, False)), 1)
        self.assertEqual(len(build_lora.trainable_lengths([short, truth], 4096, True)), 2)

    def test_the_smoke_plan_is_a_short_run(self):
        s = build_lora.smoke_plan(4000, gpus=2, steps=4, max_new_tokens=256, rates=self.R)
        self.assertLess(s["total_upper_s"], 1.5 * 3600)
        self.assertIn("smoke run", build_lora.describe(s, 8.0, 7.5)[0])


class Training(unittest.TestCase):
    def test_the_learning_rate_warms_up_then_decays_by_a_cosine(self):
        lrs = [lora_eval.lr_at(s, 100, 1e-3) for s in range(100)]
        self.assertAlmostEqual(lrs[0], 1e-3 / 5)
        self.assertAlmostEqual(lrs[4], 1e-3)
        self.assertTrue(all(a >= b for a, b in zip(lrs[4:], lrs[5:])), "never up again after the warmup")
        self.assertLess(lrs[-1], 1e-5)
        self.assertAlmostEqual(lora_eval.lr_at(52, 100, 1e-3), 1e-3 * 0.5 * (1 + math.cos(math.pi * 47 / 95)))
        self.assertEqual(lora_eval.lr_at(0, 1, 1e-3), 1e-3)

    def test_the_steps_are_the_epochs_unless_the_cap_binds(self):
        self.assertEqual(lora_eval.total_steps(100, 2.0, 16), 13)
        self.assertEqual(lora_eval.total_steps(100, 2.0, 16, cap=5), 5)
        self.assertEqual(lora_eval.total_steps(100, 2.0, 16, cap=50), 13)
        self.assertEqual(lora_eval.total_steps(3, 0.1, 16), 1)

    def test_a_step_is_the_same_on_every_process_and_on_a_resume(self):
        n, g = 20, 8
        stream = [i for step in range(5) for i in lora_eval.step_indices(n, g, step, 7)]
        self.assertEqual(stream, [i for step in range(5) for i in lora_eval.step_indices(n, g, step, 7)])
        self.assertEqual(sorted(stream[:20]), list(range(20)), "an epoch is a permutation: every example once")
        self.assertEqual(sorted(stream[20:40]), list(range(20)))
        self.assertNotEqual(stream[:20], stream[20:40], "another epoch, another order")
        self.assertNotEqual(lora_eval.step_indices(n, g, 0, 7), lora_eval.step_indices(n, g, 0, 8))
        crossing = lora_eval.step_indices(n, g, 2, 7)           # examples 16..23: the end of epoch 0 and the start of epoch 1
        self.assertEqual(crossing, list(lora_eval.epoch_perm(n, 0, 7))[16:] + list(lora_eval.epoch_perm(n, 1, 7))[:4])

    def test_the_processes_share_a_step_without_overlap(self):
        step = lora_eval.step_indices(20, 16, 1, 7)
        halves = [lora_eval.rank_slice(step, r, 2) for r in range(2)]
        self.assertEqual(sorted(halves[0] + halves[1]), sorted(step))
        self.assertEqual([len(h) for h in halves], [8, 8])
        self.assertEqual(lora_eval.rank_slice(step, 0, 1), step)

    def test_the_loss_is_computed_in_slices(self):
        self.assertEqual(lora_eval.chunk_bounds(1100, 512), [(0, 512), (512, 1024), (1024, 1100)])
        self.assertEqual(lora_eval.chunk_bounds(0, 512), [])
        self.assertEqual(lora_eval.chunk_bounds(512, 512), [(0, 512)])

    def test_training_stops_in_time_for_the_evaluation(self):
        self.assertEqual(lora_eval.train_deadline(1000, 1000 + 7.5 * 3600, 3.0, 4000), 1000 + 3 * 3600)
        self.assertEqual(lora_eval.train_deadline(1000, 1000 + 4 * 3600, 3.0, 4000), 1000 + 4 * 3600 - 4000, "the evaluation's reserve comes off the session first")

    def test_a_smoke_run_trains_on_the_longest_records(self):
        ex = [{"input_ids": [0] * n} for n in (5, 50, 20, 40, 10)]
        self.assertEqual([len(e["input_ids"]) for e in lora_eval.smoke_examples(ex, 2, 2)], [50, 40, 20, 10])
        self.assertEqual(len(lora_eval.smoke_examples(ex, 4, 4)), 5)


class Tok:
    """A tokenizer in the shape of Qwen3.5's chat template: the generation prompt ends in `<think>\\n` (thinking) or the empty block; words and newlines are the tokens."""
    eos_token = "<|im_end|>"

    def __init__(self):
        self.ids = {}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert not tokenize and add_generation_prompt
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.ids.setdefault(w, len(self.ids) + 1) for w in re.findall(r"<\|[a-z_]+\|>|</?think>|[^\s<]+|\n|<", text)]}


class Records(unittest.TestCase):
    def test_a_record_is_encoded_by_the_modal_trainers_code(self):
        rows = sft_rows(6)
        tok = Tok()
        examples, counts = lora_eval.encode_records(tok, rows, 4096, mix=False)
        self.assertEqual((len(examples), counts["think"], counts["skipped_plain"], counts["plain"]), (6, 6, 6, 0))
        same_tok = Tok()          # ids are numbered by first sight: the same tokenizer, the same order
        direct = [sft.encode_record(same_tok, r["messages"], 4096, True, False)[0] for r in rows]
        self.assertEqual(examples, [d for d in direct if d])
        ex = examples[0]
        first_target = next(i for i, x in enumerate(ex["labels"]) if x != -100)
        self.assertTrue(all(x == -100 for x in ex["labels"][:first_target]), "the prompt is not trained")
        self.assertEqual(ex["labels"][first_target:], ex["input_ids"][first_target:])
        self.assertEqual(ex["input_ids"][-1], tok.ids["<|im_end|>"], "the target ends with the end of turn")

    def test_mix_trains_the_truth_records_with_an_empty_think_block(self):
        examples, counts = lora_eval.encode_records(Tok(), sft_rows(4), 4096, mix=True)
        self.assertEqual((len(examples), counts["think"], counts["plain"]), (8, 4, 4))

    def test_a_malformed_record_is_dropped_and_a_long_one_is_dropped_not_cut(self):
        rows = sft_rows(3) + [row(make_tasks(7, 4)[3], text="<think>never closed")]
        examples, counts = lora_eval.encode_records(Tok(), rows, 4096, mix=False)
        self.assertEqual((len(examples), counts["malformed"]), (3, 1))
        examples, counts = lora_eval.encode_records(Tok(), sft_rows(3), 20, mix=False)
        self.assertEqual((examples, counts["too_long"]), ([], 3))

    def test_the_trainers_share_one_module(self):
        self.assertIs(lora_eval.sft, sft)
        for name in ("encode_ids", "parse_assistant", "record_fate", "render_record", "encode_record", "fate_counts", "is_thinking_config", "LORA_TARGETS"):
            self.assertTrue(hasattr(sft, name), name)
        self.assertEqual(sft.LORA_TARGETS, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkvz", "in_proj_ba", "out_proj"])

    def test_the_modal_trainer_still_has_every_name_it_moved(self):
        text = (ROOT / "distill" / "backends" / "modal_train.py").read_text()
        for name in ("encode_ids", "parse_assistant", "record_fate", "render_record", "encode_record", "fate_counts", "is_thinking_config", "THINK_OPEN", "EMPTY_THINK", "FATES", "LORA_TARGETS"):
            self.assertRegex(text, rf"(?m)^from \.\.sft import \(.*\b{name}\b|^\s+.*\b{name}\b.*\)$", name)
        self.assertNotRegex(text, r"(?m)^def (encode_ids|parse_assistant|record_fate|render_record|encode_record|fate_counts|is_thinking_config)\(")
        self.assertIn("target_modules=LORA_TARGETS", text)

    def test_the_pins_are_the_ones_the_modal_trainer_trains_with(self):
        text = (ROOT / "distill" / "backends" / "modal_train.py").read_text()
        for pkg in ("transformers", "peft"):
            pin = re.search(rf'"({pkg}==[\d.]+)"', text).group(1)
            self.assertIn(pin, lora_eval.DEFAULTS["pins"])
        self.assertTrue(any(p.startswith("bitsandbytes") for p in lora_eval.DEFAULTS["pins"]), "4-bit needs it, and the Modal image does not")

    def test_the_defaults_follow_the_modal_run_where_a_t4_allows_it(self):
        d = lora_eval.DEFAULTS
        self.assertEqual((d["lr"], d["epochs"], d["global_batch"], d["dropout"], d["seed"], d["student"]), (1.5e-4, 2.0, 16, 0.05, 7, config.DEFAULT_STUDENT))
        self.assertEqual((d["max_len"], d["sampling"]), (4096, config.THINKING_SAMPLING))
        self.assertEqual(d["max_new_tokens"], config.DEFAULT_THINKING_MAX_TOKENS)
        self.assertEqual((d["gpus"], d["gpu_name"], d["mix"]), (2, "T4", False))

    def test_the_config_of_the_kernel_is_checked(self):
        for bad in ({"sampling": {"temperature": 0}}, {"sampling": {"top_k": 1}}, {"k": 0}, {"max_len": True}, {"epochs": 0}, {"gpus": -1}):
            with self.assertRaises((ValueError, SystemExit), msg=str(bad)):
                lora_eval.resolve_config(bad)
        got = lora_eval.resolve_config({"sampling": {"temperature": 0.5}})
        self.assertEqual(got["sampling"], {"temperature": 0.5, "top_p": 0.95, "top_k": 20, "min_p": 0.0})


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.cfg = {"k": 1, "max_new_tokens": 4096}
        self.tasks = [{"id": t["id"], "kind": t["kind"], "messages": ARITH.messages(t, "thinking")} for t in HELDOUT[:4]]

    def line(self, task, outs):
        return lora_eval.reply_line(task, run_teacher.prompt_sha(task["messages"]), outs)

    def out(self, task, finish="stop", budget=4096, sha=None):
        return {"text": "<think>\nx\n</think>\n\nAnswer: 1", "finish": finish, "tokens": 10, "prompt_tokens": 5, "secs": 1.0, "prompt_sha": sha or run_teacher.prompt_sha(task["messages"]), "max_tokens": budget}

    def previous(self, base, lora):
        return {"base": {t["id"]: self.line(t, [self.out(t)]) for t in base}, "lora": {t["id"]: self.line(t, [self.out(t)]) for t in lora}}

    def test_nothing_is_asked_when_both_arms_have_every_reply(self):
        self.assertEqual(lora_eval.work_items(self.tasks, self.previous(self.tasks, self.tasks), self.cfg), [])

    def test_a_reply_that_one_arm_lacks_is_asked_for_both_so_they_stay_paired(self):
        items = lora_eval.work_items(self.tasks, self.previous(self.tasks, self.tasks[:2]), self.cfg)
        self.assertEqual(sorted(t["id"] for t, _ in items), sorted(t["id"] for t in self.tasks[2:]))
        self.assertEqual(len(lora_eval.work_items(self.tasks, {}, self.cfg)), 4)

    def test_a_failed_a_stale_and_a_cut_off_reply_are_asked_again_but_a_cut_off_one_at_the_same_budget_is_not(self):
        t0, t1, t2, t3 = self.tasks
        prev = self.previous(self.tasks, self.tasks)
        prev["base"][t0["id"]] = self.line(t0, [{**self.out(t0), "finish": "error", "text": "[error x]"}])
        prev["lora"][t1["id"]] = self.line(t1, [self.out(t1, sha="0" * 16)])
        prev["lora"][t2["id"]] = self.line(t2, [self.out(t2, finish="length", budget=2048)])
        prev["base"][t3["id"]] = self.line(t3, [self.out(t3, finish="length", budget=4096)])
        self.assertEqual(sorted(t["id"] for t, _ in lora_eval.work_items(self.tasks, prev, self.cfg)), sorted([t0["id"], t1["id"], t2["id"]]))

    def test_with_k_samples_only_the_missing_ones_are_asked(self):
        t = self.tasks[0]
        prev = {"base": {t["id"]: self.line(t, [self.out(t), self.out(t)])}, "lora": {t["id"]: self.line(t, [self.out(t), None])}}
        items = lora_eval.work_items([t], prev, {"k": 3, "max_new_tokens": 4096})
        self.assertEqual([i for _, i in items], [1, 2])

    def test_the_longest_prompts_come_first_and_the_order_is_stable(self):
        a, b = {**self.tasks[0], "messages": [{"role": "user", "content": "x" * 10}]}, {**self.tasks[1], "messages": [{"role": "user", "content": "x" * 99}]}
        self.assertEqual([t["id"] for t, _ in lora_eval.work_items([a, b], {}, self.cfg)], [b["id"], a["id"]])

    def test_the_processes_split_whole_tasks_so_no_shard_loses_a_sample(self):
        items = lora_eval.work_items(self.tasks, {}, {"k": 3, "max_new_tokens": 4096})
        shares = [lora_eval.assign(items, r, 2) for r in range(2)]
        self.assertEqual(sorted((t["id"], i) for share in shares for t, i in share), sorted((t["id"], i) for t, i in items))
        for a in shares[0]:
            self.assertNotIn(a[0]["id"], {t["id"] for t, _ in shares[1]})
        self.assertEqual(lora_eval.assign(items, 0, 1), items)

    def test_the_shards_and_the_attached_replies_become_one_file_and_the_last_line_wins(self):
        out, inp = self.root / "out", self.root / "in"
        (inp / "earlier").mkdir(parents=True)
        out.mkdir()
        t0, t1, t2 = self.tasks[:3]
        write_jsonl(inp / "earlier" / "replies-base.jsonl", [self.line(t0, [{**self.out(t0), "text": "old"}]), self.line(t2, [self.out(t2)])])
        write_jsonl(lora_eval.shard_path(out, "base", 0), [self.line(t0, [{**self.out(t0), "text": "new"}])])
        write_jsonl(lora_eval.shard_path(out, "base", 1), [self.line(t1, [self.out(t1)])])
        self.assertEqual(lora_eval.merge_shards(out, "base", input_root=inp), 3)
        merged, info = score_replies.read_replies(out / "replies-base.jsonl")
        self.assertEqual(sorted(merged), sorted([t0["id"], t1["id"], t2["id"]]))
        self.assertEqual(merged[t0["id"]]["outs"][0]["text"], "new")
        self.assertEqual(info["malformed"], 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["replies-base.jsonl"], "the shards and the .part file are gone")
        self.assertEqual(lora_eval.merge_shards(out, "base", input_root=inp), 3, "a second merge changes nothing")

    def test_a_smoke_run_never_mixes_with_a_real_one(self):
        out, inp = self.root / "out", self.root / "in"
        out.mkdir(), inp.mkdir()
        for name in ("replies-base.jsonl", "smoke-replies-base.jsonl", "replies-base.r0.jsonl", "replies-lora.jsonl"):
            (out / name).write_text("")
        (inp / "replies-base.jsonl").write_text("")
        real = [p.name for p in lora_eval.previous_files("base", out, inp, "")]
        self.assertEqual(real, ["replies-base.jsonl", "replies-base.jsonl", "replies-base.r0.jsonl"], "attached first, then this kernel's own, shards after the merged file")
        self.assertEqual([p.name for p in lora_eval.previous_files("base", out, inp, "smoke-")], ["smoke-replies-base.jsonl"])

    def test_a_finished_adapter_is_found_and_an_unfinished_or_smoke_one_is_not(self):
        out, inp = self.root / "out", self.root / "in"

        def adapter(where, status, weights=True, name="adapter"):
            d = where / name
            d.mkdir(parents=True)
            (d / "train.json").write_text(json.dumps({"status": status}))
            if weights:
                (d / "adapter_model.safetensors").write_text("x")
            return d
        adapter(out, "running"), adapter(inp / "a", "done", weights=False), adapter(out / "s", "done", name="adapter-smoke")
        self.assertIsNone(lora_eval.find_adapter([out, inp]))
        partial = adapter(inp / "b", "partial")
        self.assertEqual(lora_eval.find_adapter([out, inp]), partial)
        done = adapter(out / "x", "done")
        self.assertEqual(lora_eval.find_adapter([out, inp]), done, "this kernel's own output first")
        self.assertIsNone(lora_eval.find_adapter([self.root / "nothing"]))

    def test_the_newest_complete_checkpoint_is_the_one_to_resume(self):
        out, inp = self.root / "out", self.root / "in"
        for where, step, done in ((out, 10, True), (out, 20, False), (inp / "prev", 30, True), (inp / "prev", 40, False)):
            d = where / "ckpt" / f"step-{step}"
            d.mkdir(parents=True)
            if done:
                (d / "DONE").write_text(str(step))
        (out / "elsewhere" / "step-99").mkdir(parents=True)
        (out / "elsewhere" / "step-99" / "DONE").write_text("99")
        self.assertEqual(lora_eval.latest_checkpoint([out, inp]), inp / "prev" / "ckpt" / "step-30", "a checkpoint without DONE was cut by a killed session; one outside ckpt/ is not ours")
        self.assertEqual(lora_eval.latest_checkpoint([out]), out / "ckpt" / "step-10")
        self.assertIsNone(lora_eval.latest_checkpoint([self.root / "nothing"]))

    def test_the_deadline_is_the_teacher_kernels_rule(self):
        self.assertIs(lora_eval.run_teacher, run_teacher)
        self.assertTrue(run_teacher.can_start(100, 200, 50))
        self.assertFalse(run_teacher.can_start(100, 200, 150))


class Generation(unittest.TestCase):
    def test_the_reasoning_and_the_reply_are_split_at_the_end_of_the_think_block(self):
        self.assertEqual(lora_eval.split_generation("first I count\nthen I add\n</think>\n\nAnswer: 7<|im_end|>"), ("first I count\nthen I add", "Answer: 7"))
        self.assertEqual(lora_eval.split_generation("<think>\nit repeated the tag\n</think>\n\nAnswer: 7"), ("it repeated the tag", "Answer: 7"))

    def test_the_first_end_of_the_think_block_is_the_one_that_splits(self):
        self.assertEqual(lora_eval.split_generation("a</think>b</think>c"), ("a", "b</think>c"), "as the chat template reads a record: the first `</think>`, nothing is split at the last")

    def test_a_block_that_never_closed_is_all_reasoning_and_has_no_reply(self):
        self.assertEqual(lora_eval.split_generation("still thinking and thinking"), ("still thinking and thinking", ""))
        self.assertEqual(lora_eval.split_generation(""), ("", ""))

    def test_the_end_of_turn_decides_stop_or_length(self):
        self.assertEqual(lora_eval.finish_and_tokens([5, 6, 99, 0, 0], {99}), ("stop", 3, [5, 6]))
        self.assertEqual(lora_eval.finish_and_tokens([5, 6, 7], {99}), ("length", 3, [5, 6, 7]))
        self.assertEqual(lora_eval.finish_and_tokens([99], {99}), ("stop", 1, []))

    def test_a_generated_reply_is_stored_in_the_form_the_scorer_reads(self):
        reasoning, content = lora_eval.split_generation("count the apples\n</think>\n\nAnswer: 7")
        stored = run_teacher.stored_text(content, reasoning, True)
        self.assertEqual(stored, "<think>\ncount the apples\n</think>\n\nAnswer: 7")
        task = {"id": "t", "kind": "add", "question": "q", "answer": 7}
        self.assertTrue(traces.score_reasoning(score_replies.reply_of({"text": stored, "finish": "stop"}, thinking=True), task, domain=ARITH))
        cut = run_teacher.stored_text(*reversed(lora_eval.split_generation("count the apples and count")), True)
        self.assertEqual(traces.verdict(score_replies.reply_of({"text": cut, "finish": "length"}, thinking=True), task, domain=ARITH), "truncated")


class Packaging(unittest.TestCase):
    def test_the_kernel_imports_nothing_heavy_at_the_top(self):
        tree = ast.parse((KIT / "lora_eval.py").read_text())
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 0}
        self.assertLessEqual(top - {"run_teacher", "distill"}, set(sys.stdlib_module_names), "torch, transformers and peft are imported inside the workers only")
        for heavy in ("torch", "transformers", "peft", "bitsandbytes", "huggingface_hub", "datasets"):
            self.assertNotIn(heavy, top)

    def test_the_flat_files_become_the_package_and_nothing_else_is_needed(self):
        with tempfile.TemporaryDirectory() as d:
            src, dest = pathlib.Path(d) / "data", pathlib.Path(d) / "lib"
            src.mkdir()
            for name, path in build_lora.kernel_files().items():
                (src / name).write_bytes(path.read_bytes())
            with mock.patch.object(sys, "path", list(sys.path)):
                self.assertTrue(lora_eval.unpack_lib(src, dest))
                self.assertEqual(sorted(p.name for p in (dest / "distill").iterdir()), sorted(f"{m}.py" for m in lora_eval.LIB_MODULES))
                self.assertEqual(sys.path[0], str(dest))
                self.assertTrue(lora_eval.unpack_lib(src, dest), "idempotent")
            code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import lora_eval, distill.sft, distill.config, distill.traces, distill.domain; "
                    "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('rea_bench', 'modal', 'torch', 'transformers', 'peft', 'huggingface_hub', 'datasets')); assert not bad, bad; "
                    "assert lora_eval.sft.__file__.startswith(%r), lora_eval.sft.__file__; print(lora_eval.resolve_config({})['gpu_name'])" % (str(src), str(dest), str(dest)))
            r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=d, env={**os.environ, "LORA_LIB": str(dest)})
            self.assertEqual((r.returncode, r.stderr, r.stdout.strip()), (0, "", "T4"), "the dataset's files run with the repository nowhere on the path")

    def test_without_flat_files_distill_is_the_repositorys(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(lora_eval.unpack_lib(pathlib.Path(d), pathlib.Path(d) / "lib"))
            self.assertFalse((pathlib.Path(d) / "lib").exists())

    def test_the_library_modules_are_the_stdlib_only_core(self):
        for m in lora_eval.LIB_MODULES:
            tree = ast.parse((ROOT / "distill" / f"{m}.py").read_text())
            for n in ast.walk(tree):
                names = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else []
                for name in names:
                    self.assertIn(name, sys.stdlib_module_names, f"distill/{m}.py imports {name}")
                if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
                    self.assertIn(n.module, lora_eval.LIB_MODULES, f"distill/{m}.py imports .{n.module}, which the dataset does not ship")


class GpuGlue(unittest.TestCase):
    """The workers need torch, so they cannot run here: these pin, in the source, the invariants a T4 and two processes depend on, so that an edit cannot drop one without a test saying so."""
    SRC = (KIT / "lora_eval.py").read_text()

    def body(self, name: str) -> str:
        node = next(n for n in ast.parse(self.SRC).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(self.SRC, node)

    def test_a_t4_trains_in_fp16_with_a_4_bit_base_and_never_bf16(self):
        for needle in ("load_in_4bit=True", 'bnb_4bit_quant_type="nf4"', "bnb_4bit_compute_dtype=torch.float16", "torch_dtype=torch.float16", 'torch.autocast("cuda", dtype=torch.float16)', 'torch.amp.GradScaler("cuda")'):
            self.assertIn(needle, self.SRC, needle)
        self.assertNotIn("bfloat16", self.SRC.replace("bf16", ""), "a T4 has no bf16")
        self.assertNotIn("bf16=True", self.SRC)

    def test_memory_is_kept_down_by_checkpointing_and_by_slicing_the_loss(self):
        train = self.body("train_worker")
        self.assertIn("gradient_checkpointing_enable", self.body("load_student"))
        self.assertIn("use_reentrant", self.body("load_student"))
        self.assertIn("chunked_loss(", train)
        self.assertNotIn(".logits", train, "the logits of a whole sequence are never made")
        self.assertIn("torch.utils.checkpoint.checkpoint", self.body("chunked_loss"))
        self.assertIn("use_reentrant=False", self.body("chunked_loss"))
        self.assertIn("y != -100", self.body("chunked_loss"), "only the target positions go through the head")

    def test_the_processes_stop_together_and_average_before_the_scaler_looks(self):
        train = self.body("train_worker")
        self.assertIn("torch.distributed.broadcast(stop, 0)", train, "one process stopping alone leaves the other in the all-reduce")
        self.assertLess(train.index("torch.distributed.broadcast(stop, 0)"), train.index("all_reduce_mean("))
        self.assertLess(train.index("all_reduce_mean("), train.index("scaler.unscale_("), "an overflow on one process must be seen by all of them")
        self.assertLess(train.index("scaler.unscale_("), train.index("clip_grad_norm_"))
        self.assertLess(train.index("clip_grad_norm_"), train.index("scaler.step("))

    def test_only_the_adapter_is_saved_and_a_checkpoint_is_renamed_into_place(self):
        train, ckpt = self.body("train_worker"), self.body("save_checkpoint")
        self.assertIn("model.save_pretrained(str(adapter))", train)
        self.assertNotIn("merge_and_unload", train + self.body("eval_worker"), "the adapter is evaluated as trained, never merged")
        self.assertLess(ckpt.index('"DONE").write_text'), ckpt.index("os.replace(tmp, final)"), "DONE is written last, then the folder takes its name")

    def test_the_base_arm_is_the_adapter_switched_off_on_the_same_weights(self):
        gen = self.body("generate_batch")
        self.assertIn("model.disable_adapter()", gen)
        self.assertIn("enable_thinking=True", gen)
        self.assertIn('add_generation_prompt=True', gen)
        self.assertIn("**cfg[\"sampling\"]", gen)
        self.assertEqual(self.body("eval_worker").count("load_student("), 1, "one model load for both arms")

    def test_a_gpu_that_is_not_a_t4_stops_the_kernel_before_the_download(self):
        main = self.body("main")
        self.assertLess(main.index('cfg["gpu_name"]'), main.index("pip"))
        self.assertLess(main.index('cfg["gpu_name"]'), main.index("snapshot_download"))


class Scoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        self.heldout = write_jsonl(self.dir / "heldout.jsonl", HELDOUT)
        self.by_kind = {k: [t for t in HELDOUT if t["kind"] == k] for k in ("add", "sub", "mul")}

    def out(self, task, right, tokens=100, finish="stop", text=None, sha=None):
        answer = task["answer"] if right else task["answer"] + 1
        return {"text": text if text is not None else THINK.format("thinking", answer), "finish": finish, "tokens": tokens, "prompt_tokens": 50, "secs": 1.0,
                "prompt_sha": sha or outputs.prompt_sha(ARITH.messages(task, "thinking")), "max_tokens": 4096}

    def replies(self, name, make):
        lines = [{"id": t["id"], "kind": t["kind"], "prompt_sha": outputs.prompt_sha(ARITH.messages(t, "thinking")), "outs": make(t)} for t in HELDOUT]
        return write_jsonl(self.dir / name, lines)

    def files(self):
        """base: right on every add and on the first sub; adapter: right on every add but the first (cut off) and on every mul."""
        add0, sub0 = self.by_kind["add"][0]["id"], self.by_kind["sub"][0]["id"]
        base = self.replies("base.jsonl", lambda t: [self.out(t, t["kind"] == "add" or t["id"] == sub0, tokens=100)])
        lora = self.replies("lora.jsonl", lambda t: [self.out(t, False, tokens=4096, finish="length", text="<think>\nstill thinking\n</think>\n\n") if t["id"] == add0 else self.out(t, t["kind"] in ("add", "mul"), tokens=200)])
        return base, lora

    def test_the_two_arms_are_scored_per_family_and_overall(self):
        base, lora = self.files()
        r = score_lora.score(base, lora, [self.heldout], domain=ARITH)
        b, l = r["arms"]["base"], r["arms"]["lora"]
        self.assertEqual((b["overall"]["accepted"], l["overall"]["accepted"], b["overall"]["replies"]), (5, 7, 12))
        self.assertEqual({k: v["accepted"] for k, v in b["by_kind"].items()}, {"add": 4, "mul": 0, "sub": 1})
        self.assertEqual({k: v["accepted"] for k, v in l["by_kind"].items()}, {"add": 3, "mul": 4, "sub": 0})
        self.assertAlmostEqual(b["overall"]["accuracy"], 5 / 12, places=4)
        self.assertEqual((b["overall"]["mean_tokens"], round(l["overall"]["mean_tokens"], 1)), (100.0, round((4096 + 11 * 200) / 12, 1)))
        self.assertEqual((b["overall"]["cut_off"], l["overall"]["cut_off"], l["by_kind"]["add"]["cut_off"]), (0.0, round(1 / 12, 4), 0.25))
        self.assertEqual((l["overall"]["truncated"], l["overall"]["wrong"]), (1, 4), "a cut-off reply is `truncated`, not `wrong`")
        self.assertEqual((r["tasks_answered"], r["tasks_in_files"]), (12, 12))

    def test_the_arms_are_paired_and_the_sign_test_counts_only_the_discordant_pairs(self):
        r = score_lora.score(*self.files(), [self.heldout], domain=ARITH)
        p = r["paired"]
        self.assertEqual({k: p["overall"][k] for k in ("both", "base_only", "lora_only", "neither", "n")}, {"both": 3, "base_only": 2, "lora_only": 4, "neither": 3, "n": 12})
        self.assertAlmostEqual(p["overall"]["sign_p"], 2 * (1 + 6 + 15) / 64, places=4)
        self.assertEqual((p["by_kind"]["mul"]["lora_only"], p["by_kind"]["mul"]["sign_p"]), (4, 0.125))
        self.assertEqual((p["by_kind"]["add"]["base_only"], p["by_kind"]["sub"]["base_only"], p["unpaired"]), (1, 1, 0))

    def test_the_statistics(self):
        lo, hi = score_lora.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.237, places=3)
        self.assertAlmostEqual(hi, 0.763, places=3)
        self.assertEqual(score_lora.wilson(0, 0), (0.0, 0.0))
        self.assertEqual(score_lora.wilson(10, 10)[1], 1.0)
        self.assertEqual((score_lora.sign_test_p(0, 0), score_lora.sign_test_p(3, 3), score_lora.sign_test_p(0, 5)), (1.0, 1.0, 0.0625))
        self.assertAlmostEqual(score_lora.sign_test_p(2, 9), 2 * sum(math.comb(11, i) for i in range(3)) / 2 ** 11)

    def test_the_scorer_is_the_one_kaggle_score_uses(self):
        """A reply is accepted exactly when `traces.score_reasoning` of `score_replies.reply_of` accepts it, and its reason is `traces.verdict`'s."""
        t = self.by_kind["add"][0]
        cases = {"right": self.out(t, True), "wrong": self.out(t, False), "cut": self.out(t, True, finish="length"), "no answer line": self.out(t, True, text="<think>\nx\n</think>\n\nit is 7"),
                 "text after": self.out(t, True, text=THINK.format("x", t["answer"]) + "\nthanks"), "no think": self.out(t, True, text=f"Answer: {t['answer']}"), "error": {**self.out(t, True), "finish": "error", "text": "[error boom]"}}
        verdicts = {}
        for name, out in cases.items():
            line = {"id": t["id"], "kind": t["kind"], "prompt_sha": out["prompt_sha"], "outs": [out]}
            path = write_jsonl(self.dir / "one.jsonl", [line])
            got = score_lora.read_arm(path, {t["id"]: t}, domain=ARITH)["rows"][(t["id"], 0)]["verdict"]
            self.assertEqual(got, traces.verdict(score_replies.reply_of(out, thinking=True), t, domain=ARITH), name)
            self.assertEqual(got == "accepted", traces.score_reasoning(score_replies.reply_of(out, thinking=True), t, domain=ARITH), name)
            verdicts[name] = got
        self.assertEqual(verdicts, {"right": "accepted", "wrong": "wrong", "cut": "truncated", "no answer line": "no_answer_line", "text after": "no_answer_line", "no think": "accepted", "error": "error"})

    def test_a_reply_to_another_prompt_is_refused(self):
        t = self.by_kind["add"][0]
        out = self.out(t, True, sha="deadbeefdeadbeef")
        path = write_jsonl(self.dir / "one.jsonl", [{"id": t["id"], "kind": t["kind"], "prompt_sha": out["prompt_sha"], "outs": [out]}])
        with self.assertRaisesRegex(SystemExit, "another prompt"):
            score_lora.read_arm(path, {t["id"]: t}, domain=ARITH)

    def test_missing_replies_unknown_tasks_and_unchecked_replies_are_counted_not_scored(self):
        t = self.by_kind["add"][0]
        sha = outputs.prompt_sha(ARITH.messages(t, "thinking"))
        lines = [{"id": t["id"], "kind": "add", "prompt_sha": sha, "outs": [None, {**self.out(t, True), "prompt_sha": None}, self.out(t, True)]}, {"id": "nobody", "kind": "add", "outs": [self.out(t, True)]}]
        got = score_lora.read_arm(write_jsonl(self.dir / "one.jsonl", lines), {t["id"]: t}, domain=ARITH)
        self.assertEqual(sorted(got["rows"]), [(t["id"], 2)])
        self.assertEqual((got["info"]["missing"], got["info"]["unchecked"], got["info"]["unknown"]), (1, 1, 1))

    def test_an_arm_that_has_fewer_replies_leaves_the_rest_unpaired(self):
        base, lora = self.files()
        short = write_jsonl(self.dir / "short.jsonl", [json.loads(line) for line in lora.read_text().splitlines()][:6])
        r = score_lora.score(base, short, [self.heldout], domain=ARITH)
        self.assertEqual((r["paired"]["overall"]["n"], r["paired"]["unpaired"], r["arms"]["base"]["overall"]["replies"], r["arms"]["lora"]["overall"]["replies"]), (6, 6, 12, 6))

    def test_the_table_and_the_command(self):
        base, lora = self.files()
        info = self.dir / "run.json"
        info.write_text(json.dumps({"smoke": True, "train": {"status": "partial", "steps_done": 40, "steps_planned": 99, "loss_first": 1.5, "loss_last": 0.7, "records": {"think": 800}, "deadline_hit": True}}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score_lora.main(["--domain", "example_arith", "--heldout", str(self.heldout), "--base", str(base), "--lora", str(lora), "--run-info", str(info), "--out", str(self.dir / "report.json")])
        text = buf.getvalue()
        for needle in ("SMOKE run", "partial, 40 of 99 steps", "stopped by the deadline", "OVERALL", "add", "sub", "mul", "sign p", "paired replies 12, unpaired 0"):
            self.assertIn(needle, text)
        self.assertEqual(json.loads((self.dir / "report.json").read_text())["paired"]["overall"]["n"], 12)
        with self.assertRaises(SystemExit):
            score_lora.main(["--domain", "example_arith", "--heldout", str(self.heldout), "--base", str(base), "--lora", str(lora), "--out", str(self.dir / "report.json")])


class ReaWiring(unittest.TestCase):
    """The REA domain: its held-out files are readable for an evaluation and still refused for training, and its scorer is the one that judges the replies."""

    @classmethod
    def setUpClass(cls):
        from distill.domains.rea import DOMAIN
        from .distill_fixtures import load_tasks, right
        cls.domain, cls.right = DOMAIN, staticmethod(right)
        cls.tasks = [{**t, "id": t["id"].replace("-7-", "-1001-")} for t in load_tasks() if t["kind"] in ("xor-key", "password")][:4]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        self.file = write_jsonl(self.dir / "decompiled-seed1001-all.jsonl", self.tasks)

    def test_the_held_out_files_are_read_for_an_evaluation_and_refused_for_training(self):
        self.assertEqual(sorted(t["id"] for t in self.domain.load_heldout([self.file])), sorted(t["id"] for t in self.tasks))
        with self.assertRaises(SystemExit):
            self.domain.load_tasks([self.file])
        self.assertTrue(all(t["id"] in self.domain.heldout_ids() for t in self.tasks))

    def test_a_file_of_another_seed_is_still_read_by_load_tasks_as_it_always_was(self):
        other = write_jsonl(self.dir / "decompiled-seed7-all.jsonl", [{**t, "id": t["id"].replace("-1001-", "-7-")} for t in self.tasks])
        self.assertEqual(len(self.domain.load_tasks([other])), 4)
        self.assertEqual(len(self.domain.load_heldout([other])), 4)
        with self.assertRaisesRegex(SystemExit, "not held-out ids"):
            build_lora.pick_eval([other], "", 2, ["xor-key"], domain=self.domain, train_tasks={})

    def test_the_evaluation_subset_of_rea_is_checked_by_the_domain(self):
        chosen, _ = build_lora.pick_eval([self.file], "", 1, ["xor-key", "password"], domain=self.domain, train_tasks={})
        self.assertEqual(sorted(t["kind"] for t in chosen), ["password", "xor-key"])
        with self.assertRaisesRegex(SystemExit, "no task kind|which is no task kind"):
            build_lora.pick_eval([self.file], "nonsense=3", 1, [], domain=self.domain, train_tasks={})

    def test_a_reply_is_judged_by_the_rea_scorer(self):
        t = self.tasks[0]
        sha = outputs.prompt_sha(self.domain.messages(t, "thinking"))
        good = {"text": THINK.format("looking at the loop", self.right(t)), "finish": "stop", "tokens": 300, "prompt_sha": sha}
        bad = {"text": THINK.format("looking at the loop", "zzzz"), "finish": "stop", "tokens": 300, "prompt_sha": sha}
        got = score_lora.read_arm(write_jsonl(self.dir / "r.jsonl", [{"id": t["id"], "kind": t["kind"], "outs": [good, bad]}]), {t["id"]: t}, domain=self.domain)["rows"]
        self.assertEqual((got[(t["id"], 0)]["verdict"], got[(t["id"], 1)]["verdict"]), ("accepted", "wrong"))
        from rea_bench import scoring
        self.assertTrue(scoring.score_strict(self.right(t), t))
        self.assertFalse(scoring.score_strict("zzzz", t))


class Docs(unittest.TestCase):
    README = (KIT / "README.md").read_text()

    def section(self) -> str:
        m = re.search(r"(?ms)^## LoRA training and held-out evaluation on Kaggle\b.*?(?=^## |\Z)", self.README)
        self.assertIsNotNone(m, "README.md has the section")
        return m.group(0)

    def test_every_flag_the_readme_shows_exists(self):
        import argparse
        known = {}
        for name, add in (("build_lora", build_lora.add_arguments), ("score_lora", None)):
            ap = argparse.ArgumentParser()
            if add:
                add(ap)
            known[name] = {o for a in ap._actions for o in a.option_strings}
        known["score_lora"] = {"--domain", "--heldout", "--base", "--lora", "--run-info", "--out", "--force"}
        blocks = re.findall(r"(?s)```bash\n(.*?)```", self.section())
        shown = {"build_lora": set(), "score_lora": set()}
        for block in blocks:
            for cmd in re.split(r"\n(?=python3 )", block):
                for tool in shown:
                    if f"kaggle.{tool}" in cmd:
                        shown[tool] |= set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", cmd.replace("\\\n", " ")))
        self.assertTrue(shown["build_lora"] and shown["score_lora"], "both commands are shown")
        for tool, flags in shown.items():
            self.assertLessEqual(flags, known[tool] | {"--domain"}, tool)

    def test_the_section_says_the_things_an_owner_must_know(self):
        text = self.section()
        for needle in ("T4 x2", "--smoke", "--measured", "fp16", "NF4", "chunk", "not run on a GPU", "llama-bin", "kaggle datasets create", "kaggle kernels push", "--resume-from", "score_lora", "heldout-1001", "Wilson",
                       "build_lora", "not tried"):
            self.assertIn(needle, text, needle)

    def test_the_section_explains_the_content_check_and_the_commands_that_go_with_it(self):
        text = self.section()
        for needle in ("not by id", "code+question", "question+answer", "before the quota", "--train-tasks", "--min-family", "--strict-families", "built_from.eval.excluded", "build_lora overlap", "structs-7-0292", "structs-1001-0028",
                       "FUN_", "truth-template record that has think text", "user message is an evaluation task's"):
            self.assertIn(needle, text, needle)
        self.assertRegex(text, r"(?s)build_lora overlap --train-tasks .*?--heldout", "the overlap command is shown with both its inputs")

    def test_the_files_the_readme_names_exist(self):
        for name in re.findall(r"`((?:tests/|distill/)[\w./-]+\.py)`", self.section()):
            self.assertTrue((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()

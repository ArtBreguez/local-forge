"""Everything that determines an artifact is in that artifact's key (PR 34 fourth review, m1-m4).

The ADAPTER key holds every setting that can change a weight, the length of the schedule (`max_steps`) and the smoke limits among them (m1, m3); the EVALUATION key holds every setting that can change a
reply: the budget, the sampling, the seed, `k` (m2), the smoke limits (m3). Every setting of the kernel's `DEFAULTS` is classified (`TRAIN_KEYS`, `EVAL_KEYS` or `NOT_KEYED` with a reason), and the first
test here fails for a setting that is added without being classified. The kernel computes both keys again from the files it is given (`check_keys`), and `main` is run on stand-ins, so that the lines that
carry the reserve and the export into the two workers are executed and not re-implemented (m4).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import re
import stat
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from distill import config

from .test_kaggle_lora import ARITH, HELDOUT, Kit, build_lora, lora_eval, run_teacher, score_lora, sft_rows, write_jsonl
from .test_kaggle_lora_evalset import keys_of, line, reply

ADAPTER = {"student", "max_len", "rank", "alpha_ratio", "dropout", "epochs", "lr", "warmup", "max_steps", "global_batch", "max_grad_norm", "loss_chunk", "mix", "seed", "gpus", "fla", "pins", "smoke", "smoke_steps"}
EVALUATION = {"k", "max_new_tokens", "sampling", "seed_base", "eval_batch", "smoke", "smoke_new_tokens"}
NEITHER = {"gpu_name", "save_every", "train_hours", "deadline_h", "eval_reserve_s", "export_reserve_s", "export_gguf", "adapter_sha", "eval_sha"}


def other(value):
    """Another value of the same kind as `value`."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, dict):
        return {**value, "temperature": value["temperature"] + 0.1}
    if isinstance(value, list):
        return [*value, "another==1.0"]
    return value + 1


def kernel_cfg(cfg: dict) -> dict:
    """The config as the kernel resolves it from the dataset's file."""
    return lora_eval.resolve_config(json.loads(json.dumps({k: v for k, v in cfg.items() if k != "plan_lines"})))


class Classification(unittest.TestCase):
    def test_every_setting_of_the_kernel_is_in_a_key_or_is_left_out_with_a_reason(self):
        d = set(lora_eval.DEFAULTS)
        train, ev, neither = set(lora_eval.TRAIN_KEYS), set(lora_eval.EVAL_KEYS), set(lora_eval.NOT_KEYED)
        self.assertEqual(d - train - ev - neither, set(), "a new setting of DEFAULTS: say which key it belongs to (TRAIN_KEYS, EVAL_KEYS), or why to neither (NOT_KEYED)")
        self.assertEqual((train | ev | neither) - d, set(), "a classified name that is no setting")
        self.assertEqual(neither & (train | ev), set(), "keyed, or not")
        self.assertEqual((len(lora_eval.TRAIN_KEYS), len(lora_eval.EVAL_KEYS)), (len(train), len(ev)), "no name twice")
        for name, why in lora_eval.NOT_KEYED.items():
            self.assertTrue(isinstance(why, str) and len(why) > 20, name)

    def test_the_classification_is_this_one(self):
        """By name: a setting moved out of a key, or from one key to the other, is a decision, and it is made here too."""
        self.assertEqual(set(lora_eval.TRAIN_KEYS), ADAPTER)
        self.assertEqual(set(lora_eval.EVAL_KEYS), EVALUATION)
        self.assertEqual(set(lora_eval.NOT_KEYED), NEITHER)
        self.assertEqual(ADAPTER & EVALUATION, {"smoke"}, "the smoke flag makes another adapter AND another evaluation")

    def test_the_readme_table_is_the_classification(self):
        rows = [x for x in (pathlib.Path(lora_eval.__file__).parent / "README.md").read_text().splitlines() if x.startswith("| ")]
        cell = lambda start: next(set(re.findall(r"`([\w.]+)`", x.split("|")[2])) for x in rows if x.startswith(start))
        self.assertEqual(cell("| **adapter key** (`TRAIN_KEYS`)"), ADAPTER | {"train.jsonl", "sft.LORA_TARGETS"})
        self.assertEqual(cell("| **evaluation key** (`EVAL_KEYS`)"), EVALUATION | {"eval_tasks.jsonl"})
        start = next(i for i, x in enumerate(rows) if x.startswith("| **neither** (`NOT_KEYED`"))
        self.assertEqual(set().union(*(set(re.findall(r"`(\w+)`", x.split("|")[2])) for x in rows[start:start + 3])), NEITHER)

    def test_the_kernel_reads_no_setting_that_is_not_one(self):
        src = pathlib.Path(lora_eval.__file__).read_text()
        read = set(re.findall(r"""cfg\[["'](\w+)["']\]""", src)) | set(re.findall(r"""cfg\.get\(["'](\w+)["']""", src))
        self.assertEqual(read - set(lora_eval.DEFAULTS), {"built_from"}, "a setting the kernel reads from its config is a setting of DEFAULTS (and so is classified); `built_from` is the builder's record, read by check_keys")
        self.assertGreater(len(read), 30)

    def test_the_builder_writes_no_setting_the_kernel_does_not_know(self):
        k = Kit("build")
        k.setUp()
        self.addCleanup(k.doCleanups)
        for smoke in (False, True):
            k.build(out=k.dir / str(smoke), smoke=smoke)
            on_disk = json.loads((k.dir / str(smoke) / "data" / "lora-config.json").read_text())
            self.assertEqual(set(on_disk) - set(lora_eval.DEFAULTS), {"built_from"})
            self.assertEqual(set(lora_eval.DEFAULTS) - set(on_disk), {"pins"}, "the pins are the kernel file's own: a dataset cannot carry others without another adapter key")

    def test_what_is_not_a_setting_is_a_constant_the_keys_hold(self):
        cfg = lora_eval.resolve_config({})
        a, e = lora_eval.adapter_key("abc", cfg), lora_eval.eval_key("adapter", "ex", "tasks", cfg)
        with mock.patch.object(lora_eval.sft, "LORA_TARGETS", [*lora_eval.sft.LORA_TARGETS, "another_proj"]):
            self.assertNotEqual(lora_eval.adapter_key("abc", cfg), a, "the modules the adapter sits on")
            self.assertEqual(lora_eval.eval_key("adapter", "ex", "tasks", cfg), e)
        with mock.patch.object(lora_eval, "THINKING", False):
            self.assertNotEqual(lora_eval.eval_key("adapter", "ex", "tasks", cfg), e, "a reply that did not think is of another evaluation")
            self.assertEqual(lora_eval.adapter_key("abc", cfg), a)


class AdapterKeyReads(unittest.TestCase):
    def setUp(self):
        self.cfg = lora_eval.resolve_config({})
        self.base = lora_eval.adapter_key("abc", self.cfg)

    def test_the_records_and_every_training_setting_by_name(self):
        self.assertRegex(self.base, r"^[0-9a-f]{16}$")
        self.assertNotEqual(lora_eval.adapter_key("abd", self.cfg), self.base)
        for key in sorted(ADAPTER):
            self.assertNotEqual(lora_eval.adapter_key("abc", {**self.cfg, key: other(self.cfg[key])}), self.base, key)

    def test_nothing_of_the_evaluation_and_no_clock(self):
        for key in sorted((EVALUATION | NEITHER) - ADAPTER):
            self.assertEqual(lora_eval.adapter_key("abc", {**self.cfg, key: other(self.cfg[key])}), self.base, key)

    def test_the_schedule_is_what_the_worker_runs(self):
        """What the key holds is what the training worker uses: the warmup of the config, the alpha of the config."""
        import ast
        src = pathlib.Path(lora_eval.__file__).read_text()
        worker = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "train_worker"))
        self.assertEqual(worker.count('lr_at(step, steps, cfg["lr"], cfg["warmup"])') + worker.count("lr_at(step, steps, cfg['lr'], cfg['warmup'])"), 2, "the rate that is set, and the one that is logged")
        self.assertEqual(worker.count("lr_at("), 2)
        self.assertIn('total_steps(len(examples), cfg["epochs"], gb, cfg["max_steps"])', worker)
        self.assertIn('clip_grad_norm_(params, cfg["max_grad_norm"])', worker)
        self.assertEqual(lora_eval.lr_at(0, 100, 1.0), lora_eval.lr_at(0, 100, 1.0, lora_eval.DEFAULTS["warmup"]), "one default, the config's")
        self.assertEqual(lora_eval.lr_at(0, 100, 1.0, 0.5), 1 / 50)
        info = lora_eval.train_info({**self.cfg, "warmup": 0.5, "lr": 1.0, "smoke": True}, last=10, start=0, steps=100, counts={}, examples=1, gb=16, world=2, losses=[], wall_s=1, tokens=1, step_secs=[1.0], peak_gib=1)
        self.assertEqual(info["lr_last"], 10 / 50, "train.json's last rate is the schedule's of this config, not of a default")
        self.assertIs(info["smoke"], True)
        self.assertIs(lora_eval.train_info(self.cfg, last=10, start=0, steps=100, counts={}, examples=1, gb=16, world=2, losses=[], wall_s=1, tokens=1, step_secs=[1.0], peak_gib=1)["smoke"], False)

    def test_the_adapter_is_made_with_the_alpha_and_the_targets_of_the_key(self):
        made = {}
        peft = SimpleNamespace(LoraConfig=lambda **kw: made.setdefault("lora", kw), PeftModel=None, get_peft_model=lambda model, c: "peft model")
        base = SimpleNamespace(config=SimpleNamespace(), gradient_checkpointing_enable=lambda **kw: None, enable_input_require_grads=lambda: None)
        transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: base), BitsAndBytesConfig=lambda **kw: kw)
        torch = SimpleNamespace(float16="fp16", cuda=SimpleNamespace(empty_cache=lambda: None))
        with mock.patch.dict(sys.modules, {"peft": peft, "transformers": transformers}):
            self.assertEqual(lora_eval.load_student(torch, {**self.cfg, "rank": 8, "alpha_ratio": 3, "dropout": 0.1}, 0, train=True), "peft model")
        self.assertEqual(made["lora"], {"r": 8, "lora_alpha": 24, "lora_dropout": 0.1, "task_type": "CAUSAL_LM", "target_modules": lora_eval.sft.LORA_TARGETS})


class EvalKeyReads(unittest.TestCase):
    def setUp(self):
        self.cfg = lora_eval.resolve_config({})
        self.key = lambda cfg=self.cfg, adapter="adapter", ex="ex", tasks="tasks": lora_eval.eval_key(adapter, ex, tasks, cfg)
        self.base = self.key()

    def test_the_adapter_the_exclusion_and_the_tasks_asked(self):
        self.assertRegex(self.base, r"^[0-9a-f]{16}$")
        self.assertEqual(self.key(), self.base)
        for name, kw in (("another adapter", {"adapter": "adapteR"}), ("another exclusion", {"ex": "ey"}), ("another file of tasks", {"tasks": "taskS"})):
            self.assertNotEqual(self.key(**kw), self.base, name)
        self.assertEqual(len({self.key(adapter="a", ex="b", tasks="c"), self.key(adapter="b", ex="a", tasks="c"), self.key(adapter="a", ex="c", tasks="b"), self.key(adapter="c", ex="b", tasks="a")}), 4, "each in its own place")

    def test_every_evaluation_setting_by_name(self):
        for key in sorted(EVALUATION):
            self.assertNotEqual(self.key({**self.cfg, key: other(self.cfg[key])}), self.base, key)
        for name in ("temperature", "top_p", "top_k", "min_p"):
            self.assertNotEqual(self.key({**self.cfg, "sampling": {**self.cfg["sampling"], name: self.cfg["sampling"][name] + 0.01}}), self.base, name)

    def test_a_training_setting_reaches_it_through_the_adapter_key_and_a_clock_never(self):
        for key in sorted((ADAPTER | NEITHER) - EVALUATION):
            self.assertEqual(self.key({**self.cfg, key: other(self.cfg[key])}), self.base, key)

    def test_the_replies_are_made_with_what_the_key_holds(self):
        import ast
        src = pathlib.Path(lora_eval.__file__).read_text()
        body = lambda name: ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name))
        gen = body("generate_batch")
        for needle in ('max_new_tokens=cfg["max_new_tokens"]', '**cfg["sampling"]', 'torch.manual_seed(cfg["seed_base"] + items[0][1])', "enable_thinking=THINKING"):
            self.assertIn(needle, gen)
        self.assertIn("config.sampling_for(THINKING,", body("resolve_config"))
        with self.assertRaises(SystemExit), mock.patch.object(lora_eval, "THINKING", True):
            lora_eval.resolve_config({"sampling": {"temperature": 0}})
        with mock.patch.object(lora_eval, "THINKING", False):
            self.assertEqual(lora_eval.resolve_config({"sampling": {"temperature": 0}})["sampling"]["temperature"], 0, "the sampling is resolved for the mode the key names")


class Built(Kit):
    """The keys of real builds of the fixture."""

    def slow(self) -> pathlib.Path:
        """A smoke run's run.json with a training speed at which the hours cap the steps."""
        path = self.dir / "smoke-run.json"
        path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": 0.47}}))
        return path

    def test_hours_that_cap_the_steps_elsewhere_are_another_adapter(self):
        """m1: `max_steps` is the length of the cosine. The builder derives it from --train-hours, --measured and --gpus."""
        a = self.build(out=self.dir / "a", measured=self.slow(), train_hours=2.0)
        b = self.build(out=self.dir / "b", measured=self.slow(), train_hours=3.0)
        full = a["built_from"]["estimate"]["steps_full"]
        self.assertEqual((a["max_steps"], b["max_steps"], full), (2, 3, 4))
        self.assertNotEqual(a["adapter_sha"], b["adapter_sha"])
        self.assertNotEqual(a["eval_sha"], b["eval_sha"], "and so another evaluation")
        again = self.build(out=self.dir / "again", measured=self.slow(), train_hours=2.0, gpu_hours=7.0)
        self.assertEqual(keys_of(again), keys_of(a), "the same cap, with less quota left: the same adapter (the clock is in no key)")
        self.assertNotEqual(again["deadline_h"], a["deadline_h"])
        unmeasured = self.build(out=self.dir / "c", train_hours=3.0)
        self.assertEqual(unmeasured["max_steps"], 0)
        self.assertNotEqual(unmeasured["adapter_sha"], a["adapter_sha"], "--measured changes the cap too")
        self.assertEqual(keys_of(self.build(out=self.dir / "d", train_hours=5.0)), keys_of(unmeasured), "hours that do not bind change nothing: the epochs decide in both")
        one = self.build(out=self.dir / "e", gpus=1)
        self.assertNotEqual(one["adapter_sha"], unmeasured["adapter_sha"], "--gpus: a step's share per process")

    def test_a_checkpoint_or_an_adapter_of_another_schedule_is_refused(self):
        a = self.build(out=self.dir / "a", measured=self.slow(), train_hours=2.0)
        b = self.build(out=self.dir / "b", measured=self.slow(), train_hours=3.0)
        inp = self.dir / "in"
        ckpt = inp / "earlier" / "ckpt" / "step-1"
        ckpt.mkdir(parents=True)
        (ckpt / "DONE").write_text(json.dumps({"step": 1, "adapter_sha": a["adapter_sha"]}))
        self.assertEqual(lora_eval.latest_checkpoint([inp], a["adapter_sha"]), ckpt)
        with self.assertRaisesRegex(SystemExit, "--train-hours, --measured and --gpus are training settings when they change the cap on the steps"):
            lora_eval.latest_checkpoint([inp], b["adapter_sha"])
        adapter = inp / "earlier" / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_text("x")
        (adapter / "train.json").write_text(json.dumps({"status": "partial", "adapter_sha": a["adapter_sha"]}))
        self.assertEqual(lora_eval.find_adapter([inp], a["adapter_sha"]), adapter)
        with self.assertRaisesRegex(SystemExit, "--train-hours, --measured and --gpus are training settings when they change the cap on the steps"):
            lora_eval.find_adapter([inp], b["adapter_sha"])

    def test_a_training_setting_no_flag_sets_is_another_adapter_too(self):
        base = keys_of(self.build(out=self.dir / "a"))
        for name, value in (("max_grad_norm", 5.0), ("seed", 8), ("dropout", 0.1), ("warmup", 0.1), ("alpha_ratio", 1), ("loss_chunk", 256), ("smoke_steps", 5)):
            with mock.patch.dict(lora_eval.DEFAULTS, {name: value}):
                got = keys_of(self.build(out=self.dir / name))
            self.assertNotEqual(got["adapter_sha"], base["adapter_sha"], name)
        with mock.patch.dict(lora_eval.DEFAULTS, {"pins": ["transformers==9.9.9"]}):
            self.assertNotEqual(self.build(out=self.dir / "pins")["adapter_sha"], base["adapter_sha"], "the library versions that train")
        self.assertNotEqual(self.build(out=self.dir / "fla", fla=True)["adapter_sha"], base["adapter_sha"])

    def test_the_budget_the_sampling_the_seed_and_k_are_another_evaluation_of_the_same_adapter(self):
        """m2: the review's repro. `--max-new-tokens 1024`, `--k 3`, and an edit of the sampling or of `seed_base` between two builds."""
        base = keys_of(self.build(out=self.dir / "a"))
        got = {"budget": self.build(out=self.dir / "b", max_new_tokens=1024), "k": self.build(out=self.dir / "c", k=3), "batch": self.build(out=self.dir / "d", eval_batch=4)}
        with mock.patch.dict(lora_eval.DEFAULTS, {"seed_base": 99}):
            got["seed"] = self.build(out=self.dir / "e")
        with mock.patch.dict(lora_eval.DEFAULTS, {"sampling": {**config.THINKING_SAMPLING, "temperature": 1.2}}):
            got["temperature"] = self.build(out=self.dir / "f")
        with mock.patch.dict(lora_eval.DEFAULTS, {"smoke_new_tokens": 128}):
            got["smoke budget"] = self.build(out=self.dir / "g")
        self.assertEqual((got["temperature"]["sampling"]["temperature"], got["seed"]["seed_base"]), (1.2, 99))
        for name, cfg in got.items():
            self.assertEqual(cfg["adapter_sha"], base["adapter_sha"], f"{name}: the adapter is kept")
            self.assertNotEqual(cfg["eval_sha"], base["eval_sha"], f"{name}: another evaluation")
        self.assertEqual(len({cfg["eval_sha"] for cfg in got.values()}), len(got))

    def test_what_is_in_no_key_changes_no_key(self):
        base = keys_of(self.build(out=self.dir / "a"))
        for name, kw in (("quota", {"gpu_hours": 7.0}), ("deadline", {"deadline_h": 5.0}), ("export", {"export_gguf": True}), ("resume", {"resume_from": "someone/earlier"}), ("slugs", {"dataset_slug": "other-data", "kernel_slug": "other-kernel"})):
            self.assertEqual(keys_of(self.build(out=self.dir / name, **kw)), base, name)
        for name, value in (("save_every", 5), ("gpu_name", "")):
            with mock.patch.dict(lora_eval.DEFAULTS, {name: value}):
                self.assertEqual(keys_of(self.build(out=self.dir / name)), base, name)

    def test_replies_asked_under_another_budget_are_asked_again_and_never_scored_as_this_builds(self):
        """m2's second repro: tasks answered under a budget of 4096, then a resume built with `--max-new-tokens 1024` (the builder's own advice when a plan does not fit)."""
        first = self.build(out=self.dir / "first")
        resume = self.build(out=self.dir / "resume", max_new_tokens=1024)
        ids = first["built_from"]["eval"]["ids"]
        self.assertEqual(resume["built_from"]["eval"]["ids"], ids)
        earlier = self.dir / "in" / "earlier"
        earlier.mkdir(parents=True)
        for arm in lora_eval.ARMS:
            write_jsonl(earlier / f"replies-{arm}.jsonl", [line(t, [{**reply(t, arm == "lora", keys_of(first)), "tokens": 3000}]) for t in HELDOUT if t["id"] in ids])
        tasks = [json.loads(x) for x in (self.dir / "resume" / "data" / "eval_tasks.jsonl").read_text().splitlines()]
        previous = {arm: run_teacher.load_previous(lora_eval.previous_files(arm, self.dir / "work", self.dir / "in")) for arm in lora_eval.ARMS}
        self.assertEqual(lora_eval.work_items(tasks, previous, kernel_cfg(first)), [], "the build that asked them has nothing left")
        self.assertEqual(sorted(t["id"] for t, _ in lora_eval.work_items(tasks, previous, kernel_cfg(resume))), ids, "the review: 0 asked again")
        score = lambda cfg_dir: score_lora.score(earlier / "replies-base.jsonl", earlier / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / cfg_dir / "data" / "lora-config.json"))
        r = score("resume")
        self.assertEqual((r["paired"]["overall"]["n"], r["tasks_answered"], r["other_adapter"]["replies"], r["other_eval"]["replies"]), (0, 0, 0, 2 * len(ids)), "the review: 16 paired replies of 3,000 tokens under a 1,024 budget")
        self.assertIn(f"{2 * len(ids)} replies of this adapter written under another evaluation key than {resume['eval_sha']} (another exclusion, set of tasks, prompt, reply budget, sampling or k: a resume asks them again), not scored "
                      f"(base {len(ids)}, adapter {len(ids)})", score_lora.lines_of(r))
        self.assertEqual(score("first")["paired"]["overall"]["n"], len(ids), "under the build that asked them they are scored")


class Smoke(Kit):
    """m3: a smoke build has keys of its own, so neither its adapter nor a reply of it is a full run's, whatever the files are called."""

    def setUp(self):
        super().setUp()
        self.full, self.smoke = self.build(out=self.dir / "full"), self.build(out=self.dir / "smoke", smoke=True)
        self.ids = self.full["built_from"]["eval"]["ids"]

    def config(self, name) -> pathlib.Path:
        return self.dir / name / "data" / "lora-config.json"

    def replies(self, keys: dict, where: str = "replies") -> tuple:
        d = self.dir / where
        d.mkdir()
        for arm in lora_eval.ARMS:
            write_jsonl(d / f"{arm}.jsonl", [line(t, [{**reply(t, arm == "lora", keys), "max_tokens": 256}]) for t in HELDOUT if t["id"] in self.ids])
        return d / "base.jsonl", d / "lora.jsonl"

    def test_both_keys_of_a_smoke_build_are_its_own(self):
        self.assertEqual(self.smoke["built_from"]["eval"]["ids"], self.ids, "the same evaluation set, the same records")
        self.assertEqual(self.smoke["built_from"]["train"]["train_sha256"], self.full["built_from"]["train"]["train_sha256"])
        self.assertNotEqual(self.smoke["adapter_sha"], self.full["adapter_sha"], "the review: equal")
        self.assertNotEqual(self.smoke["eval_sha"], self.full["eval_sha"])
        cfg = kernel_cfg(self.full)
        self.assertEqual(lora_eval.adapter_key(self.full["built_from"]["train"]["train_sha256"], {**cfg, "smoke": True}), self.smoke["adapter_sha"], "the flag alone is the difference")
        self.assertNotEqual(lora_eval.eval_key(self.full["adapter_sha"], self.full["built_from"]["eval"]["excluded"]["sha"], self.full["built_from"]["eval"]["sha256"], {**cfg, "smoke": True}), self.full["eval_sha"],
                            "and it is in the evaluation key itself, not only through the adapter key")

    def test_smoke_replies_given_by_hand_to_the_scorer_of_the_full_build_are_not_scored(self):
        """The review's repro: `smoke-replies-*.jsonl` given as --base and --lora with the full build's --config and no --run-info: 16 paired replies, a table, no warning."""
        base, lora = self.replies(keys_of(self.smoke))
        r = score_lora.score(base, lora, [self.heldout], domain=ARITH, build=score_lora.read_build(self.config("full")))
        self.assertEqual((r["paired"]["overall"]["n"], r["tasks_answered"], r["other_adapter"]["replies"], r["other_eval"]["replies"]), (0, 0, 2 * len(self.ids), 0))
        self.assertIn(f"{2 * len(self.ids)} replies of another adapter than {self.full['adapter_sha']} (another --sft or training setting, or a smoke run's), not scored (base {len(self.ids)}, adapter {len(self.ids)})", score_lora.lines_of(r))

    def test_scored_with_the_smoke_builds_own_config_the_report_says_smoke_with_or_without_a_run_json(self):
        base, lora = self.replies(keys_of(self.smoke))
        self.assertIs(score_lora.read_build(self.config("smoke"))["smoke"], True)
        self.assertIs(score_lora.read_build(self.config("full"))["smoke"], False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score_lora.main(["--domain", "example_arith", "--config", str(self.config("smoke")), "--heldout", str(self.heldout), "--base", str(base), "--lora", str(lora), "--out", str(self.dir / "report.json")])
        out = buf.getvalue().splitlines()
        self.assertEqual(out[0], "WARNING: this is a SMOKE run (4 steps, one batch of replies): its numbers say nothing about the adapter")
        self.assertEqual(sum("SMOKE" in x for x in out), 1)
        self.assertIs(json.loads((self.dir / "report.json").read_text())["smoke"], True)
        r = score_lora.score(base, lora, [self.heldout], domain=ARITH, build=score_lora.read_build(self.config("smoke")))
        self.assertEqual(r["paired"]["overall"]["n"], len(self.ids))
        self.assertEqual(sum("SMOKE" in x for x in score_lora.lines_of(r, {"smoke": True, "train": {}})), 1, "said once, by the config or by the run.json")
        full = score_lora.score(*self.replies(keys_of(self.full), "full-replies"), [self.heldout], domain=ARITH, build=score_lora.read_build(self.config("full")))
        self.assertIs(full["smoke"], False)
        self.assertFalse(any("SMOKE" in x for x in score_lora.lines_of(full)))
        self.assertFalse(any("SMOKE" in x for x in score_lora.lines_of(full, {"train": {}})))
        self.assertEqual(sum("SMOKE" in x for x in score_lora.lines_of(full, {"smoke": True, "train": {}})), 1, "a run.json that says smoke still says it")

    def test_a_smoke_config_edited_to_say_it_is_not_one_is_refused_by_the_scorer(self):
        """The seventh review's first note: a smoke build's `lora-config.json` with `"smoke": false` written into it by hand, no --run-info: the 16 smoke replies were scored (the keys are the file's,
        unchanged) and no warning was printed. The kernel refuses that file (`check_keys`); the scorer now does, by the same function."""
        base, lora = self.replies(keys_of(self.smoke))
        data = json.loads(self.config("smoke").read_text())
        self.assertIs(data["smoke"], True)
        edited = self.dir / "edited.json"
        edited.write_text(json.dumps({**data, "smoke": False}))
        args = ["--domain", "example_arith", "--heldout", str(self.heldout), "--base", str(base), "--lora", str(lora), "--out", str(self.dir / "report.json")]
        buf = io.StringIO()
        made = lora_eval.adapter_key(data["built_from"]["train"]["train_sha256"], kernel_cfg({**data, "smoke": False}))
        self.assertEqual(made, self.full["adapter_sha"], "the fixture: what the edit pretends to be is the full build of the same files")
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as e:
            score_lora.main([*args, "--config", str(edited)])
        self.assertEqual(str(e.exception), f"{edited}: lora-config.json says adapter key {data['adapter_sha']!r}, and its own settings and the two files it records (built_from) make {made!r}: the config was changed "
                                           "after the build (or it is of another version of the kit). Give the lora-config.json as build_lora.py wrote it (the dataset's); a setting is never edited in it")
        self.assertEqual((buf.getvalue(), (self.dir / "report.json").exists()), ("", False), "no table and no report: the review got 16 paired replies and no warning")
        for value in ("yes", 1, 0, None):
            edited.write_text(json.dumps({**data, "smoke": value}))
            with self.assertRaisesRegex(SystemExit, "says adapter key", msg=repr(value)):
                score_lora.read_build(edited)
        edited.write_text(json.dumps({k: v for k, v in data.items() if k != "smoke"}))
        with self.assertRaisesRegex(SystemExit, "says adapter key", msg="no `smoke` at all is the kernel's default, false"):
            score_lora.read_build(edited)
        edited.write_text(json.dumps({**data, "smoke": False, **keys_of(self.full)}))
        r = score_lora.score(base, lora, [self.heldout], domain=ARITH, build=score_lora.read_build(edited))
        self.assertEqual((r["paired"]["overall"]["n"], r["other_adapter"]["replies"]), (0, 2 * len(self.ids)), "with the keys edited too it is the full build's config, and no smoke reply carries its keys")
        full = json.loads(self.config("full").read_text())
        edited.write_text(json.dumps({**full, "smoke": True}))
        with self.assertRaisesRegex(SystemExit, "says adapter key", msg="and the other way: a full build is not made a smoke one"):
            score_lora.read_build(edited)
        with contextlib.redirect_stdout(buf):
            score_lora.main([*args, "--config", str(self.config("smoke"))])
        self.assertTrue(buf.getvalue().startswith("WARNING: this is a SMOKE run"), "the file as it was built is scored, under the warning")

    def test_the_scorer_refuses_every_edit_of_a_config_that_the_kernel_refuses(self):
        """One function, `lora_eval.check_keys`: with the dataset (the kernel) and with the shas the config records (the scorer). An edit of a keyed setting, of a key or of what the config records of its
        files is refused by both; a setting that is in no key is refused by neither."""
        data = json.loads(self.config("full").read_text())
        folder, path = self.dir / "full" / "data", self.dir / "c.json"
        built = lambda **eval_edit: {"built_from": {**data["built_from"], "eval": {**data["built_from"]["eval"], **eval_edit}}}
        refused = [({"max_new_tokens": 1024}, "evaluation key"), ({"k": 2}, "evaluation key"), ({"seed_base": 1}, "evaluation key"), ({"sampling": {"temperature": 1.0}}, "evaluation key"), ({"smoke_new_tokens": 8}, "evaluation key"),
                   ({"max_steps": 3}, "adapter key"), ({"lr": 1e-3}, "adapter key"), ({"gpus": 1}, "adapter key"), ({"pins": ["transformers==9.9.9"]}, "adapter key"), ({"smoke": True}, "adapter key"),
                   ({"adapter_sha": "0" * 16}, "adapter key"), ({"eval_sha": "0" * 16}, "evaluation key"), ({"eval_sha": self.smoke["eval_sha"]}, "evaluation key"),
                   (built(excluded={**data["built_from"]["eval"]["excluded"], "sha": "0" * 16}), "evaluation key"), (built(excluded="x"), "evaluation key"), (built(sha256="0" * 16), "evaluation key"),
                   ({"built_from": {**data["built_from"], "train": {**data["built_from"]["train"], "train_sha256": "0" * 16}}}, "adapter key"),
                   ({"built_from": {k: v for k, v in data["built_from"].items() if k != "train"}}, "adapter key"), ({"built_from": {**data["built_from"], "train": "x"}}, "adapter key")]
        for edit, what in refused:
            path.write_text(json.dumps({**data, **edit}))
            with self.assertRaisesRegex(SystemExit, f"^{re.escape(str(path))}: lora-config.json says {what} .* its own settings and the two files it records", msg=edit):
                score_lora.read_build(path)
        for edit, what in refused[:15]:          # the edits of a setting, of a key and of the exclusion; the kernel reads the two files themselves, not what the config records of them
            with self.assertRaisesRegex(SystemExit, f"lora-config.json says {what} .* the files and settings of this dataset make", msg=edit):
                lora_eval.check_keys(lora_eval.resolve_config({**data, **edit}), folder)
        for edit in ({"train_hours": 1.0}, {"deadline_h": 2.0}, {"save_every": 3}, {"gpu_name": ""}, {"export_gguf": True}, {"eval_reserve_s": 1}):
            path.write_text(json.dumps({**data, **edit}))
            self.assertEqual(score_lora.read_build(path), {"ids": set(self.ids), **keys_of(self.full), "smoke": False}, edit)
            lora_eval.check_keys(lora_eval.resolve_config({**data, **edit}), folder)
        partial = {**data, "sampling": {"temperature": 1.0}}          # a sampling the kernel completes (`resolve_config`): the keys are of the config as the kernel resolves it, here as there
        self.assertNotEqual(kernel_cfg(partial)["sampling"], partial["sampling"])
        partial["adapter_sha"] = lora_eval.adapter_key(data["built_from"]["train"]["train_sha256"], kernel_cfg(partial))
        partial["eval_sha"] = lora_eval.eval_key(partial["adapter_sha"], data["built_from"]["eval"]["excluded"]["sha"], data["built_from"]["eval"]["sha256"], kernel_cfg(partial))
        path.write_text(json.dumps(partial))
        self.assertEqual(score_lora.read_build(path), {"ids": set(self.ids), **keys_of(partial), "smoke": False})
        lora_eval.check_keys(kernel_cfg(partial), folder)
        for edit in ({"max_len": 0}, {"k": "two"}, {"epochs": "x"}, {"sampling": 5}):
            path.write_text(json.dumps({**data, **edit}))
            with self.assertRaisesRegex(SystemExit, "is not the lora-config.json of a build_lora build \\(the kernel would not take its settings: ", msg=edit):
                score_lora.read_build(path)

    def test_the_scorer_checks_the_keys_of_a_config_without_the_dataset(self):
        """`score_lora` is given the config, not the dataset: the two files are the shas the config records, and the set of ids, which the kernel compares with `eval_tasks.jsonl`, is the config's."""
        data, folder = json.loads(self.config("full").read_text()), self.dir / "full" / "data"
        moved = self.dir / "alone" / "lora-config.json"
        moved.parent.mkdir()
        moved.write_text(json.dumps(data))
        self.assertEqual(score_lora.read_build(moved), {"ids": set(self.ids), **keys_of(self.full), "smoke": False})
        lora_eval.check_keys(kernel_cfg(data))
        with self.assertRaises(FileNotFoundError, msg="with a folder, the files are read"):
            lora_eval.check_keys(kernel_cfg(data), moved.parent)
        fewer = {**data, "built_from": {**data["built_from"], "eval": {**data["built_from"]["eval"], "ids": self.ids[1:]}}}
        lora_eval.check_keys(kernel_cfg(fewer))
        with self.assertRaisesRegex(SystemExit, "names another evaluation set"):
            lora_eval.check_keys(kernel_cfg(fewer), folder)
        moved.write_text(json.dumps(fewer))
        base, lora = self.replies(keys_of(self.full))
        r = score_lora.score(base, lora, [self.heldout], domain=ARITH, build=score_lora.read_build(moved))
        self.assertEqual((r["paired"]["overall"]["n"], r["outside"]["replies"], r["outside"]["tasks"]), (len(self.ids) - 1, 2, self.ids[:1]), "a task taken out of the set by hand is not hidden: its replies are printed as outside the set")

    def test_a_smoke_adapter_under_a_full_runs_name_is_refused(self):
        out = self.dir / "in" / "earlier" / "adapter"
        out.mkdir(parents=True)
        (out / "adapter_model.safetensors").write_text("x")
        info = lora_eval.train_info(kernel_cfg(self.smoke), last=4, start=0, steps=4, counts={}, examples=8, gb=4, world=2, losses=[1.0], wall_s=1, tokens=1, step_secs=[1.0], peak_gib=1)
        (out / "train.json").write_text(json.dumps(info))
        self.assertEqual((info["status"], info["smoke"], info["adapter_sha"]), ("done", True, self.smoke["adapter_sha"]))
        with self.assertRaisesRegex(SystemExit, "trained under adapter key"):
            lora_eval.find_adapter([self.dir / "in"], self.full["adapter_sha"])

    def test_a_run_json_of_the_smoke_run_is_refused_with_the_full_builds_config(self):
        with self.assertRaisesRegex(SystemExit, "is the run of adapter key"):
            score_lora.check_run({"smoke": True, "config": keys_of(self.smoke)}, score_lora.read_build(self.config("full")))

    def test_the_build_says_its_keys_are_a_smoke_runs(self):
        self.assertIn("[SMOKE build: keys of its own, never a full run's]", build_lora.keys_line(self.smoke))
        self.assertNotIn("SMOKE", build_lora.keys_line(self.full))


class KernelChecksItsKeys(Kit):
    """The keys are the builder's, and the kernel does not take its word for them: it computes both again from the files of the dataset."""

    def setUp(self):
        super().setUp()
        self.build()
        self.data = self.dir / "out" / "data"

    def cfg(self, **edit) -> dict:
        return lora_eval.resolve_config({**json.loads((self.data / "lora-config.json").read_text()), **edit})

    def test_a_dataset_as_it_was_built_passes(self):
        lora_eval.check_keys(self.cfg(), self.data)
        lora_eval.check_keys(self.cfg())
        for name, kw in (("smoke", {"smoke": True}), ("export", {"export_gguf": True}), ("k", {"k": 2}), ("mix", {"mix": True})):
            self.build(out=self.dir / name, **kw)
            d = self.dir / name / "data"
            lora_eval.check_keys(lora_eval.resolve_config(json.loads((d / "lora-config.json").read_text())), d)
            lora_eval.check_keys(lora_eval.resolve_config(json.loads((d / "lora-config.json").read_text())))          # and without the dataset, as the scorer checks it
            self.assertEqual(score_lora.read_build(d / "lora-config.json")["smoke"], name == "smoke")

    def test_a_setting_edited_in_the_dataset_stops_the_kernel(self):
        for name, value, what in (("max_new_tokens", 1024, "evaluation key"), ("k", 2, "evaluation key"), ("seed_base", 1, "evaluation key"), ("sampling", {"temperature": 1.0}, "evaluation key"),
                                  ("max_steps", 3, "adapter key"), ("lr", 1e-3, "adapter key"), ("smoke", True, "adapter key"), ("pins", ["transformers==9.9.9"], "adapter key")):
            with self.assertRaisesRegex(SystemExit, f"lora-config.json says {what} .* was changed after the build", msg=name):
                lora_eval.check_keys(self.cfg(**{name: value}), self.data)
        for name, value in (("train_hours", 1.0), ("deadline_h", 2.0), ("save_every", 3), ("gpu_name", ""), ("export_gguf", True)):
            lora_eval.check_keys(self.cfg(**{name: value}), self.data)

    def test_a_key_edited_or_missing_stops_the_kernel(self):
        mine = self.cfg()
        for name, what in (("adapter_sha", "adapter key"), ("eval_sha", "evaluation key")):
            for value in ("0" * 16, "", None):
                with self.assertRaises(SystemExit) as e:
                    lora_eval.check_keys({**mine, name: value}, self.data)
                self.assertEqual(str(e.exception), f"lora-config.json says {what} {value!r}, and the files and settings of this dataset make {mine[name]!r}: the config, train.jsonl or eval_tasks.jsonl was changed "
                                                   "after the build (or the dataset is of another version of the kit). Build again with build_lora.py; a setting is never edited in the dataset")
        with self.assertRaisesRegex(SystemExit, "says evaluation key"):
            lora_eval.check_keys({k: v for k, v in mine.items() if k != "built_from"}, self.data)
        with self.assertRaisesRegex(SystemExit, "says evaluation key"):
            lora_eval.check_keys({**mine, "built_from": {**mine["built_from"], "eval": {**mine["built_from"]["eval"], "excluded": {**mine["built_from"]["eval"]["excluded"], "sha": "0" * 16}}}}, self.data)
        with self.assertRaisesRegex(SystemExit, "says adapter key ''"):
            lora_eval.check_keys(lora_eval.resolve_config({}), self.data)

    def test_other_records_or_other_tasks_than_the_builds_stop_the_kernel(self):
        train, tasks = self.data / "train.jsonl", self.data / "eval_tasks.jsonl"
        kept = train.read_text()
        train.write_text(kept + kept.splitlines()[0] + "\n")
        with self.assertRaisesRegex(SystemExit, "says adapter key"):
            lora_eval.check_keys(self.cfg(), self.data)
        train.write_text(kept)
        lora_eval.check_keys(self.cfg(), self.data)
        tasks.write_text("".join(x + "\n" for x in tasks.read_text().splitlines()[1:]))
        with self.assertRaisesRegex(SystemExit, "says evaluation key"):
            lora_eval.check_keys(self.cfg(), self.data)

    def test_the_adapter_key_is_of_the_records_the_kernel_reads_not_of_the_bytes_of_the_sft_file(self):
        """The same records written another way (other separators, a blank line): another SFT file by its bytes, the same `train.jsonl`, the same adapter."""
        spaced = self.dir / "spaced.jsonl"
        spaced.write_text("\n" + "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in sft_rows()))
        cfg = self.build(sft=spaced, out=self.dir / "spaced")
        mine = json.loads((self.data / "lora-config.json").read_text())
        d = self.dir / "spaced" / "data"
        self.assertEqual((d / "train.jsonl").read_bytes(), (self.data / "train.jsonl").read_bytes())
        self.assertNotEqual(cfg["built_from"]["train"]["sha256"], mine["built_from"]["train"]["sha256"], "the SFT file's own sha, kept for the record")
        self.assertNotEqual(cfg["built_from"]["train"]["sha256"], cfg["built_from"]["train"]["train_sha256"])
        self.assertEqual(keys_of(cfg), keys_of(mine))
        lora_eval.check_keys(lora_eval.resolve_config(json.loads((d / "lora-config.json").read_text())), d)
        lora_eval.check_keys(lora_eval.resolve_config(json.loads((d / "lora-config.json").read_text())))          # without the dataset it is `train_sha256` that stands for the file, not the SFT file's sha
        self.assertEqual(score_lora.read_build(d / "lora-config.json")["adapter_sha"], mine["adapter_sha"])

    def test_the_file_sha_is_of_the_bytes(self):
        import hashlib
        self.assertEqual(lora_eval.file_sha(self.data / "train.jsonl"), hashlib.sha256((self.data / "train.jsonl").read_bytes()).hexdigest()[:16])


class KernelMain(Kit):
    """`lora_eval.main` itself, with the workers, pip, the download and the clock as stand-ins: what it hands the two workers, and what it refuses before it spends anything."""

    NOW, T0 = 1000.0, 400.0

    def run_main(self, names=("Tesla T4", "Tesla T4"), trains=True, **build_kw):
        self.build(**build_kw)
        data, out, inp = pathlib.Path(build_kw.get("out", self.dir / "out")) / "data", self.dir / "working", self.dir / "input"
        out.mkdir(exist_ok=True)
        cfg = lora_eval.resolve_config(json.loads((data / "lora-config.json").read_text()))
        self.calls, self.sh, self.downloads, self.exports = [], [], [], []

        def run_worker(stage, gpus, data_dir, env):
            self.calls.append((stage, gpus, data_dir, env))
            if stage == "train" and trains:
                adapter = out / ("adapter-smoke" if cfg["smoke"] else "adapter")
                adapter.mkdir()
                (adapter / "adapter_model.safetensors").write_text("x")
                (adapter / "train.json").write_text(json.dumps({"status": "done", "adapter_sha": cfg["adapter_sha"], "tok_per_s_per_gpu": 1.0, "step_s": 1.0, "peak_gib": 1.0}))
            return 0
        hub = SimpleNamespace(snapshot_download=lambda *a, **kw: self.downloads.append(a))
        clock = SimpleNamespace(time=lambda: self.NOW, strftime=time.strftime, gmtime=time.gmtime)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"huggingface_hub": hub}))
            stack.enter_context(mock.patch.object(sys, "argv", ["lora_eval.py"]))
            stack.enter_context(mock.patch.dict(os.environ, {}))
            stack.enter_context(mock.patch.object(lora_eval.run_teacher, "sh", lambda *a, **kw: self.sh.append(a)))
            for name, value in (("OUT", out), ("INPUT", inp), ("T0", self.T0), ("time", clock), ("find_data", lambda: data), ("gpu_names", lambda: list(names)), ("run_worker", run_worker), ("log", lambda *a: None),
                                ("subprocess", SimpleNamespace(run=lambda *a, **kw: None)), ("export_gguf", lambda c, adapter: self.exports.append(adapter) or {"gguf": "x"})):
                stack.enter_context(mock.patch.object(lora_eval, name, value))
            lora_eval.main()
        self.cfg, self.out = cfg, out
        return json.loads((out / "run.json").read_text())

    def env(self, stage):
        return next(env for s, _, _, env in self.calls if s == stage)

    def test_training_is_told_to_stop_the_builds_reserve_before_the_deadline(self):
        """The fourth review's surviving mutant: `train_deadline(..., 0)` in `main`. A deadline at which the reserve binds, not the hours."""
        self.run_main(deadline_h=3.0, export_gguf=True)
        deadline = self.T0 + 3.0 * 3600
        self.assertGreater(self.cfg["eval_reserve_s"], self.cfg["export_reserve_s"])
        self.assertLess(deadline - self.cfg["eval_reserve_s"], self.NOW + self.cfg["train_hours"] * 3600, "the reserve binds")
        self.assertEqual(self.env("train"), {"LORA_TRAIN_DEADLINE": str(deadline - self.cfg["eval_reserve_s"])})
        self.assertEqual(float(self.env("train")["LORA_TRAIN_DEADLINE"]), lora_eval.train_deadline(self.NOW, deadline, self.cfg["train_hours"], self.cfg["eval_reserve_s"]))

    def test_training_is_told_to_stop_at_its_hours_when_they_come_first(self):
        self.run_main(train_hours=0.5)
        self.assertEqual(self.env("train"), {"LORA_TRAIN_DEADLINE": str(self.NOW + 0.5 * 3600)}, "from now (the download is behind it), not from the kernel's start")

    def test_the_evaluation_ends_the_exports_reserve_before_the_deadline_and_only_with_the_export(self):
        """The other surviving mutant: the evaluation's deadline without `- export_reserve_s`."""
        run = self.run_main(export_gguf=True)
        deadline = self.T0 + self.cfg["deadline_h"] * 3600
        self.assertGreater(self.cfg["export_reserve_s"], 0)
        self.assertEqual(self.env("eval"), {"LORA_DEADLINE": str(deadline - self.cfg["export_reserve_s"])})
        self.assertEqual(self.exports, [self.out / "adapter"])
        self.assertEqual(run["gguf"], {"gguf": "x"})

    def test_without_the_export_the_evaluation_has_the_whole_session(self):
        run = self.run_main()
        self.assertEqual(self.env("eval"), {"LORA_DEADLINE": str(self.T0 + self.cfg["deadline_h"] * 3600)})
        self.assertEqual((self.exports, "gguf" in run), ([], False))
        self.assertEqual([(s, g) for s, g, _, _ in self.calls], [("train", 2), ("eval", 2)])
        self.assertEqual((run["gpus"], run["smoke"], run["replies"], run["train"]["status"]), (2, False, {"base": 0, "lora": 0}, "done"))
        self.assertEqual(keys_of(run["config"]), keys_of(self.cfg))

    def test_the_evaluations_deadline_is_a_function_of_the_config(self):
        self.assertEqual(lora_eval.eval_deadline({"export_gguf": True, "export_reserve_s": 1800}, 9000.0), 7200.0)
        self.assertEqual(lora_eval.eval_deadline({"export_gguf": False, "export_reserve_s": 1800}, 9000.0), 9000.0, "a reserve of an export that is off is not taken")

    def test_a_dataset_that_was_edited_stops_the_kernel_before_it_installs_or_downloads_anything(self):
        self.build(out=self.dir / "edited")
        path = self.dir / "edited" / "data" / "lora-config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "max_new_tokens": 1024}))
        with mock.patch.object(build_lora, "build", lambda *a, **kw: None), self.assertRaisesRegex(SystemExit, "lora-config.json says evaluation key"):
            self.run_main(out=self.dir / "edited")
        self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))

    def run_built(self, out, **kw):
        """`run_main` on the dataset under `out`, which is built already."""
        with mock.patch.object(build_lora, "build", lambda *a, **k: None):
            return self.run_main(out=out, **kw)

    def test_fewer_gpus_than_the_build_is_for_stop_the_kernel_before_it_installs_anything(self):
        with self.assertRaisesRegex(SystemExit, "the kernel got 1 GPU\\(s\\) and the build is for 2: .* build again with --gpus 1"):
            self.run_main(names=("Tesla T4",))
        self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))

    def test_the_gpus_the_build_is_for_are_the_processes_even_when_the_kernel_has_more(self):
        self.run_main(gpus=1)
        self.assertEqual([(s, g) for s, g, _, _ in self.calls], [("train", 1), ("eval", 1)], "--gpus 1 on two cards: one process, as the adapter key says")
        self.assertEqual(len(self.sh), 1, "one pip install")
        self.assertEqual(len(self.downloads), 1)

    def test_a_kept_adapter_is_not_trained_again_and_a_smoke_run_never_keeps_one(self):
        self.build()
        cfg = json.loads((self.dir / "out" / "data" / "lora-config.json").read_text())
        earlier = self.dir / "input" / "earlier" / "adapter"
        earlier.mkdir(parents=True)
        (earlier / "adapter_model.safetensors").write_text("kept")
        (earlier / "train.json").write_text(json.dumps({"status": "partial", "adapter_sha": cfg["adapter_sha"], "tok_per_s_per_gpu": 1.0, "step_s": 1.0, "peak_gib": 1.0}))
        run = self.run_built(self.dir / "out", trains=False)
        self.assertEqual([s for s, _, _, _ in self.calls], ["eval"])
        self.assertEqual((self.out / "adapter" / "adapter_model.safetensors").read_text(), "kept")
        self.assertEqual((run["train"]["status"], "train_rc" in run), ("partial", False))
        self.build(out=self.dir / "smoke", smoke=True)
        run = self.run_built(self.dir / "smoke")
        self.assertEqual([s for s, _, _, _ in self.calls], ["train", "eval"], "the smoke run trains its own four steps")
        self.assertTrue((self.out / "adapter-smoke" / "train.json").exists())
        self.assertIs(run["smoke"], True)

    def test_an_attached_adapter_of_another_schedule_stops_the_kernel(self):
        self.build()
        earlier = self.dir / "input" / "earlier" / "adapter"
        earlier.mkdir(parents=True)
        (earlier / "adapter_model.safetensors").write_text("x")
        (earlier / "train.json").write_text(json.dumps({"status": "done", "adapter_sha": "0" * 16}))
        with self.assertRaisesRegex(SystemExit, "trained under adapter key '0000000000000000'"):
            self.run_built(self.dir / "out")
        self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []), "before the pip install and the download (PR 34 fifth review, m3)")


class CarryAdapter(unittest.TestCase):
    """The fourth review's two surviving mutants of `carry_adapter`: the `adapter/` that is in the way, and the modes of a read-only attached output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.found = self.root / "in" / "earlier" / "adapter"
        (self.found / "sub").mkdir(parents=True)
        (self.found / "adapter_model.safetensors").write_bytes(b"\x00weights\xff")
        (self.found / "train.json").write_text(json.dumps({"status": "done", "adapter_sha": "aaaa"}))
        (self.found / "sub" / "README.md").write_text("card")
        self.out = self.root / "out"
        self.out.mkdir()

    def test_an_adapter_folder_that_is_not_a_finished_one_is_replaced(self):
        """A session that was killed while it saved left `adapter/` with no train.json: `find_adapter` does not take it, and the rename onto a folder that is not empty would fail."""
        stale = self.out / "adapter"
        stale.mkdir()
        (stale / "adapter_model.safetensors").write_text("half written")
        (stale / "leftover.bin").write_text("x")
        self.assertIsNone(lora_eval.find_adapter([self.out]))
        kept = lora_eval.carry_adapter(self.found, self.out)
        self.assertEqual(kept, stale)
        self.assertEqual(sorted(p.name for p in kept.iterdir()), ["adapter_model.safetensors", "sub", "train.json"], "nothing of the folder that was in the way")
        self.assertEqual((kept / "adapter_model.safetensors").read_bytes(), b"\x00weights\xff")
        self.assertEqual((kept / "sub" / "README.md").read_text(), "card")
        self.assertEqual(lora_eval.find_adapter([self.out], "aaaa"), kept)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["adapter"])

    def test_the_copy_of_a_read_only_output_can_be_written_and_removed(self):
        """An attached output is mounted read-only. A copy that kept the modes (of a file: `copy2`; of a folder: `copytree` whatever the copy function) could not be given a new train.json nor removed."""
        paths = [self.found / "sub" / "README.md", self.found / "adapter_model.safetensors", self.found / "train.json", self.found / "sub", self.found]
        for p in paths:
            p.chmod(0o555 if p.is_dir() else 0o444)
        self.addCleanup(lambda: [p.chmod(0o755 if p.is_dir() else 0o644) for p in reversed(paths)])
        kept = lora_eval.carry_adapter(self.found, self.out)
        for p in (kept, kept / "sub", kept / "adapter_model.safetensors", kept / "train.json", kept / "sub" / "README.md"):
            self.assertTrue(p.stat().st_mode & stat.S_IWUSR, f"{p.name}: writable by the session that owns the copy")
        (kept / "train.json").write_text("{}")
        (kept / "sub" / "new").write_text("x")
        self.assertEqual(stat.S_IMODE(self.found.stat().st_mode), 0o555, "the attached output is not touched")
        again = lora_eval.carry_adapter(self.found, self.out)
        self.assertEqual(json.loads((again / "train.json").read_text())["adapter_sha"], "aaaa", "and a later copy replaces it")
        self.assertFalse((again / "sub" / "new").exists())

    def test_copy_bytes_refuses_a_folder_that_exists(self):
        with self.assertRaises(FileExistsError):
            lora_eval.copy_bytes(self.found, self.out)


if __name__ == "__main__":
    unittest.main()

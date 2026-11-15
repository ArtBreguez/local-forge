"""The tasks that are asked are in the evaluation key by their bytes, a resume of another adapter is refused before it costs anything, and the worker uses what the keys hold (PR 34 fifth review, m1-m3).

m1: `eval_tasks.jsonl` is in the evaluation key as `train.jsonl` is in the adapter key (the id, the family and the prompt of every task), the kernel refuses a file that is not the build's, and the
scorer takes a row's family from the held-out file. m2: the mutants that survived (a `--gpus` that moves the cap on the steps, `k` > 1 in `work_items`, and the training worker's own use of the seed, the
loss chunk, the smoke steps and the smoke guard, run on stand-ins). m3: the builder says what a resume with another `--train-hours` does (it is refused), and refuses it itself, with the reason, when it is
given the fetched output of the run that is resumed; the kernel refuses it before it installs anything.

Sixth review. m1: a smoke run takes no reply of an attached output (the builder said so, the kernel read them, asked nothing and measured nothing). m2: the kernel's refusals and the builder's last line name
the accelerator the build asks for, not always the two T4. Its notes: the estimate of a resume that skips training counts no training, and the mutants that survived (`any` for `all` over the cards, the
default pins, the smoke guard of the checkpoints, an empty list of differences).
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from distill import outputs

from .test_kaggle_lora import ARITH, HELDOUT, Kit, build_lora, lora_eval, run_teacher, score_lora, sft_rows, write_jsonl
from .test_kaggle_lora_evalset import keys_of, line, reply
from . import test_kaggle_lora_keys as keys_tests

kernel_cfg = keys_tests.kernel_cfg

OTHER = "0" * 16


def slow(where: pathlib.Path) -> pathlib.Path:
    """A smoke run's run.json with a training speed at which the hours cap the steps."""
    path = where / "smoke-run.json"
    path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": 0.47}}))
    return path


def tasks_of(data: pathlib.Path) -> list:
    return [json.loads(x) for x in (data / "eval_tasks.jsonl").read_text().splitlines()]


def put_adapter(root: pathlib.Path, sha: str, status: str = "done", name: str = "adapter") -> pathlib.Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_text("x")
    (d / "train.json").write_text(json.dumps({"status": status, "adapter_sha": sha, "tok_per_s_per_gpu": 1.0, "step_s": 1.0, "peak_gib": 1.0}))
    return d


def put_checkpoint(root: pathlib.Path, sha: str, step: int = 10) -> pathlib.Path:
    d = root / "ckpt" / f"step-{step}"
    d.mkdir(parents=True)
    (d / "DONE").write_text(json.dumps({"step": step, "adapter_sha": sha}))
    return d


class TasksInTheKey(Kit):
    """m1: the text and the family of the tasks of `eval_tasks.jsonl` were in no key, and the kernel did not check them."""

    def setUp(self):
        super().setUp()
        self.cfg = self.build()
        self.data = self.dir / "out" / "data"
        self.file = self.data / "eval_tasks.jsonl"

    def kernel(self) -> dict:
        return lora_eval.resolve_config(json.loads((self.data / "lora-config.json").read_text()))

    def rewrite(self, edit) -> None:
        write_jsonl(self.file, [edit(t) for t in tasks_of(self.data)])

    def test_the_evaluation_key_is_of_the_bytes_of_the_file_the_kernel_reads(self):
        ev = self.cfg["built_from"]["eval"]
        self.assertEqual(ev["sha256"], lora_eval.file_sha(self.file))
        self.assertEqual(self.cfg["eval_sha"], lora_eval.eval_key(self.cfg["adapter_sha"], ev["excluded"]["sha"], lora_eval.file_sha(self.file), kernel_cfg(self.cfg)))
        self.assertEqual(json.loads((self.data / "lora-config.json").read_text())["built_from"]["eval"]["sha256"], ev["sha256"])
        self.assertEqual(build_lora.lines_sha(self.file.read_text().splitlines()), ev["sha256"])
        self.assertNotEqual(ev["sha256"], self.cfg["built_from"]["train"]["train_sha256"])
        lora_eval.check_keys(self.kernel(), self.data)

    def test_a_sentence_added_to_every_prompt_stops_the_kernel(self):
        """The review's first repro: `check_keys` passed, and the whole evaluation was spent before the scorer refused the first reply."""
        self.rewrite(lambda t: {**t, "messages": [*t["messages"][:-1], {**t["messages"][-1], "content": t["messages"][-1]["content"] + " Think twice."}]})
        with self.assertRaisesRegex(SystemExit, "lora-config.json says evaluation key .* eval_tasks.jsonl was changed after the build"):
            lora_eval.check_keys(self.kernel(), self.data)

    def test_a_family_edited_in_the_file_stops_the_kernel(self):
        """The review's second repro: `kind` set to one family in every task passed, and the report had one family row."""
        self.rewrite(lambda t: {**t, "kind": "mul"})
        with self.assertRaisesRegex(SystemExit, "lora-config.json says evaluation key"):
            lora_eval.check_keys(self.kernel(), self.data)

    def test_one_task_changed_by_one_character_stops_the_kernel(self):
        kept = self.file.read_text()
        first = tasks_of(self.data)[0]["id"]
        for name, edit in (("the system message of one task", lambda t: {**t, "messages": [{**t["messages"][0], "content": t["messages"][0]["content"] + "."}, *t["messages"][1:]]} if t["id"] == first else t),
                           ("the family of one task", lambda t: {**t, "kind": "sub" if t["kind"] != "sub" else "add"} if t["id"] == first else t)):
            self.rewrite(edit)
            self.assertEqual(lora_eval.eval_ids(self.data), set(self.cfg["built_from"]["eval"]["ids"]), "the ids are the build's: only the text differs")
            with self.assertRaisesRegex(SystemExit, "says evaluation key", msg=name):
                lora_eval.check_keys(self.kernel(), self.data)
            self.file.write_text(kept)
            lora_eval.check_keys(self.kernel(), self.data)

    def test_a_config_that_names_another_set_than_the_file_stops_the_kernel(self):
        """What the scorer scores (`built_from.eval.ids`) is what the kernel asks (the file): both are the build's."""
        cfg = self.kernel()
        ev = cfg["built_from"]["eval"]
        for name, ids in (("a task fewer", ev["ids"][1:]), ("a task more", [*ev["ids"], "arith-9999-003"]), ("none", None), ("not sorted as the build writes them", ev["ids"][::-1])):
            with self.assertRaisesRegex(SystemExit, "names another evaluation set \\(built_from.eval.ids, what score_lora scores\\) than the tasks of eval_tasks.jsonl", msg=name):
                lora_eval.check_keys({**cfg, "built_from": {**cfg["built_from"], "eval": {**ev, "ids": ids}}}, self.data)

    def changed_prompt(self):
        """The subject's prompt code changed between a build and its resume: every held-out task is asked with another user message."""
        asked = type(ARITH).messages
        return mock.patch.object(type(ARITH), "messages", lambda dom, task, mode: [*asked(dom, task, mode)[:-1], {"role": "user", "content": asked(dom, task, mode)[-1]["content"] + " Show the steps."}])

    def test_a_prompt_that_changed_is_another_evaluation_of_the_same_adapter(self):
        with self.changed_prompt():
            resume = self.build(out=self.dir / "resume")
        self.assertEqual(resume["built_from"]["eval"]["ids"], self.cfg["built_from"]["eval"]["ids"], "the same ids")
        self.assertEqual(resume["built_from"]["eval"]["excluded"]["sha"], self.cfg["built_from"]["eval"]["excluded"]["sha"], "the same exclusion")
        self.assertEqual(resume["adapter_sha"], self.cfg["adapter_sha"], "the adapter is kept")
        self.assertNotEqual(resume["eval_sha"], self.cfg["eval_sha"], "the review: the same evaluation key")
        self.assertNotEqual(resume["built_from"]["eval"]["sha256"], self.cfg["built_from"]["eval"]["sha256"])

    def test_replies_to_the_old_prompt_are_asked_again_and_counted_not_scored_and_do_not_stop_the_scorer(self):
        """The review: with the prompt in no key, a reply to the old prompt that the resume could not ask again before the deadline was not "counted and not scored": it stopped the scorer."""
        with self.changed_prompt():
            resume = self.build(out=self.dir / "resume")
        ids = self.cfg["built_from"]["eval"]["ids"]
        earlier = self.dir / "in" / "earlier"
        earlier.mkdir(parents=True)
        for arm in lora_eval.ARMS:
            write_jsonl(earlier / f"replies-{arm}.jsonl", [line(t, [reply(t, arm == "lora", keys_of(self.cfg))]) for t in HELDOUT if t["id"] in ids])
        previous = {arm: run_teacher.load_previous(lora_eval.previous_files(arm, self.dir / "work", self.dir / "in")) for arm in lora_eval.ARMS}
        self.assertEqual(lora_eval.work_items(tasks_of(self.data), previous, kernel_cfg(self.cfg)), [])
        self.assertEqual(sorted(t["id"] for t, _ in lora_eval.work_items(tasks_of(self.dir / "resume" / "data"), previous, kernel_cfg(resume))), ids)
        score = lambda name: score_lora.score(earlier / "replies-base.jsonl", earlier / "replies-lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / name / "data" / "lora-config.json"))
        with self.changed_prompt():
            r = score("resume")
        self.assertEqual((r["paired"]["overall"]["n"], r["other_eval"]["replies"], r["other_adapter"]["replies"]), (0, 2 * len(ids), 0))
        self.assertEqual(score("out")["paired"]["overall"]["n"], len(ids), "under the build that asked them, and its prompt, they are scored")
        stamped = self.dir / "stamped"
        stamped.mkdir()
        for arm in lora_eval.ARMS:
            write_jsonl(stamped / f"{arm}.jsonl", [line(t, [reply(t, True, keys_of(resume))]) for t in HELDOUT if t["id"] in ids])
        with self.changed_prompt(), self.assertRaisesRegex(SystemExit, "answered another prompt than the task's now"):
            score_lora.score(stamped / "base.jsonl", stamped / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / "resume" / "data" / "lora-config.json"))


class FamilyOfARow(Kit):
    """m1: `score_lora` took a row's family from the replies line. It is the task's, in the held-out file."""

    def score(self, kind_of) -> dict:
        cfg = self.build()
        ids = cfg["built_from"]["eval"]["ids"]
        for arm in lora_eval.ARMS:
            rows = [line(t, [reply(t, arm == "lora", keys_of(cfg))]) for t in HELDOUT if t["id"] in ids]
            write_jsonl(self.dir / f"{arm}.jsonl", [kind_of(r) for r in rows])
        self.by_kind = cfg["built_from"]["eval"]["by_kind"]
        return score_lora.score(self.dir / "base.jsonl", self.dir / "lora.jsonl", [self.heldout], domain=ARITH, build=score_lora.read_build(self.dir / "out" / "data" / "lora-config.json"))

    def check(self, r: dict) -> None:
        self.assertEqual(self.by_kind, {"add": 3, "mul": 3, "sub": 3})
        self.assertEqual({k: v["n"] for k, v in r["paired"]["by_kind"].items()}, self.by_kind)
        for arm in lora_eval.ARMS:
            self.assertEqual({k: v["replies"] for k, v in r["arms"][arm]["by_kind"].items()}, self.by_kind, arm)
        self.assertEqual({k: v["lora_only"] for k, v in r["paired"]["by_kind"].items()}, self.by_kind)

    def test_a_replies_file_that_names_one_family_for_every_task_is_scored_by_the_tasks_families(self):
        """The review's repro: `crc: 57` for a build of six families."""
        self.check(self.score(lambda rec: {**rec, "kind": "mul"}))

    def test_a_family_no_task_has_makes_no_row(self):
        r = self.score(lambda rec: {**rec, "kind": "crc"})
        self.check(r)
        self.assertFalse(any("crc" in x for x in score_lora.lines_of(r)))

    def test_a_line_that_names_no_family_is_scored_by_the_tasks(self):
        self.check(self.score(lambda rec: {k: v for k, v in rec.items() if k != "kind"}))


class GpusAndK(Kit):
    """m2: a `--gpus` that moves the cap on the steps, and `k` > 1 in what is asked again."""

    def test_gpus_reach_the_cap_on_the_steps(self):
        """The mutant: `plan(..., gpus=<the default>)`. One process takes twice as long for a step, so the same hours hold half the steps."""
        two = self.build(out=self.dir / "two", measured=slow(self.dir), train_hours=2.0)
        one = self.build(out=self.dir / "one", measured=slow(self.dir), train_hours=2.0, gpus=1)
        est = lambda cfg: cfg["built_from"]["estimate"]
        self.assertEqual((two["max_steps"], one["max_steps"]), (2, 1))
        self.assertEqual((est(two)["gpus"], est(one)["gpus"], est(one)["step_s"]), (2, 1, 2 * est(two)["step_s"]))
        self.assertEqual((est(two)["steps_full"], est(one)["steps_full"]), (4, 4))
        with_gpus = lora_eval.adapter_key(two["built_from"]["train"]["train_sha256"], {**kernel_cfg(two), "gpus": 1})
        self.assertNotEqual(one["adapter_sha"], with_gpus, "another adapter by the cap too, not only by the number of processes")
        self.assertEqual(one["adapter_sha"], lora_eval.adapter_key(two["built_from"]["train"]["train_sha256"], {**kernel_cfg(two), "gpus": 1, "max_steps": 1}))

    def test_every_sample_of_another_evaluation_is_asked_again_with_k_above_one(self):
        """The mutant: `other_eval(..., cfg, 1)`. With `k` 3, the second and third replies of another evaluation key were never asked again."""
        cfg = self.build(k=3)
        mine, tasks = keys_of(cfg), tasks_of(self.dir / "out" / "data")
        t0, t1, t2 = (next(h for h in HELDOUT if h["id"] == t["id"]) for t in tasks[:3])
        theirs = {"adapter_sha": cfg["adapter_sha"], "eval_sha": OTHER}
        rows = {"base": [line(t0, [reply(t0, True, mine), reply(t0, True, theirs), reply(t0, True, mine)]), line(t1, [reply(t1, True, mine)] * 3), line(t2, [reply(t2, True, mine), reply(t2, True, mine), reply(t2, True, mine)])],
                "lora": [line(t0, [reply(t0, True, mine)] * 3), line(t1, [reply(t1, True, mine)] * 3), line(t2, [reply(t2, True, mine), reply(t2, True, mine), reply(t2, True, {**mine, "adapter_sha": OTHER})])]}
        previous = {arm: {r["id"]: r for r in rows[arm]} for arm in lora_eval.ARMS}
        items = lora_eval.work_items(tasks[:3], previous, kernel_cfg(cfg))
        self.assertEqual(sorted((t["id"], i) for t, i in items), sorted([(t0["id"], 1), (t2["id"], 2)]), "the second sample of one task (the base's), the third of another (the adapter's): for both arms")
        self.assertEqual(lora_eval.other_eval(rows["base"][0], kernel_cfg(cfg), 3), [1])
        self.assertEqual(lora_eval.other_eval(rows["base"][0], kernel_cfg(cfg), 1), [], "below k only")


class Tensor:
    def __init__(self, data):
        self.data = data

    def __int__(self):
        return int(self.data[0])


class Loss:
    def detach(self):
        return 1.0

    def __truediv__(self, n):
        return SimpleNamespace(scaled=None)


class TrainWorker(unittest.TestCase):
    """m2: `train_worker` itself on stand-ins for torch, the tokenizer and the student. What it does with the settings of the adapter key: the shuffle's seed, the loss chunk, the smoke steps, and that a
    smoke run looks for no checkpoint (the review's four surviving mutants), and with them the window, `mix`, the batch, the learning rate and the checkpoints."""

    N = 10

    def run_worker(self, **settings) -> dict:
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        out, data = root / "working", root / "data"
        out.mkdir()
        data.mkdir()
        write_jsonl(data / "train.jsonl", [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(self.N)])
        cfg = lora_eval.resolve_config({"adapter_sha": "feedfacefeedface", "eval_sha": "0ddba11c0ffee000", **settings})
        seen = {"steps": [], "chunks": [], "encode": [], "checkpoint_lookups": [], "saved": [], "lrs": [], "clips": [], "adamw": []}
        examples = [{"input_ids": [i] * (i + 1), "labels": [i] * (i + 1)} for i in range(self.N)]          # example i is i+1 tokens of the value i: the longest are the last

        def base_model(input_ids, attention_mask, use_cache):
            seen["steps"][-1].append(input_ids.data[0][0])
            return SimpleNamespace(last_hidden_state="hidden")

        inner = SimpleNamespace(model=base_model, lm_head="head")
        model = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=True), SimpleNamespace(requires_grad=False)], get_base_model=lambda: inner, train=lambda: None,
                                save_pretrained=lambda path: pathlib.Path(path).mkdir(parents=True, exist_ok=True))
        groups = [{"lr": None}]
        opt = SimpleNamespace(param_groups=groups, zero_grad=lambda set_to_none: None, state_dict=lambda: {}, load_state_dict=lambda s: None)
        scaler = SimpleNamespace(scale=lambda x: SimpleNamespace(backward=lambda: None), unscale_=lambda o: None, update=lambda: None, state_dict=lambda: {}, load_state_dict=lambda s: None,
                                 step=lambda o: (seen["lrs"].append(groups[0]["lr"]), seen["steps"].append([])))
        torch = SimpleNamespace(float16="fp16", tensor=lambda data, device=None: Tensor(data), ones_like=lambda x: x, autocast=lambda *a, **kw: contextlib.nullcontext(),
                                cuda=SimpleNamespace(set_device=lambda i: None, max_memory_allocated=lambda: 0),
                                optim=SimpleNamespace(AdamW=lambda params, lr, weight_decay: (seen["adamw"].append((len(params), lr, weight_decay)), opt)[1]),
                                amp=SimpleNamespace(GradScaler=lambda device: scaler), nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda params, norm: seen["clips"].append(norm))))
        torch.utils = SimpleNamespace(checkpoint=SimpleNamespace())
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: "tok"))

        def encode_records(tok, rows, max_len, mix):
            seen["encode"].append((tok, len(rows), max_len, mix))
            return list(examples), {"think": self.N}

        def chunked_loss(t, hidden, labels, head, chunk):
            seen["chunks"].append(chunk)
            return Loss(), len(labels.data[0])

        def latest_checkpoint(roots, sha=None):
            seen["checkpoint_lookups"].append((list(roots), sha))
            return None
        seen["steps"].append([])
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"torch": torch, "torch.utils": torch.utils, "torch.utils.checkpoint": torch.utils.checkpoint, "transformers": transformers}))
            stack.enter_context(mock.patch.dict(os.environ, {"LORA_TRAIN_DEADLINE": "1e18"}))
            for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
                os.environ.pop(k, None)
            for name, value in (("OUT", out), ("INPUT", root / "input"), ("log", lambda *a: None), ("encode_records", encode_records), ("chunked_loss", chunked_loss), ("latest_checkpoint", latest_checkpoint),
                                ("load_student", lambda t, c, local, adapter=None, train=False: model), ("save_checkpoint", lambda t, m, o, s, step, ckpt, sha="": seen["saved"].append((step, sha)))):
                stack.enter_context(mock.patch.object(lora_eval, name, value))
            lora_eval.train_worker(cfg, data)
        seen["steps"] = seen["steps"][:-1]
        seen["train"] = json.loads((out / ("adapter-smoke" if cfg["smoke"] else "adapter") / "train.json").read_text())
        seen["roots"] = [out, root / "input"]
        return seen

    def test_the_steps_are_the_shuffle_of_the_configs_seed(self):
        """The mutant: `step_indices(..., <the default seed>)`."""
        got = self.run_worker(seed=8, global_batch=4, epochs=2.0)
        want = [lora_eval.step_indices(self.N, 4, step, 8) for step in range(5)]
        self.assertEqual(got["steps"], want, "2 epochs of 10 examples, 4 per step: 5 steps, each the stream of the seed 8")
        self.assertNotEqual(want, [lora_eval.step_indices(self.N, 4, step, lora_eval.DEFAULTS["seed"]) for step in range(5)], "and it is not the default seed's")
        self.assertEqual((got["train"]["status"], got["train"]["steps_done"], got["train"]["steps_planned"], got["train"]["global_batch"]), ("done", 5, 5, 4))
        self.assertEqual(self.run_worker(global_batch=4)["steps"], [lora_eval.step_indices(self.N, 4, step, 7) for step in range(5)])

    def test_the_loss_is_made_in_slices_of_the_configs_chunk(self):
        """The mutant: `chunked_loss(..., <the default chunk>)`."""
        got = self.run_worker(loss_chunk=256, global_batch=4, epochs=1.0)
        self.assertEqual(set(got["chunks"]), {256})
        self.assertEqual(len(got["chunks"]), 3 * 4, "3 steps of 4 sequences")
        self.assertEqual(set(self.run_worker(global_batch=4, epochs=1.0)["chunks"]), {512})

    def test_a_smoke_run_takes_the_configs_steps_on_the_longest_records_and_looks_for_no_checkpoint(self):
        """The mutants: the default `smoke_steps` for the steps (and for the records), and `latest_checkpoint` without the smoke guard: a smoke run would refuse an attached full checkpoint."""
        got = self.run_worker(smoke=True, smoke_steps=3, save_every=2)          # a step at which a full run would save: the smoke guard decides, not the 4 steps under a `save_every` of 10
        self.assertEqual((got["train"]["steps_planned"], got["train"]["steps_done"], got["train"]["status"], got["train"]["smoke"]), (3, 3, "done", True))
        self.assertEqual(got["train"]["examples"], 6, "3 steps of 2 per process, one process")
        self.assertEqual(len(got["steps"]), 3)
        self.assertEqual({i for step in got["steps"] for i in step}, {4, 5, 6, 7, 8, 9}, "the six longest records, each once")
        self.assertEqual(got["checkpoint_lookups"], [], "a smoke run resumes nothing")
        self.assertEqual(got["saved"], [], "and it writes no checkpoint")
        self.assertEqual(self.run_worker(global_batch=4, epochs=1.0, save_every=2)["saved"], [(2, "feedfacefeedface")], "a full run of 3 steps does, at the same step")
        five = self.run_worker(smoke=True, smoke_steps=5)
        self.assertEqual((five["train"]["steps_planned"], five["train"]["examples"]), (5, 10))

    def test_a_full_run_looks_for_a_checkpoint_of_its_adapter_key_and_saves_under_it(self):
        got = self.run_worker(global_batch=2, epochs=1.0, save_every=2)
        self.assertEqual(got["checkpoint_lookups"], [(got["roots"], "feedfacefeedface")])
        self.assertEqual(got["saved"], [(2, "feedfacefeedface"), (4, "feedfacefeedface")], "every 2 steps, and not at the last one (the adapter is saved there)")
        self.assertEqual(self.run_worker(global_batch=2, epochs=1.0, save_every=3)["saved"], [(3, "feedfacefeedface")])

    def test_the_window_mix_the_rate_and_the_clip_are_the_configs(self):
        got = self.run_worker(max_len=1024, mix=True, lr=1e-3, warmup=0.5, max_grad_norm=0.5, global_batch=5, epochs=2.0, max_steps=3)
        self.assertEqual(got["encode"], [("tok", self.N, 1024, True)])
        self.assertEqual(got["adamw"], [(1, 1e-3, 0.0)], "the parameters that train, at the config's rate")
        self.assertEqual(got["lrs"], [lora_eval.lr_at(step, 3, 1e-3, 0.5) for step in range(3)], "the cap is the length of the schedule")
        self.assertEqual(got["clips"], [0.5] * 3)
        self.assertEqual([len(s) for s in got["steps"]], [5, 5, 5])
        self.assertEqual(self.run_worker()["encode"], [("tok", self.N, 4096, False)])


class Advice(Kit):
    """m3: the builder said a resume with another `--train-hours` "trains again". The kernel refuses it."""

    def test_the_advice_of_a_plan_that_does_not_fit_says_what_the_kernel_does_on_a_resume(self):
        cfg = self.build(out=self.dir / "a", gpu_hours=1.0, deadline_h=0.1)
        self.assertTrue(cfg["plan_lines"][-2].endswith("lower --train-hours, --k, --per-family or --max-new-tokens (on a resume: a --train-hours that changes the cap on the steps is another adapter, and the kernel "
                                                     "REFUSES the adapter or the checkpoint it has attached: keep --train-hours, or build without --resume-from to train again; the other three are another "
                                                     "evaluation of the same adapter, and it asks again)"), cfg["plan_lines"][-2])
        self.assertNotIn("and it trains again", "\n".join(cfg["plan_lines"]))

    def test_the_refusals_say_how_to_train_again(self):
        put_adapter(self.dir / "in", OTHER)
        with self.assertRaisesRegex(SystemExit, "build again with the same ones, or without --resume-from to train again$"):
            lora_eval.find_adapter([self.dir / "in"], "feedfacefeedface")
        put_checkpoint(self.dir / "ck", OTHER)
        with self.assertRaisesRegex(SystemExit, "build again with the same ones, or without --resume-from to train again$"):
            lora_eval.latest_checkpoint([self.dir / "ck"], "feedfacefeedface")


class BuilderRefusesTheResume(Kit):
    """m3: the review's repro, at the builder: the build with other hours, given the fetched output of the run that is resumed, is refused before anything is written, with the reason."""

    def setUp(self):
        super().setUp()
        self.first = self.build(out=self.dir / "first", measured=slow(self.dir), train_hours=2.0)
        self.fetched = self.dir / "fetched"
        self.fetched.mkdir()

    def run_json(self, cfg: dict) -> None:
        (self.fetched / "run.json").write_text(json.dumps({"smoke": cfg["smoke"], "config": kernel_cfg(cfg)}))

    def resume(self, **kw) -> dict:
        return self.build(**{"out": self.dir / "resume", "measured": slow(self.dir), "train_hours": 2.0, "resume_from": "someone/example-arith-lora-eval", "resume_output": self.fetched, **kw})

    def test_other_hours_with_the_adapter_of_the_run_are_refused_with_what_differs_and_nothing_is_written(self):
        adapter = put_adapter(self.fetched, self.first["adapter_sha"], "partial")
        self.run_json(self.first)
        with self.assertRaises(SystemExit) as caught:
            self.resume(train_hours=3.0, max_new_tokens=1024, k=2)          # the evaluation differs too: it is not what is refused, and it is not named
        text = str(caught.exception)
        mine = self.build(out=self.dir / "alone", measured=slow(self.dir), train_hours=3.0)["adapter_sha"]
        self.assertEqual(text, f"--resume-output: the adapter {adapter} was trained under adapter key {self.first['adapter_sha']!r}, this build's is {mine!r} (another --sft or another training setting; --train-hours, "
                               "--measured and --gpus are training settings when they change the cap on the steps): build again with the same ones, or without --resume-from to train again. What differs from the run "
                               "of that output (its run.json), then -> now: max_steps 2 -> 3 (the cap on the steps: --train-hours, --measured and --gpus decide it). Nothing was written")
        self.assertFalse((self.dir / "resume").exists(), "before anything is written")

    def test_a_checkpoint_of_another_schedule_is_refused_too(self):
        ckpt = put_checkpoint(self.fetched, self.first["adapter_sha"])
        self.run_json(self.first)
        with self.assertRaisesRegex(SystemExit, f"^--resume-output: the checkpoint {ckpt} was written under adapter key .* max_steps 2 -> 3 .* Nothing was written$"):
            self.resume(train_hours=3.0)
        self.assertFalse((self.dir / "resume").exists())

    def test_every_training_setting_that_differs_is_named_and_the_records_too(self):
        put_adapter(self.fetched, self.first["adapter_sha"])
        self.run_json(self.first)
        with self.assertRaisesRegex(SystemExit, "then -> now: rank 16 -> 8; lr 0.00015 -> 0.0003; fla False -> True\\. Nothing was written$"):
            self.resume(rank=8, lr=3e-4, fla=True)
        other = write_jsonl(self.dir / "other-sft.jsonl", sft_rows()[:-2])
        with self.assertRaises(SystemExit) as caught:
            self.resume(sft=other)
        first, now = self.first["built_from"]["train"]["train_sha256"], build_lora.lines_sha([json.dumps(r) for r in sft_rows()[:-2]])
        self.assertIn(f"then -> now: the training records (train.jsonl {first} -> {now}: another --sft). Nothing was written", str(caught.exception))

    def test_without_a_run_json_of_that_adapter_the_two_keys_are_the_reason(self):
        put_adapter(self.fetched, self.first["adapter_sha"])
        for name, write in (("no run.json", lambda: None), ("not JSON", lambda: (self.fetched / "run.json").write_text("{")), ("no config", lambda: (self.fetched / "run.json").write_text("{}")),
                            ("a list", lambda: (self.fetched / "run.json").write_text("[]")), ("a config with no key", lambda: (self.fetched / "run.json").write_text(json.dumps({"config": {"max_steps": 9}})))):
            write()
            with self.assertRaises(SystemExit, msg=name) as caught:
                self.resume(train_hours=3.0)
            self.assertRegex(str(caught.exception), "^--resume-output: the adapter .* was trained under adapter key .* to train again\\. Nothing was written$", name)
        later = self.build(out=self.dir / "later", measured=slow(self.dir), train_hours=3.0)
        self.run_json(later)
        with self.assertRaises(SystemExit) as caught:
            self.resume(train_hours=3.0)
        self.assertNotIn("What differs", str(caught.exception), "the run.json is of a run of this build's key: the adapter is an older one's, and there is nothing to compare it with")
        self.run_json(self.build(out=self.dir / "third", measured=slow(self.dir), train_hours=2.0, rank=8))
        with self.assertRaises(SystemExit) as caught:
            self.resume(train_hours=3.0)
        self.assertNotIn("What differs", str(caught.exception), "nor with a run.json of a third key: the settings of the run that made the adapter are not known")

    def test_the_same_training_and_another_evaluation_is_built_and_says_what_is_kept(self):
        """A legitimate resume is never refused: the adapter key is the same whatever the evaluation asks."""
        adapter = put_adapter(self.fetched, self.first["adapter_sha"], "partial")
        self.run_json(self.first)
        for name, kw in (("the same build", {}), ("another budget", {"max_new_tokens": 1024}), ("k", {"k": 2}), ("fewer tasks", {"per_family": 2}), ("less quota", {"gpu_hours": 7.0}), ("the export", {"export_gguf": True})):
            cfg = self.resume(out=self.dir / name, **kw)
            self.assertEqual(cfg["adapter_sha"], self.first["adapter_sha"], name)
            self.assertEqual(cfg["plan_lines"][-1], f"resume:   {self.fetched} holds the finished adapter of this build's adapter key ({adapter}): training is skipped; the replies of another evaluation key are asked again", name)
            self.assertTrue((self.dir / name / "data" / "lora-config.json").exists())
            self.assertNotIn("plan_lines", json.loads((self.dir / name / "data" / "lora-config.json").read_text()))

    def test_an_output_with_nothing_of_another_key_is_built(self):
        nothing = "no finished adapter and nothing of another adapter key: the kernel continues the newest complete checkpoint of this build's adapter key, or trains from the start; the replies of another evaluation key are asked again"
        self.assertEqual(self.resume(out=self.dir / "empty", train_hours=3.0)["plan_lines"][-1], f"resume:   {self.fetched} holds {nothing}", "a session that left nothing: any training may follow it")
        put_checkpoint(self.fetched, self.first["adapter_sha"])
        (self.fetched / "ckpt" / "step-20").mkdir()          # a checkpoint a killed session left half written has no DONE: the kernel does not take it, and it refuses nothing
        self.assertEqual(self.resume(out=self.dir / "ckpt")["plan_lines"][-1], f"resume:   {self.fetched} holds {nothing}")

    def test_a_finished_adapter_of_this_key_is_kept_whatever_checkpoint_lies_next_to_it(self):
        """The kernel skips training when it finds its adapter, and then looks at no checkpoint: the builder refuses what the kernel refuses, and nothing else."""
        put_adapter(self.fetched, self.first["adapter_sha"])
        put_checkpoint(self.fetched, OTHER)
        self.assertIn("training is skipped", self.resume()["plan_lines"][-1])

    def test_the_output_of_a_smoke_run_refuses_nothing_and_a_smoke_build_looks_at_nothing(self):
        smoke = self.build(out=self.dir / "smoke", smoke=True)
        put_adapter(self.fetched, smoke["adapter_sha"], name="adapter-smoke")
        self.run_json(smoke)
        self.assertIn("no finished adapter and nothing of another adapter key", self.resume(out=self.dir / "after-smoke")["plan_lines"][-1], "the README's step 3: the full build after the smoke run")
        put_adapter(self.fetched, OTHER)
        put_checkpoint(self.fetched, OTHER)
        again = self.resume(out=self.dir / "smoke-again", smoke=True)
        self.assertEqual(again["plan_lines"][-1], "resume:   a smoke run resumes nothing: no adapter, no checkpoint and no reply of the attached output is used")
        self.assertIsNone(build_lora.check_resume(again, self.fetched))
        # the sentence is the builder's; what is used is the kernel's (sixth review, m1: it read the attached smoke replies). The same output, with the replies of that very smoke build, as the kernel sees it:
        tasks = tasks_of(self.dir / "smoke-again" / "data")
        self.assertEqual(keys_of(again), keys_of(smoke), "the same smoke build: its replies carry this build's two keys")
        for arm in lora_eval.ARMS:
            write_jsonl(self.fetched / f"smoke-replies-{arm}.jsonl", [line(t, [reply(t, True, keys_of(smoke))]) for t in HELDOUT if t["id"] in smoke["built_from"]["eval"]["ids"]])
        with mock.patch.object(lora_eval, "INPUT", self.fetched):
            for cfg, used in ((kernel_cfg(again), 0), ({**kernel_cfg(again), "smoke": False}, 1)):
                root = lora_eval.replies_root(cfg)
                previous = {arm: run_teacher.load_previous(lora_eval.previous_files(arm, self.dir / "working", root, "smoke-")) for arm in lora_eval.ARMS}
                self.assertEqual({arm: len(v) for arm, v in previous.items()}, {arm: used * len(tasks) for arm in lora_eval.ARMS}, "no reply of the attached output is used")
                self.assertEqual(len(lora_eval.work_items(tasks, previous, {**kernel_cfg(again), "k": 1, "max_tokens": 4096, "max_new_tokens": 4096})), (1 - used) * len(tasks), "and every task is asked")
        with self.assertRaisesRegex(SystemExit, "^--resume-output: the adapter"):
            self.resume(out=self.dir / "full")

    def test_without_the_output_the_build_says_it_did_not_check(self):
        cfg = self.resume(resume_output=None, train_hours=3.0)
        self.assertEqual(cfg["plan_lines"][-1], f"resume:   not checked here (no --resume-output): the kernel refuses an attached adapter or checkpoint of another adapter key than {cfg['adapter_sha']} before it installs anything; "
                                                "give --resume-output <the fetched output> to know now")
        self.assertFalse(any(x.startswith("resume:") for x in self.first["plan_lines"]), "a build that resumes nothing says nothing of a resume")

    def test_the_output_needs_the_run_it_is_of_and_a_folder(self):
        with self.assertRaisesRegex(SystemExit, "--resume-output is the fetched output of the run that --resume-from names: give --resume-from too"):
            self.resume(resume_from="")
        with self.assertRaisesRegex(SystemExit, "--resume-output .*: not a folder"):
            self.resume(resume_output=self.dir / "nowhere")
        with self.assertRaisesRegex(SystemExit, "--resume-output .*: not a folder"):
            self.resume(resume_output=self.sft)
        self.assertFalse((self.dir / "resume").exists())

    def test_the_builder_refuses_exactly_what_the_kernel_refuses(self):
        """One rule, `lora_eval.attached_adapter`, on the fetched output here and on the attached one there."""
        full, smoke = kernel_cfg(self.first), kernel_cfg(self.build(out=self.dir / "smoke", smoke=True))
        cases = {"nothing": lambda d: None, "its adapter": lambda d: put_adapter(d, full["adapter_sha"]), "another adapter": lambda d: put_adapter(d, OTHER), "its checkpoint": lambda d: put_checkpoint(d, full["adapter_sha"]),
                 "another checkpoint": lambda d: put_checkpoint(d, OTHER), "its adapter and another checkpoint": lambda d: (put_adapter(d, full["adapter_sha"]), put_checkpoint(d, OTHER)),
                 "an unfinished adapter and another checkpoint": lambda d: (put_adapter(d, full["adapter_sha"], "running"), put_checkpoint(d, OTHER))}
        refused = {"another adapter", "another checkpoint", "an unfinished adapter and another checkpoint"}
        for name, put in cases.items():
            d = self.dir / "case" / name
            d.mkdir(parents=True)
            put(d)
            for cfg, refuses in ((full, name in refused), (smoke, False)):
                outcome = []
                for check in (lambda: lora_eval.attached_adapter(cfg, [d]), lambda: build_lora.check_resume(cfg, d)):
                    try:
                        outcome.append(check())
                    except SystemExit:
                        outcome.append("refused")
                self.assertEqual(outcome[0], outcome[1], name)
                self.assertEqual(outcome[0] == "refused", refuses, name)
        self.assertEqual(lora_eval.attached_adapter(full, [self.dir / "case" / "its adapter"]), self.dir / "case" / "its adapter" / "adapter")
        self.assertIsNone(lora_eval.attached_adapter(full, [self.dir / "case" / "its checkpoint"]))

    def test_the_command_takes_the_output_and_the_card(self):
        put_adapter(self.fetched, self.first["adapter_sha"])
        args = ["--domain", "example_arith", "--sft", str(self.sft), "--train-tasks", str(self.train_tasks), "--heldout", str(self.heldout), "--owner", "someone", "--per-family", "3", "--min-family", "1",
                "--measured", str(slow(self.dir)), "--resume-from", "someone/example-arith-lora-eval", "--resume-output", str(self.fetched)]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "^--resume-output: the adapter"):
            build_lora.main([*args, "--out", str(self.dir / "cli-refused"), "--train-hours", "3"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_lora.main([*args, "--out", str(self.dir / "cli"), "--train-hours", "2", "--gpu-name", ""])
        self.assertIn("holds the finished adapter of this build's adapter key", buf.getvalue())
        cfg = json.loads((self.dir / "cli" / "data" / "lora-config.json").read_text())
        self.assertEqual((cfg["gpu_name"], keys_of(cfg)), ("", keys_of(self.first)), "any card: the same adapter, the same evaluation")
        self.assertEqual(self.first["gpu_name"], "T4")
        self.assertEqual(self.build(out=self.dir / "p100", gpu_name="P100", gpus=1)["gpu_name"], "P100")

    def test_the_last_line_names_the_accelerator_the_build_asks_for(self):
        """Sixth review, m2, at the builder: "set the kernel's accelerator to GPU T4 x2" was printed for every build, the one with --gpu-name P100 too."""
        args = ["--domain", "example_arith", "--sft", str(self.sft), "--train-tasks", str(self.train_tasks), "--heldout", str(self.heldout), "--owner", "someone", "--per-family", "3", "--min-family", "1"]
        for name, more, want in (("t4", [], "GPU T4 x2"), ("p100", ["--gpu-name", "P100", "--gpus", "1"], "GPU P100"), ("any", ["--gpu-name", ""], "GPU T4 x2"), ("a100", ["--gpu-name", "A100"], "the one whose GPUs are named 'A100'")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                build_lora.main([*args, "--out", str(self.dir / name), *more])
            self.assertTrue(buf.getvalue().rstrip().endswith(f"(set the kernel's accelerator to {want}; distill/backends/kaggle/README.md)"), buf.getvalue()[-200:])

    def test_a_card_whose_accelerator_has_fewer_gpus_than_the_build_is_for_is_refused(self):
        """A build the kernel could only refuse (it stops when it finds fewer GPUs than the build is for): --gpu-name P100 with the two processes of the default."""
        with self.assertRaisesRegex(SystemExit, "^--gpu-name 'P100' --gpus 2: the accelerator that has this card is GPU P100, 1 GPU\\(s\\), and the kernel stops when it finds fewer GPUs than the build is for: "
                                                "give --gpus 1, or another --gpu-name \\('' = any card\\)$"):
            self.build(out=self.dir / "two-p100", gpu_name="P100")
        with self.assertRaisesRegex(SystemExit, "^--gpu-name 'T4' --gpus 3: the accelerator that has this card is GPU T4 x2, 2 GPU\\(s\\)"):
            self.build(out=self.dir / "three-t4", gpus=3)
        self.assertFalse((self.dir / "two-p100").exists() or (self.dir / "three-t4").exists(), "before anything is written")
        for name, kw in (("any card", {"gpu_name": "", "gpus": 3}), ("a card that is not known here", {"gpu_name": "A100", "gpus": 3}), ("one T4", {"gpus": 1}), ("two T4", {"gpu_name": "Tesla T4"})):
            self.assertEqual(self.build(out=self.dir / name, **kw)["gpus"], kw.get("gpus", 2), name)

    def test_the_accelerator_of_a_card_name(self):
        for name, want in (("T4", ("GPU T4 x2", 2)), ("Tesla T4", ("GPU T4 x2", 2)), ("", ("GPU T4 x2", 2)), ("P100", ("GPU P100", 1)), ("Tesla P100-PCIE-16GB", ("GPU P100", 1)), ("Tesla", ("GPU T4 x2", 2)),
                           ("L4", ("the one whose GPUs are named 'L4'", None)), ("Tesla T4 x2", ("the one whose GPUs are named 'Tesla T4 x2'", None))):
            self.assertEqual(lora_eval.accelerator(name), want, name)
        self.assertEqual(lora_eval.accelerator(lora_eval.DEFAULTS["gpu_name"]), ("GPU T4 x2", 2), "the default build is for the two T4")

    def test_a_resume_that_skips_training_is_judged_without_the_training(self):
        """The sixth review's note: "training is skipped" under a plan that counted the training again, so a session with only the evaluation left was told IT DOES NOT FIT."""
        hours = 2.0
        whole = self.build(out=self.dir / "whole", measured=slow(self.dir), train_hours=2.0, gpu_hours=hours)
        total = lambda cfg: next(x for x in cfg["plan_lines"] if x.startswith("total:"))
        self.assertIn("IT DOES NOT FIT (more than the quota)", total(whole), "the build that trains does not fit these hours")
        nothing = self.resume(out=self.dir / "nothing", gpu_hours=hours)
        self.assertEqual(nothing["plan_lines"][:-1], whole["plan_lines"], "an output with no finished adapter: the kernel trains (from a checkpoint or from the start), and the plan counts it")
        self.assertEqual(self.resume(out=self.dir / "unchecked", gpu_hours=hours, resume_output=None)["plan_lines"][:-1], whole["plan_lines"], "an output that was not given: nothing is known")
        self.assertFalse([x for x in whole["plan_lines"] + nothing["plan_lines"] if "NOT in this session" in x or "one model load" in x], "a session that trains is not told that it does not")
        put_adapter(self.fetched, self.first["adapter_sha"], "partial")
        cfg = self.resume(out=self.dir / "kept", gpu_hours=hours)
        est, h = cfg["built_from"]["estimate"], lambda s: f"{s / 3600:.1f} h"
        left = est["total_upper_s"] - est["train_s"] - 300
        self.assertGreater(est["total_upper_s"], hours * 3600)
        self.assertLess(left, cfg["deadline_h"] * 3600)
        self.assertTrue(total(cfg).startswith(f"total:    {h(left)} assuming every batch of the evaluation runs to the token budget"), total(cfg))
        self.assertTrue(total(cfg).endswith(f"of the {hours:g} h of quota left: it fits"), total(cfg))
        self.assertIn(f"          NOT in this session: the finished adapter is attached and training is skipped (`resume:` below), so the {h(est['train_s'] + 300)} of the training and of its model load are not in the total", cfg["plan_lines"])
        self.assertIn(f"overhead: {h(est['overhead_s'] - 300)} (pip install, download, one model load)", cfg["plan_lines"])
        self.assertEqual(cfg["plan_lines"][-2], f"the kernel starts nothing after {cfg['deadline_h']:g} h", "no word of a training that stops early")
        self.assertIn("training is skipped", cfg["plan_lines"][-1])
        self.assertEqual((keys_of(cfg), est), (keys_of(whole), whole["built_from"]["estimate"]), "the config is the whole plan's: the cap on the steps and the reserves are its numbers, and the keys with them")
        self.assertEqual(json.loads((self.dir / "kept" / "data" / "lora-config.json").read_text())["built_from"]["estimate"], est)

    def test_a_resume_built_without_measured_that_skips_training_has_to_fit_what_its_session_does(self):
        """The seventh review's second note: the finished adapter in --resume-output, no --measured (the run that is resumed was built without it, and the resume keeps that): the smoke run "that comes
        first" was added to a session that runs none, and the plan did not fit hours that hold the setup, one model load and the evaluation."""
        first = self.build(out=self.dir / "plain")
        put_adapter(self.fetched, first["adapter_sha"])
        est = first["built_from"]["estimate"]
        left = est["total_upper_s"] - est["train_s"] - 300
        hours = math.ceil((left / 3600 + build_lora.DEADLINE_MARGIN_H) * 100) / 100          # the least quota whose deadline (the quota less the margin) holds what is left to do
        self.assertGreater(left + est["smoke_s"], hours * 3600, "the fixture: with the smoke run on top it is more than the quota")
        cfg = self.resume(out=self.dir / "kept", measured=None, train_hours=None, gpu_hours=hours)
        total = next(x for x in cfg["plan_lines"] if x.startswith("total:"))
        self.assertEqual(total, f"total:    {left / 3600:.1f} h assuming every batch of the evaluation runs to the token budget, ~{(est['total_typical_s'] - est['train_s'] - 300) / 3600:.1f} h assuming 75% of it, of the {hours:g} h of quota left: it fits")
        self.assertEqual(cfg["deadline_h"], math.floor((hours - build_lora.DEADLINE_MARGIN_H) * 100) / 100, "the quota less the margin: no smoke run is taken out of it")
        self.assertIn("smoke:    none is counted: a resume (--resume-from) continues a run that was started, and no smoke run comes before it: --gpu-hours is the quota that is left now", cfg["plan_lines"])
        self.assertFalse([x for x in cfg["plan_lines"] if x.startswith("smoke:    +")])
        self.assertIn("overhead: 0.4 h (pip install, download, one model load)", cfg["plan_lines"])
        self.assertIn("training is skipped", cfg["plan_lines"][-1])
        self.assertEqual(keys_of(cfg), keys_of(first))
        self.assertIn("IT DOES NOT FIT", next(x for x in self.build(out=self.dir / "fresh", gpu_hours=hours)["plan_lines"] if x.startswith("total:")), "a build that resumes nothing still pays its smoke run and its training")

    def test_a_resume_that_skips_training_and_still_does_not_fit_is_told_what_is_left_to_lower(self):
        put_adapter(self.fetched, self.first["adapter_sha"])
        cfg = self.resume(out=self.dir / "short", gpu_hours=4.0, deadline_h=0.2, export_gguf=True)
        est = cfg["built_from"]["estimate"]
        over = -(-(est["total_upper_s"] - est["train_s"] - 300 - 0.2 * 3600) // 60)
        self.assertTrue(cfg["plan_lines"][-3].endswith(
            f"IT DOES NOT FIT (inside the quota, but this run needs {(est['total_upper_s'] - est['train_s'] - 300) / 3600:.2f} h, {over:.0f} min past the 0.2 h after which the kernel starts nothing: it would leave the last "
            "batches unasked): lower --k, --per-family or --max-new-tokens (another evaluation of the same adapter: the adapter is kept and the replies are asked again); --train-hours is the adapter's and "
            "no part of this total: keep it"), cfg["plan_lines"][-3])
        self.assertEqual(cfg["plan_lines"][-2], "the kernel starts nothing after 0.2 h, and the evaluation ends 0.5 h before that for the GGUF export")
        whole = self.build(out=self.dir / "whole", measured=slow(self.dir), train_hours=2.0, gpu_hours=4.0, deadline_h=0.2, export_gguf=True)
        self.assertIn("it would stop training early or leave the last batches unasked): lower --train-hours, --k, --per-family or --max-new-tokens (on a resume", whole["plan_lines"][-2])
        self.assertIn("(pip install, download, two model loads); GGUF export", "\n".join(whole["plan_lines"]))

    def test_the_plan_of_a_resume_that_skips_training(self):
        p = build_lora.plan([3000] * 1200, [2000] * 70, epochs=2.0, gpus=2, global_batch=16, train_hours=3.0, k=1, max_new_tokens=4096, eval_batch=8, export=True, rates=build_lora.RATES)
        r = build_lora.resumed_plan(p)
        load = build_lora.RATES["load_s"]
        self.assertEqual((r["train_skipped_s"], r["overhead_s"], r["total_upper_s"], r["total_typical_s"]), (p["train_s"] + load, p["overhead_s"] - load, p["total_upper_s"] - p["train_s"] - load, p["total_typical_s"] - p["train_s"] - load))
        self.assertEqual(r["total_upper_s"], round(build_lora.RATES["setup_s"] + load) + p["eval_upper_s"] + p["export_s"], "the setup, one load, the evaluation and the export")
        self.assertEqual({k: v for k, v in r.items() if k not in ("resumed", "train_skipped_s", "overhead_s", "total_upper_s", "total_typical_s")}, {k: v for k, v in p.items() if k not in ("overhead_s", "total_upper_s", "total_typical_s")})
        self.assertNotIn("resumed", p)
        self.assertEqual(build_lora.fit_verdict(r, r["total_upper_s"] / 3600, 7.5), "")
        self.assertEqual(build_lora.fit_verdict(r, (r["total_upper_s"] - 1) / 3600, 7.5), "more than the quota")
        self.assertEqual(build_lora.fit_verdict(p, (p["total_upper_s"] - 1) / 3600, 7.5), "more than the quota")

    def test_the_difference_is_not_printed_when_no_setting_differs(self):
        """The review's surviving mutant: an empty "What differs ..." list. The run.json is of the key that is refused and every setting and the records are this build's: the LoRA target modules moved."""
        with mock.patch.object(lora_eval.sft, "LORA_TARGETS", ("q_proj",)):
            older = self.build(out=self.dir / "older", measured=slow(self.dir), train_hours=2.0)
        self.assertNotEqual(older["adapter_sha"], self.first["adapter_sha"])
        put_adapter(self.fetched, older["adapter_sha"])
        self.run_json(older)
        with self.assertRaises(SystemExit) as caught:
            self.resume()
        self.assertTrue(str(caught.exception).endswith("or without --resume-from to train again. Nothing was written"), str(caught.exception))


class KernelRefusesBeforeTheSetup(Kit):
    """m3: the kernel refused an adapter of another key after the pip install and the 18 GB download, and a checkpoint of another key later still, in the workers ("training left no adapter").
    `lora_eval.main` itself on the stand-ins of `test_kaggle_lora_keys.KernelMain`."""

    NOW, T0 = keys_tests.KernelMain.NOW, keys_tests.KernelMain.T0
    run_main, run_built = keys_tests.KernelMain.run_main, keys_tests.KernelMain.run_built

    def attach(self, put) -> None:
        self.build()
        put(self.dir / "input" / "earlier")

    def test_an_adapter_of_another_key_stops_the_kernel_before_it_installs_or_downloads_anything(self):
        self.attach(lambda d: put_adapter(d, OTHER))
        with self.assertRaisesRegex(SystemExit, "trained under adapter key '0000000000000000'"):
            self.run_built(self.dir / "out")
        self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))

    def test_a_checkpoint_of_another_key_stops_the_kernel_before_it_installs_or_downloads_anything(self):
        self.attach(lambda d: put_checkpoint(d, OTHER))
        with self.assertRaisesRegex(SystemExit, "the checkpoint .* was written under adapter key '0000000000000000'"):
            self.run_built(self.dir / "out")
        self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))

    def test_a_checkpoint_of_this_key_is_left_to_the_training_worker(self):
        self.build()
        cfg = json.loads((self.dir / "out" / "data" / "lora-config.json").read_text())
        put_checkpoint(self.dir / "input" / "earlier", cfg["adapter_sha"])
        self.run_built(self.dir / "out")
        self.assertEqual([s for s, _, _, _ in self.calls], ["train", "eval"])

    def test_a_smoke_run_ignores_whatever_is_attached(self):
        self.build(out=self.dir / "smoke", smoke=True)
        put_adapter(self.dir / "input" / "earlier", OTHER)
        put_checkpoint(self.dir / "input" / "earlier", OTHER)
        cfg = json.loads((self.dir / "smoke" / "data" / "lora-config.json").read_text())
        ids = cfg["built_from"]["eval"]["ids"]
        for arm in lora_eval.ARMS:          # the replies of a smoke run of this very build (sixth review, m1: the merge took them for this session's)
            write_jsonl(self.dir / "input" / "earlier" / f"smoke-replies-{arm}.jsonl", [line(t, [reply(t, True, keys_of(cfg))]) for t in HELDOUT if t["id"] in ids])
        run = self.run_built(self.dir / "smoke")
        self.assertEqual(([s for s, _, _, _ in self.calls], run["smoke"]), (["train", "eval"], True))
        self.assertEqual(run["replies"], {"base": 0, "lora": 0}, "the stand-in worker asked nothing, and no reply of the attached output is this session's")
        self.assertEqual([(self.out / f"smoke-replies-{arm}.jsonl").read_text() for arm in lora_eval.ARMS], ["", ""])

    def test_a_full_run_merges_the_replies_that_are_attached(self):
        self.build()
        cfg = json.loads((self.dir / "out" / "data" / "lora-config.json").read_text())
        ids = cfg["built_from"]["eval"]["ids"]
        (self.dir / "input" / "earlier").mkdir(parents=True)
        for arm in lora_eval.ARMS:
            write_jsonl(self.dir / "input" / "earlier" / f"replies-{arm}.jsonl", [line(t, [reply(t, True, keys_of(cfg))]) for t in HELDOUT if t["id"] in ids])
        self.assertEqual(self.run_built(self.dir / "out")["replies"], {"base": len(ids), "lora": len(ids)})

    def test_a_dataset_that_was_edited_is_said_before_the_cards_are_looked_at(self):
        self.build()
        path = self.dir / "out" / "data" / "lora-config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "max_new_tokens": 1024}))
        for names in ((), ("Tesla P100-PCIE-16GB",), ("Tesla T4",)):
            with self.assertRaisesRegex(SystemExit, "lora-config.json says evaluation key", msg=str(names)):
                self.run_built(self.dir / "out", names=names)

    def test_the_card_message_names_the_flag_and_no_edit_of_the_dataset(self):
        with self.assertRaisesRegex(SystemExit, "^the kernel got \\['Tesla P100-PCIE-16GB'\\], not T4: set the accelerator to GPU T4 x2 in the kernel's settings \\(or build again with --gpu-name '' to run on whatever it is: the card is in no key\\)$"):
            self.run_main(names=("Tesla P100-PCIE-16GB",))
        self.assertEqual((self.sh, self.downloads), ([], []))
        self.run_main(names=("Tesla P100-PCIE-16GB", "Tesla P100-PCIE-16GB"), out=self.dir / "any", gpu_name="")
        self.assertEqual([s for s, _, _, _ in self.calls], ["train", "eval"])

    def test_the_card_message_names_the_accelerator_the_build_asks_for(self):
        """Sixth review, m2: a build with --gpu-name P100 on two T4 was told to set the accelerator to GPU T4 x2, which it has."""
        with self.assertRaises(SystemExit) as caught:
            self.run_main(gpu_name="P100", gpus=1)
        self.assertEqual(str(caught.exception), "the kernel got ['Tesla T4', 'Tesla T4'], not P100: set the accelerator to GPU P100 in the kernel's settings (or build again with --gpu-name '' to run on whatever it is: "
                                                "the card is in no key)")
        self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))
        run = self.run_built(self.dir / "out", names=("Tesla P100-PCIE-16GB",))
        self.assertEqual(([(s, g) for s, g, _, _ in self.calls], run["gpus"]), ([("train", 1), ("eval", 1)], 1), "on the card it asks for the same build runs")

    def test_a_card_that_is_not_one_of_kaggles_is_named_as_it_was_asked(self):
        with self.assertRaisesRegex(SystemExit, "^the kernel got \\['Tesla T4', 'Tesla T4'\\], not A100: set the accelerator to the one whose GPUs are named 'A100' in the kernel's settings \\(or build again"):
            self.run_main(gpu_name="A100")

    def test_every_card_has_to_be_the_one_asked_for(self):
        """The review's surviving mutant: `any` for `all` in the card check."""
        for names in (("Tesla T4", "Tesla P100-PCIE-16GB"), ("Tesla P100-PCIE-16GB", "Tesla T4")):
            with self.assertRaisesRegex(SystemExit, "not T4: set the accelerator to GPU T4 x2"):
                self.run_main(names=names, out=self.dir / "mixed", force=True)
            self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))

    def with_card(self, out, gpu_name: str) -> None:
        """The dataset under `out` with another `gpu_name`: it is in no key, so the kernel takes it (a builder refuses the pairs below, and a dataset can still say them)."""
        path = out / "data" / "lora-config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "gpu_name": gpu_name}))

    def test_too_few_gpus_are_told_the_accelerator_only_when_it_has_the_gpus_of_the_build(self):
        """With m2: "or set the accelerator to GPU T4 x2" was the advice for every build that found too few GPUs, the one whose --gpu-name refuses a T4 too."""
        said = "the kernel got 1 GPU\\(s\\) and the build is for 2: .* so build again with --gpus 1"
        with self.assertRaisesRegex(SystemExit, said + ", or set the accelerator to GPU T4 x2$"):
            self.run_main(names=("Tesla T4",))
        with self.assertRaisesRegex(SystemExit, said + ", or set the accelerator to GPU T4 x2$"):
            self.run_main(names=("Tesla P100-PCIE-16GB",), out=self.dir / "any", gpu_name="")
        for card, names in (("P100", ("Tesla P100-PCIE-16GB",)), ("A100", ("NVIDIA A100-SXM4-40GB",))):
            self.with_card(self.dir / "out", card)
            with self.assertRaisesRegex(SystemExit, said + "$", msg=card):
                self.run_built(self.dir / "out", names=names)
            self.assertEqual((self.sh, self.downloads, self.calls), ([], [], []))

    def test_the_pins_installed_are_the_configs(self):
        """The review's surviving mutant: `DEFAULTS["pins"]` for `cfg["pins"]` in `main`. The builder writes no pins; a dataset whose config names its own, under the keys they make, installs those."""
        self.build()
        data = self.dir / "out" / "data"
        cfg = {**json.loads((data / "lora-config.json").read_text()), "pins": ["onelib==1.0"]}
        resolved = lora_eval.resolve_config(cfg)
        cfg["adapter_sha"] = lora_eval.adapter_key(lora_eval.file_sha(data / "train.jsonl"), resolved)
        cfg["eval_sha"] = lora_eval.eval_key(cfg["adapter_sha"], cfg["built_from"]["eval"]["excluded"]["sha"], lora_eval.file_sha(data / "eval_tasks.jsonl"), resolved)
        (data / "lora-config.json").write_text(json.dumps(cfg))
        self.run_built(self.dir / "out")
        self.assertEqual(self.sh[0][0][5:], ["onelib==1.0"])

    def test_the_setup_installs_the_pins_and_downloads_the_student_of_the_config(self):
        self.run_main(student="someone/another-student", fla=True)
        (cmd,), = self.sh
        self.assertEqual(cmd[:5], [sys.executable, "-m", "pip", "install", "-q"])
        self.assertEqual(cmd[5:], [*lora_eval.DEFAULTS["pins"], "flash-linear-attention"])
        self.assertEqual([a[0] for a in self.downloads], ["someone/another-student"])

    def test_the_setup_of_a_default_build(self):
        self.run_main()
        self.assertEqual(self.sh[0][0][5:], lora_eval.DEFAULTS["pins"])
        self.assertEqual([a[0] for a in self.downloads], [lora_eval.DEFAULTS["student"]])


class Lines(unittest.TestCase):
    """The review's note on Windows: a jsonl written in text mode has `\\r\\n` there, and the builder's sha of the lines joined with `\\n` would not be the file's."""

    def test_a_jsonl_is_written_with_one_newline_byte_per_line_on_every_system(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x.jsonl"
            opened = []
            real = open
            with mock.patch("builtins.open", lambda *a, **kw: opened.append(kw) or real(*a, **kw)):
                self.assertEqual(outputs.write_atomic(path, ["a", "b"]), 2)
            self.assertEqual(opened, [{"newline": "\n"}])
            self.assertEqual(path.read_bytes(), b"a\nb\n")
            self.assertEqual(build_lora.lines_sha(["a", "b"]), lora_eval.file_sha(path))


if __name__ == "__main__":
    unittest.main()

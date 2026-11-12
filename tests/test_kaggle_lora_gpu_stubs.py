"""The GPU glue of the Kaggle LoRA kernel (`distill/backends/kaggle/lora_eval.py`) run on stubs, with no torch, no transformers and no GPU (PR 34 review, M1).

`GpuGlue` in test_kaggle_lora.py only looks for a substring in the source, so an edit that swapped the two arms, gave them other seeds or other budgets, dropped the deadline margin, lost the next-token shift of the
loss or summed the gradients of the processes instead of averaging them passed every test. These run the real functions (`generate_batch`, `eval_worker`, `chunked_loss`, `all_reduce_mean`) against stand-ins that
record what they are asked, so that each of those edits fails a test. What a stub cannot say is what torch does: the real numbers are the smoke run's.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from .test_kaggle_lora import ARITH, HELDOUT, lora_eval, run_teacher, write_jsonl

SHA = "0123456789abcdef"


# ---- a tiny torch -----------------------------------------------------------------------------------------------------------------------------

class Vec:
    """A flat tensor: what `all_reduce_mean` does to the gradients."""

    def __init__(self, data):
        self.data = [float(x) for x in data]

    def reshape(self, *_):
        return Vec(self.data)

    def numel(self):
        return len(self.data)

    def __getitem__(self, key):
        return Vec(self.data[key])

    def view_as(self, other):
        return Vec(self.data)

    def clone(self):
        return Vec(self.data)

    def __itruediv__(self, k):
        self.data = [x / k for x in self.data]
        return self


class Param:
    def __init__(self, n, grad=None):
        self.n, self.grad = n, None if grad is None else Vec(grad)

    def numel(self):
        return self.n


class ReduceTorch:
    """`torch.cat`, `zeros_like` and `distributed.all_reduce` (a sum with what `peer`, the other process's flat gradient, holds)."""

    def __init__(self, peer):
        self.peer, self.reduced = peer, 0
        self.distributed = SimpleNamespace(all_reduce=self.all_reduce)

    def cat(self, parts):
        return Vec([x for v in parts for x in v.data])

    def zeros_like(self, p):
        return Vec([0] * p.numel())

    def all_reduce(self, flat):
        self.reduced += 1
        flat.data = [a + b for a, b in zip(flat.data, self.peer)]


class Arr:
    """Nested lists with the indexing forms `chunked_loss` uses: `a[0, :-1]`, `a[mask]`, `a[i:j]`, `a != v`."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if isinstance(key, Arr):
            return Arr([x for x, keep in zip(self.data, key.data) if keep])
        if isinstance(key, tuple):
            i, sl = key
            return Arr(self.data[i][sl])
        return Arr(self.data[key])

    def __ne__(self, v):
        return Arr([x != v for x in self.data])

    def numel(self):
        return len(self.data)

    def float(self):
        return self


class LossTorch:
    """`cross_entropy` (the sum over positions of -log softmax) and `utils.checkpoint.checkpoint` (a call, that remembers how many positions it was given)."""

    def __init__(self):
        self.chunks = []
        self.nn = SimpleNamespace(functional=SimpleNamespace(cross_entropy=self.cross_entropy))
        self.utils = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint=self.checkpoint))

    def cross_entropy(self, logits, y, reduction):
        assert reduction == "sum"
        return sum(-math.log(math.exp(row[t]) / sum(math.exp(v) for v in row)) for row, t in zip(logits.data, y.data))

    def checkpoint(self, fn, *args, use_reentrant):
        assert use_reentrant is False
        self.chunks.append(len(args[1].data))
        return fn(*args)


def nll(logits, target):
    return -math.log(math.exp(logits[target]) / sum(math.exp(v) for v in logits))


class AllReduceMean(unittest.TestCase):
    def test_the_gradients_are_the_mean_over_the_processes(self):
        a, b, c = Param(2, [1, 2]), Param(3, None), Param(1, [5])
        torch = ReduceTorch(peer=[3, 4, 6, 6, 6, 7])
        lora_eval.all_reduce_mean(torch, [a, b, c], 2)
        self.assertEqual((torch.reduced, a.grad.data, b.grad.data, c.grad.data), (1, [2.0, 3.0], [3.0, 3.0, 3.0], [6.0]), "(this process's + the other's) / 2; a parameter with no gradient counts as zeros")

    def test_one_process_has_nothing_to_average(self):
        a = Param(2, [1, 2])
        torch = ReduceTorch(peer=[9, 9])
        lora_eval.all_reduce_mean(torch, [a], 1)
        self.assertEqual((torch.reduced, a.grad.data), (0, [1.0, 2.0]))

    def test_the_average_is_over_the_number_of_processes_not_two(self):
        a = Param(1, [3])
        lora_eval.all_reduce_mean(ReduceTorch(peer=[9]), [a], 4)
        self.assertEqual(a.grad.data, [3.0], "a stand-in all-reduce that adds one peer: (3 + 9) / 4")


class ChunkedLoss(unittest.TestCase):
    W = [[0.5, -1.0], [1.5, 0.25], [-0.75, 1.0]]                         # lm_head: 2 hidden units -> 3 tokens
    H = [[0.1, 0.2], [0.3, -0.1], [0.5, 0.4], [-0.2, 0.6], [0.0, 0.3]]    # the hidden state of 5 positions
    Y = [-100, -100, 2, 1, 0]                                               # the prompt is not trained; the targets are the tokens that FOLLOW a position

    def run_loss(self, chunk):
        torch = LossTorch()
        head = lambda hh: Arr([[sum(w * x for w, x in zip(row, v)) for row in self.W] for v in hh.data])
        loss, n = lora_eval.chunked_loss(torch, Arr([self.H]), Arr([self.Y]), head, chunk)
        return loss, n, torch.chunks

    def logits(self, pos):
        return [sum(w * x for w, x in zip(row, self.H[pos])) for row in self.W]

    def test_the_position_before_a_token_predicts_it(self):
        """Position i of the hidden state is scored against the label of position i + 1: the next token."""
        expected = (nll(self.logits(1), 2) + nll(self.logits(2), 1) + nll(self.logits(3), 0)) / 3
        loss, n, _ = self.run_loss(512)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(loss, expected, places=12)
        no_shift = (nll(self.logits(2), 2) + nll(self.logits(3), 1) + nll(self.logits(4), 0)) / 3
        self.assertNotAlmostEqual(loss, no_shift, places=3, msg="the loss of a model trained to repeat its input")

    def test_the_loss_is_the_same_whatever_the_slice_and_is_made_one_slice_at_a_time(self):
        whole, n, chunks = self.run_loss(512)
        self.assertEqual(chunks, [3])
        for chunk, sizes in ((1, [1, 1, 1]), (2, [2, 1])):
            loss, _, chunks = self.run_loss(chunk)
            self.assertAlmostEqual(loss, whole, places=12, msg=chunk)
            self.assertEqual(chunks, sizes)

    def test_only_target_positions_count(self):
        torch = LossTorch()
        loss, n = lora_eval.chunked_loss(torch, Arr([self.H]), Arr([[-100] * 5]), lambda hh: Arr([]), 512)
        self.assertEqual((n, loss), (0, 0), "nothing to learn from: a zero, not a division by zero")


# ---- a tiny tokenizer, model and clock --------------------------------------------------------------------------------------------------------

class Row(list):
    def sum(self):
        return sum(self)


class Mat:
    def __init__(self, rows):
        self.rows = [Row(r) for r in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Mat([r[key[1]] for r in self.rows])
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def tolist(self):
        return [list(r) for r in self.rows]


class Enc(dict):
    def to(self, device):
        self.device = device
        return self


class GenTok:
    """A tokenizer for `generate_batch`: left padding, the generation prompt ends in `<think>\\n`, and a reply is decoded from a small vocabulary: 5 thinking, 6 the end of the think block, 7 the base arm's
    answer, 8 the adapter's, 10 + i the row it belongs to, 99 `<|im_end|>`, 77 `<|endoftext|>`."""
    pad_token_id, eos_token_id = 0, 99
    VOCAB = {5: "thinking ", 6: "\n</think>\n\n", 7: "Answer: BASE", 8: "Answer: LORA", 99: "<|im_end|>", 77: "<|endoftext|>"}

    def __init__(self):
        self.templates, self.ids = [], {}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        self.templates.append((tokenize, add_generation_prompt, enable_thinking))
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n<think>\n"

    def __call__(self, prompts, return_tensors=None, padding=False, add_special_tokens=True):
        assert (return_tensors, padding, add_special_tokens) == ("pt", True, False)
        ids = [[self.ids.setdefault(w, 100 + len(self.ids)) for w in p.split()] for p in prompts]
        width = max(map(len, ids))
        return Enc({"input_ids": Mat([[0] * (width - len(r)) + r for r in ids]), "attention_mask": Mat([[0] * (width - len(r)) + [1] * len(r) for r in ids])})

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is False, "the end of turn is cut by ids, the text keeps its special tokens for split_generation"
        return "".join(f"r{i - 10} " if 10 <= i < 50 else self.VOCAB.get(i, "?") for i in ids)

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<|im_end|>" else None

    @classmethod
    def from_pretrained(cls, *args, **kw):
        return cls._current


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def time(self):
        return self.now


class GenTorch:
    def __init__(self):
        self.seeds = []
        self.cuda = SimpleNamespace(max_memory_allocated=lambda: 0)

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def no_grad(self):
        return contextlib.nullcontext()


class StubModel:
    """`generate` records what it is asked and under which adapter state and seed, and answers each row `thinking r<i> </think> Answer: BASE|LORA` then `end`; the clock moves 100 s per call."""

    def __init__(self, torch, clock=None, end=99, eos=None):
        self.torch, self.clock, self.end, self.adapter_on, self.calls = torch, clock, end, True, []
        self.generation_config = SimpleNamespace(eos_token_id=eos)

    def eval(self):
        return self

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True

    def generate(self, **kw):
        ids, mask = kw.pop("input_ids"), kw.pop("attention_mask")
        self.calls.append({"adapter_on": self.adapter_on, "seed": self.torch.seeds[-1] if self.torch.seeds else None, "input_ids": ids.tolist(), "attention_mask": mask.tolist(), "kwargs": kw})
        if self.clock:
            self.clock.now += 100
        answer = 8 if self.adapter_on else 7
        return Mat([list(row) + [5, 10 + i, 6, answer, self.end] for i, row in enumerate(ids)])


def task(i: int) -> dict:
    t = HELDOUT[i]
    return {"id": t["id"], "kind": t["kind"], "messages": ARITH.messages(t, "thinking")}


class GenerateBatch(unittest.TestCase):
    def cfg(self, **kw):
        return lora_eval.resolve_config({"max_new_tokens": 64, "seed_base": 1234, **kw})

    def run_arms(self, items, **kw):
        torch, tok = GenTorch(), GenTok()
        model = StubModel(torch)
        cfg = self.cfg(**kw)
        outs = {arm: lora_eval.generate_batch(torch, model, tok, items, arm, cfg, {99}, 0) for arm in lora_eval.ARMS}
        return model, tok, torch, cfg, outs

    def test_the_two_arms_are_asked_the_same_thing_and_only_the_base_runs_without_the_adapter(self):
        items = [(task(0), 2), (task(1), 0), (task(2), 1)]
        model, tok, torch, cfg, outs = self.run_arms(items)
        base, lora = model.calls
        self.assertEqual((base["adapter_on"], lora["adapter_on"]), (False, True), "the base arm is the adapter switched off; the other has it on")
        self.assertTrue(model.adapter_on, "and it is back on after the base arm")
        self.assertEqual(base["input_ids"], lora["input_ids"], "the same prompts, padded the same")
        self.assertEqual(base["attention_mask"], lora["attention_mask"])
        self.assertEqual(base["kwargs"], lora["kwargs"], "the same budget, sampling, padding id and cache for both arms")
        self.assertEqual(base["kwargs"], {"max_new_tokens": 64, "do_sample": True, "pad_token_id": 0, "use_cache": True, **cfg["sampling"]})
        self.assertEqual((base["seed"], lora["seed"]), (1234 + 2, 1234 + 2), "the RNG is seeded with the batch's first item before each arm, the same for both")
        self.assertEqual(torch.seeds, [1236, 1236])
        self.assertEqual(tok.templates, [(False, True, True)] * 6, "thinking template, generation prompt, for every prompt of both arms")

    def test_each_reply_comes_back_to_the_task_that_asked_it_under_its_arm(self):
        items = [(task(0), 0), (task(1), 0)]
        _, _, _, cfg, outs = self.run_arms(items)
        (base, steps_b), (lora, steps_l) = outs["base"], outs["lora"]
        self.assertEqual((steps_b, steps_l), (5, 5))
        self.assertEqual([o["text"] for o in base], [f"<think>\nthinking r{i}\n</think>\n\nAnswer: BASE" for i in (0, 1)])
        self.assertEqual([o["text"] for o in lora], [f"<think>\nthinking r{i}\n</think>\n\nAnswer: LORA" for i in (0, 1)])
        for out, (t, _) in zip(base, items):
            self.assertEqual((out["finish"], out["tokens"], out["max_tokens"], out["prompt_sha"]), ("stop", 5, 64, run_teacher.prompt_sha(t["messages"])))

    def test_the_seed_follows_the_first_item_of_the_batch(self):
        for first, want in ((0, 1234), (3, 1237)):
            model, _, _, _, _ = self.run_arms([(task(0), first), (task(1), 0)])
            self.assertEqual([c["seed"] for c in model.calls], [want, want])
        model, _, _, _, _ = self.run_arms([(task(0), 0)], seed_base=99)
        self.assertEqual([c["seed"] for c in model.calls], [99, 99])

class EvalOnStubs(unittest.TestCase):
    """`eval_worker` end to end: the files it reads and writes are real (in a temporary directory), the GPU libraries, the model and the clock are stand-ins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.out, self.inp, self.data = self.root / "out", self.root / "in", self.root / "data"
        for d in (self.out, self.inp, self.data):
            d.mkdir()
        self.logs = []

    def adapter(self, where, name="adapter", sha=SHA):
        d = where / name
        d.mkdir(parents=True)
        (d / "train.json").write_text(json.dumps({"status": "done", "config_sha": sha}))
        (d / "adapter_model.safetensors").write_text("x")
        return d

    def run_eval(self, n=4, deadline=10**9, clock=None, end=99, eos=None, smoke=False, **cfg_kw):
        write_jsonl(self.data / "eval_tasks.jsonl", [task(i) for i in range(n)])
        adapter = self.adapter(self.out, "adapter-smoke" if smoke else "adapter")
        torch, clock = GenTorch(), clock or Clock()
        model, tok = StubModel(torch, clock, end=end, eos=eos), GenTok()
        GenTok._current = tok
        cfg = lora_eval.resolve_config({"eval_batch": 2, "k": 1, "max_new_tokens": 64, "config_sha": SHA, "smoke": smoke, **cfg_kw})
        loads = []

        def load(torch_, cfg_, local, adapter=None, train=False):
            loads.append((adapter, train))
            return model
        stubs = {"torch": torch, "transformers": SimpleNamespace(AutoTokenizer=GenTok)}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, stubs))
            stack.enter_context(mock.patch.dict(os.environ, {"LORA_DEADLINE": str(deadline)}))
            for name, value in (("OUT", self.out), ("INPUT", self.inp), ("time", clock), ("dist_setup", lambda t: (0, 1, 0)), ("load_student", load), ("log", lambda *a: self.logs.append(a))):
                stack.enter_context(mock.patch.object(lora_eval, name, value))
            lora_eval.eval_worker(cfg, self.data)
        self.model, self.tok, self.loads, self.cfg, self.adapter_dir = model, tok, loads, cfg, adapter
        return model

    def shard(self, arm, prefix=""):
        path = lora_eval.shard_path(self.out, arm, 0, prefix)
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_every_batch_asks_the_base_arm_then_the_adapter_arm_the_same_prompts(self):
        model = self.run_eval()
        self.assertEqual([c["adapter_on"] for c in model.calls], [False, True, False, True], "base, adapter, base, adapter: the two arms of a batch next to each other")
        for base, lora in zip(model.calls[::2], model.calls[1::2]):
            self.assertEqual((base["input_ids"], base["attention_mask"], base["kwargs"], base["seed"]), (lora["input_ids"], lora["attention_mask"], lora["kwargs"], lora["seed"]))
        self.assertEqual(model.calls[0]["kwargs"]["max_new_tokens"], 64)

    def test_the_model_is_loaded_once_with_the_adapter_that_was_found(self):
        self.run_eval()
        self.assertEqual(self.loads, [(self.adapter_dir, False)], "one load for both arms")

    def test_each_arms_replies_are_filed_in_its_own_shard_with_the_config_they_were_made_under(self):
        self.run_eval()
        base, lora = self.shard("base"), self.shard("lora")
        self.assertEqual(sorted(l["id"] for l in base), sorted(l["id"] for l in lora))
        self.assertEqual(len(base), 4)
        for line in base:
            self.assertTrue(all("Answer: BASE" in o["text"] for o in line["outs"]), line)
        for line in lora:
            self.assertTrue(all("Answer: LORA" in o["text"] for o in line["outs"]), line)
            self.assertEqual(line["prompt_sha"], run_teacher.prompt_sha(next(task(i)["messages"] for i in range(4) if task(i)["id"] == line["id"])))

    def test_the_deadline_stops_the_next_batch_when_the_mean_batch_would_pass_it(self):
        """A batch of both arms takes 200 s (100 per generate): after the first, one arm's mean is 100 s, and the second batch needs 2 x that before the deadline."""
        clock = Clock()
        self.run_eval(deadline=1250, clock=clock)
        self.assertEqual(len(self.model.calls), 2, "the first batch always starts; at 1200 s the next one (2 x 100 s) would end at 1400, past 1250")
        self.assertTrue(any("deadline, 2 replies left" in " ".join(map(str, a)) for a in self.logs), self.logs)
        self.assertEqual(sum(len(l["outs"]) for l in self.shard("lora")), 2)

    def test_the_deadline_lets_a_batch_start_that_fits(self):
        self.run_eval(deadline=1000 + 200 + 200 + 1, clock=Clock())
        self.assertEqual(len(self.model.calls), 4, "the second batch is expected to end at 1400, before 1401")

    def test_the_smoke_run_asks_one_full_batch_per_process_and_arm(self):
        self.run_eval(n=12, smoke=True, eval_batch=8, k=3, max_new_tokens=4096, smoke_new_tokens=32)
        self.assertEqual([c["adapter_on"] for c in self.model.calls], [False, True])
        self.assertEqual([len(c["input_ids"]) for c in self.model.calls], [8, 8], "the batch the full run uses: the gen_step_s it measures is the estimate's")
        self.assertEqual(self.model.calls[0]["kwargs"]["max_new_tokens"], 32)
        self.assertEqual(sorted(p.name for p in self.out.glob("smoke-replies-*.jsonl")), ["smoke-replies-base.r0.jsonl", "smoke-replies-lora.r0.jsonl"], "never named like a result")
        self.assertFalse(list(self.out.glob("replies-*")))
        self.assertEqual(len(self.shard("base", "smoke-")), 8)


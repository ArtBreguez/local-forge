"""Round 8, the Modal stage (stubbed: nothing here rents anything): M2 (4)-(6), m1 (the Modal-side re-check), m12.

M2   (4) the K samples have a temperature above 0 and one seed each, (5) the Modal build uses the task's -O level and the gate gets the task's compiler,
     (6) the referenced-data block is in the prompt.
m1   the Modal-side re-check refuses a legacy lcg task whose multiplier the Modal compilers removed. m12 `modal_lora` drops an over-long pair instead of cutting it.
"""
from __future__ import annotations

import collections
import itertools
import os
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_r8 import GCC_OBJDUMP
from .test_rea_modal import ModalCase, StubbedSampleStage

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import families  # noqa: E402


# ---- M2 (4)-(6): the Modal stage ------------------------------------------------------------------------------------------------------

def private_mkdtemp(case) -> None:
    """`_decompile_task` makes its work directory with tempfile.mkdtemp and never removes it (the container is thrown away): in a test it would leave a `t.c` in /tmp per call."""
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    counter = itertools.count()

    def mkdtemp(*a, **k):
        path = os.path.join(tmp.name, f"d{next(counter)}")
        os.makedirs(path)
        return path

    patch = mock.patch("tempfile.mkdtemp", mkdtemp)
    patch.start()
    case.addCleanup(patch.stop)


class ModalStageGuarantees(StubbedSampleStage):
    """The sample stage and the build of the decompile stage, with Modal and the model stubbed."""

    def setUp(self):
        super().setUp()
        private_mkdtemp(self)

    def test_the_k_samples_have_a_temperature_above_zero_and_one_seed_each(self):
        """Round 8, M2, D34: at temperature 0 the K samples are one greedy answer and the K-agreement rule counts it K times (D33, one seed for all, was caught)."""
        self.run_stage(resume=False)
        self.assertEqual(len(self.bodies), len(self.tasks) * self.k)
        self.assertTrue(all(b["temperature"] > 0 for b in self.bodies), {b["temperature"] for b in self.bodies})
        per_task = collections.defaultdict(set)
        for (tid, _i), b in zip(self.requests, self.bodies):
            per_task[tid].add(b["seed"])
        self.assertTrue(all(len(seeds) == self.k for seeds in per_task.values()), "every one of the K samples of a task has its own seed")

    def test_the_prompt_the_sampler_sends_holds_the_referenced_data_block(self):
        """Round 8, M2, D19: the sampler's prompt must be the SFT record's prompt; without the block every task with DAT_/PTR_ is trained on data it was never shown."""
        t = {"pseudocode": "int f(void){return DAT_00102010[1];}", "question": "Q?", "refs": "0x102010: 01 02 03"}
        user = self.data._messages(t)[1]["content"]
        self.assertIn("Data referenced by the function (bytes at those addresses):\n0x102010: 01 02 03", user)
        self.assertLess(user.index("```c"), user.index("Data referenced"))
        self.assertLess(user.index("Data referenced"), user.index("Question: Q?"))
        self.assertNotIn("Data referenced", self.data._messages({**t, "refs": ""})[1]["content"])
        self.assertEqual(self.data._messages(t)[0]["role"], "system")

    def test_the_stripped_binary_is_built_with_the_tasks_cc_and_opt(self):
        """Round 8, M2, D35: the address of proc comes from t.sym and the decompile is of t: both are built with `{cc} {opt}`; a `t` without the -O level is another layout."""
        for cc, opt in (("gcc", "-O0"), ("clang", "-O3"), ("gcc", "-Os"), ("clang", "-O1")):
            cmds: list = []
            task = {"id": "x-1", "kind": "x", "opt": opt, "cc": cc, "score": "tokens", "accept": [["1"]]}
            with mock.patch.object(self.data, "_sh_rc", lambda cmd, d, timeout=900: (cmds.append(cmd), (1, "stop"))[1]):
                rec = self.data._decompile_task(task, "int proc(void){return 0;}", "seed7-crc")
            self.assertTrue(rec["error"].startswith(f"compile failed ({cc} {opt}"), rec)
            self.assertEqual(len(cmds), 1)
            self.assertRegex(cmds[0], rf"^{cc} {opt} -fno-lto -o t\.sym t\.c && {cc} {opt} -fno-lto -s -o t t\.c$")

    def test_the_modal_side_gate_gets_the_tasks_facts_source_cc_and_opt(self):
        task = {"id": "x-1", "kind": "x", "variant": "v", "opt": "-O3", "cc": "clang", "facts": [[5]], "score": "tokens", "accept": [["1"]]}
        seen: list = []
        with mock.patch.object(self.data, "_sh_rc", return_value=(0, "")), \
                mock.patch.object(self.data.survive, "check", lambda *a: (seen.append(a), (False, "gone"))[1]):
            rec = self.data._decompile_task(task, "SRC", "seed7-crc")
        self.assertEqual(seen, [([[5]], "SRC", "clang", "-O3")])
        self.assertIn("fact not in the compiled code", rec["error"])

@unittest.skipUnless(GCC_OBJDUMP, "gcc and objdump are needed")
class LcgModalRecheck(ModalCase):
    def setUp(self):
        private_mkdtemp(self)

    def test_the_modal_side_recheck_refuses_a_folded_task_and_lets_a_readable_one_through_to_the_decompile(self):
        by = {t.id: t for t in families.generate("legacy", 3000, 2027)}
        t = by["lcg-2027-0013"]
        with mock.patch.object(self.data, "_sh_rc", return_value=(0, "")):
            rec = self.data._decompile_task(t.meta(), t.source(), "seed2027-legacy")
        self.assertIn("fact not in the compiled code", rec.get("error", ""), rec)
        ok = by["lcg-2027-0173"]
        with mock.patch.object(self.data, "_sh_rc", return_value=(0, "")), mock.patch.object(self.data, "_decompile_fn", side_effect=KeyError("reached the decompile")):
            with self.assertRaises(KeyError):
                self.data._decompile_task(ok.meta(), ok.source(), "seed2027-legacy")


# ---- m12 -------------------------------------------------------------------------------------------------------------------------------

class LoraEncode(ModalCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.modules.pop("modal_lora", None)
        import modal_lora
        cls.lora = modal_lora

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("modal_lora", None)
        super().tearDownClass()

    def test_a_pair_over_the_window_is_dropped_not_cut(self):
        """Round 8, m12: `tok(full)[:2048]` cut the answer (and its EOS) of a long pair, and gave an all-masked example when the prompt alone was longer."""
        enc, n = self.lora.encode_ids, self.lora.MAX_LEN
        self.assertEqual(n, 2048)
        p, a = list(range(100, 160)), [7, 8, 9, 2]
        ok = enc(p, p + a)
        self.assertEqual(ok["input_ids"], p + a)
        self.assertEqual(ok["labels"], [-100] * len(p) + a)
        self.assertEqual(ok["attention_mask"], [1] * len(p + a))
        self.assertIsNotNone(enc(list(range(n - 4)), list(range(n - 4)) + a), "exactly the window")
        self.assertIsNone(enc(list(range(n - 3)), list(range(n - 3)) + a), "one token over the window: dropped")
        self.assertIsNone(enc(list(range(3000)), list(range(3000)) + a), "a prompt that fills the window leaves nothing to learn")
        self.assertIsNone(enc(p, p), "no answer tokens")
        self.assertIsNone(enc(p, p[:10]))

    def test_train_uses_it_and_counts_what_it_dropped(self):
        import inspect
        src = inspect.getsource(self.lora.train)
        self.assertIn("encode_record(", src, "round 13: train encodes through encode_record, which ends in encode_ids")
        self.assertIn("encode_ids(", inspect.getsource(self.lora.encode_record))
        self.assertNotIn("[:2048]", src)
        self.assertIn("dropped", src)

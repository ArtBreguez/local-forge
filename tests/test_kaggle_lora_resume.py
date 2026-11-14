"""What an attached output is tied to: the two keys of the Kaggle LoRA build (PR 34 review, m4; third review, the owner's decision). The ADAPTER key is the training identity (the SFT file and the
training settings), the EVALUATION key is that adapter key with the evaluation set (the content exclusion and the ids asked) and the evaluation settings. What each key reads, setting by setting, is
`tests/test_kaggle_lora_keys.py` (fourth review, m1-m4).
"""
from __future__ import annotations

import json
import unittest

from .test_kaggle_lora import Kit, sft_rows, write_jsonl


class AdapterKey(Kit):
    """An adapter is tied to the records and settings it was made with (PR 34 review, m4), and to nothing of the evaluation."""

    def keys(self, out, **kw):
        cfg = self.build(out=self.dir / out, **kw)
        return cfg["adapter_sha"], cfg["eval_sha"]

    def test_the_same_build_has_the_same_keys_and_the_dataset_says_so(self):
        a = self.keys("a")
        self.assertEqual(a, self.keys("b"))
        on_disk = json.loads((self.dir / "a" / "data" / "lora-config.json").read_text())
        self.assertEqual((on_disk["adapter_sha"], on_disk["eval_sha"]), a)
        self.assertNotIn("config_sha", on_disk, "one name for each identity: nothing reads a third")
        for key in a:
            self.assertRegex(key, r"^[0-9a-f]{16}$")
        self.assertNotEqual(a[0], a[1])

    def test_other_records_or_other_training_settings_are_another_adapter_and_so_another_evaluation(self):
        base = self.keys("a")
        other = write_jsonl(self.dir / "other.jsonl", sft_rows(20))
        for name, kw in (("sft", {"sft": other}), ("rank", {"rank": 8}), ("max_len", {"max_len": 3072}), ("epochs", {"epochs": 1.0}), ("lr", {"lr": 1e-4}), ("batch", {"global_batch": 8}), ("mix", {"mix": True}),
                         ("student", {"student": "someone/another-student"})):
            got = self.keys(name, **kw)
            self.assertNotEqual(got[0], base[0], name)
            self.assertNotEqual(got[1], base[1], f"{name}: the replies of another adapter are another evaluation")

    def test_what_neither_the_adapter_nor_the_evaluation_depends_on_leaves_both_keys(self):
        base = self.keys("a")
        for name, kw in (("hours that do not cap the steps", {"train_hours": 1.0}), ("quota", {"gpu_hours": 9.0}), ("resume", {"resume_from": "someone/earlier"})):
            self.assertEqual(self.keys(name.replace(" ", "-"), **kw), base, f"{name}: neither the adapter nor the evaluation")
        for name, kw in (("eval", {"per_family": 2}), ("subset", {"quota": "add=2,mul=1"}), ("k", {"k": 2}), ("budget", {"max_new_tokens": 2048})):
            got = self.keys(name, **kw)
            self.assertEqual(got[0], base[0], f"{name}: the adapter is kept")
            self.assertNotEqual(got[1], base[1], f"{name}: other tasks, another number of replies or another budget: another evaluation (fourth review, m2)")


if __name__ == "__main__":
    unittest.main()

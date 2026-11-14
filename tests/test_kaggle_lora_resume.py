"""What an attached output is tied to: the two keys of the Kaggle LoRA build (PR 34 review, m4; third review, the owner's decision). The ADAPTER key is the training identity (the SFT file and the
training settings), the EVALUATION key is that adapter key with the evaluation set (the content exclusion and the ids asked).
"""
from __future__ import annotations

import json
import unittest

from .test_kaggle_lora import Kit, lora_eval, sft_rows, write_jsonl


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

    def test_what_the_adapter_does_not_depend_on_leaves_the_adapter_key(self):
        base = self.keys("a")
        for name, kw in (("k", {"k": 2}), ("budget", {"max_new_tokens": 2048}), ("hours", {"train_hours": 1.0}), ("quota", {"gpu_hours": 9.0}), ("resume", {"resume_from": "someone/earlier"})):
            self.assertEqual(self.keys(name, **kw), base, f"{name}: neither the adapter nor the evaluation set")
        for name, kw in (("eval", {"per_family": 2}), ("subset", {"quota": "add=2,mul=1"})):
            got = self.keys(name, **kw)
            self.assertEqual(got[0], base[0], f"{name}: the adapter is kept")
            self.assertNotEqual(got[1], base[1], f"{name}: other tasks are asked, another evaluation")

    def test_the_adapter_key_reads_the_records_and_exactly_the_training_keys(self):
        cfg = lora_eval.resolve_config({})
        base = lora_eval.adapter_key("abc", cfg)
        self.assertNotEqual(lora_eval.adapter_key("abd", cfg), base)
        self.assertIn("student", lora_eval.TRAIN_KEYS, "the base model is part of what an adapter is")
        for key in lora_eval.TRAIN_KEYS:
            changed = {**cfg, key: "x" if isinstance(cfg[key], str) else (not cfg[key]) if isinstance(cfg[key], bool) else cfg[key] + 1}
            self.assertNotEqual(lora_eval.adapter_key("abc", changed), base, key)
        for key in ("k", "max_steps", "train_hours", "deadline_h", "gpus", "max_new_tokens", "seed_base", "adapter_sha", "eval_sha", "eval_reserve_s", "export_gguf"):
            self.assertEqual(lora_eval.adapter_key("abc", {**cfg, key: 12345}), base, key)

    def test_the_evaluation_key_reads_the_adapter_the_exclusion_and_the_ids_asked(self):
        base = lora_eval.eval_key("adapter", "ex", ["t1", "t2"])
        self.assertRegex(base, r"^[0-9a-f]{16}$")
        self.assertEqual(lora_eval.eval_key("adapter", "ex", ["t2", "t1"]), base, "a set: the order of the ids is not part of it")
        self.assertEqual(lora_eval.eval_key("adapter", "ex", {"t1", "t2"}), base)
        for name, other in (("another adapter", ("adapteR", "ex", ["t1", "t2"])), ("another exclusion", ("adapter", "ey", ["t1", "t2"])), ("a task fewer", ("adapter", "ex", ["t1"])), ("a task more", ("adapter", "ex", ["t1", "t2", "t3"])),
                            ("another task", ("adapter", "ex", ["t1", "t3"]))):
            self.assertNotEqual(lora_eval.eval_key(*other), base, name)


if __name__ == "__main__":
    unittest.main()

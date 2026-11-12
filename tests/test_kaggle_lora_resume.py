"""What an attached output is tied to: the `config_sha` of the Kaggle LoRA build (PR 34 review, m4).
"""
from __future__ import annotations

import json
import unittest

from .test_kaggle_lora import Kit, lora_eval, sft_rows, write_jsonl


class ConfigSha(Kit):
    """An adapter and its replies are tied to the records and settings they were made with (PR 34 review, m4)."""

    def sha(self, out, **kw):
        return self.build(out=self.dir / out, **kw)["config_sha"]

    def test_the_same_build_has_the_same_sha_and_the_dataset_says_so(self):
        a = self.sha("a")
        self.assertEqual(a, self.sha("b"))
        self.assertEqual(json.loads((self.dir / "a" / "data" / "lora-config.json").read_text())["config_sha"], a)
        self.assertRegex(a, r"[0-9a-f]{16}")

    def test_other_records_or_other_training_settings_are_another_sha(self):
        base = self.sha("a")
        other = write_jsonl(self.dir / "other.jsonl", sft_rows(20))
        for name, kw in (("sft", {"sft": other}), ("rank", {"rank": 8}), ("max_len", {"max_len": 3072}), ("epochs", {"epochs": 1.0}), ("lr", {"lr": 1e-4}), ("batch", {"global_batch": 8}), ("mix", {"mix": True}),
                         ("student", {"student": "someone/another-student"})):
            self.assertNotEqual(self.sha(name, **kw), base, name)

    def test_what_the_adapter_does_not_depend_on_leaves_it(self):
        base = self.sha("a")
        for name, kw in (("k", {"k": 2}), ("budget", {"max_new_tokens": 2048}), ("hours", {"train_hours": 1.0}), ("eval", {"per_family": 2}), ("quota", {"gpu_hours": 9.0}), ("resume", {"resume_from": "someone/earlier"})):
            self.assertEqual(self.sha(name, **kw), base, name)

    def test_the_function_reads_the_records_and_exactly_the_training_keys(self):
        cfg = lora_eval.resolve_config({})
        base = lora_eval.config_sha("abc", cfg)
        self.assertNotEqual(lora_eval.config_sha("abd", cfg), base)
        for key in lora_eval.TRAIN_KEYS:
            changed = {**cfg, key: "x" if isinstance(cfg[key], str) else (not cfg[key]) if isinstance(cfg[key], bool) else cfg[key] + 1}
            self.assertNotEqual(lora_eval.config_sha("abc", changed), base, key)
        for key in ("k", "max_steps", "train_hours", "deadline_h", "gpus", "max_new_tokens", "seed_base", "config_sha"):
            self.assertEqual(lora_eval.config_sha("abc", {**cfg, key: 12345}), base, key)


if __name__ == "__main__":
    unittest.main()

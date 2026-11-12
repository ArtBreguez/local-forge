"""The deadline of the Kaggle LoRA kernel, tied to the quota that is left and to the smoke run that comes first (PR 34 review, m1).
"""
from __future__ import annotations

import json
import math
import unittest

from .test_kaggle_lora import Kit, build_lora, lora_eval


class Deadline(Kit):
    """The kernel's deadline is tied to the quota that is left and to the smoke run that comes first (PR 34 review, m1)."""

    def measured(self):
        path = self.dir / "smoke-run.json"
        path.write_text(json.dumps({"measured": {"train_tok_per_s_per_gpu": 150, "gen_step_s": 0.3}}))
        return path

    def test_before_the_smoke_run_it_is_the_quota_less_the_smoke_run_less_the_margin(self):
        cfg = self.build()
        smoke_h = cfg["built_from"]["estimate"]["smoke_s"] / 3600
        self.assertTrue(0.3 < smoke_h < 1.0, smoke_h)
        self.assertEqual(cfg["deadline_h"], math.floor((8.0 - smoke_h - build_lora.DEADLINE_MARGIN_H) * 100) / 100)
        self.assertLess(cfg["deadline_h"], lora_eval.DEFAULTS["deadline_h"], "7.5 h after a 0.7 h smoke run would be past the 8 h quota")
        self.assertIn("smoke:    + ", "\n".join(cfg["plan_lines"]))

    def test_after_the_smoke_run_the_quota_given_is_what_is_left(self):
        cfg = self.build(measured=self.measured(), out=self.dir / "m")
        self.assertEqual((cfg["built_from"]["estimate"]["smoke_s"], cfg["built_from"]["estimate"]["smoke_done"]), (0, True))
        self.assertEqual(cfg["deadline_h"], 7.5, "8 h left: the default 7.5 h, less than 8 - 0.25")
        cfg = self.build(measured=self.measured(), gpu_hours=7.3, out=self.dir / "m2")
        self.assertEqual(cfg["deadline_h"], 7.05, "7.3 h left: 7.3 less the margin")
        self.assertNotIn("smoke:    +", "\n".join(cfg["plan_lines"]))

    def test_a_deadline_past_the_quota_is_refused_and_one_inside_it_is_kept(self):
        for kw in ({"deadline_h": 7.5}, {"gpu_hours": 0.5}, {"gpu_hours": 0.0}):
            with self.assertRaisesRegex(SystemExit, "kernel's deadline can be at most"):
                self.build(out=self.dir / "bad", **kw)
        self.assertFalse((self.dir / "bad").exists())
        self.assertEqual(self.build(deadline_h=5.0, out=self.dir / "ok")["deadline_h"], 5.0)

    def test_the_smoke_build_itself_has_no_smoke_run_to_wait_for(self):
        cfg = self.build(smoke=True)
        self.assertEqual(cfg["deadline_h"], 7.5)
        self.assertNotIn("smoke_s", cfg["built_from"]["estimate"])

    def test_the_deadline_is_what_the_kernel_reads(self):
        cfg = self.build()
        got = lora_eval.resolve_config(json.loads((self.dir / "out" / "data" / "lora-config.json").read_text()))
        self.assertEqual(got["deadline_h"], cfg["deadline_h"])


if __name__ == "__main__":
    unittest.main()

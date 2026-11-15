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

    def test_a_resume_is_charged_no_smoke_run_with_or_without_measured(self):
        """The seventh review's second note: a resume built without --measured (as the run it resumes was) was charged the smoke run "that comes first". None comes before a resume."""
        first = self.build()
        smoke_s = first["built_from"]["estimate"]["smoke_s"]
        self.assertGreater(smoke_s, 0)
        self.assertNotIn("smoke_none", first["built_from"]["estimate"])
        none = "smoke:    none is counted: a resume (--resume-from) continues a run that was started, and no smoke run comes before it: --gpu-hours is the quota that is left now"
        self.assertNotIn(none, first["plan_lines"])
        cfg = self.build(resume_from="someone/example-arith-lora-eval", out=self.dir / "r")
        est = cfg["built_from"]["estimate"]
        self.assertEqual((est["smoke_s"], est["smoke_done"], est["smoke_none"]), (0, False, True))
        self.assertEqual(cfg["deadline_h"], 7.5, "8 h left: the default 7.5 h, as after a smoke run that was measured")
        self.assertEqual(self.build(resume_from="someone/example-arith-lora-eval", gpu_hours=7.3, out=self.dir / "r2")["deadline_h"], 7.05, "7.3 h left: 7.3 less the margin, and no smoke run")
        self.assertIn(none, cfg["plan_lines"])
        text = "\n".join(cfg["plan_lines"])
        self.assertNotIn("smoke:    +", text)
        self.assertNotIn("(smoke run included)", text)
        self.assertNotIn("the smoke run is done", text)
        total = lambda c: next(x for x in c["plan_lines"] if x.startswith("total:"))
        h = lambda s: f"{s / 3600:.1f} h"
        self.assertTrue(total(cfg).startswith(f"total:    {h(est['total_upper_s'])} assuming"), total(cfg))
        self.assertTrue(total(first).startswith(f"total:    {h(est['total_upper_s'] + smoke_s)} assuming"), total(first))
        self.assertIn("(smoke run included)", total(first))
        same = ("adapter_sha", "eval_sha", "max_steps", "eval_reserve_s", "export_reserve_s")
        self.assertEqual({k: cfg[k] for k in same}, {k: first[k] for k in same}, "the clock is in no key: the resume keeps the adapter and the replies of the run that was built without --measured")
        self.assertEqual({k: v for k, v in est.items() if k not in ("smoke_s", "smoke_none")}, {k: v for k, v in first["built_from"]["estimate"].items() if k != "smoke_s"})
        with self.assertRaises(SystemExit) as e:
            self.build(resume_from="someone/example-arith-lora-eval", deadline_h=7.8, out=self.dir / "late")
        self.assertEqual(str(e.exception), "--gpu-hours 8: the kernel's deadline can be at most 7.75 h (the quota left, less 0.25 h), and --deadline-h is 7.8")
        with self.assertRaises(SystemExit) as e:
            self.build(deadline_h=7.8, out=self.dir / "late")
        self.assertEqual(str(e.exception), f"--gpu-hours 8: the kernel's deadline can be at most {8 - smoke_s / 3600 - 0.25:.2f} h (the quota left, less {smoke_s / 3600:.1f} h for the smoke run that is still to come, less 0.25 h), "
                                           "and --deadline-h is 7.8")
        with self.assertRaises(SystemExit) as e:
            self.build(gpu_hours=0.2, smoke=True, out=self.dir / "late")
        self.assertEqual(str(e.exception), "--gpu-hours 0.2: the kernel's deadline can be at most 0.00 h (the quota left, less 0.25 h)")
        self.assertEqual(self.build(resume_from="someone/example-arith-lora-eval", deadline_h=7.7, out=self.dir / "r3")["deadline_h"], 7.7, "past what a build that still has its smoke run to pay may ask")
        measured = self.build(resume_from="someone/example-arith-lora-eval", measured=self.measured(), out=self.dir / "m")
        self.assertEqual((measured["built_from"]["estimate"]["smoke_s"], measured["built_from"]["estimate"]["smoke_done"], "smoke_none" in measured["built_from"]["estimate"]), (0, True, False))
        self.assertIn("smoke:    the smoke run is done (--measured): --gpu-hours is the quota that is left after it", measured["plan_lines"])
        self.assertNotIn(none, measured["plan_lines"])
        bare = build_lora.plan([3000] * 40, [2000] * 9, epochs=2.0, gpus=2, global_batch=16, train_hours=3.0, k=1, max_new_tokens=4096, eval_batch=8, export=False, rates=build_lora.RATES)
        self.assertFalse([x for x in build_lora.describe(bare, 8.0, 7.5) if x.startswith("smoke:")], "a plan that says nothing of a smoke run gets no line about one")
        smoke = self.build(resume_from="someone/example-arith-lora-eval", smoke=True, out=self.dir / "s")
        self.assertFalse({"smoke_s", "smoke_none"} & set(smoke["built_from"]["estimate"]))
        self.assertFalse([x for x in smoke["plan_lines"] if x.startswith("smoke:")])

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

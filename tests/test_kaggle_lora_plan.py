"""The deadline of the Kaggle LoRA kernel, tied to the quota that is left and to the smoke run that comes first (PR 34 review, m1).
"""
from __future__ import annotations

import json
import math
import re
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

    def test_a_smoke_build_that_does_not_fit_is_not_told_to_lower_what_a_smoke_run_does_not_read(self):
        """PR 34 eighth review, m1: it got the full run's advice, "lower --train-hours, --k, --per-family or --max-new-tokens", and a smoke build with each of them lowered is the same run and the same line."""
        total = lambda cfg: next(x for x in cfg["plan_lines"] if x.startswith("total:"))
        cfg = self.build(smoke=True, gpu_hours=0.7)
        est = cfg["built_from"]["estimate"]
        size, need = est["total_upper_s"] / 3600, float(re.search(r"it needs --gpu-hours ([\d.]+) or more", total(cfg)).group(1))
        self.assertEqual(cfg["deadline_h"], 0.44, "0.7 less the margin, in hundredths and never later")
        self.assertTrue(size + 0.25 <= need < size + 0.25 + 0.02, (size, need))
        self.assertEqual(total(cfg), f"total:    {size:.1f} h with its one batch run to its token budget, of the 0.7 h of quota left: IT DOES NOT FIT (inside the quota, but this run needs {size:.2f} h, {math.ceil((size - 0.44) * 60)} min past the 0.44 h after which the kernel starts nothing: "
                         f"it would stop training early or leave the last batches unasked): a smoke run has one size, {size:.2f} h here, and the kernel's deadline is at most the quota left less 0.25 h: it needs "
                         f"--gpu-hours {need:g} or more (and no --deadline-h under its size). --train-hours, --k, --per-family and --max-new-tokens change nothing of a smoke run; only a lower --max-len shortens it, "
                         f"by part of the {est['steps'] * est['step_s']:.0f} s of its {est['steps']} steps, which run on the longest records it keeps (and the memory the run measures is then that window's)")
        self.assertNotIn("lower --train-hours", total(cfg))
        same = self.build(smoke=True, gpu_hours=0.7, train_hours=0.5, k=3, per_family=5, max_new_tokens=512, out=self.dir / "lowered")
        self.assertEqual((total(same), same["built_from"]["estimate"]), (total(cfg), est), "the four flags the advice names change nothing of a smoke run")
        over = self.build(smoke=True, gpu_hours=0.5, out=self.dir / "over")
        self.assertIn(f"IT DOES NOT FIT (more than the quota): a smoke run has one size, {size:.2f} h here", total(over))
        self.assertIn("it fits", total(self.build(smoke=True, gpu_hours=need, out=self.dir / "enough")), "the hours the advice names are enough")
        self.assertIn("IT DOES NOT FIT", total(self.build(smoke=True, gpu_hours=round(need - 0.01, 2), out=self.dir / "short")), "and a hundredth less is not")
        self.assertIn("IT DOES NOT FIT (inside the quota", total(self.build(smoke=True, deadline_h=0.5, out=self.dir / "early")), "a --deadline-h under its size")
        lengths = sorted(set(build_lora.trainable_lengths(build_lora.read_sft(self.sft), cfg["max_len"], False)))
        window = self.build(smoke=True, gpu_hours=0.7, max_len=lengths[-2], out=self.dir / "window")["built_from"]["estimate"]
        self.assertEqual((est["step_s"], window["step_s"]), (round(2 * lengths[-1] / 150, 1), round(2 * lengths[-2] / 150, 1)), "--max-len is the one flag that does: the steps run on the longest record it keeps")
        kw = dict(gpus=2, steps=4, max_new_tokens=256, eval_batch=8, rates=build_lora.RATES)
        long, short = build_lora.smoke_plan(4000, **kw), build_lora.smoke_plan(1000, **kw)
        self.assertEqual(long["total_upper_s"] - short["total_upper_s"], round(4 * 2 * 3000 / 150))
        self.assertGreater(build_lora.smoke_plan(1, **kw)["total_upper_s"], long["total_upper_s"] - 4 * long["step_s"] - 1, "and by no more than its steps: the rest of a smoke run follows no flag")
        self.assertIn(f"by part of the {4 * long['step_s'] / 60:.0f} min of its 4 steps", build_lora.describe(long, 0.5, 0.25)[-2])
        for kind in ({}, {"resume_from": "someone/example-arith-lora-eval"}):
            full = self.build(gpu_hours=1.0, deadline_h=0.1, out=self.dir / f"full{len(kind)}", **kind)
            self.assertIn("lower --train-hours, --k, --per-family or --max-new-tokens", total(full), "a full run reads all four")
            self.assertNotIn("a smoke run has one size", total(full))

    def test_the_deadline_is_what_the_kernel_reads(self):
        cfg = self.build()
        got = lora_eval.resolve_config(json.loads((self.dir / "out" / "data" / "lora-config.json").read_text()))
        self.assertEqual(got["deadline_h"], cfg["deadline_h"])


if __name__ == "__main__":
    unittest.main()

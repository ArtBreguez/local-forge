import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from forge import autotune, plan, util

from .helpers import FAKE_BENCH, fixture, profile, real_model


class Parsing(unittest.TestCase):
    def test_real_llama_bench_jsonl(self):
        m = autotune.parse_bench_jsonl(fixture("llama-bench-sweepc-snapshot.jsonl"))
        # the file holds 12 configs; the dict keeps the last one per (depth, kind): ngl=16 FA on q8_0
        self.assertEqual(set(m), {(0, "pp"), (0, "tg"), (8192, "pp"), (8192, "tg")})
        one = "\n".join(l for l in fixture("llama-bench-sweepc-snapshot.jsonl").splitlines()
                        if '"n_gpu_layers": 16, ' in l and '"flash_attn": 0' in l)
        r = autotune.parse_bench_jsonl(one)
        self.assertAlmostEqual(r[(8192, "tg")][0], 9.15, places=2)
        self.assertAlmostEqual(r[(8192, "pp")][0], 194.61, places=2)
        self.assertAlmostEqual(r[(0, "tg")][0], 9.83, places=2)

    def test_ignores_chatter(self):
        text = "ggml_vulkan: Found 1 Vulkan devices:\n{broken json\n" + fixture("llama-bench-sweepc-snapshot.jsonl").splitlines()[1]
        self.assertEqual(len(autotune.parse_bench_jsonl(text)), 1)

    def test_real_oom_text_is_classified_as_oom(self):
        txt = fixture("llama-bench-oom-stderr.txt")
        self.assertEqual(autotune.classify_failure(1, txt, 99), "oom")
        # the generic llama-bench messages from sweep C count as OOM only when layers are on the GPU
        generic = "llama_bench: error: failed to create context with model\n'x.gguf'"
        self.assertEqual(autotune.classify_failure(1, generic, 20), "oom")
        self.assertEqual(autotune.classify_failure(1, generic, 0), "error")
        self.assertEqual(autotune.classify_failure(1, "illegal instruction", 8), "error")
        self.assertEqual(autotune.classify_failure(None, "", 8), "timeout")


class CandidateOrder(unittest.TestCase):
    def test_plan_point_first_then_nearest_upward_first(self):
        order = autotune.candidate_order(16, (False, "f16"), 32, True)
        self.assertEqual(order[0], (16, False, "f16"))
        self.assertEqual(order[1:3], [(18, False, "f16"), (14, False, "f16")])   # same setting before another setting at equal distance
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual({n for n, _, _ in order}, {12, 14, 16, 18, 20})
        self.assertEqual({(f, k) for _, f, k in order}, set(autotune.COMBOS))

    def test_cpu_only_has_no_layer_axis(self):
        order = autotune.candidate_order(0, (False, "f16"), 32, False)
        self.assertEqual({n for n, _, _ in order}, {0})
        self.assertEqual(len(order), 3)

    def test_clamped_to_valid_layer_counts(self):
        order = autotune.candidate_order(1, (False, "f16"), 32, True)
        self.assertTrue(all(0 <= n <= 33 for n, _, _ in order))
        order = autotune.candidate_order(33, (False, "f16"), 32, True)
        self.assertTrue(all(n <= 33 for n, _, _ in order))


class Objective(unittest.TestCase):
    def pt(self, ngl, fa, kv, pp, tg):
        return autotune.Point(ngl, fa, kv, "ok", ppd=pp, tgd=tg)

    def test_floor_excludes_faster_generation(self):
        pts = [self.pt(16, False, "f16", 190, 9.0), self.pt(16, True, "f16", 60, 9.8)]
        best, why = autotune.pick_best(pts, 100)
        self.assertEqual((best.fa, best.tgd), (False, 9.0))
        best, _ = autotune.pick_best(pts, 0)
        self.assertEqual(best.fa, True)

    def test_nothing_meets_floor_picks_best_prompt_speed(self):
        best, why = autotune.pick_best([self.pt(1, False, "f16", 40, 9), self.pt(2, False, "f16", 50, 3)], 100)
        self.assertEqual(best.ngl, 2)
        self.assertIn("floor", why)

    def test_failures_never_win(self):
        pts = [autotune.Point(20, False, "f16", "oom"), self.pt(16, False, "f16", 150, 5)]
        self.assertEqual(autotune.pick_best(pts, 100)[0].ngl, 16)
        self.assertIsNone(autotune.pick_best([autotune.Point(20, False, "f16", "oom")], 100)[0])


class WithFakeBench(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / "bench.log"
        self.env = mock.patch.dict(os.environ, {"FAKE_BENCH_LOG": str(self.log), "FAKE_BENCH_OOM_NGL": "18", "FAKE_BENCH_BUILD": "12207"})
        self.env.start()
        self.m = replace(real_model(), path=str(self.dir / "model.gguf"))
        self.prof = profile()
        self.plan = plan.build_plan(self.prof, self.m)            # ngl 15, FA off, f16, ctx 16384

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def calls(self):
        return [tuple(l.split()) for l in self.log.read_text().splitlines()] if self.log.exists() else []

    def run_tune(self, **kw):
        return autotune.tune(self.prof, self.m, self.plan, FAKE_BENCH, **kw)

    def test_finds_best_and_stops_at_oom(self):
        res = self.run_tune(budget_s=3600)
        self.assertTrue(res.complete)
        self.assertEqual(self.plan.config.ngl, 15)
        best = res.best
        self.assertEqual((best.fa, best.kv), (False, "f16"))      # FA on breaks the prompt floor in the fake
        self.assertEqual(best.ngl, 17)                            # the fake's generation speed grows with layers up to 16; 17 is the nearest probe above
        self.assertEqual(res.config.ngl, best.ngl)
        with mock.patch.dict(os.environ, {"FAKE_BENCH_OOM_NGL": "17"}):
            self.log.unlink()
            res = self.run_tune(budget_s=3600)
        statuses = {(p.ngl, p.fa, p.kv): p.status for p in res.points}
        self.assertEqual(statuses[(17, False, "f16")], "oom")
        # early stop: after the first failure for a setting, larger counts for that setting are skipped, not run
        self.assertEqual(statuses[(19, False, "f16")], "skipped")
        ran = [(int(n), fa, kv) for n, fa, kv, _ in self.calls()]
        self.assertNotIn((19, "off", "f16"), ran)
        self.assertNotIn((19, "on", "q8_0"), ran)
        self.assertEqual(sum(1 for r in ran if r[0] >= 17), len(autotune.COMBOS))   # one OOM probe per setting, no more
        self.assertEqual(res.best.ngl, 16)                        # refinement probed the count just below the failure and it won
        self.assertIn((16, "off", "f16"), [(int(n), fa, kv) for n, fa, kv, _ in self.calls()])
        self.assertNotIn((16, "on", "f16"), [(int(n), fa, kv) for n, fa, kv, _ in self.calls()])   # settings that are not winning are not refined

    def test_calls_use_both_depths_and_valid_flags(self):
        self.run_tune(budget_s=3600)
        self.assertTrue(all(d == "0,8192" for *_, d in self.calls()))
        self.assertFalse(("15", "off", "q8_0", "0,8192") in self.calls())      # q8_0 V cache is never tried without FA

    def test_objective_flips_when_the_floor_is_lifted(self):
        with mock.patch.dict(os.environ, {"FAKE_BENCH_FA_FAST": "1"}):
            strict = self.run_tune(budget_s=3600, pp_floor=100)
            loose = self.run_tune(budget_s=3600, pp_floor=0)
        self.assertFalse(strict.best.fa)
        self.assertTrue(loose.best.fa)

    def test_time_budget_returns_best_so_far_and_says_incomplete(self):
        t = {"now": 0.0}
        def clock():
            return t["now"]
        def slow_run(args, timeout=20, env=None):
            t["now"] += 400                                     # every llama-bench call "takes" 400 s
            return util.run_cmd(args, timeout=timeout)
        res = self.run_tune(budget_s=1000, clock=clock, run=slow_run)
        self.assertFalse(res.complete)
        measured = [p for p in res.points if p.status in ("ok", "oom")]
        self.assertLessEqual(len(measured), 3)                  # 1000 s / 400 s per call
        self.assertIsNotNone(res.best)
        self.assertEqual((res.points[0].ngl, res.points[0].fa, res.points[0].kv), (15, False, "f16"))   # the plan itself goes first
        self.assertTrue(any("budget" in n for n in res.notes))

    def test_every_point_fails_falls_back_to_the_plan(self):
        with mock.patch.dict(os.environ, {"FAKE_BENCH_OOM_NGL": "0"}):
            res = self.run_tune(budget_s=3600)
        self.assertIsNone(res.best)
        self.assertEqual(res.config, self.plan.config)

    def test_device_is_passed_to_llama_bench(self):
        cfg = replace(self.plan.config, device="Vulkan1")
        self.assertIn("-dev", autotune.bench_args(FAKE_BENCH, "m", cfg, 8192, 1, True))
        self.assertNotIn("-dev", autotune.bench_args(FAKE_BENCH, "m", self.plan.config, 8192, 1, True))

    def test_hanging_benchmark_is_killed_not_awaited(self):
        with mock.patch.dict(os.environ, {"FAKE_BENCH_SLEEP": "30"}):
            res = self.run_tune(budget_s=3600, per_point_timeout=0.3)
        self.assertTrue(all(p.status in ("timeout", "skipped") for p in res.points))

    def test_estimate_guard_skips_points_that_cannot_fit_the_card(self):
        res = self.run_tune(budget_s=3600)
        # 99 layers (whole model + output) never reaches llama-bench on a 4 GB card
        big = replace(self.plan, config=replace(self.plan.config, ngl=32))
        res = autotune.tune(self.prof, self.m, big, FAKE_BENCH, budget_s=3600)
        self.assertTrue(any(p.status == "skipped" and "total VRAM" in p.note for p in res.points))


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / "bench.log"
        self.env = mock.patch.dict(os.environ, {"FAKE_BENCH_LOG": str(self.log), "FAKE_BENCH_OOM_NGL": "18", "FAKE_BENCH_BUILD": "12207"})
        self.env.start()
        self.m = replace(real_model(), path=str(self.dir / "model.gguf"))
        self.prof = profile()
        self.plan = plan.build_plan(self.prof, self.m)
        self.cache = autotune.Cache(self.dir / "cache")

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def n_measurements(self):
        return len(self.log.read_text().splitlines()) if self.log.exists() else 0

    def go(self, prof=None, model=None, **kw):
        return autotune.tune_cached(prof or self.prof, model or self.m, self.plan, FAKE_BENCH, self.cache, budget_s=3600, **kw)

    def test_second_start_is_instant_and_identical(self):
        first, k1 = self.go()
        n = self.n_measurements()
        self.assertGreater(n, 0)
        self.assertFalse(first.from_cache)
        second, k2 = self.go()
        self.assertTrue(second.from_cache)
        self.assertEqual(k1, k2)
        self.assertEqual(self.n_measurements(), n)                # no llama-bench measurement the second time
        self.assertEqual(second.config, first.config)

    def test_changed_hardware_retunes(self):
        self.go()
        n = self.n_measurements()
        other = profile(gpu_name="AMD Radeon RX 6600", vram_total=8176, free_vram=7000)
        res, _ = self.go(prof=other)
        self.assertFalse(res.from_cache)
        self.assertGreater(self.n_measurements(), n)

    def test_load_changes_do_not_invalidate(self):
        self.go()
        n = self.n_measurements()
        busy = profile(free_vram=2500, ram_free=3000, battery=True)
        res, _ = self.go(prof=busy)
        self.assertTrue(res.from_cache)
        self.assertEqual(self.n_measurements(), n)

    def test_changed_model_retunes(self):
        self.go()
        res, _ = self.go(model=replace(self.m, fingerprint="0" * 32))
        self.assertFalse(res.from_cache)

    def test_changed_llama_build_retunes(self):
        self.go()
        with mock.patch.dict(os.environ, {"FAKE_BENCH_BUILD": "12999"}):
            res, _ = self.go()
        self.assertFalse(res.from_cache)

    def test_truncated_result_is_not_reused_for_a_request_with_more_time(self):
        t = {"now": 0.0}
        def slow_run(args, timeout=20, env=None):
            t["now"] += 400
            return util.run_cmd(args, timeout=timeout)
        short, _ = autotune.tune_cached(self.prof, self.m, self.plan, FAKE_BENCH, self.cache, budget_s=1000, clock=lambda: t["now"], run=slow_run)
        self.assertFalse(short.complete)
        n = self.n_measurements()
        again, _ = autotune.tune_cached(self.prof, self.m, self.plan, FAKE_BENCH, self.cache, budget_s=900, clock=lambda: t["now"], run=slow_run)
        self.assertTrue(again.from_cache)                       # no more time than before: nothing new to learn
        self.assertEqual(self.n_measurements(), n)
        longer, _ = autotune.tune_cached(self.prof, self.m, self.plan, FAKE_BENCH, self.cache, budget_s=3600)
        self.assertFalse(longer.from_cache)                     # more time allowed: measure the rest
        self.assertTrue(longer.complete)
        final, _ = autotune.tune_cached(self.prof, self.m, self.plan, FAKE_BENCH, self.cache, budget_s=3600)
        self.assertTrue(final.from_cache)

    def test_force_ignores_cache(self):
        self.go()
        res, _ = self.go(force=True)
        self.assertFalse(res.from_cache)

    def test_keys_differ_per_component(self):
        base = autotune.cache_key("hw", "model", "b1", "v")
        self.assertEqual(base, autotune.cache_key("hw", "model", "b1", "v"))
        for other in (autotune.cache_key("hw2", "model", "b1", "v"), autotune.cache_key("hw", "model2", "b1", "v"),
                      autotune.cache_key("hw", "model", "b2", "v"), autotune.cache_key("hw", "model", "b1", "v2")):
            self.assertNotEqual(base, other)
        # separator is unambiguous
        self.assertNotEqual(autotune.cache_key("a", "bc", "d"), autotune.cache_key("ab", "c", "d"))

    def test_failed_tuning_is_not_cached(self):
        with mock.patch.dict(os.environ, {"FAKE_BENCH_OOM_NGL": "0"}):
            res, key = self.go()
        self.assertIsNone(res.best)
        self.assertIsNone(self.cache.load(key))

    def test_corrupt_cache_file_is_ignored(self):
        self.cache.dir.mkdir(parents=True)
        self.cache.path.write_text("{not json")
        res, _ = self.go()
        self.assertFalse(res.from_cache)
        self.assertIsNotNone(res.best)

    def test_detect_build_parses_version(self):
        real = fixture("llama-bench-version-b12207.txt")      # the real b12207 output: "0.9.0-dev (build 12207, commit ...)"
        self.assertEqual(autotune.detect_build(["llama-bench"], run=lambda a, timeout=0: (0, real)), "b12207-998200491")
        self.assertEqual(autotune.detect_build(["x"], run=lambda a, timeout=0: (0, "version: 4567 (abcdef0)")), "b4567-abcdef0")
        self.assertNotEqual(autotune.detect_build(["x"], run=lambda a, timeout=0: (0, real)),
                            autotune.detect_build(["x"], run=lambda a, timeout=0: (0, real.replace("12207", "12999"))))
        self.assertTrue(autotune.detect_build([str(self.dir / "nope")], run=lambda a, timeout=0: None).startswith("unknown"))


if __name__ == "__main__":
    unittest.main()

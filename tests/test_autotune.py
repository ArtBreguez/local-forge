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
    CENTERS = {(False, "f16", 512): 14, (False, "f16", 256): 16, (True, "f16", 512): 16, (True, "f16", 256): 18,
               (True, "q8_0", 512): 16, (True, "q8_0", 256): 18}
    PLAN = (False, "f16", 512)

    def cost(self, o):
        n, fa, kv, ub = o
        return (autotune.OFFSET_COST * (self.CENTERS[(fa, kv, ub)] - n) + (autotune.COMBO_COST if (fa, kv) != self.PLAN[:2] else 0)
                + (autotune.UBATCH_COST if ub != self.PLAN[2] else 0))

    def test_plan_point_first_then_the_other_ubatch_then_other_attention_settings(self):
        order = autotune.candidate_order(self.CENTERS, self.PLAN, 33, True)
        self.assertEqual(order[0], (14, False, "f16", 512))
        self.assertEqual(order[1], (16, False, "f16", 256))                                 # a different ubatch costs 1
        self.assertEqual({o[1:] for o in order[2:4]}, {(True, "f16", 512), (True, "q8_0", 512)})   # a different attention/KV costs 2
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(len(order), 6 * len(autotune.LAYER_OFFSETS))
        costs = [self.cost(o) for o in order]
        self.assertEqual(costs, sorted(costs))
        for n, fa, kv, ub in order:
            self.assertLessEqual(n, self.CENTERS[(fa, kv, ub)])                             # never above what the server can hold
        self.assertEqual({(f, k) for _, f, k, _ in order}, set(autotune.COMBOS))
        self.assertEqual({u for *_, u in order}, {512, 256})

    def test_cpu_only_has_no_layer_axis(self):
        centers = {(f, k, 512): 0 for f, k in autotune.COMBOS}
        order = autotune.candidate_order(centers, (False, "f16", 512), 33, False)
        self.assertEqual({n for n, *_ in order}, {0})
        self.assertEqual(len(order), 3)

    def test_clamped_to_valid_layer_counts_and_settings_without_room_are_skipped(self):
        order = autotune.candidate_order({(False, "f16", 512): 2, (False, "f16", 256): 0, (True, "f16", 512): 40}, (False, "f16", 512), 34, True)
        self.assertTrue(all(1 <= n <= 34 for n, *_ in order))
        self.assertEqual({o[1:] for o in order}, {(False, "f16", 512), (True, "f16", 512)})        # nothing fits with ubatch 256: no point
        self.assertEqual([n for n, f, *_ in order if not f], [2, 1])


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
        self.env = mock.patch.dict(os.environ, {"FAKE_BENCH_LOG": str(self.log), "FAKE_BENCH_OOM_NGL": "99", "FAKE_BENCH_BUILD": "12207"})
        self.env.start()
        self.m = replace(real_model(), path=str(self.dir / "model.gguf"))
        self.prof = profile()
        self.plan = plan.build_plan(self.prof, self.m)            # ngl 14, FA off, f16, ubatch 512, ctx 16384

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def calls(self):
        return [tuple(l.split()) for l in self.log.read_text().splitlines()] if self.log.exists() else []

    def run_tune(self, **kw):
        return autotune.tune(self.prof, self.m, self.plan, FAKE_BENCH, **kw)

    def statuses(self, res):
        return {(p.ngl, p.fa, p.kv, p.ub): p.status for p in res.points}

    def test_picks_the_best_setting_and_measures_one_point_per_setting(self):
        res = self.run_tune(budget_s=3600)
        self.assertTrue(res.complete)
        self.assertEqual((self.plan.config.ngl, self.plan.config.ubatch), (14, 512))
        self.assertEqual(len(self.calls()), 6)                    # 3 attention/KV settings x 2 ubatch, the top layer count of each; lower counts skipped
        self.assertEqual(self.statuses(res)[(16, False, "f16", 256)], "ok")
        self.assertEqual(self.statuses(res)[(13, False, "f16", 512)], "skipped")
        best = res.best
        self.assertEqual((best.fa, best.kv, best.ub, best.ngl), (False, "f16", 256, 16))     # more layers (sweep D: 9.60 vs 9.13 tok/s) and pp above the floor
        self.assertEqual((res.config.ngl, res.config.ubatch), (16, 256))
        self.assertIn("-ub 256", " ".join(plan.server_command("llama-server", self.m.path, res.config)))
        self.assertEqual(res.conditions["server_ctx"], 16384)

    def test_never_measures_more_layers_than_the_server_can_hold(self):
        """llama-bench would happily load 16 layers at ubatch 512 (it runs at ~8.8K context); the server cannot at 16K, so that point is not worth a minute."""
        self.run_tune(budget_s=3600)
        for n, fa, kv, d, ub in self.calls():
            limit = plan.max_layers(self.m, replace(self.plan.config, fa=fa == "on", kv=kv, ubatch=int(ub)), 3443)
            self.assertLessEqual(int(n), limit, (n, fa, kv, ub))
        ran = {(int(n), fa, kv, int(ub)) for n, fa, kv, _, ub in self.calls()}
        self.assertIn((14, "off", "f16", 512), ran)
        self.assertNotIn((15, "off", "f16", 512), ran)
        self.assertNotIn((16, "off", "f16", 512), ran)

    def test_prompt_floor_decides_between_ubatches(self):
        # fake: ubatch 256 prefills at 148 tok/s, 512 at 190 (sweep D: 151 vs 195)
        strict = self.run_tune(budget_s=3600, pp_floor=160)
        loose = self.run_tune(budget_s=3600, pp_floor=100)
        self.assertEqual((strict.config.ubatch, strict.config.ngl), (512, 14))
        self.assertEqual((loose.config.ubatch, loose.config.ngl), (256, 16))

    def test_ubatch_axis_can_be_restricted(self):
        res = self.run_tune(budget_s=3600, ubatches=(512,))
        self.assertEqual({u for *_, u in self.calls()}, {"512"})
        self.assertEqual(res.config.ubatch, 512)

    def test_top_count_failing_falls_back_one_layer_at_a_time(self):
        with mock.patch.dict(os.environ, {"FAKE_BENCH_OOM_NGL": "15"}):        # the estimate was optimistic by 2 layers at ubatch 256
            res = self.run_tune(budget_s=3600)
        st = self.statuses(res)
        self.assertEqual(st[(16, False, "f16", 256)], "oom")
        self.assertEqual(st[(15, False, "f16", 256)], "oom")
        self.assertEqual(st[(14, False, "f16", 256)], "ok")
        self.assertEqual(st[(12, False, "f16", 256)], "skipped")                # the setting ran at 14: lower counts are not measured
        self.assertEqual(st[(14, False, "f16", 512)], "ok")
        self.assertLessEqual(res.config.ngl, 14)
        self.assertTrue(all(n < 15 for (n, *_), s_ in st.items() if s_ == "ok"))

    def test_calls_use_both_depths_and_valid_flags(self):
        self.run_tune(budget_s=3600)
        self.assertTrue(all(d == "0,8192" for _, _, _, d, _ in self.calls()))
        self.assertFalse([c for c in self.calls() if c[1] == "off" and c[2] == "q8_0"])      # q8_0 V cache is never tried without FA

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
        self.assertEqual((res.points[0].ngl, res.points[0].fa, res.points[0].kv, res.points[0].ub), (14, False, "f16", 512))   # the plan itself goes first
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
        self.assertEqual(autotune.bench_args(FAKE_BENCH, "m", replace(cfg, ubatch=256), 8192, 1, True)[autotune.bench_args(FAKE_BENCH, "m", replace(cfg, ubatch=256), 8192, 1, True).index("-ub") + 1], "256")

    def test_hanging_benchmark_is_killed_not_awaited(self):
        with mock.patch.dict(os.environ, {"FAKE_BENCH_SLEEP": "30"}):
            res = self.run_tune(budget_s=3600, per_point_timeout=0.3)
        self.assertTrue(all(p.status in ("timeout", "skipped") for p in res.points))
        self.assertEqual(len(self.calls()), 6)                  # a timeout settles the setting: no slower, lower layer count is tried after it

    def test_estimate_guard_skips_points_the_server_cannot_hold(self):
        # a plan with 32 layers (the whole model) never reaches llama-bench on a 4 GB card
        big = replace(self.plan, config=replace(self.plan.config, ngl=32))
        res = autotune.tune(self.prof, self.m, big, FAKE_BENCH, budget_s=3600)
        skipped = [p for p in res.points if p.status == "skipped" and "server cannot hold" in p.note]
        self.assertTrue(skipped)
        self.assertTrue(all(p.ngl >= 28 for p in skipped))
        for n, fa, kv, d, ub in self.calls():
            self.assertLessEqual(int(n), plan.max_layers(self.m, replace(self.plan.config, fa=fa == "on", kv=kv, ubatch=int(ub)), 3443), (n, fa, kv, ub))

    def test_cpu_plan_tunes_attention_settings_only(self):
        cpu = plan.build_plan(profile(ram_free=12000), self.m, mode="cpu")
        res = autotune.tune(profile(ram_free=12000), self.m, cpu, FAKE_BENCH, budget_s=3600)
        self.assertEqual({(p.ngl, p.ub) for p in res.points}, {(0, 512)})
        self.assertEqual(len([p for p in res.points if p.status == "ok"]), 3)


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

    def test_entry_from_the_buggy_first_host_tune_is_inert_and_documented(self):
        """results/forge-tune-host-20261108.json: build 'b0' (version regex bug), 15 layers (not loadable by llama-server at 16K), no ubatch axis."""
        import json
        from .helpers import ROOT
        raw = json.loads((ROOT / "results" / "forge-tune-host-20261108.json").read_text())
        self.assertIn("b0", raw["_note"])
        self.assertIn("NOT a valid cache entry", raw["_note"])
        entries = {k: v for k, v in raw.items() if not k.startswith("_")}
        (key, entry), = entries.items()
        self.assertEqual(entry["llama_build"], "b0")                              # left as recorded: it is evidence
        real_build = autotune.detect_build(["llama-bench"], run=lambda a, timeout=0: (0, fixture("llama-bench-version-b12207.txt")))
        self.assertEqual(real_build, "b12207-998200491")
        cache = autotune.Cache(self.dir / "stale")
        cache.dir.mkdir(parents=True)
        cache.path.write_text(json.dumps(entries))
        self.assertIsNotNone(cache.load(key))
        new_key = autotune.cache_key(entry["hardware"], entry["model"], real_build, autotune.variant_of(self.plan))
        self.assertNotEqual(key, new_key)
        self.assertIsNone(cache.load(new_key))
        # an entry stored under the CURRENT build and the previous search space (same request, no ubatch axis, layers not bounded by the server) is not reused
        old_variant = f"ctx={self.plan.requested_ctx};depth=8192;floor=100;plan=gpu"
        self.assertNotEqual(old_variant, autotune.variant_of(self.plan))
        fresh, _ = self.go()
        old_cache = autotune.Cache(self.dir / "old-version")
        old_cache.save(autotune.cache_key(self.prof["fingerprint"], self.m.fingerprint, real_build, old_variant),
                       {"result": fresh.to_dict(), "variant": old_variant})
        n = self.n_measurements()
        again, _ = autotune.tune_cached(self.prof, self.m, self.plan, FAKE_BENCH, old_cache, budget_s=3600)
        self.assertFalse(again.from_cache)
        self.assertGreater(self.n_measurements(), n)

    def test_detect_build_parses_version(self):
        real = fixture("llama-bench-version-b12207.txt")      # the real b12207 output: "0.9.0-dev (build 12207, commit ...)"
        self.assertEqual(autotune.detect_build(["llama-bench"], run=lambda a, timeout=0: (0, real)), "b12207-998200491")
        self.assertEqual(autotune.detect_build(["x"], run=lambda a, timeout=0: (0, "version: 4567 (abcdef0)")), "b4567-abcdef0")
        self.assertNotEqual(autotune.detect_build(["x"], run=lambda a, timeout=0: (0, real)),
                            autotune.detect_build(["x"], run=lambda a, timeout=0: (0, real.replace("12207", "12999"))))
        self.assertTrue(autotune.detect_build([str(self.dir / "nope")], run=lambda a, timeout=0: None).startswith("unknown"))


if __name__ == "__main__":
    unittest.main()

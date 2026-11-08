import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from dataclasses import replace
from pathlib import Path
from unittest import mock

from forge import autotune, plan, serve
from forge.plan import Caps, Config

from .helpers import FAKE_SERVER, ROOT, free_port, profile, real_model, tiny_gguf


def tuned_for(prof, model, ngl=14, **over):
    """A tuning result as if `forge tune` had measured the plan's neighbourhood on `prof` (14 layers = what the server holds at 16K)."""
    p = plan.build_plan(prof, model)
    cfg = replace(p.config, ngl=ngl, **over)
    best = autotune.Point(ngl, cfg.fa, cfg.kv, "ok", ppd=190.0, tgd=9.15, ub=cfg.ubatch)
    return autotune.TuneResult(cfg, [best], best, "test", True, 0.0, plan_name="gpu",
                               conditions={"free_vram_mib": prof["gpus"][0]["free_mib"] if prof["gpus"] else None,
                                           "ram_available_mib": prof["ram"]["available_mib"], "bench_ctx": 8832})


class Selection(unittest.TestCase):
    def setUp(self):
        self.m = real_model()

    def select(self, prof, tuned):
        return serve.select_config(prof, self.m, plan.build_all(prof, self.m), tuned)

    def test_untouched_machine_uses_the_tuned_config(self):
        prof = profile()
        sel = self.select(prof, tuned_for(prof, self.m, ngl=16, kv="q8_0", fa=True))
        self.assertEqual((sel.name, sel.source, sel.config.ngl, sel.config.kv), ("gpu", "tuned", 16, "q8_0"))
        sel = self.select(prof, tuned_for(prof, self.m))
        self.assertEqual((sel.source, sel.config.ngl), ("tuned", 14))

    def test_without_a_cache_entry_the_plan_is_used(self):
        sel = self.select(profile(), None)
        self.assertEqual((sel.name, sel.source), ("gpu", "plan"))

    def test_vram_dropped_since_tuning_falls_back_to_fewer_layers(self):
        tuned = tuned_for(profile(), self.m, ngl=16, kv="q8_0", fa=True)   # tuned when 3443 MiB were free
        now = profile(free_vram=2300)                                     # another program took ~1.1 GB
        sel = self.select(now, tuned)
        self.assertEqual(sel.source, "plan")
        self.assertLess(sel.config.ngl, tuned.config.ngl)
        self.assertEqual((sel.config.fa, sel.config.kv), (True, "q8_0"))  # what tuning learned about FA/KV is kept
        self.assertLessEqual(plan.estimate(self.m, sel.config).vram, plan.vram_budget(2300))
        self.assertTrue(any("VRAM" in r for r in sel.reasons))

    def test_ram_dropped_since_tuning_shrinks_the_plan(self):
        tuned = tuned_for(profile(), self.m)                              # f16 KV, 14 layers, 16K context
        now = profile(ram_free=2900)                                      # the VM / other programs took ~2.7 GB
        sel = self.select(now, tuned)
        self.assertEqual(sel.source, "plan")
        self.assertTrue(any("RAM" in r for r in sel.reasons))
        self.assertLessEqual(plan.estimate(self.m, sel.config).ram, plan.ram_budget(2900) + 1e-6)

    def test_battery_selects_low_power(self):
        tuned = tuned_for(profile(), self.m)
        sel = self.select(profile(battery=True), tuned)           # a GPU-plan tuning must not leak into the battery profile
        self.assertEqual((sel.name, sel.source), ("low-power", "plan"))
        self.assertLess(sel.config.threads, 6)
        self.assertLessEqual(sel.config.ctx, 8192)
        self.assertTrue(any("ignored" in r for r in sel.reasons))
        own = replace(tuned, plan_name="low-power", config=replace(tuned.config, threads=3, ctx=8192))
        self.assertEqual(self.select(profile(battery=True), own).source, "tuned")

    def test_gpu_gone_selects_cpu_and_ignores_the_gpu_tuning(self):
        tuned = tuned_for(profile(), self.m)
        sel = self.select(profile(gpu=False), tuned)
        self.assertEqual((sel.name, sel.config.ngl), ("cpu", 0))

    def test_fits_now(self):
        tuned = tuned_for(profile(), self.m)
        cfg = tuned.config
        self.assertTrue(serve.fits_now(self.m, cfg, profile())[0])
        ok, why = serve.fits_now(self.m, cfg, profile(free_vram=3105))     # the 338 MiB drop seen in the fixtures
        self.assertFalse(ok)
        self.assertIn("VRAM", why)
        self.assertIn("compute", why)
        self.assertFalse(serve.fits_now(self.m, cfg, profile(gpu=False))[0])
        self.assertFalse(serve.fits_now(self.m, cfg, profile(free_vram=2000))[0])
        # exactly at the budget edge, from the planner's own estimate: one MiB either side
        need = plan.estimate(self.m, cfg).vram
        edge = (need + plan.VRAM_MARGIN_MIB) / (1 - plan.VRAM_MARGIN_FRACTION)
        self.assertTrue(serve.fits_now(self.m, cfg, profile(free_vram=int(edge) + 2))[0])
        self.assertFalse(serve.fits_now(self.m, cfg, profile(free_vram=int(edge) - 2))[0])
        self.assertTrue(serve.fits_now(self.m, replace(cfg, ngl=0), profile(gpu=False, ram_free=12000))[0])      # CPU-only configs do not look at VRAM

    def test_tuned_layers_from_llama_bench_are_rechecked_against_the_server(self):
        """The first host tune chose 15 layers, measured by llama-bench at 8832 tokens of context; llama-server allocates 673 MiB of compute
        buffer at 16K and cannot load 15. The tuned result is an upper bound, not a promise (H-SERVER-RECHECK)."""
        import json
        raw = json.loads((ROOT / "results" / "forge-tune-host-20261108.json").read_text())
        entry = next(v for k, v in raw.items() if not k.startswith("_"))
        tuned = autotune.TuneResult.from_dict(entry["result"])                 # the real result from the host
        self.assertEqual((tuned.config.ngl, tuned.config.ctx, tuned.config.ubatch), (15, 16384, 512))
        prof = profile()                                                        # the host: 3443 MiB free
        sel = self.select(prof, tuned)
        self.assertEqual((sel.source, sel.config.ngl), ("plan", 14))
        self.assertEqual((sel.config.fa, sel.config.kv, sel.config.ubatch, sel.config.ctx), (False, "f16", 512, 16384))
        self.assertTrue(any("tuned configuration skipped" in r and "compute" in r for r in sel.reasons), sel.reasons)
        ok, why = serve.fits_now(self.m, tuned.config, prof)
        self.assertFalse(ok)
        # the same 15 layers are fine at the context llama-bench used: the check is about the SERVER's context
        self.assertTrue(serve.fits_now(self.m, replace(tuned.config, ctx=8832), prof)[0])

    def test_a_ubatch_outside_the_modelled_range_is_rejected_not_a_crash(self):
        prof = profile()
        odd = tuned_for(prof, self.m, ubatch=2048)                      # nothing in forge writes this; a hand-edited cache could
        ok, why = serve.fits_now(self.m, odd.config, prof)
        self.assertFalse(ok)
        self.assertIn("ubatch", why)
        sel = self.select(prof, odd)
        self.assertEqual((sel.source, sel.config.ubatch), ("plan", 512))

    def test_tuned_ubatch_survives_the_recheck_and_reaches_the_command_line(self):
        prof = profile()
        fits = self.select(prof, tuned_for(prof, self.m, ngl=16, ubatch=256))              # ubatch 256 holds 16 layers at 16K
        self.assertEqual((fits.source, fits.config.ngl, fits.config.ubatch), ("tuned", 16, 256))
        too_many = self.select(prof, tuned_for(prof, self.m, ngl=18, ubatch=256))          # 18 loads in llama-bench at 8.7K, not in the server at 16K
        self.assertEqual((too_many.source, too_many.config.ngl, too_many.config.ubatch), ("plan", 16, 256))
        for sel in (fits, too_many):
            cmd = plan.server_command("llama-server", "m.gguf", sel.config)
            self.assertEqual(cmd[cmd.index("-ub") + 1], "256")

    def test_speculation_choice_is_made_at_serve_time_not_by_the_tuning(self):
        prof = profile()
        plans = plan.build_all(prof, self.m, spec="ngram-mod")
        kept = serve.select_config(prof, self.m, plans, tuned_for(prof, self.m))
        self.assertEqual((kept.source, kept.config.spec_type), ("tuned", "ngram-mod"))
        replanned = serve.select_config(prof, self.m, plans, tuned_for(prof, self.m, ngl=15))
        self.assertEqual((replanned.source, replanned.config.spec_type, replanned.config.ngl), ("plan", "ngram-mod", 14))
        off = serve.select_config(prof, self.m, plan.build_all(prof, self.m), tuned_for(prof, self.m))
        self.assertIsNone(off.config.spec_type)

    def test_fallback_ladder_steps_down_one_layer_at_a_time_then_by_four_then_context(self):
        ladder = serve.fallback_ladder(Config(99, 16384, 6, 6, False), self.m)
        ngls = [c.ngl for c in ladder]
        # 99 = everything + output layer (= 34 here); first rungs are one layer apart, then steps of 4, then the CPU with smaller contexts
        self.assertEqual(ngls[:8], [99, 33, 32, 31, 30, 29, 28, 24])
        self.assertEqual(ngls[8:14], [20, 16, 12, 8, 4, 0])
        self.assertEqual([c.ctx for c in ladder[:14]], [16384] * 14)
        self.assertEqual([(c.ngl, c.ctx) for c in ladder[14:]], [(0, 8192), (0, 4096)])
        small = [c.ngl for c in serve.fallback_ladder(Config(15, 16384, 6, 6, False), self.m)]
        self.assertEqual(small[:8], [15, 14, 13, 12, 11, 10, 9, 5])                # 15 -> 14 first, not 15 -> 11
        self.assertEqual(serve.fallback_ladder(Config(1, 16384, 6, 6, False), self.m)[1].ngl, 0)
        self.assertEqual([c.ngl for c in serve.fallback_ladder(Config(0, 8192, 6, 6, False), self.m)], [0, 0])   # CPU-only: context only
        self.assertEqual(ladder[0], Config(99, 16384, 6, 6, False))

    def test_fallback_ladder_reaches_the_smaller_contexts_within_the_attempt_limit(self):
        for start in (99, 34, 33, 20, 15, 14, 3, 1, 0):
            ladder = serve.fallback_ladder(Config(start, 16384, 6, 6, False), self.m)
            self.assertEqual(ladder[-1].ctx, plan.CTX_MIN, start)
            self.assertLessEqual(len(ladder), serve.MAX_ATTEMPTS, start)
            self.assertTrue(all(c.ngl == 0 for c in ladder if c.ctx < 16384))
            self.assertEqual(len({(c.ngl, c.ctx) for c in ladder}), len(ladder), "no rung repeated")
            ngls = [c.ngl for c in ladder if c.ctx == 16384]
            self.assertEqual(ngls, sorted(ngls, reverse=True))

    def test_cpu_tuned_result_is_used_on_a_machine_without_gpu(self):
        prof = profile(gpu=False, ram_free=12000)
        p = plan.build_plan(prof, self.m, mode="cpu")
        cfg = replace(p.config, fa=True, kv="q8_0")
        best = autotune.Point(0, True, "q8_0", "ok", ppd=150.0, tgd=5.0)
        tuned = autotune.TuneResult(cfg, [best], best, "t", True, 0.0, plan_name="cpu", conditions={"free_vram_mib": None, "ram_available_mib": 12000})
        sel = self.select(prof, tuned)
        self.assertEqual((sel.name, sel.source, sel.config.kv), ("cpu", "tuned", "q8_0"))

    def test_context_reduced_by_old_memory_pressure_is_restored_when_memory_is_back(self):
        prof = profile()
        tuned = tuned_for(prof, self.m)
        tuned.config = replace(tuned.config, ctx=8192)                      # tuned while RAM was short
        sel = self.select(prof, tuned)
        self.assertEqual((sel.source, sel.config.ctx), ("tuned", 16384))
        self.assertEqual(sel.config.ngl, tuned.config.ngl)

    def test_connect_host(self):
        self.assertEqual(serve.connect_host("0.0.0.0"), "127.0.0.1")
        self.assertEqual(serve.connect_host("::"), "::1")
        self.assertEqual(serve.connect_host("192.168.1.5"), "192.168.1.5")


class Launch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / "server.log"
        self.env = mock.patch.dict(os.environ, {"FAKE_SERVER_LOG": str(self.log), "FAKE_SERVER_LOAD_S": "0.6"})
        self.env.start()
        self.m = replace(real_model(), path="model.gguf")
        self.port = free_port()
        self.started = []

    def tearDown(self):
        for s in self.started:
            s.stop()
        self.env.stop()
        self.tmp.cleanup()

    def start(self, cfg, **kw):
        srv = serve.start_server(FAKE_SERVER, self.m, cfg, host="127.0.0.1", port=self.port, log_dir=self.dir / "logs",
                                 caps=Caps(), health_timeout=kw.pop("health_timeout", 20), **kw)
        self.started.append(srv)
        return srv

    def test_starts_waits_for_health_and_exposes_the_endpoint(self):
        srv = self.start(Config(16, 16384, 6, 6, False))
        self.assertEqual(srv.endpoint, f"http://127.0.0.1:{self.port}/v1")
        with urllib.request.urlopen(srv.endpoint + "/models", timeout=5) as r:
            self.assertEqual(r.status, 200)
        line = self.log.read_text().splitlines()[0]
        for frag in ("-ngl 16", "-c 16384", "-t 6", "-fa off", "--fit off", f"--port {self.port}"):
            self.assertIn(frag, line)
        self.assertEqual([a["ok"] for a in srv.attempts], [True])

    def test_out_of_memory_at_load_falls_back_one_layer_at_a_time(self):
        with mock.patch.dict(os.environ, {"FAKE_SERVER_OOM_NGL": "10"}):
            srv = self.start(Config(16, 16384, 6, 6, False))
        self.assertEqual([a["ngl"] for a in srv.attempts], [16, 15, 14, 13, 12, 11, 10])     # every count, none skipped
        self.assertEqual(srv.config.ngl, 10)
        self.assertEqual([a["ok"] for a in srv.attempts], [False] * 6 + [True])
        self.assertEqual({a["ctx"] for a in srv.attempts}, {16384})                           # layers first, context only after the GPU is exhausted
        self.assertIn("allocate", Path(srv.attempts[0]["log"]).read_text())

    def test_one_layer_over_the_limit_costs_one_layer_not_four(self):
        """The host case: 15 layers fail at 16K, 14 load. The old ladder went 15 -> 11 and served at 8.0 instead of 9.0 tok/s."""
        with mock.patch.dict(os.environ, {"FAKE_SERVER_OOM_NGL": "14"}):
            srv = self.start(Config(15, 16384, 6, 6, False))
        self.assertEqual((srv.config.ngl, [a["ngl"] for a in srv.attempts]), (14, [15, 14]))
        self.assertEqual(self.log.read_text().count("-ngl 14"), 1)

    def test_ladder_continues_with_bigger_steps_and_then_context(self):
        with mock.patch.dict(os.environ, {"FAKE_SERVER_OOM_NGL": "0"}):              # nothing with a GPU layer loads
            srv = self.start(Config(12, 16384, 6, 6, False))
        self.assertEqual([a["ngl"] for a in srv.attempts], [12, 11, 10, 9, 8, 7, 6, 2, 0])
        self.assertEqual((srv.config.ngl, srv.config.ctx), (0, 16384))

    def test_gives_up_with_a_clear_error_when_nothing_starts(self):
        with mock.patch.dict(os.environ, {"FAKE_SERVER_OOM_NGL": "-1"}):
            with self.assertRaises(RuntimeError) as cm:
                self.start(Config(8, 8192, 6, 6, False), max_attempts=3)
        self.assertIn("did not start", str(cm.exception))

    def test_refuses_to_start_when_something_already_answers_on_the_port(self):
        first = self.start(Config(8, 8192, 6, 6, False))                  # a server is up on self.port
        with self.assertRaises(RuntimeError) as cm:
            self.start(Config(8, 8192, 6, 6, False))                      # the second must not be declared healthy by the first's /health
        self.assertIn("already in use", str(cm.exception))
        self.assertIsNone(first.proc.poll())

    def test_unhealthy_server_is_killed_after_the_timeout(self):
        with mock.patch.dict(os.environ, {"FAKE_SERVER_NEVER": "1"}):
            with self.assertRaises(RuntimeError):
                self.start(Config(0, 4096, 2, 2, False), health_timeout=1.5, max_attempts=1)


class EndToEnd(unittest.TestCase):
    """serve() with a real (synthetic) GGUF, a saved profile and the fake server: probe -> plan -> select -> launch."""

    def test_serve_uses_cache_then_falls_back_when_conditions_change(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model_path = str(tiny_gguf(d / "tiny.gguf", layers=8, layer_kib=256))
            log = d / "server.log"
            env = {"FAKE_SERVER_LOG": str(log), "FAKE_SERVER_LOAD_S": "0.3"}
            prof = profile(free_vram=3443, ram_free=12000)
            m = plan.load_model(model_path)
            plans = plan.build_all(prof, m)
            started = []
            try:
                with mock.patch.dict(os.environ, env):
                    # 1. no cache entry: the plan
                    srv, sel = serve.serve(model_path, host="127.0.0.1", port=free_port(), cache_dir=str(d / "c"), server_prefix=FAKE_SERVER,
                                           profile=prof, health_timeout=20, progress=lambda s: None)
                    started.append(srv)
                    self.assertEqual((sel.name, sel.source), ("gpu", "plan"))
                    srv.stop()
                    # 2. battery: low-power plan, fewer threads
                    srv, sel = serve.serve(model_path, host="127.0.0.1", port=free_port(), cache_dir=str(d / "c"), server_prefix=FAKE_SERVER,
                                           profile=profile(free_vram=3443, ram_free=12000, battery=True), health_timeout=20, progress=lambda s: None)
                    started.append(srv)
                    self.assertEqual(sel.name, "low-power")
                    self.assertIn("-t 3", log.read_text().splitlines()[-1])
                    srv.stop()
                    # 3. speculative decoding: off unless asked for
                    self.assertIn("--spec-type none", log.read_text().splitlines()[0])
                    srv, sel = serve.serve(model_path, host="127.0.0.1", port=free_port(), cache_dir=str(d / "c"), server_prefix=FAKE_SERVER,
                                           profile=prof, spec="ngram-mod", health_timeout=20, progress=lambda s: None)
                    started.append(srv)
                    self.assertEqual(sel.config.spec_type, "ngram-mod")
                    self.assertIn("--spec-type ngram-mod", log.read_text().splitlines()[-1])
            finally:
                for s in started:
                    s.stop()


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "forge", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_plan_json_and_heuristics(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = str(tiny_gguf(Path(d) / "tiny.gguf"))
            prof = Path(d) / "profile.json"
            prof.write_text(json.dumps(profile()))
            r = self.run_cli("plan", "--model", model_path, "--profile", str(prof), "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertEqual(set(out["plans"]), {"gpu", "cpu", "low-power"})
            self.assertEqual(out["applies"], "gpu")
            r = self.run_cli("plan", "--model", model_path, "--profile", str(prof), "--explain")
            self.assertIn("H-LAYERS-VRAM-FIT", r.stdout)
        r = self.run_cli("heuristics")
        self.assertEqual(r.returncode, 0)
        self.assertIn("H-FA-BACKEND", r.stdout)

    def test_spec_and_ubatch_flags(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = str(tiny_gguf(Path(d) / "tiny.gguf"))
            prof = Path(d) / "profile.json"
            prof.write_text(json.dumps(profile()))
            base = ["plan", "--model", model_path, "--profile", str(prof), "--json"]
            default = json.loads(self.run_cli(*base).stdout)["plans"]["gpu"]["config"]
            self.assertEqual((default["spec_type"], default["ubatch"]), (None, 512))
            r = self.run_cli(*base, "--spec", "ngram-mod", "--ubatch", "256")
            self.assertEqual(r.returncode, 0, r.stderr)
            cfg = json.loads(r.stdout)["plans"]["gpu"]["config"]
            self.assertEqual((cfg["spec_type"], cfg["ubatch"]), ("ngram-mod", 256))
            bad = self.run_cli(*base, "--spec", "ngram-cache")                 # measured worse: not an option
            self.assertNotEqual(bad.returncode, 0)
            helptxt = self.run_cli("serve", "--help").stdout
            self.assertIn("--spec", helptxt)
            self.assertIn("ngram-mod", helptxt)

    def test_missing_model_is_a_clean_error(self):
        r = self.run_cli("plan", "--model", "/no/such.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()

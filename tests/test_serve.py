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


def tuned_for(prof, model, **over):
    """A tuning result as if `forge tune` had measured the plan's neighbourhood on `prof`."""
    p = plan.build_plan(prof, model)
    cfg = replace(p.config, ngl=16, **over)
    best = autotune.Point(16, cfg.fa, cfg.kv, "ok", ppd=190.0, tgd=9.15)
    return autotune.TuneResult(cfg, [best], best, "test", True, 0.0, plan_name="gpu",
                               conditions={"free_vram_mib": prof["gpus"][0]["free_mib"] if prof["gpus"] else None,
                                           "ram_available_mib": prof["ram"]["available_mib"]})


class Selection(unittest.TestCase):
    def setUp(self):
        self.m = real_model()

    def select(self, prof, tuned):
        return serve.select_config(prof, self.m, plan.build_all(prof, self.m), tuned)

    def test_untouched_machine_uses_the_tuned_config(self):
        prof = profile()
        sel = self.select(prof, tuned_for(prof, self.m, kv="q8_0", fa=True))
        self.assertEqual((sel.name, sel.source, sel.config.ngl, sel.config.kv), ("gpu", "tuned", 16, "q8_0"))

    def test_without_a_cache_entry_the_plan_is_used(self):
        sel = self.select(profile(), None)
        self.assertEqual((sel.name, sel.source), ("gpu", "plan"))

    def test_vram_dropped_since_tuning_falls_back_to_fewer_layers(self):
        tuned = tuned_for(profile(), self.m, kv="q8_0", fa=True)          # tuned when 3443 MiB were free
        now = profile(free_vram=2300)                                     # another program took ~1.1 GB
        sel = self.select(now, tuned)
        self.assertEqual(sel.source, "plan")
        self.assertLess(sel.config.ngl, tuned.config.ngl)
        self.assertEqual((sel.config.fa, sel.config.kv), (True, "q8_0"))  # what tuning learned about FA/KV is kept
        self.assertLessEqual(plan.estimate(self.m, sel.config).vram, plan.vram_budget(2300))
        self.assertTrue(any("VRAM" in r for r in sel.reasons))

    def test_ram_dropped_since_tuning_shrinks_the_plan(self):
        tuned = tuned_for(profile(), self.m)                              # f16 KV, 16 layers, 16K context
        now = profile(ram_free=3500)                                      # the VM / other programs took ~2 GB
        sel = self.select(now, tuned)
        self.assertEqual(sel.source, "plan")
        self.assertTrue(any("RAM" in r for r in sel.reasons))
        self.assertLessEqual(plan.estimate(self.m, sel.config).ram, plan.ram_budget(3500) + 1e-6)

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
        cfg, cond = tuned.config, tuned.conditions
        self.assertTrue(serve.fits_now(self.m, cfg, profile(), cond)[0])
        self.assertTrue(serve.fits_now(self.m, cfg, profile(free_vram=3200), cond)[0])     # small drift is tolerated
        ok, why = serve.fits_now(self.m, cfg, profile(free_vram=3105), cond)               # the 338 MiB drop seen in the fixtures is not
        self.assertFalse(ok)
        self.assertIn("VRAM", why)
        self.assertFalse(serve.fits_now(self.m, cfg, profile(gpu=False), cond)[0])
        self.assertFalse(serve.fits_now(self.m, cfg, profile(free_vram=2000))[0])         # no record -> the estimator decides

    def test_fallback_ladder_goes_down_in_layers_then_context(self):
        ladder = serve.fallback_ladder(Config(99, 16384, 6, 6, False), self.m)
        ngls = [c.ngl for c in ladder]
        self.assertEqual(ngls[:10], [99, 32, 28, 24, 20, 16, 12, 8, 4, 0])      # 99 = all + output layer; 32 = blocks only
        self.assertEqual(ladder[-1].ctx, plan.CTX_MIN)
        self.assertTrue(all(c.ngl == 0 for c in ladder if c.ctx < 16384))
        self.assertLessEqual(len(ladder), serve.MAX_ATTEMPTS)                   # the smaller contexts must be reachable

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

    def test_fits_now_accounts_for_serving_at_a_bigger_context_than_llama_bench_used(self):
        tuned = tuned_for(profile(), self.m)
        cond = {**tuned.conditions, "bench_ctx": 8832}                       # llama-bench ran at depth 8192 + 640
        delta = plan.estimate(self.m, tuned.config).vram - plan.estimate(self.m, replace(tuned.config, ctx=8832)).vram
        self.assertGreater(delta, 100)
        floor = 3443 - (plan.VRAM_MARGIN_MIB + plan.VRAM_MARGIN_FRACTION * 3443) + delta
        self.assertTrue(serve.fits_now(self.m, tuned.config, profile(free_vram=int(floor) + 2), cond)[0])
        self.assertFalse(serve.fits_now(self.m, tuned.config, profile(free_vram=int(floor) - 2), cond)[0])
        self.assertTrue(serve.fits_now(self.m, tuned.config, profile(free_vram=int(floor) - 2), tuned.conditions)[0])   # no record of bench ctx: not applied

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

    def test_out_of_memory_at_load_falls_back_instead_of_failing(self):
        with mock.patch.dict(os.environ, {"FAKE_SERVER_OOM_NGL": "10"}):
            srv = self.start(Config(16, 16384, 6, 6, False))
        self.assertEqual(srv.config.ngl, 8)                             # 16 -> 12 -> 8
        self.assertEqual([a["ok"] for a in srv.attempts], [False, False, True])
        self.assertIn("allocate", Path(srv.attempts[0]["log"]).read_text())

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

    def test_missing_model_is_a_clean_error(self):
        r = self.run_cli("plan", "--model", "/no/such.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()

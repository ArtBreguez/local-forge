import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from forge import gguf, plan
from forge.plan import Caps, Config

from .helpers import FIX, ROOT, profile, real_model, tiny_gguf


class ModelArithmetic(unittest.TestCase):
    def test_real_model_kv_is_attention_layers_only(self):
        m = real_model()
        self.assertEqual((m.arch, m.quant, m.trunk_layers, m.block_count), ("qwen35", "IQ4_XS", 32, 33))
        self.assertEqual(m.attn_layers, [3, 7, 11, 15, 19, 23, 27, 31])
        # 8 layers x 4 kv heads x (256 + 256) x 2 bytes = 32 KiB per token, not 128 KiB
        self.assertEqual(m.kv_bytes_per_token("f16"), 32 * 1024)
        self.assertAlmostEqual(m.kv_bytes_per_token("q8_0") / m.kv_bytes_per_token("f16"), 34 / 32 / 2)

    def test_real_model_weights_add_up_to_the_file(self):
        m = real_model()
        total = sum(m.layer_bytes) + m.output_bytes + m.embd_bytes + m.other_bytes
        header = int(next(l for l in (FIX / "gguf-dump-dsv4-9b.txt").read_text().splitlines() if l.startswith("parse_s")).split("data_offset ")[1].split()[0])
        self.assertAlmostEqual((total + header) / 2**20, m.file_mib, delta=0.01)   # tensors + header (metadata, tokenizer) = file

    def test_mtp_declared_but_tensors_missing_in_the_real_file(self):
        st = plan.mtp_status(real_model())
        self.assertTrue(st["declared"])
        self.assertFalse(st["tensors_present"])
        self.assertFalse(st["available"])

    def test_mtp_available_only_with_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            with_t = plan.load_model(str(tiny_gguf(Path(d) / "a.gguf", nextn=1, mtp_tensors=True)))
            without = plan.load_model(str(tiny_gguf(Path(d) / "b.gguf", nextn=1, mtp_tensors=False)))
            none = plan.load_model(str(tiny_gguf(Path(d) / "c.gguf", nextn=0)))
        self.assertTrue(plan.mtp_status(with_t)["available"])
        self.assertFalse(plan.mtp_status(without)["available"])
        self.assertFalse(plan.mtp_status(none)["declared"])
        self.assertEqual(with_t.trunk_layers, 8)                      # block_count 9 - 1 nextn
        p = plan.build_plan(profile(), with_t, enable_mtp=True)
        self.assertEqual(p.config.spec_type, "draft-mtp")
        self.assertIn("--spec-type", plan.server_command("llama-server", with_t.path, p.config))
        p2 = plan.build_plan(profile(), without, enable_mtp=True)      # asked for, but not possible
        self.assertIsNone(p2.config.spec_type)

    def test_tied_embeddings_charge_an_output_matrix_to_the_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            tied = plan.load_model(str(tiny_gguf(Path(d) / "t.gguf", tied=True)))
            untied = plan.load_model(str(tiny_gguf(Path(d) / "u.gguf", tied=False)))
        self.assertEqual(tied.output_bytes, tied.embd_bytes + 256)      # output_norm + a token_embd-sized copy
        self.assertEqual(untied.output_bytes, 16384 + 256)
        self.assertTrue(any("tied" in w for w in tied.warnings))

    def test_all_layers_sentinel_covers_very_deep_models(self):
        self.assertEqual(plan.all_layers(real_model()), 99)
        deep = replace(real_model(), trunk_layers=126)
        self.assertEqual(plan.all_layers(deep), 127)                     # -ngl 99 would be a partial offload of a 126-block model

    def test_summary_says_all_for_the_sentinel(self):
        p = plan.build_plan(profile(free_vram=24000, vram_total=24576), real_model())
        self.assertIn("ngl=all", p.summary())

    def test_layer_kinds_from_synthetic_headers(self):
        with tempfile.TemporaryDirectory() as d:
            hyb = plan.load_model(str(tiny_gguf(Path(d) / "h.gguf", layers=8, hybrid=True)))
            dense = plan.load_model(str(tiny_gguf(Path(d) / "d.gguf", layers=8, hybrid=False)))
        self.assertEqual(hyb.attn_layers, [3, 7])
        self.assertEqual(dense.attn_layers, list(range(8)))
        self.assertGreater(hyb.state_bytes_per_recurrent_layer, 0)
        self.assertEqual(dense.state_bytes_per_recurrent_layer, 0)
        # dense: 8 layers x 2 kv heads x 32 x 2 B = 1 KiB per token
        self.assertEqual(dense.kv_bytes_per_token("f16"), 8 * 2 * 32 * 2)


class IndependentOracle(unittest.TestCase):
    """Expected VRAM computed here from the raw fixture numbers, not with the planner's own helpers,
    so a bug in estimate()/max_layers() cannot cancel itself out."""

    def expected_vram_mib(self, ngl, ctx, kv_bytes_per_elem, scratch=plan.COMPUTE_BASE_MIB):
        d = json.loads((FIX / "modelinfo-dsv4-9b.json").read_text())
        n = d["trunk_layers"]
        on_gpu = range(n - min(ngl, n), n)
        weights = sum(d["layer_bytes"][i] for i in on_gpu) + (d["output_bytes"] if ngl > n else 0)
        kv = sum(d["kv_elems_per_token"][i] for i in on_gpu) * kv_bytes_per_elem * ctx
        state = sum(d["state_bytes_per_recurrent_layer"] for i in on_gpu if d["kv_elems_per_token"][i] == 0)
        return (weights + kv + state) / 2**20 + (scratch if ngl else 0)

    def test_estimate_matches_hand_arithmetic(self):
        m = real_model()
        for ngl, ctx, kv, bpe in ((16, 16384, "f16", 2.0), (16, 32768, "f16", 2.0), (16, 16384, "q8_0", 34 / 32),
                                  (8, 8192, "f16", 2.0), (32, 16384, "f16", 2.0), (33, 16384, "f16", 2.0), (0, 16384, "f16", 2.0),
                                  (15, 16384, "f16", 2.0), (13, 8192, "q8_0", 34 / 32), (18, 16384, "f16", 2.0), (21, 32768, "f16", 2.0)):
            # (the non-multiples of 4 matter: this model repeats every 4 layers, so the first n and the last n
            #  blocks weigh the same when n is a multiple of 4 and a wrong end would go unnoticed)
            got = plan.estimate(m, Config(ngl, ctx, 6, 6, kv != "f16", kv)).vram
            self.assertAlmostEqual(got, self.expected_vram_mib(ngl, ctx, bpe), places=3, msg=(ngl, ctx, kv))

    def test_kv_doubles_with_context_and_shrinks_with_q8(self):
        m = real_model()
        k16 = plan.estimate(m, Config(32, 16384, 6, 6, False)).vram_kv
        k32 = plan.estimate(m, Config(32, 32768, 6, 6, False)).vram_kv
        kq8 = plan.estimate(m, Config(32, 16384, 6, 6, True, "q8_0")).vram_kv
        self.assertAlmostEqual(k16, 512.0)                  # 8 layers x 4096 B/token x 16384 / 2^20
        self.assertAlmostEqual(k32, 2 * k16)
        self.assertAlmostEqual(kq8, k16 * 34 / 32 / 2)

    def test_scratch_reserve_sits_inside_the_window_the_sweeps_leave_open(self):
        """16 layers load -> scratch <= free - need16; 17 fail -> scratch > free - need17 (non-scratch need by hand arithmetic)."""
        need = lambda n: self.expected_vram_mib(n, 8704, 2.0, scratch=0)
        lo, hi = 3443 - need(17), 3443 - need(16)
        reserve = plan.COMPUTE_BASE_MIB + plan.VRAM_MARGIN_MIB + plan.VRAM_MARGIN_FRACTION * 3443
        self.assertLess(lo, reserve)
        self.assertLessEqual(reserve, hi)

    def test_chosen_layers_never_exceed_free_vram_even_ignoring_the_margin(self):
        """Physical bound, independent of the margin constants: the true need must be below what is free."""
        m = real_model()
        for free in range(1000, 8000, 250):
            p = plan.build_plan(profile(free_vram=free, vram_total=free + 600), m)
            need = self.expected_vram_mib(min(p.config.ngl, 33), p.config.ctx, plan.KV_BYTES_PER_ELEM[p.config.kv], plan.COMPUTE_BASE_MIB)
            self.assertLess(need, free, f"free={free}: plan {p.summary()} needs {need:.0f} MiB by hand arithmetic")

    def test_plan_for_the_idle_card_survives_the_drift_seen_in_the_fixtures(self):
        """H-VRAM-HEADROOM: free VRAM was 3443 MiB idle and 3105 MiB with a benchmark running. A plan made on the idle
        card must still fit the busy one (by hand arithmetic, not by the planner's own budget function)."""
        from forge import probe
        idle = probe.parse_list_devices((FIX / "list-devices-rx6500xt-idle.txt").read_text())[1][0]["free_mib"]
        busy = probe.parse_list_devices((FIX / "list-devices-rx6500xt-busy.txt").read_text())[1][0]["free_mib"]
        self.assertEqual((idle, busy), (3443, 3105))
        for kv in (None, "q8_0"):
            p = plan.build_plan(profile(free_vram=idle), real_model(), kv=kv)
            bpe = plan.KV_BYTES_PER_ELEM[p.config.kv]
            need = self.expected_vram_mib(min(p.config.ngl, 33), p.config.ctx, bpe)
            self.assertLessEqual(need, busy, f"{p.summary()} needs {need:.0f} MiB but the card had {busy} free when busy")

    def test_this_machine_never_gets_layers_that_failed_to_load_in_llama_cpp(self):
        """Measured: ngl=20 failed at ~8.7K context on 3443 MiB free (all settings). The plan must stay below it at any context >= that."""
        m = real_model()
        for ctx in (8192, 16384, 32768):
            for kv in (None, "q8_0"):
                self.assertLess(plan.build_plan(profile(), m, ctx=ctx, kv=kv).config.ngl, 20, (ctx, kv))


class VramFit(unittest.TestCase):
    """The rule that must never be wrong: do not plan more layers than the card holds."""

    def test_four_gb_card_gets_no_more_layers_than_fit(self):
        m = real_model()
        for free in (3443, 3105, 2000):
            p = plan.build_plan(profile(free_vram=free), m)
            budget = plan.vram_budget(free)
            self.assertLessEqual(plan.estimate(m, p.config).vram, budget, f"free={free}")
            bigger = replace(p.config, ngl=p.config.ngl + 1)
            self.assertGreater(plan.estimate(m, bigger).vram, budget, f"one more layer should not fit, free={free}")

    def test_never_exceeds_free_vram_for_any_card_size(self):
        m = real_model()
        last = -1
        for free in range(0, 26000, 500):
            p = plan.build_plan(profile(free_vram=free, vram_total=max(free, 1)), m)
            self.assertLessEqual(p.config.ngl, 99)
            if p.config.ngl > 0:
                self.assertLessEqual(p.estimate["vram"], plan.vram_budget(free))
            n = p.config.ngl
            self.assertGreaterEqual(min(n, m.trunk_layers + 1), min(last, m.trunk_layers + 1))  # more VRAM never means fewer layers
            last = n

    def test_everything_offloads_on_a_big_card_and_nothing_on_a_tiny_one(self):
        m = real_model()
        self.assertEqual(plan.build_plan(profile(free_vram=24000, vram_total=24576), m).config.ngl, 99)
        p = plan.build_plan(profile(free_vram=900), m)
        self.assertEqual(p.config.ngl, 0)
        self.assertTrue(any("CPU-only" in w for w in p.warnings))

    def test_this_machine_lands_near_the_measured_best(self):
        """JOURNAL.md: best measured is ~16 layers at 16K (sweep A: 16; sweep B with q8_0 KV: 16)."""
        m = real_model()
        default = plan.build_plan(profile(), m)
        self.assertEqual(default.config.ctx, 16384)
        self.assertEqual(default.config.threads, 6)
        self.assertLessEqual(abs(default.config.ngl - 16), 2, default.summary())
        q8 = plan.build_plan(profile(), m, kv="q8_0")
        self.assertEqual((q8.config.fa, q8.config.kv), (True, "q8_0"))
        self.assertLessEqual(abs(q8.config.ngl - 16), 2, q8.summary())

    def test_matches_the_llama_cpp_load_boundary_measured_in_sweeps_c_and_d(self):
        """llama-bench at ~8.7K context, 3443 MiB free. Sweep C: ngl=16 loaded and ngl=20 failed for FA off/on and f16/q8_0.
        Sweep D (ubatch 512, FA off, f16): 16 ran, 17 failed at depth, 18-20 failed. The plan must put the line between 16 and 17."""
        m = real_model()
        budget = plan.vram_budget(3443)
        for fa, kv in ((False, "f16"), (True, "f16"), (True, "q8_0")):
            self.assertLessEqual(plan.estimate(m, Config(16, 8704, 6, 6, fa, kv)).vram, budget, (fa, kv))
            self.assertGreater(plan.estimate(m, Config(20, 8704, 6, 6, fa, kv)).vram, budget, (fa, kv))
        self.assertGreater(plan.estimate(m, Config(17, 8704, 6, 6, False, "f16")).vram, budget)
        log = (FIX / "llama-bench-sweepd-snapshot.log").read_text()      # and the sweeps' logs really say that
        self.assertIn("ngl=16 fa=off kv=f16 ub=512 rc=0", log)
        self.assertIn("ngl=17 fa=off kv=f16 ub=512 rc=1", log)

    def test_longer_context_costs_layers(self):
        m = real_model()
        a = plan.build_plan(profile(), m, ctx=16384, kv="f16", fa=False)
        b = plan.build_plan(profile(), m, ctx=32768, kv="f16", fa=False)
        self.assertLessEqual(b.config.ngl, a.config.ngl)

    def test_flash_attention_off_on_vulkan_and_cpu_on_for_cuda(self):
        m = real_model()
        self.assertFalse(plan.build_plan(profile(), m).config.fa)
        self.assertFalse(plan.build_plan(profile(gpu=False), m).config.fa)
        cuda = plan.build_plan(profile(backend="CUDA", free_vram=11000, vram_total=12288, gpu_name="RTX 3060"), m)
        self.assertTrue(cuda.config.fa)

    def test_q8_kv_requires_flash_attention(self):
        with self.assertRaises(ValueError):
            plan.build_plan(profile(), real_model(), kv="q8_0", fa=False)

    def test_os_only_gpu_with_unknown_free_vram_is_handled(self):
        p = profile()
        p["gpus"][0]["free_mib"] = None
        out = plan.build_plan(p, real_model())
        self.assertGreater(out.config.ngl, 0)
        self.assertTrue(any(d.rule == "H-UNKNOWN-FREE-VRAM" for d in out.decisions))
        p["gpus"][0]["backend"] = None            # OS saw a GPU but nothing can drive it
        self.assertEqual(plan.build_plan(p, real_model()).config.ngl, 0)


class OtherPlans(unittest.TestCase):
    def test_cpu_plan(self):
        p = plan.build_plan(profile(ram_free=12000), real_model(), mode="cpu")
        self.assertEqual((p.config.ngl, p.config.threads, p.config.fa, p.config.kv), (0, 6, False, "f16"))
        self.assertTrue(p.feasible)
        self.assertEqual(p.estimate["vram"], 0)

    def test_cpu_plan_shrinks_context_then_reports_infeasible(self):
        m = real_model()
        tight = plan.build_plan(profile(ram_free=8000), m, mode="cpu", ctx=131072)
        self.assertLess(tight.config.ctx, 131072)
        self.assertTrue(tight.feasible)
        starved = plan.build_plan(profile(ram_free=3000), m, mode="cpu")
        self.assertEqual(starved.config.ctx, plan.CTX_MIN)
        self.assertFalse(starved.feasible)
        self.assertTrue(any("RAM" in w for w in starved.warnings))

    def test_the_cpu_only_run_that_worked_in_sweep_c_is_feasible(self):
        """Sweep C ran ngl=0 to completion at ~8.7K context with 5.5 GiB free (fixtures/llama-bench-sweepc-snapshot.log).
        A plan that calls that infeasible, or shrinks its context, contradicts the measurement."""
        log = (FIX / "llama-bench-sweepc-snapshot.log").read_text()
        self.assertIn("free RAM GB before: 5.5", log)
        self.assertIn("ngl=0 fa=off kv=f16 rc=0", log)
        p = plan.build_plan(profile(ram_free=int(5.5 * 1024)), real_model(), mode="cpu", ctx=8704)
        self.assertTrue(p.feasible, p.summary())
        self.assertEqual(p.config.ctx, 8704)
        host_now = plan.build_plan(profile(ram_free=5784), real_model(), mode="cpu")      # the host as probed
        self.assertTrue(host_now.feasible)
        self.assertGreaterEqual(host_now.config.ctx, 8192)

    def test_low_power_plan(self):
        p = plan.build_plan(profile(battery=True), real_model(), mode="low-power")
        self.assertEqual(p.config.threads, 3)                  # half of 6 physical cores
        self.assertLessEqual(p.config.ctx, 8192)
        self.assertTrue(any(d.rule == "H-LOW-POWER" for d in p.decisions))

    def test_which_plan_applies(self):
        m = real_model()
        for prof, expect in ((profile(), "gpu"), (profile(battery=True), "low-power"), (profile(gpu=False), "cpu"),
                             (profile(free_vram=900), "cpu")):
            plans = plan.build_all(prof, m)
            self.assertEqual(plan.choose_plan(prof, plans)[0], expect)

    def test_threads_fall_back_when_cores_unknown(self):
        p = plan.build_plan(profile(cores=None, threads=8), real_model())
        self.assertEqual(p.config.threads, 4)
        p = profile(cores=None, threads=None)
        p["cpu"]["logical_threads"] = None
        self.assertGreaterEqual(plan.build_plan(p, real_model()).config.threads, 1)

    def test_multi_gpu_uses_the_roomier_one_and_pins_it(self):
        p = profile()
        p["gpus"].append({"index": 1, "device": "Vulkan1", "backend": "Vulkan", "name": "big", "total_mib": 12000, "free_mib": 11000})
        out = plan.build_plan(p, real_model())
        self.assertEqual(out.config.device, "Vulkan1")
        self.assertIn("--device", plan.server_command("s", "m", out.config))


class CommandLine(unittest.TestCase):
    def test_modern_flags(self):
        cfg = plan.build_plan(profile(), real_model()).config
        cmd = plan.server_command("llama-server", "m.gguf", cfg, "127.0.0.1", 8080)
        s = " ".join(cmd)
        for frag in ("-m m.gguf", "--port 8080", f"-ngl {cfg.ngl}", f"-c {cfg.ctx}", "-t 6", "-fa off", "-ctk f16", "-ctv f16", "-np 1", "--fit off"):
            self.assertIn(frag, s)
        self.assertNotIn("--spec-type", s)

    def test_old_builds_get_a_bare_fa_switch_and_no_unknown_flags(self):
        cfg = Config(10, 8192, 4, 4, True, "q8_0")
        old = Caps(fa_takes_value=False, fit=False, spec_type=False, threads_batch=False)
        cmd = plan.server_command("s", "m", cfg, caps=old)
        self.assertIn("-fa", cmd)
        self.assertNotIn("on", cmd)
        self.assertNotIn("--fit", cmd)
        self.assertNotIn("-tb", cmd)

    def test_every_flag_we_emit_exists_in_the_recorded_b12207_help(self):
        """The real llama-server was not started for this; at least no flag may be one the b12207 help text has never heard of."""
        help_text = (FIX / "llama-server-help-b12207.txt").read_text(encoding="utf-8")
        cfg = Config(16, 16384, 6, 6, False, "f16", device="Vulkan0", spec_type="draft-mtp")
        cmd = plan.server_command("llama-server", "m.gguf", cfg, "127.0.0.1", 8080)
        flags = [a for a in cmd[1:] if a.startswith("-")]
        self.assertGreater(len(flags), 10)
        for f in flags:
            self.assertRegex(help_text, r"(^|[\s,])" + __import__("re").escape(f) + r"([\s,\[<]|$)", f"{f} not in the b12207 llama-server help")
        bench_help = (FIX / "llama-bench-help-b12207.txt").read_text(encoding="utf-8")
        from forge import autotune
        bench_flags = [a for a in autotune.bench_args(["llama-bench"], "m", cfg, 8192, 1, True)[1:] if a.startswith("-") and not a[1:].isdigit()]
        for f in bench_flags:
            self.assertRegex(bench_help, r"(^|[\s,])" + __import__("re").escape(f) + r"([\s,\[<]|$)", f"{f} not in the b12207 llama-bench help")

    def test_caps_from_real_help_text(self):
        c = Caps.from_help((FIX / "llama-server-help-b12207.txt").read_text(encoding="utf-8"))
        self.assertEqual((c.fa_takes_value, c.fit, c.spec_type, c.threads_batch), (True, True, True, True))
        c = Caps.from_help("-fa, --flash-attn            enable Flash Attention (default: disabled)\n")
        self.assertFalse(c.fa_takes_value)


class HeuristicRegistry(unittest.TestCase):
    """Every rule names its measurement or says it is untested."""

    def test_every_decision_cites_a_registered_heuristic(self):
        m = real_model()
        seen = set()
        for prof in (profile(), profile(gpu=False), profile(battery=True), profile(ram_free=3000)):
            for p in plan.build_all(prof, m).values():
                seen |= {d.rule for d in p.decisions}
        self.assertTrue(seen)
        self.assertLessEqual(seen, set(plan.HEURISTICS))

    def test_registry_has_every_rule_the_docs_promise(self):
        expected = {"H-LAYERS-VRAM-FIT", "H-VRAM-HEADROOM", "H-OUTPUT-LAYER", "H-KV-ATTN-ONLY", "H-FA-BACKEND", "H-KV-Q8-GAIN",
                    "H-THREADS-PHYSICAL", "H-CTX-TARGET", "H-CTX-RAM", "H-MTP", "H-PARALLEL-1", "H-FIT-OFF", "H-UBATCH", "H-CPU-ONLY",
                    "H-LOW-POWER", "H-UNKNOWN-FREE-VRAM"}
        self.assertEqual(set(plan.HEURISTICS), expected)

    def test_evidence_files_exist_and_untested_is_admitted(self):
        for h in plan.HEURISTICS.values():
            self.assertIn(h.status, ("measured", "partially measured", "derived", "untested"), h.id)
            self.assertTrue(h.evidence, h.id)
            for path, note in h.evidence:
                self.assertTrue(note, h.id)
                if path:
                    self.assertTrue((ROOT / path).exists(), f"{h.id}: {path} does not exist")
            if h.status in ("untested", "partially measured"):
                self.assertTrue(any(p is None for p, _ in h.evidence), f"{h.id}: says {h.status} but names no gap")
            if h.status == "measured":
                self.assertTrue(any(p for p, _ in h.evidence), f"{h.id}: 'measured' without a file")

    def test_cited_numbers_are_in_the_cited_files(self):
        import json
        rows = [json.loads(l) for l in (FIX / "llama-bench-sweepc-snapshot.jsonl").read_text().splitlines() if l.strip()]
        def tg(ngl, fa, d=8192):
            return next(r["avg_ts"] for r in rows if r["n_gpu_layers"] == ngl and r["flash_attn"] == fa and r["type_k"] == "f16"
                        and r["n_gen"] == 128 and r["n_depth"] == d)
        for ngl, off, on in ((0, 5.45, 4.44), (8, 6.78, 5.76), (12, 7.65, 6.59), (16, 9.15, 7.83)):
            self.assertAlmostEqual(tg(ngl, 0), off, places=1)
            self.assertAlmostEqual(tg(ngl, 1), on, places=1)
        journal = (ROOT / "docs" / "JOURNAL.md").read_text(encoding="utf-8")
        for needle in ("9.19", "8.32", "8.00", "7.81"):
            self.assertIn(needle, journal)


if __name__ == "__main__":
    unittest.main()

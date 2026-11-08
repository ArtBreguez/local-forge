import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from forge import gguf, plan
from forge.plan import Caps, Config

from .helpers import FIX, ROOT, fit_params_grid, profile, real_model, tiny_gguf


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
    """Expected VRAM computed here from the raw fixture numbers and llama.cpp's own estimate (results/fit-params-grid-*.txt),
    with the placement rule written out again, so a bug in estimate()/max_layers() cannot cancel itself out."""

    def placed_mib(self, ngl, ctx, kv_bytes_per_elem):
        """Weights + KV + recurrent state of what llama.cpp puts on the GPU for -ngl N: block slots >= n_layer + 1 - N, output matrix first."""
        d = json.loads((FIX / "modelinfo-dsv4-9b.json").read_text())
        if ngl <= 0:
            return 0.0
        n_layer, trunk = d["block_count"], d["trunk_layers"]
        on_gpu = range(max(0, n_layer + 1 - ngl), trunk)
        weights = sum(d["layer_bytes"][i] for i in on_gpu) + d["output_bytes"]
        kv = sum(d["kv_elems_per_token"][i] for i in on_gpu) * kv_bytes_per_elem * ctx
        state = sum(d["state_bytes_per_recurrent_layer"] for i in on_gpu if d["kv_elems_per_token"][i] == 0)
        return (weights + kv + state) / 2**20

    def test_estimate_matches_hand_arithmetic(self):
        m = real_model()
        for ngl, ctx, kv, bpe in ((16, 16384, "f16", 2.0), (16, 32768, "f16", 2.0), (16, 16384, "q8_0", 34 / 32),
                                  (8, 8192, "f16", 2.0), (32, 16384, "f16", 2.0), (33, 16384, "f16", 2.0), (99, 16384, "f16", 2.0),
                                  (15, 16384, "f16", 2.0), (13, 8192, "q8_0", 34 / 32), (18, 16384, "f16", 2.0), (21, 32768, "f16", 2.0),
                                  (1, 16384, "f16", 2.0), (2, 16384, "f16", 2.0), (3, 16384, "f16", 2.0)):
            # (the non-multiples of 4 matter: this model repeats every 4 layers, so the first n and the last n
            #  blocks weigh the same when n is a multiple of 4 and a wrong end would go unnoticed)
            cfg = Config(ngl, ctx, 6, 6, kv != "f16", kv)
            e = plan.estimate(m, cfg)
            self.assertAlmostEqual(e.vram_weights + e.vram_kv + e.vram_state, self.placed_mib(ngl, ctx, bpe), places=3, msg=(ngl, ctx, kv))
            self.assertAlmostEqual(e.vram, self.placed_mib(ngl, ctx, bpe) + plan.compute_mib(m, cfg), places=3, msg=(ngl, ctx, kv))
        self.assertEqual(plan.estimate(m, Config(0, 16384, 6, 6, False)).vram, 0)

    def test_mtp_block_weights_are_charged_when_the_file_has_them(self):
        """tiny_gguf(nextn=1, mtp_tensors=True): the MTP block owns the slot offloaded right after the output matrix."""
        with tempfile.TemporaryDirectory() as d:
            with_t = plan.load_model(str(tiny_gguf(Path(d) / "a.gguf", nextn=1, mtp_tensors=True)))
            without = plan.load_model(str(tiny_gguf(Path(d) / "b.gguf", nextn=1, mtp_tensors=False)))
        self.assertEqual((with_t.mtp_bytes, without.mtp_bytes), (8192, 0))
        total = sum(with_t.layer_bytes) + with_t.output_bytes + with_t.embd_bytes + with_t.other_bytes + with_t.mtp_bytes
        self.assertEqual(total, sum(without.layer_bytes) + without.output_bytes + without.embd_bytes + without.other_bytes + 8192)
        g = lambda m, ngl: plan.estimate(m, Config(ngl, 4096, 2, 2, False))
        self.assertAlmostEqual(g(with_t, 1).vram_weights, g(without, 1).vram_weights)               # -ngl 1: the output matrix only
        self.assertAlmostEqual(g(with_t, 2).vram_weights - g(without, 2).vram_weights, 8192 / 2**20)   # -ngl 2: the MTP slot comes next
        self.assertAlmostEqual(g(with_t, 99).vram_weights - g(without, 99).vram_weights, 8192 / 2**20)
        self.assertAlmostEqual(g(with_t, 1).ram_weights - g(without, 1).ram_weights, 8192 / 2**20)   # off the GPU it sits in RAM instead
        self.assertAlmostEqual(g(with_t, 2).ram_weights, g(without, 2).ram_weights)
        self.assertEqual(real_model().mtp_bytes, 0)                                                   # the host's GGUF has no MTP tensors

    def test_offload_order_matches_llama_cpp(self):
        """H-OUTPUT-LAYER, from llama.cpp's own numbers: -ngl 1 is the output matrix alone, -ngl 2 adds only the empty MTP slot, -ngl N is N-2 blocks."""
        m = real_model()
        grid = fit_params_grid()
        out_mib = m.output_bytes / 2**20
        self.assertEqual(grid[(1, 16384, 512, "off", "f16")][0], int(out_mib))      # fit-params prints whole MiB, truncated
        self.assertEqual(grid[(2, 16384, 512, "off", "f16")][0], grid[(1, 16384, 512, "off", "f16")][0])
        self.assertEqual(plan.gpu_layer_ids(m, 1), [])
        self.assertEqual(plan.gpu_layer_ids(m, 2), [])
        self.assertEqual(plan.gpu_layer_ids(m, 3), [31])
        self.assertEqual(plan.gpu_layer_ids(m, 16), list(range(18, 32)))
        self.assertEqual(plan.gpu_layer_ids(m, 34), list(range(32)))
        self.assertEqual(plan.gpu_layer_ids(m, 99), list(range(32)))
        self.assertEqual(plan.gpu_layer_ids(m, 0), [])
        e1 = plan.estimate(m, Config(1, 16384, 6, 6, False))
        self.assertAlmostEqual(e1.vram_weights, out_mib, places=3)       # the output matrix is charged from the first layer on
        self.assertEqual(plan.all_layers(m), 99)
        self.assertEqual(plan.all_layers(replace(m, block_count=126, trunk_layers=126)), 127)

    def test_weights_and_kv_agree_with_llama_cpps_own_estimate_everywhere(self):
        """135 (ngl, ctx, ubatch <= 512, FA, KV) points of llama-fit-params: model and context buffers within 2-3 MiB, by the hand rule and by estimate()."""
        m = real_model()
        checked = 0
        for (ngl, ctx, ub, fa, kv), (model_mib, ctx_mib, _compute) in fit_params_grid().items():
            if ub > plan.COMPUTE_MAX_UBATCH:
                continue
            bpe = plan.KV_BYTES_PER_ELEM[kv]
            self.assertAlmostEqual(self.placed_mib(ngl, ctx, bpe), model_mib + ctx_mib, delta=3, msg=(ngl, ctx, ub, fa, kv))
            e = plan.estimate(m, Config(ngl, ctx, 6, 6, fa == "on", kv, ub))
            self.assertAlmostEqual(e.vram_weights, model_mib, delta=2, msg=(ngl, ctx, ub, fa, kv))
            self.assertAlmostEqual(e.vram_kv + e.vram_state, ctx_mib, delta=2, msg=(ngl, ctx, ub, fa, kv))
            checked += 1
        self.assertGreaterEqual(checked, 135)

    def test_compute_model_tracks_llama_cpps_estimate(self):
        """H-COMPUTE-SCRATCH: never below llama.cpp's number by more than 36 MiB, at most 36 above for every layer count a 4 GB card can hold, 100 above at -ngl 99."""
        m = real_model()
        worst_under = worst_over_small = worst_over = 0.0
        for (ngl, ctx, ub, fa, kv), (_m, _c, compute) in fit_params_grid().items():
            if ngl == 0 or ub > plan.COMPUTE_MAX_UBATCH:
                continue
            err = plan.compute_mib(m, Config(ngl, ctx, 6, 6, fa == "on", kv, ub)) - compute
            worst_under = min(worst_under, err)
            worst_over = max(worst_over, err)
            if ngl <= 20:
                worst_over_small = max(worst_over_small, err)
        self.assertGreaterEqual(worst_under, -36)
        self.assertLessEqual(worst_over_small, 36)
        self.assertLessEqual(worst_over, 100)

    def test_compute_model_is_exact_where_the_facts_are(self):
        m = real_model()
        grid = fit_params_grid()
        for (ctx, ub, fa), want in (((8704, 512, False), 517), ((8704, 256, False), 258), ((16384, 512, False), 673), ((16384, 256, False), 377),
                                    ((8704, 512, True), 517), ((8704, 256, True), 258), ((16384, 512, True), 517)):
            kv = "f16"
            self.assertEqual(grid[(16, ctx, ub, "on" if fa else "off", kv)][2], want)          # the ground truth itself
            self.assertAlmostEqual(plan.compute_mib(m, Config(16, ctx, 6, 6, fa, kv, ub)), want, delta=25, msg=(ctx, ub, fa))

    def test_the_failed_server_allocation_is_the_compute_buffer(self):
        """705695744 bytes (llama-server, ngl 15, 16K, ubatch 512) = 673 MiB = what llama-fit-params says that buffer is."""
        log = (ROOT / "results" / "llama-server-fail-ngl15-ctx16384-host-20261108.log").read_text()
        self.assertIn("Device memory allocation of size 705695744 failed", log)
        self.assertIn("failed to allocate compute pp buffers", log)
        self.assertEqual(int(705695744 / 2**20), 673)
        self.assertEqual(fit_params_grid()[(15, 16384, 512, "off", "f16")][2], 673)
        bench_oom = (FIX / "llama-bench-sweepd-snapshot.log").read_text()          # sweep D's last failure: FA on, q8_0, ubatch 256, ngl 20
        self.assertEqual(int(271058944 / 2**20), 258)
        self.assertEqual(fit_params_grid()[(16, 8704, 256, "on", "q8_0")][2], 258)
        self.assertIn("ngl=20 fa=on kv=q8_0 ub=256 rc=1", bench_oom)

    def test_kv_doubles_with_context_and_shrinks_with_q8(self):
        m = real_model()
        k16 = plan.estimate(m, Config(34, 16384, 6, 6, False)).vram_kv
        k32 = plan.estimate(m, Config(34, 32768, 6, 6, False)).vram_kv
        kq8 = plan.estimate(m, Config(34, 16384, 6, 6, True, "q8_0")).vram_kv
        self.assertAlmostEqual(k16, 512.0)                  # 8 layers x 4096 B/token x 16384 / 2^20
        self.assertAlmostEqual(k32, 2 * k16)
        self.assertAlmostEqual(kq8, k16 * 34 / 32 / 2)

    def test_compute_buffer_grows_with_context_and_ubatch_and_ignores_context_with_flash_attention(self):
        m = real_model()
        c = lambda ctx, ub, fa: plan.compute_mib(m, Config(16, ctx, 6, 6, fa, "f16", ub))
        self.assertLess(c(8704, 512, False), c(16384, 512, False))
        self.assertLess(c(16384, 512, False), c(32768, 512, False))
        self.assertLess(c(16384, 256, False), c(16384, 512, False))
        self.assertEqual(c(4096, 512, True), c(32768, 512, True))
        self.assertLess(c(16384, 512, True), c(16384, 512, False))
        self.assertEqual(c(8704, 512, False), c(8192, 512, False))             # below ~12K the logits phase dominates: context is free
        with self.assertRaises(ValueError):
            c(16384, 1024, False)
        with_no_vocab = replace(m, n_vocab=0)                                 # a header without a vocabulary size: assume the large end
        self.assertGreater(plan.compute_mib(with_no_vocab, Config(16, 16384, 6, 6, True, "f16", 512)), c(16384, 512, True))

    def test_reserve_sits_inside_the_window_the_facts_leave_open(self):
        """Six load/fail facts on 3443 MiB free, with llama.cpp's own model+context+compute numbers (the grid, not the planner):
        a loading run needs reserve <= free - total, a failing run needs reserve > free - total."""
        g = fit_params_grid()
        total = lambda ngl, ctx, ub: sum(g[(ngl, ctx, ub, "off", "f16")])
        loads = [(16, 8704, 512), (18, 8704, 256), (14, 16384, 512)]           # llama-bench, llama-bench, llama-server
        fails = [(17, 8704, 512), (19, 8704, 256), (15, 16384, 512)]
        hi = min(3443 - total(*c) for c in loads)
        lo = max(3443 - total(*c) for c in fails)
        self.assertEqual((lo, hi), (271, 389))                                  # llama.cpp-side window: (271, 389] MiB
        reserve = 3443 - plan.vram_budget(3443)
        self.assertLess(lo + 40, reserve)                                      # at least 40 MiB inside the window on either side
        self.assertLessEqual(reserve + 40, hi)

    def test_chosen_layers_never_exceed_free_vram_even_ignoring_the_margin(self):
        """Physical bound, independent of the margin constants: the true need must be below what is free."""
        m = real_model()
        for free in range(1000, 8000, 250):
            p = plan.build_plan(profile(free_vram=free, vram_total=free + 600), m)
            need = self.placed_mib(p.config.ngl, p.config.ctx, plan.KV_BYTES_PER_ELEM[p.config.kv]) + (plan.compute_mib(m, p.config) if p.config.ngl else 0)
            self.assertLess(need, free, f"free={free}: plan {p.summary()} needs {need:.0f} MiB by hand arithmetic")

    def test_plan_follows_free_vram_down_to_the_busy_card(self):
        """3443 MiB free idle, 3105 with a benchmark running (fixtures). The reserve does not cover a drop that large (H-VRAM-HEADROOM says so):
        what keeps the server loading is the re-probe at every start (and the one-layer fallback), so the busy plan must be smaller and fit its own budget."""
        from forge import probe
        idle = probe.parse_list_devices((FIX / "list-devices-rx6500xt-idle.txt").read_text())[1][0]["free_mib"]
        busy = probe.parse_list_devices((FIX / "list-devices-rx6500xt-busy.txt").read_text())[1][0]["free_mib"]
        self.assertEqual((idle, busy), (3443, 3105))
        for kv in (None, "q8_0"):
            a = plan.build_plan(profile(free_vram=idle), real_model(), kv=kv)
            b = plan.build_plan(profile(free_vram=busy), real_model(), kv=kv)
            self.assertLess(b.config.ngl, a.config.ngl, (kv, a.summary(), b.summary()))
            need = self.placed_mib(b.config.ngl, b.config.ctx, plan.KV_BYTES_PER_ELEM[b.config.kv]) + plan.compute_mib(real_model(), b.config)
            self.assertLessEqual(need, busy)
            need_idle = self.placed_mib(a.config.ngl, a.config.ctx, plan.KV_BYTES_PER_ELEM[a.config.kv]) + plan.compute_mib(real_model(), a.config)
            self.assertGreater(need_idle + 271, busy,          # 271 MiB = the least unexplained overhead the load/fail facts allow
                               "if this fails the busy card now holds the idle plan: H-VRAM-HEADROOM's 'not covered' sentence is out of date")

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

    def test_this_machine_lands_on_the_measured_server_limit(self):
        """llama-server at 16K, FA off, f16, ubatch 512: 15 layers fail to load, 14 load (results/phase2.log). The plan must say 14, not 15."""
        m = real_model()
        default = plan.build_plan(profile(), m)
        self.assertEqual((default.config.ctx, default.config.threads, default.config.ubatch), (16384, 6, 512))
        self.assertEqual(default.config.ngl, 14, default.summary())
        log = (ROOT / "results" / "phase2.log").read_text()
        self.assertIn("load ngl=15 ctx=16384 ok=False", log)
        self.assertIn("load ngl=14 ctx=16384 ok=True", log)
        q8 = plan.build_plan(profile(), m, kv="q8_0")
        self.assertEqual((q8.config.fa, q8.config.kv), (True, "q8_0"))
        self.assertLessEqual(abs(q8.config.ngl - 16), 2, q8.summary())

    def test_matches_the_load_boundary_measured_in_llama_bench_and_llama_server(self):
        """Six facts on 3443 MiB free, FA off, f16: llama-bench (context 8704) loads 16 / fails 17 at ubatch 512 and loads 18 / fails 19 at 256;
        llama-server (context 16384, ubatch 512) loads 14 / fails 15. One reserve, one compute model, all six."""
        m = real_model()
        budget = plan.vram_budget(3443)
        fits = lambda n, ctx, ub: plan.estimate(m, Config(n, ctx, 6, 6, False, "f16", ub)).vram <= budget
        for n, ctx, ub, loads in ((16, 8704, 512, True), (17, 8704, 512, False), (18, 8704, 256, True), (19, 8704, 256, False),
                                  (14, 16384, 512, True), (15, 16384, 512, False)):
            self.assertEqual(fits(n, ctx, ub), loads, (n, ctx, ub))
        # and the plan is exactly at those lines
        for ctx, ub, want in ((8704, 512, 16), (8704, 256, 18), (16384, 512, 14), (16384, 256, 16)):
            self.assertEqual(plan.max_layers(m, Config(0, ctx, 6, 6, False, "f16", ub), 3443), want, (ctx, ub))
        log = (FIX / "llama-bench-sweepd-snapshot.log").read_text()      # and the sweeps' logs really say that
        for line in ("ngl=16 fa=off kv=f16 ub=512 rc=0", "ngl=17 fa=off kv=f16 ub=512 rc=1", "ngl=18 fa=off kv=f16 ub=256 rc=0", "ngl=19 fa=off kv=f16 ub=256 rc=1"):
            self.assertIn(line, log)
        # sweep C (ubatch 512, ~8.7K): 16 loads for FA off/on and f16/q8_0, 20 fails for all three
        for fa, kv in ((False, "f16"), (True, "f16"), (True, "q8_0")):
            self.assertLessEqual(plan.estimate(m, Config(16, 8704, 6, 6, fa, kv)).vram, budget, (fa, kv))
            self.assertGreater(plan.estimate(m, Config(20, 8704, 6, 6, fa, kv)).vram, budget, (fa, kv))
        # sweep D, FA on + q8_0 at ubatch 256: 18 loads, 20 fails
        self.assertLessEqual(plan.estimate(m, Config(18, 8704, 6, 6, True, "q8_0", 256)).vram, budget)
        self.assertGreater(plan.estimate(m, Config(20, 8704, 6, 6, True, "q8_0", 256)).vram, budget)

    def test_a_smaller_ubatch_buys_layers_and_the_server_plan_carries_it(self):
        m = real_model()
        a = plan.build_plan(profile(), m)
        b = plan.build_plan(profile(), m, ubatch=256)
        self.assertEqual((a.config.ubatch, b.config.ubatch), (512, 256))
        self.assertGreaterEqual(b.config.ngl - a.config.ngl, 2)                       # sweep D: 18 vs 16 at 8.7K; here 16 vs 14 at 16K
        self.assertIn("-ub 256", " ".join(plan.server_command("llama-server", "m.gguf", b.config)))
        self.assertIn("-ub 512", " ".join(plan.server_command("llama-server", "m.gguf", a.config)))
        with self.assertRaises(ValueError):
            plan.build_plan(profile(), m, ubatch=1024)
        # at 32K the compute buffer alone (2.3 GiB at ubatch 512) leaves no layer; a smaller ubatch does
        roomy = profile(ram_free=20000)
        self.assertEqual(plan.build_plan(roomy, m, ctx=32768, kv="f16", fa=False).config.ngl, 0)
        self.assertGreater(plan.build_plan(roomy, m, ctx=32768, kv="f16", fa=False, ubatch=256).config.ngl, 8)

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
    @classmethod
    def setUpClass(cls):
        import json
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = tiny_gguf(Path(cls._tmp.name) / "tiny.gguf")
        cls.prof = Path(cls._tmp.name) / "profile.json"
        cls.prof.write_text(json.dumps(profile()))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_modern_flags(self):
        cfg = plan.build_plan(profile(), real_model()).config
        cmd = plan.server_command("llama-server", "m.gguf", cfg, "127.0.0.1", 8080)
        s = " ".join(cmd)
        for frag in ("-m m.gguf", "--port 8080", f"-ngl {cfg.ngl}", f"-c {cfg.ctx}", "-t 6", "-fa off", "-ctk f16", "-ctv f16", "-np 1", "--fit off"):
            self.assertIn(frag, s)
        self.assertIn("--spec-type none", s)                       # explicit: speculation is off unless asked for (H-SPEC-NONE)

    def test_bad_ubatch_is_a_clean_error_on_the_command_line(self):
        import subprocess, sys
        for bad in ("1024", "0"):
            r = subprocess.run([sys.executable, "-m", "forge", "plan", "--model", str(self.model), "--profile", str(self.prof), "--ubatch", bad],
                               cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("ubatch", r.stderr)

    def test_speculative_decoding_is_opt_in(self):
        m, prof = real_model(), profile()
        default = plan.build_plan(prof, m)
        self.assertIsNone(default.config.spec_type)
        self.assertTrue(any(d.rule == "H-SPEC-NONE" and "none" in d.text for d in default.decisions))
        dcmd = plan.server_command("llama-server", "m.gguf", default.config)
        self.assertEqual(dcmd[dcmd.index("--spec-type") + 1], "none")
        self.assertTrue(all(p.config.spec_type is None for p in plan.build_all(prof, m).values()))
        on = plan.build_plan(prof, m, spec="ngram-mod")
        self.assertEqual(on.config.spec_type, "ngram-mod")
        self.assertEqual(on.config.ngl, default.config.ngl)               # host-side drafting: the GPU plan does not change
        cmd = plan.server_command("llama-server", "m.gguf", on.config)
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "ngram-mod")
        for rejected in ("ngram-cache", "ngram-simple", "draft-mtp", "bogus"):          # measured worse, or not available: not offered
            with self.assertRaises(ValueError):
                plan.build_plan(prof, m, spec=rejected)
        self.assertEqual(plan.build_plan(prof, m, enable_mtp=True).config.spec_type, None)    # MTP tensors absent in this GGUF: still off
        old = plan.server_command("s", "m", plan.build_plan(prof, m).config, caps=Caps(spec_type=False))
        self.assertNotIn("--spec-type", old)                                # builds without the flag are left alone
        help_text = (FIX / "llama-server-help-b12207.txt").read_text(encoding="utf-8")
        self.assertIn("ngram-mod", help_text.split("--spec-type")[1].split("\n")[0])
        self.assertIn("default:\n                                        none", help_text.split("--spec-type")[1].split("(env")[0])

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
                    "H-LOW-POWER", "H-UNKNOWN-FREE-VRAM", "H-COMPUTE-SCRATCH", "H-SPEC-NONE", "H-FALLBACK-STEP", "H-SERVER-RECHECK", "H-TUNE-LAYERS-MONOTONE"}
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

    def test_phase2_numbers_cited_by_h_spec_none_are_in_the_raw_rows(self):
        import json
        rows = [json.loads(l) for l in (ROOT / "results" / "phase2-spec.jsonl").read_text().splitlines() if l.strip()]
        by = lambda spec: {(r["prompt"], r["rep"]): r for r in rows if r["spec"] == spec}
        none, mod = by("none"), by("ngram-mod")
        mean = lambda d: sum(r["gen_tps"] for r in d.values()) / len(d)
        self.assertAlmostEqual(mean(none), 9.03, places=2)
        self.assertAlmostEqual(mean(mod), 10.75, places=2)
        self.assertAlmostEqual(max(r["gen_tps"] for r in mod.values()), 17.49, places=2)
        self.assertEqual(sum(1 for k in none if none[k]["text_sha"] == mod[k]["text_sha"]), 6)        # 2 of 8 outputs differ at temperature 0
        self.assertEqual(sum(1 for k in none if none[k]["text_sha"] == by("ngram-cache")[k]["text_sha"]), 4)
        self.assertTrue(all(r["ngl"] == 14 for r in rows if r["spec"] != "draft-mtp"))


if __name__ == "__main__":
    unittest.main()

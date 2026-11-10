"""Round-2 review, the Modal side: m4 (the legacy default), m5 (agreement in the sample stage), m6 (robustness), m7 (held-out range),
m10 (held-out arm reported separately, with a control). Modal itself is stubbed; nothing here rents anything."""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from .helpers import ROOT

RB = ROOT / "tools" / "rea-bench"
sys.path.insert(0, str(RB))

from rea_bench import core, dedupe, families, outputs, scoring, sft  # noqa: E402


class ModalCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake = mock.MagicMock(name="modal")
        fake.App.return_value.function.side_effect = lambda *a, **k: (lambda f: f)
        fake.App.return_value.local_entrypoint.side_effect = lambda *a, **k: (lambda f: f)
        cls.patch = mock.patch.dict(sys.modules, {"modal": fake})
        cls.patch.start()
        sys.path.insert(0, str(ROOT / "tools" / "modal"))
        for name in ("modal_rea", "modal_rea_data", "modal_qbench"):
            sys.modules.pop(name, None)
        import modal_rea, modal_rea_data
        cls.rea, cls.data = modal_rea, modal_rea_data

    @classmethod
    def tearDownClass(cls):
        for name in ("modal_rea", "modal_rea_data", "modal_qbench"):
            sys.modules.pop(name, None)
        sys.path.remove(str(ROOT / "tools" / "modal"))
        cls.patch.stop()

    def run_main(self, root, starmap=None, remote_result=None, **kw):
        ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        run = mock.MagicMock(return_value=kw.pop("rm_result", ok))
        default = lambda tasks, **k: [{**meta, "pseudocode": "void f(void) {}"} for meta, _src, _tag in tasks]
        remote = mock.MagicMock(return_value=remote_result or {"sft": [], "stats": {"tasks": 0}})
        with mock.patch.object(self.data, "ROOT", root), mock.patch.object(self.data, "subprocess", types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)), \
                mock.patch.object(self.data, "decompile_task", types.SimpleNamespace(starmap=starmap or default)), \
                mock.patch.object(self.data, "sample_answers", types.SimpleNamespace(remote=remote)), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.data.main(**{"local_gate": False, **kw})
        self.printed = printed.getvalue()
        return remote, run


class SampleSelection(unittest.TestCase):
    def test_the_stored_target_is_cut_and_a_contradiction_is_not_stored(self):
        t = {"id": "s-1", "kind": "structs", "score": "tokens", "accept": [core.hx(8)], "chance": 0.07}
        msgs = lambda task: [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        results = {"s-1": {"outs": [["0x8\nThe pointer is at 0x8. Wait, actually it moves.\nMore text.", True], ["0x8\nIt is the next field. And another sentence.", True],
                                    ["0x8\nWait, actually 0x70 in some builds of the code.", True], ["0x8\nWait, actually 0x70.", True], ["0x10", False]]}}
        kept, stats = sft.assemble([t], results, "m", msgs)
        self.assertEqual([r["messages"][-1]["content"] for r in kept], ["0x8\nThe pointer is at 0x8.", "0x8\nIt is the next field.", "0x8"],
                         "a justification that contradicts itself is dropped and the headline alone is stored; a second line that is no sentence makes a list")
        self.assertEqual([r["sentence_dropped"] for r in kept], [False, False, True])
        self.assertTrue(all(r["truncated"] for r in kept))
        self.assertEqual((stats["correct"], stats["kept_records"], stats["truncated"], stats["sentence_dropped"]), (3, 3, 3, 1))
        self.assertEqual(stats["by_kind"]["structs"]["kept_tasks"], 1)

    def test_high_chance_tasks_need_two_right_samples(self):
        easy = {"id": "c-1", "kind": "crc", "score": "tokens", "accept": [core.hx(0xffff, 4)], "chance": 0.55}
        msgs = lambda task: []
        kept, st = sft.assemble([easy], {"c-1": {"outs": [["0xffff", True], ["0x0", False], ["0x1", False]]}}, "m", msgs)
        self.assertEqual((kept, st["agreement"]), ([], 1))
        kept, st = sft.assemble([easy], {"c-1": {"outs": [["0xffff", True], ["0xffff", True], ["0x1", False]]}}, "m", msgs)
        self.assertEqual((len(kept), st["agreement"]), (1, 0), "two identical stored texts are one record")
        kept, st = sft.assemble([easy], {"c-1": {"outs": [["0xffff", True], ["0xffff\nIt is the all-ones init of the register.", True]]}}, "m", msgs)
        self.assertEqual(len(kept), 2)


class Robustness(ModalCase):
    """m6."""

    def test_a_failed_volume_rm_stops_the_run_but_a_missing_directory_does_not(self):
        run = lambda rc, err: (lambda *a, **k: types.SimpleNamespace(returncode=rc, stdout="", stderr=err))
        outputs.rm_volume_dir("v", "/d", run(0, ""))
        outputs.rm_volume_dir("v", "/d", run(1, "Error: path '/d' not found"))
        with self.assertRaisesRegex(SystemExit, "could not clear /d"):
            outputs.rm_volume_dir("v", "/d", run(1, "Error: authentication failed"))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, "could not clear"):
                self.run_main(pathlib.Path(d), stage="decompile", n=6, seed=7, family="crc", rm_result=types.SimpleNamespace(returncode=2, stdout="", stderr="boom"))
            self.assertEqual(list((pathlib.Path(d) / "results" / "phase6" / "rea-data").glob("*.jsonl")), [], "nothing was written")

    def test_a_decompile_exception_becomes_an_error_record_and_the_file_is_complete(self):
        def starmap(tasks, **kw):
            assert kw.get("return_exceptions") is True
            return [subprocess.TimeoutExpired("ghidra", 900) if i == 1 else {**meta, "pseudocode": "void f(void) {}"} for i, (meta, _s, _t) in enumerate(tasks)]

        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.run_main(root, starmap=starmap, stage="decompile", n=4, seed=7, family="crc")
            recs = [json.loads(l) for l in (root / "results" / "phase6" / "rea-data" / "decompiled-seed7-crc.jsonl").read_text().splitlines()]
            self.assertEqual(len(recs), 4)
            self.assertIn("decompile crashed: TimeoutExpired", recs[1]["error"])
            self.assertIn("errors by reason: decompile crashed 1", self.printed)
            self.assertIn("crc 1", self.printed)

    def test_a_run_that_dies_half_way_leaves_nothing_that_blocks_the_next_run(self):
        def starmap(tasks, **kw):
            yield {**tasks[0][0], "pseudocode": "x"}
            raise RuntimeError("container lost")

        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            rd = root / "results" / "phase6" / "rea-data"
            with self.assertRaisesRegex(RuntimeError, "container lost"):
                self.run_main(root, starmap=starmap, stage="decompile", n=4, seed=7, family="crc")
            self.assertFalse((rd / "decompiled-seed7-crc.jsonl").exists())
            self.run_main(root, stage="decompile", n=4, seed=7, family="crc")           # no --force needed
            self.assertEqual(len((rd / "decompiled-seed7-crc.jsonl").read_text().splitlines()), 4)
            self.assertFalse(list(rd.glob("*.part")))

    def test_decompile_task_returns_an_error_record_for_a_hang_or_a_crash(self):
        task = {"id": "x-1", "kind": "x", "opt": "-O2", "cc": "gcc", "score": "tokens", "accept": [["1"]]}
        with mock.patch.object(self.data, "_decompile_task", side_effect=subprocess.TimeoutExpired("rea", 900)):
            rec = self.data.decompile_task(task, "int proc(void){return 0;}")
        self.assertIn("decompile failed: TimeoutExpired", rec["error"])
        self.assertEqual(rec["id"], "x-1")
        with mock.patch.object(self.data.subprocess, "run", side_effect=subprocess.TimeoutExpired("bash", 5)):
            self.assertIn("[timeout after 5s]", self.data._sh("sleep 9", ".", timeout=5))

    def test_error_reasons_are_summarised(self):
        recs = [{"id": "a", "kind": "crc", "error": "fact not in the compiled code (clang -O1: ...)"}, {"id": "b", "kind": "crc", "error": "fact not in the compiled code (x)"},
                {"id": "c", "kind": "cipher", "error": "no pseudocode", "raw": "[timeout after 900s] rea"}, {"id": "d", "kind": "cipher", "error": "no pseudocode", "raw": "oops"},
                {"id": "e", "kind": "crc", "error": "proc not found"}, {"id": "f", "kind": "crc", "pseudocode": "ok"}, {"id": "g", "kind": "crc", "error": "gate failed: x"}]
        s = outputs.summarize_errors(recs)
        self.assertEqual(s["total"], 6)
        self.assertEqual(s["by_reason"], {"fact gate (Modal compilers)": 2, "decompile timeout": 1, "decompile: no pseudocode": 1, "symbol not found": 1, "gate could not run": 1})
        self.assertEqual(s["by_reason_kind"]["fact gate (Modal compilers)"], {"crc": 2})

    def test_the_checkpoint_resumes_or_is_moved_aside(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d, "sft-m-seed7-crc.partial.jsonl")
            p.write_text(json.dumps({"id": "a", "outs": [["x", True]]}) + "\n" + json.dumps({"id": "b", "outs": []}) + "\n" + '{"id": "c", "ou')
            self.assertEqual(sorted(outputs.load_partial(p, True)), ["a", "b"], "a line cut by the timeout is skipped")
            self.assertEqual(outputs.load_partial(p, False), {})
            self.assertFalse(p.exists())
            self.assertEqual(len(list(pathlib.Path(d).glob("*.old-*"))), 1, "an old checkpoint is kept, never reused")
            self.assertEqual(outputs.load_partial(p, True), {})

    def test_stats_come_per_family_and_the_resume_flag_reaches_the_sampler(self):
        stats = {"tasks": 3, "answers": 6, "correct": 2, "solved_tasks": 1, "kept_tasks": 1, "kept_records": 1, "agreement": 1, "vetoed": 0,
                 "by_kind": {"crc": {"tasks": 2, "solved_tasks": 1, "kept_tasks": 1, "agreement": 1, "vetoed": 0}}}
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", resume=True,
                                      remote_result={"sft": [{"task": "x"}], "stats": stats})
            self.assertTrue(remote.call_args.kwargs["resume"])
            self.assertEqual(remote.call_args.args[-1], "seed7-crc")
            self.assertIn("crc", self.printed)
            self.assertIn("dropped for lack of agreement", self.printed)
            written = json.loads(next((pathlib.Path(d) / "results" / "phase6" / "rea-data").glob("sft-*.stats.json")).read_text())
            self.assertIn("by_kind", written)


class HeldoutLimits(ModalCase):
    """m7."""

    def test_seed_1001_beyond_120_per_family_is_refused_including_the_default_legacy_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            with self.assertRaisesRegex(SystemExit, "held-out split.*120"):
                self.run_main(root, stage="decompile", n=300, seed=1001, family="legacy")          # `--family legacy --n 300` is the default invocation
            with self.assertRaisesRegex(SystemExit, "held-out split"):
                self.run_main(root, stage="decompile", n=250, seed=1001, family="crc,cipher")
            self.run_main(root, stage="decompile", n=120, seed=1001, family="legacy")
            self.run_main(root, stage="decompile", n=240, seed=1001, family="crc,cipher")
            self.assertFalse((root / "results" / "phase6" / "rea-data" / "decompiled-seed1001-legacy+crc.jsonl").exists())

    def test_modal_rea_refuses_held_out_files_beyond_the_pinned_list(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "tools" / "modal" / "rea-bench").mkdir(parents=True)
            (root / "tools" / "modal" / "rea-bench" / "rea-tasks.json").write_text((ROOT / "tools" / "modal" / "rea-bench" / "rea-tasks.json").read_text())
            held = root / "results" / "phase6" / "rea-data"
            held.mkdir(parents=True)
            rows = [{"id": f"xor-key-1001-{i:04d}", "kind": "xor-key", "score": "exact", "accept": ["a"], "pseudocode": "void f(void) {}"} for i in range(121)]
            (held / "decompiled-seed1001.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            old, self.rea.ROOT = self.rea.ROOT, root
            try:
                with self.assertRaisesRegex(SystemExit, "legacy 121"):
                    self.rea.main("")
            finally:
                self.rea.ROOT = old


class HeldoutArm(ModalCase):
    """m10 and m6(d)."""

    def tasks(self):
        out = []
        for t in families.generate("structs", 12, 1001) + families.generate("crc", 12, 1001):
            m = t.meta()
            out.append({**m, "pseudocode": "void f(void) {}"})
        return out

    def test_strict_and_control_scores_and_a_failed_call_does_not_lose_the_run(self):
        tasks = self.tasks()
        calls = []

        def ask(messages):
            n = len(calls)
            calls.append(n)
            t = tasks[n]
            if n == 3:
                raise TimeoutError("server stalled")
            truth = scoring.canonical(t)
            return (truth if n % 3 == 0 else f"Let me read the loop.\nThe constant is used twice.\n{truth}" if n % 3 == 1 else "I do not know."), {}

        recs = self.rea.run_heldout(tasks, ask)
        self.assertEqual(len(recs), len(tasks))
        self.assertEqual(recs[3]["error"][:12], "TimeoutError")
        self.assertFalse(recs[3]["pass"] or recs[3]["pass_any_line"])
        strict = sum(r["pass"] for r in recs)
        control = sum(r["pass_any_line"] for r in recs)
        self.assertGreater(control, strict, "the control sees the answers that came after a thinking line")
        self.assertTrue(all(r["pass_any_line"] for r in recs if r["pass"]), "the control is never stricter")
        hs = outputs.heldout_summary(recs)
        self.assertEqual((hs["ALL"]["n"], hs["ALL"]["strict"], hs["ALL"]["control"], hs["ALL"]["errors"]), (len(tasks), strict, control, 1))
        self.assertEqual({k for k in hs} - {"ALL"}, {"structs", "crc"})
        self.assertEqual(sum(hs[k]["n"] for k in ("structs", "crc")), len(tasks))

    def test_main_reports_the_held_out_arm_on_its_own_line(self):
        src = (ROOT / "tools" / "modal" / "modal_rea.py").read_text()
        body = src[src.index("def main(only"):]
        self.assertIn("generated held-out set (separate from the fixed questions)", body)
        self.assertIn("format-insensitive control", body)
        self.assertIn("fixed questions: recipe", body)
        self.assertNotIn("heldout {sum(r['pass'] for r in ho)}", body, "the held-out count is no longer mixed into the fixed-question line")


class LegacyDefault(ModalCase):
    """m4."""

    def test_the_legacy_default_says_what_it_is(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_main(pathlib.Path(d), stage="decompile", n=6, seed=7, family="legacy")
            self.assertIn("frozen whole-reply substring scoring in modal_rea.py's evaluation", self.printed)
            self.assertIn("strict headline grammar", self.printed)
            self.run_main(pathlib.Path(d), stage="decompile", n=6, seed=7, family="crc")
            self.assertNotIn("whole-reply", self.printed)

    def test_k_below_two_is_warned_about(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", k=1)
            self.assertIn("none of them is kept", self.printed)


class Round3Pipeline(ModalCase):
    """Round 3: B1 (a failed request is never scored or stored), m7 (no sampling of the held-out seed), m8 (volume rm fails closed)."""

    ERRORS = ["[error <HTTPError 500: 'Internal Server Error'>]", "[error TimeoutError('The read operation timed out')]",
              "[error URLError(ConnectionRefusedError(111, 'Connection refused'))]", "[error ValueError('Expecting value: line 1 column 1 (char 0)')]"]

    def test_b1_an_error_text_is_not_a_reply_even_when_a_scorer_would_accept_it(self):
        skip = {"id": "skip-7-0055", "kind": "skip", "score": "any", "accept": ["0x0a", "0a", "10"]}        # the review's task: a legacy variant offers `a`
        for txt in self.ERRORS:
            self.assertTrue(scoring.is_error_reply(txt))
            self.assertIsNone(scoring.sft_target(txt, skip), txt)
            self.assertEqual(sft.select([(txt, True)] * 6, skip), ([], "no_correct"), "even a stale True flag does not keep it")
        self.assertFalse(scoring.is_error_reply("0x0a"))

    def test_b1_errors_are_counted_per_family_and_a_dead_server_is_loud(self):
        t = {"id": "a", "kind": "crc", "score": "tokens", "accept": [["1"]], "chance": 0.01}
        u = {"id": "b", "kind": "bounds", "score": "tokens", "accept": [["1"]], "chance": 0.01}
        st = sft.Stats()
        for task, outs in ((t, [(self.ERRORS[0], True), ("1", True), ("2", False)]), (u, [("1", True), ("1", True), ("1", True)])):
            st.add(task, outs, *sft.select(outs, task))
        d = st.as_dict()
        self.assertEqual((d["errors"], d["answers"], d["correct"]), (1, 6, 4))
        self.assertEqual((d["by_kind"]["crc"]["errors"], d["by_kind"]["bounds"]["errors"]), (1, 0))
        self.assertEqual(d["by_kind"]["crc"]["correct"], 1, "a failed request is not a correct answer")
        warning = sft.error_warning(d)
        self.assertIn("MORE THAN 5%", warning)
        self.assertIn("crc: 1/3", warning)
        self.assertIsNone(sft.error_warning({"answers": 100, "errors": 5, "by_kind": {"x": {"answers": 100, "errors": 5}}}), "5% exactly is not more")

    def test_b1_the_sampler_does_not_score_a_failed_request(self):
        src = (ROOT / "tools" / "modal" / "modal_rea_data.py").read_text()
        body = src[src.index("    def work(t):"):src.index("    try:\n        with ThreadPoolExecutor")]
        self.assertIn('txt, ok = f"[error {e!r}]", False', body)
        self.assertNotIn("score(txt, t))", body.split("except Exception")[1], "no scoring of the error text")

    def test_b1_main_prints_the_warning_for_a_run_with_many_errors(self):
        stats = {"tasks": 2, "answers": 12, "errors": 6, "correct": 0, "solved_tasks": 0, "kept_tasks": 0, "kept_records": 0, "agreement": 0, "vetoed": 0,
                 "by_kind": {"crc": {"tasks": 2, "answers": 12, "errors": 6, "solved_tasks": 0, "kept_tasks": 0, "agreement": 0, "vetoed": 0}}}
        with tempfile.TemporaryDirectory() as d:
            self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", remote_result={"sft": [], "stats": stats})
            self.assertIn("WARNING: MORE THAN 5% OF THE REQUESTS FAILED", self.printed)

    def test_m7_the_sample_stage_refuses_the_held_out_seed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, "held-out split.*--stage sample"):
                self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=1001, family="crc")
            self.assertEqual(list((pathlib.Path(d) / "results" / "phase6" / "rea-data").glob("sft-*")), [])
            self.run_main(pathlib.Path(d), stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc")        # a training seed is fine

    def test_m8_volume_rm_fails_closed_and_decides_from_a_listing_when_the_text_is_unknown(self):
        def runner(rm_rc, rm_err, ls=None):
            calls = []

            def run(cmd, **k):
                calls.append(cmd)
                if cmd[2] == "rm":
                    return types.SimpleNamespace(returncode=rm_rc, stdout="", stderr=rm_err)
                return ls if ls is not None else types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
            run.calls = calls
            return run

        listing = lambda out, rc=0: types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
        outputs.rm_volume_dir("v", "/decompiled-seed7-all", runner(1, "FileNotFound: /decompiled-seed7-all"))               # the CLI's other spelling
        outputs.rm_volume_dir("v", "/decompiled-seed7-all", runner(1, "something nobody has seen", listing("decompiled-seed8-all\nsft-x\n")))
        for ls in (listing("decompiled-seed7-all\n"), listing("decompiled-seed7-a\u2026\n"), listing("", rc=1), None):
            with self.assertRaisesRegex(SystemExit, "could not clear /decompiled-seed7-all"):
                outputs.rm_volume_dir("v", "/decompiled-seed7-all", runner(1, "something nobody has seen", ls))
        r = runner(1, "Error: authentication failed")
        with self.assertRaises(SystemExit):
            outputs.rm_volume_dir("v", "/decompiled-seed7-all", r)
        self.assertEqual([c[2] for c in r.calls], ["rm", "ls"], "an unrecognised failure is checked against a listing, then stops the run")


if __name__ == "__main__":
    unittest.main()

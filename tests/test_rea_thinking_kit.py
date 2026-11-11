"""Round 14, fix 3 of the Kaggle thinking pilot, the kit's half: `build_tasks` writes the budget per family (and takes a run config), `run_teacher` asks each task its family's budget, checks its prompt against it and
starts a server for the longest (4 slots of 8192 tokens become 2 of 12288: the same total context), `kaggle-score` shows each family's budget. The Modal half is in tests/test_rea_thinking_budget.py.

  KitBudgets     build_tasks, the kernel's geometry, the session, the run config
  KernelSizing   `run_teacher.main()` starts its server for the longest budget of the run
  ScorerReports  the scorer records the map and shows each family's budget
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from . import distill_fixtures as fx
from .test_rea_kaggle import write_replies, write_tasks
from .test_rea_kaggle_kit import StubServer, make_tasks, teacher_reply
from .thinking_pilot import build_tasks, run_teacher

from distill import config, score_replies  # noqa: E402
from rea_bench import kaggle_score  # noqa: E402


class KernelHarness(unittest.TestCase):
    """A base for the tests of `run_teacher.main()` with the world around it stubbed (the dataset, the GPU, the model download, the server process): what it wires together is the sizing of the server for the budgets of the tasks, the
    choice between an attached llama-server and a build, and what `run.json` records."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.input, self.out = self.d / "input", self.d / "working"
        (self.input / "tasks").mkdir(parents=True)
        self.out.mkdir()
        self.model = self.d / "model.gguf"
        self.model.write_text("gguf")
        self.calls = types.SimpleNamespace(popen=[], run_session=[], built=0)
        self.tasks = [{"id": f"{k}-{i}", "kind": k, "messages": [{"role": "user", "content": "q"}]} for k in ("crc", "cipher") for i in range(2)]

    def write_dataset(self, **cfg):
        (self.input / "tasks" / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in self.tasks))
        (self.input / "tasks" / "config.json").write_text(json.dumps(cfg))

    def run_main(self, prebuilt=None):
        real = __import__("subprocess")

        def fake_run(cmd, *a, **k):
            if isinstance(cmd, list) and cmd[-1] == "--version":          # the real thing: the stand-in server of the prebuilt tests
                return real.run(cmd, *a, **k)
            return types.SimpleNamespace(stdout="", returncode=0)

        def fake_popen(cmd, *a, **k):
            self.calls.popen.append(cmd)
            return types.SimpleNamespace(kill=lambda: None)

        def fake_build(cfg):
            self.calls.built += 1
            return self.d / "built" / "llama-server"

        def fake_session(cfg, tasks, previous, checkpoint, deadline, **kw):
            self.calls.run_session.append((dict(cfg), tasks))
            return {"tasks": len(tasks)}
        fake_subprocess = types.SimpleNamespace(run=fake_run, Popen=fake_popen, SubprocessError=real.SubprocessError, TimeoutExpired=real.TimeoutExpired, STDOUT=real.STDOUT)
        patches = [mock.patch.object(run_teacher, "INPUT", self.input), mock.patch.object(run_teacher, "OUT", self.out), mock.patch.object(run_teacher, "subprocess", fake_subprocess),
                   mock.patch.object(run_teacher, "find_prebuilt", lambda *a: prebuilt), mock.patch.object(run_teacher, "build_llama", fake_build),
                   mock.patch.object(run_teacher, "wait_for_server", lambda *a, **k: None), mock.patch.object(run_teacher, "run_session", fake_session),
                   mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=lambda *a, **k: str(self.model))}), mock.patch.dict(os.environ, {}), *self.more_patches()]
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            for patch in patches:
                stack.enter_context(patch)
            run_teacher.main()
        return json.loads((self.out / "run.json").read_text())

    def more_patches(self) -> list:
        return []


class KitBudgets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tasks = make_tasks((("crc", 3), ("structs", 3), ("cipher", 3)))
        self.file = write_tasks(self.d, self.tasks, "decompiled-seed9-all.jsonl")

    def build(self, out="kb", **kw):
        return build_tasks.build([self.file], self.d / out, "someone", **kw)

    def config(self, out="kb"):
        return json.loads((self.d / out / "data" / "config.json").read_text())

    def test_build_tasks_writes_the_map_only_when_there_is_one(self):
        self.build(quota="crc=2,cipher=2", max_tokens_by_family={"cipher": 8192})
        c = self.config()
        self.assertEqual((c["max_tokens"], c["max_tokens_by_family"], c["slots"], c["ctx_per_slot"]), (4096, {"cipher": 8192}, 4, 8192), "the base geometry; the kernel makes the final one")
        self.build("kb2", quota="crc=2,cipher=2")
        self.assertNotIn("max_tokens_by_family", self.config("kb2"))
        with self.assertRaisesRegex(SystemExit, "no task kind"):
            self.build("kb3", max_tokens_by_family={"cypher": 8192})
        with self.assertRaisesRegex(SystemExit, "at least 1"):
            self.build("kb4", max_tokens_by_family={"cipher": 0})

    def test_the_kernel_sizes_its_server_for_the_longest_budget_of_a_task_in_the_run(self):
        self.build(quota="crc=2,cipher=2", max_tokens_by_family={"cipher": 8192})
        cfg = run_teacher.resolve_config(self.config())
        tasks = [json.loads(line) for line in (self.d / "kb" / "data" / "tasks.jsonl").read_text().splitlines()]
        self.assertEqual(sorted({t["kind"] for t in tasks}), ["cipher", "crc"])
        self.assertEqual(run_teacher.server_geometry(cfg, tasks), (2, 12288))
        self.assertEqual(run_teacher.server_geometry(cfg, [t for t in tasks if t["kind"] == "crc"]), (4, 8192), "a run without a long task starts as the config says")
        self.assertEqual(run_teacher.server_geometry(cfg, []), (4, 8192))
        cmd = run_teacher.server_command("/x/llama-server", "/m.gguf", {**cfg, "slots": 2, "ctx_per_slot": 12288})
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), ("24576", "2"))

    def test_the_kernels_geometry_is_the_one_the_modal_stage_uses(self):
        for slots, ctx, tokens, longest in ((4, 8192, 4096, 4096), (4, 8192, 4096, 8192), (4, 8192, 4096, 6144), (4, 8192, 2048, 8192), (4, 8192, 4096, 16384), (4, 8192, 4096, 2048), (4, 8192, 768, 768)):
            cfg = {"slots": slots, "ctx_per_slot": ctx, "max_tokens": tokens, "max_tokens_by_family": {"long": longest}}
            self.assertEqual(run_teacher.server_geometry(cfg, [{"kind": "long"}, {"kind": "x"}]), config.server_plan(tokens, longest, slots), (tokens, longest))

    def test_a_hand_made_config_keeps_its_prompt_room(self):
        """The pilot ran 4 x 7168 (4096 + 3072 for the prompt): with an 8192-token family that is 2 x 11264, the same 3072, and never more total context than 4 x 7168."""
        cfg = run_teacher.resolve_config({"slots": 4, "ctx_per_slot": 7168, "max_tokens": 4096, "max_tokens_by_family": {"cipher": 8192}})
        self.assertEqual(run_teacher.server_geometry(cfg, [{"kind": "cipher"}]), (2, 11264))
        self.assertEqual(run_teacher.server_geometry(cfg, [{"kind": "crc"}]), (4, 7168))

    def test_the_configs_map_is_checked_by_the_kernel_too(self):
        for bad in ({"cipher": 0}, {"cipher": "8192"}, {"cipher": True}, [1], {3: 5}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                run_teacher.resolve_config({"max_tokens_by_family": bad})
        self.assertEqual(run_teacher.resolve_config({})["max_tokens_by_family"], {})
        self.assertEqual(run_teacher.task_max_tokens(run_teacher.resolve_config({"max_tokens_by_family": {"cipher": 8192}}), "cipher"), 8192)

    def session(self, by_family, tasks=None, ctx=12288, **over):
        cfg = run_teacher.resolve_config({"slots": 2, "ctx_per_slot": ctx, "k": 1, "max_tokens_by_family": by_family, **over})
        tasks = tasks or [{"id": t["id"], "kind": t["kind"], "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"{t['id']} question"}]} for t in self.tasks]
        by_id = {t["id"]: t for t in self.tasks}
        server = StubServer(teacher_reply(by_id, cfg))
        self.addCleanup(server.close)
        summary = run_teacher.run_session(cfg, tasks, {}, self.d / "replies.jsonl", time.time() + 600, base_url=server.url, log_fn=lambda *a: None)
        return cfg, server, summary

    def test_every_request_carries_the_budget_of_its_tasks_family(self):
        _cfg, server, summary = self.session({"cipher": 8192})
        got = {}
        for body in server.bodies:
            tid = next(t["id"] for t in self.tasks if t["id"] in body["messages"][-1]["content"])
            got.setdefault(next(t["kind"] for t in self.tasks if t["id"] == tid), set()).add(body["max_tokens"])
        self.assertEqual(got, {"crc": {4096}, "structs": {4096}, "cipher": {8192}})
        self.assertEqual(summary["skipped_too_long"], [])

    def test_a_prompt_is_checked_against_its_own_budget(self):
        crc = next(t for t in self.tasks if t["kind"] == "crc")
        cipher = next(t for t in self.tasks if t["kind"] == "cipher")
        long_prompt = lambda t: [{"role": "system", "content": "s"}, {"role": "user", "content": f"{t['id']} " + "x " * 4200}]
        tasks = [{"id": t["id"], "kind": t["kind"], "messages": long_prompt(t)} for t in (crc, cipher)]
        _cfg, server, summary = self.session({"cipher": 8192}, tasks)
        self.assertEqual(summary["skipped_too_long"], [cipher["id"]], "4,200 words and 8192 do not fit a 12288-token slot, with 4096 they do")
        self.assertEqual(len(server.bodies), 1)
        self.assertEqual(server.bodies[0]["max_tokens"], 4096)

    def test_a_longer_budget_gets_a_longer_timeout(self):
        cfg = run_teacher.resolve_config({"max_tokens_by_family": {"cipher": 8192}})
        self.assertEqual((run_teacher.request_timeout(cfg), run_teacher.request_timeout(cfg, 8192), run_teacher.request_timeout(cfg, 1024)), (1800, 3600, 1800))
        seen = []

        def urlopen(req, timeout=None):
            seen.append(timeout)
            raise OSError("stop")
        out = run_teacher.ask_one(cfg, [{"role": "user", "content": "q"}], 0, "http://127.0.0.1:1", urlopen, 8192)
        self.assertEqual((seen, out["finish"]), ([3600], "error"))
        run_teacher.ask_one(cfg, [{"role": "user", "content": "q"}], 0, "http://127.0.0.1:1", urlopen)
        self.assertEqual(seen[-1], 1800)
        self.assertEqual(run_teacher.request_body(cfg, [], 3, 8192)["max_tokens"], 8192)
        self.assertEqual(run_teacher.request_body(cfg, [], 3)["max_tokens"], cfg["max_tokens"])

    def test_the_run_config_flag_takes_the_run_from_a_file_and_refuses_a_second_source(self):
        cfgfile = self.d / "run.json"
        cfgfile.write_text(json.dumps({"mode": "thinking", "k": 3, "max_len": 8192, "quota": "crc=2,cipher=2", "max_tokens_by_family": {"cipher": 8192}}))
        argv = ["--tasks", str(self.file), "--owner", "someone", "--out", str(self.d / "kc"), "--run-config", str(cfgfile)]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            build_tasks.main(argv)
        c = self.config("kc")
        self.assertEqual((c["k"], c["max_tokens_by_family"], c["tasks"]["quota"], c["tasks"]["n"], c["thinking"]), (3, {"cipher": 8192}, "crc=2,cipher=2", 4, True))
        self.assertIn("the kernel runs 2 x 12288", printed.getvalue())
        self.assertIn("cipher=8192", printed.getvalue())
        with self.assertRaisesRegex(SystemExit, "do not also give --k, --max-tokens-by-family"):
            build_tasks.main([*argv, "--force", "--k", "2", "--max-tokens-by-family", "cipher=4096"])
        cfgfile.write_text(json.dumps({"mode": "plain"}))
        with self.assertRaisesRegex(SystemExit, "plain run"):
            build_tasks.main([*argv, "--force"])
        cfgfile.write_text(json.dumps({"domain": "example_arith", "mode": "thinking", "max_len": 8192}))
        with self.assertRaisesRegex(SystemExit, "for domain 'example_arith'"):
            build_tasks.main([*argv, "--force"])

    def test_the_flags_without_a_run_config_are_what_they_were(self):
        argv = ["--tasks", str(self.file), "--owner", "someone", "--out", str(self.d / "kf"), "--quota", "crc=2,cipher=2", "--max-tokens-by-family", "cipher=8192"]
        with contextlib.redirect_stdout(io.StringIO()):
            build_tasks.main(argv)
        c = self.config("kf")
        self.assertEqual((c["k"], c["thinking"], c["max_tokens"], c["max_tokens_by_family"]), (4, True, 4096, {"cipher": 8192}))
        with contextlib.redirect_stdout(io.StringIO()):
            build_tasks.main(["--tasks", str(self.file), "--owner", "someone", "--out", str(self.d / "kv"), "--no-thinking", "--k", "2"])
        c = self.config("kv")
        self.assertEqual((c["k"], c["thinking"], c["max_tokens"], "max_tokens_by_family" in c), (2, False, 768, False))


class KernelSizing(KernelHarness):
    """`run_teacher.main()`: the geometry of the server for the budgets of the tasks of the run, and what `run.json` records."""

    def test_the_server_is_started_for_the_longest_budget_of_a_task_of_the_run(self):
        self.write_dataset(max_tokens=4096, max_tokens_by_family={"cipher": 8192}, slots=4, ctx_per_slot=8192)
        run = self.run_main()
        cmd = self.calls.popen[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), ("24576", "2"))
        cfg, tasks = self.calls.run_session[0]
        self.assertEqual((cfg["slots"], cfg["ctx_per_slot"], len(tasks)), (2, 12288, 4), "the session asks with the geometry the server has")
        self.assertEqual((run["slots"], run["ctx_per_slot"], run["server"]), (2, 12288, {"slots": 2, "ctx_per_slot": 12288, "configured": [4, 8192]}), "run.json says what ran, and what the config said")
        self.assertEqual(run["max_tokens_by_family"], {"cipher": 8192}, "kaggle-score reads the map from here")

    def test_without_a_longer_family_the_server_is_what_the_config_says(self):
        self.write_dataset(max_tokens=4096, max_tokens_by_family={"cipher": 2048}, slots=4, ctx_per_slot=7168)
        run = self.run_main()
        cmd = self.calls.popen[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1], run["server"]["configured"]), (str(4 * 7168), "4", [4, 7168]))


class ScorerReports(unittest.TestCase):
    def test_the_scorer_records_the_map_and_shows_each_familys_budget(self):
        tasks = fx.load_tasks()
        kinds = sorted({t["kind"] for t in tasks})
        by = {kinds[0]: 8192}
        run_info = {"thinking": True, "sampling": config.THINKING_SAMPLING, "max_tokens": 4096, "max_tokens_by_family": by, "seed_base": 1234, "ctx_per_slot": 12288, "slots": 2}
        with tempfile.TemporaryDirectory() as d:
            tp = write_tasks(d, tasks)
            rp = write_replies(d, fx.kaggle_lines(tasks, "thinking"))
            _kept, stats = kaggle_score.score(rp, tp, "m", thinking=True, run_info=run_info)
            _kept, plain = kaggle_score.score(rp, tp, "m", thinking=True, run_info={k: v for k, v in run_info.items() if k != "max_tokens_by_family"})
        self.assertEqual(stats["sampling"]["max_tokens_by_family"], by)
        self.assertEqual({k: r.get("max_tokens") for k, r in stats["families"].items() if k != "ALL"}, {k: by.get(k, 4096) for k in kinds})
        self.assertNotIn("max_tokens", stats["families"]["ALL"])
        self.assertNotIn("max_tokens_by_family", plain["sampling"])
        self.assertFalse(any("max_tokens" in r for r in plain["families"].values()), "without a map the table is what it was")
        self.assertIn("max_tokens", score_replies.families_report(stats)[0])
        self.assertNotIn("max_tokens", score_replies.families_report(plain)[0])


if __name__ == "__main__":
    unittest.main()

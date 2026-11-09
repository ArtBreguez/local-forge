"""Phase 5 (03-agent): the 30-minute arms, the one-tool MCP server for OpenCode, the server timings in the metering proxy, and the report
that reads a 30-minute arm against phase 4's 15-minute one. No model is needed."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .helpers import ROOT
from .test_agent import BY_ID, HAVE_NODE, Oracle, fake_run, tmpdir

AGENT = ROOT / "03-agent"
for sub in ("harness", "orchestrator", "quality"):
    sys.path.insert(0, str(AGENT / sub))
import grid  # noqa: E402
import meter  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402
import taskbench  # noqa: E402
import testtool_mcp  # noqa: E402

PYTHON = sys.executable


def rpc_lines(*msgs):
    return "".join(json.dumps(m) + "\n" for m in msgs).encode()


class McpTool(unittest.TestCase):
    """The server speaks MCP over stdio as OpenCode will use it, and its answer is the first failing test only."""

    def server(self, task_id="py-wrap", log=None):
        t = BY_ID[task_id]
        repo = taskbench.materialize(t, tmpdir(self) / "repo")
        cmd = taskbench.expand(t.test_cmd, PYTHON, taskbench.qbench.find_node())
        return t, repo, testtool_mcp.Server(repo, cmd, t.lang, log)

    def call(self, srv, mid=1):
        return srv.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": {"name": "run_tests", "arguments": {}}})

    def test_handshake_and_one_tool(self):
        _, _, srv = self.server()
        init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(init["result"]["serverInfo"]["name"], "forge")
        self.assertIn("tools", init["result"]["capabilities"])
        self.assertIsNone(srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["run_tests"])
        self.assertEqual(tools[0]["inputSchema"]["type"], "object")
        self.assertEqual(srv.handle({"jsonrpc": "2.0", "id": 3, "method": "ping"})["result"], {})
        self.assertEqual(srv.handle({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})["error"]["code"], -32601)
        bad = srv.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "rm_rf"}})
        self.assertIn("error", bad)

    def test_failing_repository_returns_only_the_first_failure(self):
        _, _, srv = self.server("py-wrap")
        text = self.call(srv)["result"]["content"][0]["text"]
        self.assertIn("0 of 3 tests pass, 3 fail.", text)
        self.assertIn("First failing test:", text)
        self.assertIn("2 more failing tests are not shown", text)
        self.assertNotIn("unittest/case.py", text)
        self.assertLess(len(text), 1800)
        names = [ln for ln in text.splitlines() if ln.startswith(("ERROR:", "FAIL:"))]
        self.assertEqual(len(set(names)), 1, "only one failing test may be shown")

    def test_passing_repository_says_so(self):
        t, repo, srv = self.server("py-wrap")
        shutil.copytree(t.solution_dir, repo, dirs_exist_ok=True)
        res = self.call(srv)["result"]
        self.assertEqual(res["content"][0]["text"], "ALL TESTS PASS (3 tests).")
        self.assertFalse(res["isError"])

    def test_the_tool_never_shows_hidden_tests(self):
        """It runs the test command in the agent's copy, where the hidden files do not exist; a hidden-only failure cannot appear."""
        t, repo, srv = self.server("py-wrap")
        shutil.copytree(t.solution_dir, repo, dirs_exist_ok=True)
        self.assertFalse((repo / t.hidden_tests[0]).exists())
        self.assertNotIn("hidden", self.call(srv)["result"]["content"][0]["text"].lower())

    def test_a_crash_inside_the_tool_is_text_for_the_model(self):
        _, _, srv = self.server()
        srv.cmd = ["/nonexistent/python", "-m", "unittest"]
        res = self.call(srv)["result"]
        self.assertTrue(res["isError"])
        self.assertIn("the test tool failed", res["content"][0]["text"])

    def test_every_call_is_logged(self):
        log = tmpdir(self) / "calls.jsonl"
        t, repo, srv = self.server("py-wrap", log)
        self.call(srv)
        shutil.copytree(t.solution_dir, repo, dirs_exist_ok=True)
        self.call(srv, 2)
        rows = [json.loads(ln) for ln in log.read_text().splitlines()]
        self.assertEqual([(r["n"], r["ok"], r["failed"], r["total"]) for r in rows], [(1, False, 3, 3), (2, True, 0, 3)])
        self.assertEqual(runner.mcp_calls(log), {"calls": 2, "passing": 1, "last_ok": True, "secs": runner.mcp_calls(log)["secs"]})
        self.assertEqual(runner.mcp_calls(tmpdir(self) / "none.jsonl")["calls"], 0)

    @unittest.skipUnless(HAVE_NODE, "node is not installed")
    def test_javascript_failure_block(self):
        _, _, srv = self.server("js-slugify")
        text = self.call(srv)["result"]["content"][0]["text"]
        self.assertIn("First failing test:", text)
        self.assertNotIn("node:internal", text)

    def test_stdio_framing_one_json_object_per_line(self):
        t = BY_ID["py-wrap"]
        repo = taskbench.materialize(t, tmpdir(self) / "repo")
        cmd = [PYTHON, str(runner.MCP_SERVER), "--repo", str(repo), "--test-cmd", json.dumps(taskbench.expand(t.test_cmd, PYTHON)), "--lang", "py"]
        stdin = rpc_lines({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                          {"jsonrpc": "2.0", "method": "notifications/initialized"},
                          {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + b"not json\n"
        out = subprocess.run(cmd, input=stdin, capture_output=True, timeout=60).stdout
        self.assertNotIn(b"\r", out)
        lines = out.split(b"\n")
        self.assertEqual(lines[-1], b"")
        msgs = [json.loads(ln) for ln in lines[:-1]]
        self.assertEqual([m.get("id") for m in msgs], [1, 2, None])
        self.assertEqual(msgs[2]["error"]["code"], -32700)


FAKE_TOOL_AGENT = r'''#!{python}
import json, os, shutil, subprocess, sys, urllib.request
cfg = json.load(open(os.path.join(os.environ["XDG_CONFIG_HOME"], "opencode", "opencode.json")))
base = cfg["provider"]["forge"]["options"]["baseURL"]
body = {{"model": "x", "stream": True, "tools": [{{"type": "function"}}], "messages": [{{"role": "user", "content": "hi"}}]}}
urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={{"Content-Type": "application/json"}})).read()
srv = cfg.get("mcp", {{}}).get("servers", {{}}).get("forge")
seen = {{"mcp_configured": bool(srv)}}
if srv:
    assert srv["type"] == "local" and isinstance(srv["command"], list)
    p = subprocess.Popen(srv["command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)      # OpenCode starts it in the agent's directory
    def rpc(m):
        p.stdin.write((json.dumps(m) + "\n").encode()); p.stdin.flush()
        return json.loads(p.stdout.readline())
    rpc({{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {{"protocolVersion": "2024-11-05"}}}})
    p.stdin.write(b'{{"jsonrpc":"2.0","method":"notifications/initialized"}}\n'); p.stdin.flush()
    call = {{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {{"name": "run_tests", "arguments": {{}}}}}}
    seen["first"] = rpc(call)["result"]["content"][0]["text"]
    shutil.copytree(os.environ["FAKE_SOLUTION"], ".", dirs_exist_ok=True)
    call["id"] = 3
    seen["second"] = rpc(call)["result"]["content"][0]["text"]
else:
    shutil.copytree(os.environ["FAKE_SOLUTION"], ".", dirs_exist_ok=True)
open(os.environ["FAKE_OUT"], "w").write(json.dumps(seen))
'''


@unittest.skipIf(sys.platform.startswith("win"), "the fake agent is a POSIX script")
class ArmsEndToEnd(unittest.TestCase):
    def run_arm(self, arm, extra=()):
        d = tmpdir(self)
        exe = d / "fake_opencode.py"
        exe.write_text(FAKE_TOOL_AGENT.format(python=PYTHON))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        os.environ.update(FAKE_SOLUTION=str(BY_ID["py-wrap"].solution_dir), FAKE_OUT=str(d / "seen.json"))
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("FAKE_SOLUTION", "FAKE_OUT")])
        oracle = Oracle()
        self.addCleanup(oracle.close)
        runner.main(["--arm", arm, "--tasks", "py-wrap", "--endpoint", oracle.url, "--out-dir", str(d / "res"), "--work-dir", str(d / "w"),
                     "--shuffle-seed", "0", "--opencode", str(exe), "--python", PYTHON, *extra])
        rows = [json.loads(ln) for ln in (d / "res" / f"{arm}.jsonl").read_text().splitlines()]
        return rows[0], rows[1], json.loads((d / "seen.json").read_text())

    def test_o30_runs_under_its_own_limit_without_a_tool(self):
        meta, rec, seen = self.run_arm("O30")
        self.assertEqual((meta["time_limit_s"], rec["limit_s"]), (1800.0, 1800.0))
        self.assertTrue(rec["success"], rec)
        self.assertFalse(seen["mcp_configured"])
        self.assertNotIn("tool_runs", rec)
        self.assertEqual(meta["serve_flags"], [])
        self.assertFalse(meta["mcp_tests"])

    def test_the_time_limit_flag_still_overrides_the_arm(self):
        meta, rec, _ = self.run_arm("O30", ["--time-limit", "600"])
        self.assertEqual((meta["time_limit_s"], rec["limit_s"]), (600.0, 600.0))

    def test_o30_tool_gets_the_mcp_server_and_the_calls_are_recorded(self):
        meta, rec, seen = self.run_arm("O30-tool")
        self.assertTrue(seen["mcp_configured"])
        self.assertIn("First failing test:", seen["first"])
        self.assertEqual(seen["second"], "ALL TESTS PASS (3 tests).")
        self.assertEqual({k: rec["tool_runs"][k] for k in ("calls", "passing", "last_ok")}, {"calls": 2, "passing": 1, "last_ok": True})
        self.assertTrue(rec["success"], rec)
        self.assertTrue(meta["mcp_tests"])
        self.assertEqual(rec["limit_s"], 1800.0)

    def test_phase4_arm_o_keeps_fifteen_minutes(self):
        self.assertEqual(runner.ARMS["O"].get("limit_s", runner.DEFAULT_LIMIT_S), 900.0)
        self.assertEqual([a for a, v in runner.ARMS.items() if v.get("limit_s") == 1800.0], ["O30", "O30-ngram", "O30-tool"])


class McpConfig(unittest.TestCase):
    def test_config_block_for_the_tool_arm(self):
        t = BY_ID["py-wrap"]
        a = type("A", (), dict(python="C:\\Python312\\python.exe", node=None))()
        block = runner.test_tool_mcp(t, Path("R"), a, Path("W/calls.jsonl"))
        self.assertEqual(list(block), ["servers"])               # v2.2.5: mcp.servers.<name>, as `opencode mcp add` writes it
        s = block["servers"]["forge"]
        self.assertEqual(sorted(s), ["codemode", "command", "type"])
        self.assertEqual(s["type"], "local")
        self.assertIs(s["codemode"], False)                       # a native tool for the model, not JavaScript behind `execute`
        cmd = s["command"]
        self.assertEqual(cmd[0], "C:\\Python312\\python.exe")
        self.assertEqual(Path(cmd[1]).name, "testtool_mcp.py")
        self.assertEqual(json.loads(cmd[cmd.index("--test-cmd") + 1]), ["C:\\Python312\\python.exe", "-m", "unittest"])
        self.assertEqual(cmd[cmd.index("--repo") + 1], "R")
        cfg = runner.opencode_config("http://127.0.0.1:9/v1", mcp=block)
        self.assertEqual(cfg["mcp"], block)
        self.assertNotIn("mcp", runner.opencode_config("http://127.0.0.1:9/v1"))
        json.dumps(cfg)


class FakeUpstream:
    """A llama-server that streams one answer with the `timings` object of the real server in its last chunk."""

    def __init__(self, timings):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                outer.body = json.loads(self.rfile.read(n))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                ev = {"choices": [{"delta": {"content": "x"}}]}
                fin = {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 40}, "timings": timings}
                self.wfile.write(f"data: {json.dumps(ev)}\n\ndata: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class MeterTimings(unittest.TestCase):
    def ask(self, timings, n=2):
        up = FakeUpstream(timings)
        self.addCleanup(up.close)
        m = meter.Meter(up.url).start()
        self.addCleanup(m.stop)
        for _ in range(n):
            body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
            urllib.request.urlopen(urllib.request.Request(m.url + "/chat/completions", data=json.dumps(body).encode(),
                                                          headers={"Content-Type": "application/json"})).read()
        return m.totals(), m.records

    def test_generation_speed_comes_from_the_servers_clock(self):
        tot, recs = self.ask({"predicted_n": 40, "predicted_ms": 4000.0, "prompt_n": 100, "prompt_ms": 500.0, "draft_n": 30, "draft_n_accepted": 21})
        self.assertEqual(tot["gen_tok_s"], 10.0)
        self.assertEqual(tot["pp_tok_s"], 200.0)
        self.assertEqual((tot["draft_n"], tot["draft_accepted"]), (60, 42))
        self.assertEqual(recs[0]["pred_n"], 40)

    def test_no_timings_means_no_speed_not_zero(self):
        tot, _ = self.ask({})
        self.assertIsNone(tot["gen_tok_s"])
        self.assertEqual((tot["draft_n"], tot["draft_accepted"]), (0, 0))


def p5_row(task, arm, run, ok, wall, limit=1800.0, timed_out=False, **kw):
    r = fake_run(task, arm, run, ok, wall=wall, reason="ok" if ok else ("timeout" if timed_out else "tests_failed"), passed=5, total=10)
    r.update({"limit_s": limit, "timed_out": timed_out, **kw})
    return r


class Phase5Report(unittest.TestCase):
    def write(self, d, arm, rows, limit):
        with (d / f"{arm}.jsonl").open("w") as f:
            f.write(json.dumps({"type": "meta", "arm": arm, "time_limit_s": limit}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def make(self):
        p4, p5 = tmpdir(self), tmpdir(self)
        o = [p5_row("a", "O", 1, True, 120, 900), p5_row("b", "O", 1, False, 900, 900, True), p5_row("b", "O", 2, True, 450, 900),
             p5_row("c", "O", 1, False, 900, 900, True), p5_row("d", "O", 1, False, 600, 900)]
        o30 = [p5_row("a", "O30", 1, True, 130), p5_row("b", "O30", 1, True, 1000), p5_row("c", "O30", 1, False, 1800, timed_out=True),
               p5_row("d", "O30", 1, True, 300)]
        self.write(p4, "O", o, 900)
        self.write(p5, "O30", o30, 1800)
        tool = [p5_row("a", "O30-tool", 1, True, 100, tool_runs={"calls": 3, "passing": 1, "last_ok": True, "secs": 1.0}),
                p5_row("b", "O30-tool", 1, False, 1800, timed_out=True, tool_runs={"calls": 0, "passing": 0, "last_ok": False, "secs": 0})]
        self.write(p5, "O30-tool", tool, 1800)
        return p4, p5

    def test_each_arm_is_judged_under_its_own_limit(self):
        _, p5 = self.make()
        rows = report.load_runs(p5)[1]["O30"]
        s = report.summarize(rows, report.arm_limit(rows, {}))
        self.assertEqual(s["limit_s"], 1800.0)
        self.assertEqual((s["successes"], s["tasks"]), (3, 4))
        self.assertAlmostEqual(s["median_min"], (300 + 1000) / 2 / 60)

    def test_rescore_at_fifteen_minutes_and_passes_after(self):
        _, p5 = self.make()
        rows = report.load_runs(p5)[1]["O30"]
        s = report.summarize(rows, 1800.0)
        self.assertEqual((s["successes"], s["first_ok_15"], s["first_after_15"]), (3, 2, 1))      # `b` passed at 1000 s: after minute 15
        self.assertAlmostEqual(s["median_min_15cap"], 10.0)         # 2.2, 5, 15, 15 minutes
        self.assertAlmostEqual(s["rate_15"], 2 / 4)

    def test_median_of_a_long_limit_is_reported_as_measured(self):
        rows = [p5_row(f"t{i}", "O30", 1, i < 4, 1500 if i < 4 else 1800, timed_out=i >= 4) for i in range(8)]
        s = report.summarize(rows, 1800.0)
        self.assertEqual(s["median_min"], 27.5)                    # all slow: the 15-minute time criterion fails, and says so
        self.assertFalse(s["usable"])
        self.assertFalse(s["usable_time"])
        self.assertEqual(s["median_min_15cap"], 15.0)

    def test_timeout_table_lists_every_phase4_timeout_with_the_minute(self):
        p4, p5 = self.make()
        _, r4 = report.load_runs(p4)
        _, r5 = report.load_runs(p5)
        lines = report.timeout_table(r4["O"], {"O30": r5["O30"], "O30-tool": r5["O30-tool"]})
        text = "\n".join(lines)
        self.assertIn("| b |", text)
        self.assertIn("| c |", text)
        self.assertNotIn("| a |", text)            # never timed out in phase 4
        self.assertIn("ok 10/10 16.7m *", text)    # b at O30: 1000 s, passed after minute 15
        self.assertIn("timeout 5/10 30.0m", text)
        self.assertIn("run 1: timeout 5/10 15.0m; run 2: ok 10/10 7.5m", text)

    def test_full_phase5_report(self):
        p4, p5 = self.make()
        text = report.build_report(p5, phase4=p4)
        self.assertIn("# Phase 5 report", text)
        self.assertIn("O 15 min", text)
        self.assertIn("O30 30 min", text)
        self.assertIn("| O30 | 4 | 3/4 |", text)
        self.assertIn("| O30-tool | 2 | 1/2 |", text)
        self.assertIn("success within 15 min", text)
        self.assertIn("## Tasks that timed out at 15 minutes", text)
        self.assertIn("## The test tool in O30-tool", text)
        self.assertIn("runs that called it at least once: 1", text)
        self.assertIn("| O30 | O |", text)                     # paired against phase 4's O
        self.assertIn("| O30-tool | O30 |", text)              # and the tool arm against O30

    def test_speed_columns(self):
        rows = [p5_row("a", "O30-ngram", 1, True, 100, gen_tok_s=11.0, pp_tok_s=150.0, draft_n=100, draft_accepted=60, completion_tokens=900, model_seconds=100.0),
                p5_row("b", "O30-ngram", 1, False, 100, gen_tok_s=13.0, pp_tok_s=170.0, draft_n=100, draft_accepted=80, completion_tokens=900, model_seconds=100.0)]
        s = report.summarize(rows, 1800.0)
        self.assertEqual(s["gen_tok_s_median"], 12.0)
        self.assertEqual(s["draft_rate"], 0.7)
        self.assertEqual(s["eff_tok_s_median"], 9.0)

    def test_phase4_reports_still_render_without_the_new_fields(self):
        d = tmpdir(self)
        self.write(d, "O", [fake_run(f"t{i}", "O", 1, i < 2) for i in range(4)], 900)
        text = report.build_report(d)
        self.assertIn("# Phase 4 report", text)
        self.assertNotIn("success within 15 min", text)


class Phase5Grid(unittest.TestCase):
    def test_new_arms_are_valid_stages(self):
        self.assertEqual(grid.parse_stages("O30@1;O30-ngram@1; O30-tool@1;O30@2"),
                         [(["O30"], 1), (["O30-ngram"], 1), (["O30-tool"], 1), (["O30"], 2)])

    def test_arm_order_in_the_report_is_the_phase_order(self):
        self.assertEqual(report.ORDER[-3:], ["O30", "O30-ngram", "O30-tool"])


if __name__ == "__main__":
    unittest.main()

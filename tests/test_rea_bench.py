"""Phase 6 part C (03-agent/rea): the ground truth against the sources and the committed binary, scoring, REA output views, the MCP server,
arm R end to end against a fake model server, and arm O end to end against a fake OpenCode and a fake REA (the real ones are host-only)."""
import argparse
import json
import re
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .helpers import ROOT

REA_DIR = ROOT / "03-agent" / "rea"
sys.path.insert(0, str(REA_DIR))
import rea_bench  # noqa: E402
import rea_mcp  # noqa: E402
import reacli  # noqa: E402

FAKE_REA = json.dumps([sys.executable, str(ROOT / "tests" / "fake_rea.py")])
FAKE_OPENCODE = json.dumps([sys.executable, str(ROOT / "tests" / "fake_opencode.py")])
DATA = rea_bench.load_tasks()
TASKS = {t["id"]: t for t in DATA["tasks"]}
BINARY = REA_DIR / "bin" / "bench.exe"


def pe_facts(data: bytes) -> dict:
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe:pe + 4] == b"PE\0\0"
    machine, nsec = struct.unpack_from("<HH", data, pe + 4)
    opt = pe + 24
    rva, size = struct.unpack_from("<II", data, opt + 112 + 8)
    return {"machine": machine, "sections": nsec, "import_rva": rva, "import_size": size}


class GroundTruth(unittest.TestCase):
    def test_fifteen_tasks_with_unique_ids_and_known_scoring_types(self):
        self.assertEqual(len(TASKS), 15)
        self.assertEqual(len(DATA["tasks"]), 15)
        for t in DATA["tasks"]:
            self.assertIn(t["scoring"]["type"], ("hex", "exact", "yesno", "name", "choice"), t["id"])
            self.assertIn(t["file"], ("bench", "gguf"))
            self.assertTrue(t["recipe"] and t["truth"], t["id"])
            if t["scoring"]["type"] == "choice":
                self.assertIn(t["scoring"]["expected"], t["scoring"]["choices"], t["id"])

    def test_the_committed_binary_is_the_one_the_ground_truth_names(self):
        self.assertEqual(rea_bench.qbench_sha(BINARY), DATA["binary"]["sha256"])

    def test_constants_names_and_the_password_are_in_the_source(self):
        src = (REA_DIR / "src" / "bench.c").read_text(encoding="utf-8").lower()
        for tid in ("crc-poly", "mix-multiplier", "xor-key", "find-golden"):
            exp = TASKS[tid]["scoring"].get("expected")
            if isinstance(exp, int):
                self.assertIn(f"0x{exp:x}", src, tid)
        self.assertIn("0x9e3779b9", src)
        self.assertIn(f'"{TASKS["password"]["scoring"]["expected"]}"', (REA_DIR / "src" / "bench.c").read_text(encoding="utf-8"))
        for tid in ("find-strcmp", "find-xor-loop", "find-golden"):
            self.assertRegex(src, rf"\b{TASKS[tid]['scoring']['expected']}\(")

    def test_header_and_import_answers_match_the_binary(self):
        data = BINARY.read_bytes()
        facts = pe_facts(data)
        self.assertEqual(facts["machine"], 0x8664)
        self.assertEqual(TASKS["pe-import-rva"]["scoring"]["expected"], facts["import_rva"])
        for needle in (b"WS2_32.dll", b"WSAStartup", b"socket", b"fwrite", b"strcmp", b"fopen"):
            self.assertIn(needle, data)
        self.assertEqual(TASKS["imports-dll"]["scoring"]["expected"], "ws2_32")
        self.assertEqual(TASKS["imports-network"]["scoring"]["expected"], "yes")

    def test_gguf_answers_match_the_header_REA_printed_on_the_host(self):
        nr = reacli.normalized((ROOT / "results" / "rea" / "gguf.json").read_text(encoding="utf-8"))
        meta = {m["key"]: m["value"] for m in nr["metadata"]}
        self.assertEqual(meta["general.architecture"], TASKS["gguf-arch"]["scoring"]["expected"])
        self.assertEqual(int(meta["qwen35.block_count"]), TASKS["gguf-blocks"]["scoring"]["expected"])
        self.assertEqual(nr["artifact"]["header_sha256"], DATA["gguf"]["header_sha256"])


class Scoring(unittest.TestCase):
    def ok(self, tid, answer):
        return rea_bench.score(TASKS[tid], answer)[0]

    def test_hex_accepts_prefixes_case_and_decimal_for_the_same_number(self):
        for a in ("0xEDB88320", "0xedb88320.", "`edb88320`", "0xEDB88320 (reflected CRC-32 polynomial)", "3988292384"):
            self.assertTrue(self.ok("crc-poly", a), a)
        for a in ("0x04C11DB7", "EDB88321", "crc", ""):
            self.assertFalse(self.ok("crc-poly", a), a)
        self.assertTrue(self.ok("xor-key", "0x5A"))
        self.assertTrue(self.ok("xor-key", "5a"))
        self.assertFalse(self.ok("xor-key", "0xA5"))
        self.assertTrue(self.ok("pe-import-rva", "0x18000"))
        self.assertTrue(self.ok("pe-import-rva", "98304"))

    def test_decimal_only_does_not_read_hex_letters(self):
        self.assertTrue(self.ok("gguf-blocks", "33"))
        self.assertTrue(self.ok("gguf-blocks", "block_count = 33"))
        self.assertFalse(self.ok("gguf-blocks", "0x21"))
        self.assertFalse(self.ok("gguf-blocks", "32"))

    def test_choice_needs_exactly_the_right_one(self):
        self.assertTrue(self.ok("name-popcount", "popcount"))
        self.assertTrue(self.ok("name-popcount", "It computes the popcount."))
        self.assertFalse(self.ok("name-popcount", "popcount or parity"))
        self.assertFalse(self.ok("name-popcount", "parity"))
        self.assertTrue(self.ok("imports-dll", "WS2_32.dll"))
        self.assertTrue(self.ok("imports-dll", "`ws2_32`"))
        self.assertFalse(self.ok("imports-dll", "KERNEL32.dll"))
        self.assertTrue(self.ok("imports-fwrite", "fwrite"))
        self.assertFalse(self.ok("imports-fwrite", "fprintf"))
        self.assertFalse(self.ok("imports-fwrite", "fwrite and fopen"))
        self.assertTrue(self.ok("gguf-arch", "qwen35"))
        self.assertFalse(self.ok("gguf-arch", "qwen3"), "qwen3 is a different architecture from qwen35")

    def test_names_exact_strings_and_yes_no(self):
        self.assertTrue(self.ok("find-strcmp", "rb_login"))
        self.assertTrue(self.ok("find-strcmp", "`rb_login`."))
        self.assertFalse(self.ok("find-strcmp", "rb_login and rb_crc"))
        self.assertFalse(self.ok("find-strcmp", "main"))
        self.assertTrue(self.ok("password", '"tr0ub4dor&3"'))
        self.assertFalse(self.ok("password", "tr0ub4dor"))
        self.assertTrue(self.ok("imports-network", "Yes, it imports WSAStartup"))
        self.assertFalse(self.ok("imports-network", "no"))

    def test_no_answer_line_is_a_failure(self):
        self.assertEqual(rea_bench.score(TASKS["xor-key"], None), (False, "no ANSWER line"))

    def test_final_answer_takes_the_last_line_and_forgives_markdown(self):
        self.assertEqual(reacli.final_answer("thinking\nANSWER: 0x1\nmore\nANSWER: 0x5a"), "0x5a")
        self.assertEqual(reacli.final_answer("**ANSWER:** rb_login"), "rb_login")
        self.assertEqual(reacli.final_answer("> answer: `fwrite`"), "fwrite")
        self.assertIsNone(reacli.final_answer("the answer is 5"))


class Views(unittest.TestCase):
    def test_imports_view_lists_dlls_with_their_functions(self):
        nr = reacli.normalized((ROOT / "results" / "rea" / "peimports.json").read_text(encoding="utf-8"))
        text = reacli.view_imports(nr)
        self.assertIn("llama-server-impl.dll:", text)
        self.assertIn("KERNEL32.dll:", text)
        self.assertIn("import directory: rva 8672 (0x21e0)", text)

    def test_model_view_has_the_metadata_and_cuts_arrays(self):
        nr = reacli.normalized((ROOT / "results" / "rea" / "gguf.json").read_text(encoding="utf-8"))
        text = reacli.view_model(nr)
        self.assertIn("general.architecture = qwen35", text)
        self.assertIn("qwen35.block_count = 33", text)
        self.assertLessEqual(len(text), reacli.MAX_CHARS + 80)

    def test_decompile_view_is_the_pseudocode(self):
        nr = reacli.normalized((ROOT / "results" / "rea" / "decompile-forge_checksum.json").read_text(encoding="utf-8"))
        self.assertIsInstance(nr, (str, dict))
        self.assertTrue(reacli.view_decompile(nr))
        self.assertEqual(reacli.view_decompile("x" * 7000).count("cut"), 1)

    def test_fake_rea_through_the_wrapper(self):
        rea = reacli.Rea(json.loads(FAKE_REA))
        ok = rea.decompile("bench.exe", "rb_crc")
        self.assertTrue(ok.ok)
        self.assertIn("0xedb88320", ok.text)
        bad = rea.decompile("bench.exe", "nope")
        self.assertFalse(bad.ok)
        self.assertIn("procedure not found", bad.text)
        everything = rea.decompile_all("bench.exe", "rb_")
        self.assertTrue(everything.ok)
        self.assertIn("rb_login", everything.text)
        self.assertNotIn("manifest", everything.text)
        self.assertIn("KERNEL32", rea.imports("x").text)
        self.assertIn("block_count", rea.model_header("x").text)


class Mcp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "calls.jsonl"
        self.server = rea_mcp.Server(reacli.Rea(json.loads(FAKE_REA)), self.log)
        self.exe = Path(self.tmp.name) / "x.exe"
        self.exe.write_bytes(b"MZ")

    def tearDown(self):
        self.tmp.cleanup()

    def calls(self):
        return [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()]

    def test_five_tools_and_schemas(self):
        listed = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
        self.assertEqual([t["name"] for t in listed], ["decompile", "decompile_all", "search", "imports", "model_header"])
        self.assertEqual({"rea_" + t["name"] for t in listed}, rea_bench.KNOWN_TOOLS)
        self.assertLess(len(json.dumps(listed)), 4000, "five tool schemas must stay small for a 16K context")

    def test_valid_call_is_logged_as_valid_and_ok(self):
        text, err = self.server.call_tool("decompile", {"binary": str(self.exe), "procedure": "rb_crc"})
        self.assertFalse(err)
        self.assertIn("0xedb88320", text)
        c = self.calls()[0]
        self.assertEqual((c["tool"], c["valid"], c["ok"]), ("decompile", True, True))

    def test_bad_calls_are_invalid_and_never_reach_rea(self):
        for name, args in (("decompile", {"binary": str(self.exe)}), ("decompile", {"binary": "C:/nope.exe", "procedure": "f"}),
                           ("search", {"binary": str(self.exe), "pattern": "a", "kind": "everything"}), ("imports", {"binary": str(self.exe), "extra": "1"}),
                           ("disassemble", {"binary": str(self.exe)}), ("imports", "not an object"), ("imports", {"binary": 5})):
            text, err = self.server.call_tool(name, args)
            self.assertTrue(err, name)
            self.assertTrue(text.startswith("invalid call:"), text)
        self.assertEqual([c["valid"] for c in self.calls()], [False] * 7)

    def test_a_rea_failure_is_valid_but_not_ok(self):
        text, err = self.server.call_tool("decompile", {"binary": str(self.exe), "procedure": "nope"})
        self.assertTrue(err)
        self.assertIn("REA failed", text)
        c = self.calls()[0]
        self.assertEqual((c["valid"], c["ok"]), (True, False))

    def test_json_rpc_basics(self):
        init = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})["result"]
        self.assertEqual((init["protocolVersion"], init["serverInfo"]["name"]), ("2025-03-26", "rea"))
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.server.handle({"jsonrpc": "2.0", "id": 9, "method": "nope"})["error"]["code"], -32601)


class FakeModel:
    """/v1/chat/completions that answers with a fixed text and records the requests."""

    def __init__(self, text):
        self.text, self.requests = text, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": outer.text}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 300, "completion_tokens": 20}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)

    def __enter__(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def namespace(tmp, **kw):
    base = dict(endpoint="http://127.0.0.1:9/v1", seed=1, request_timeout=20, rea_root=Path(tmp), rea_cmd=FAKE_REA, rea_timeout=60, python=sys.executable,
                work_dir=str(Path(tmp) / "work"), keep_work=False, time_limit=60, opencode=FAKE_OPENCODE)
    base.update(kw)
    return argparse.Namespace(**base)


class ArmR(unittest.TestCase):
    def test_the_harness_runs_the_recipe_and_the_model_reads_the_output(self):
        with tempfile.TemporaryDirectory() as tmp, FakeModel("The polynomial is the reflected one.\nANSWER: 0xEDB88320") as srv:
            a = namespace(tmp, endpoint=srv.endpoint)
            rec = rea_bench.run_arm_r(TASKS["crc-poly"], reacli.Rea(json.loads(FAKE_REA)), {"bench": "bench.exe", "gguf": "m.gguf"}, a, 1)
            self.assertTrue(rec["success"], rec)
            self.assertEqual((rec["tool_calls"], rec["invalid_calls"], rec["tool_errors"], rec["steps"]), (1, 0, 0, 1))
            self.assertEqual(rec["recipe"], ["decompile rb_crc"])
            user = srv.requests[0]["messages"][1]["content"]
            self.assertIn("REA `decompile rb_crc`", user)
            self.assertIn("0xedb88320", user)
            self.assertEqual((srv.requests[0]["temperature"], srv.requests[0]["chat_template_kwargs"]), (0.0, {"enable_thinking": False}))

    def test_a_wrong_or_missing_answer_fails_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text, why in (("ANSWER: 0x04C11DB7", "got"), ("I think it is CRC-32.", "no ANSWER line")):
                with FakeModel(text) as srv:
                    rec = rea_bench.run_arm_r(TASKS["crc-poly"], reacli.Rea(json.loads(FAKE_REA)), {"bench": "b.exe", "gguf": "m"}, namespace(tmp, endpoint=srv.endpoint), 1)
                self.assertFalse(rec["success"])
                self.assertTrue(rec["why"].startswith(why), rec["why"])

    def test_a_rea_error_is_counted_and_the_model_still_answers(self):
        task = {**TASKS["crc-poly"], "recipe": [{"call": "decompile", "arg": "nope"}]}
        with tempfile.TemporaryDirectory() as tmp, FakeModel("ANSWER: unknown") as srv:
            rec = rea_bench.run_arm_r(task, reacli.Rea(json.loads(FAKE_REA)), {"bench": "b.exe", "gguf": "m"}, namespace(tmp, endpoint=srv.endpoint), 1)
        self.assertEqual((rec["tool_errors"], rec["success"]), (1, False))

    def test_every_recipe_names_a_known_call_and_find_tasks_get_all_functions(self):
        for t in DATA["tasks"]:
            for step in t["recipe"]:
                self.assertIn(step["call"], ("decompile", "decompile_all", "imports", "model_header"), t["id"])
        self.assertEqual(TASKS["find-strcmp"]["recipe"], [{"call": "decompile_all", "arg": "rb_"}])


class ArmO(unittest.TestCase):
    def test_config_reaches_the_mcp_server_and_good_calls_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "bench.exe"
            exe.write_bytes(b"MZ")
            import os
            os.environ["FAKE_OPENCODE_SCRIPT"] = json.dumps({"calls": [["decompile", {"binary": str(exe), "procedure": "rb_crc"}]], "final": "Found it.\nANSWER: 0xedb88320"})
            try:
                rec = rea_bench.run_arm_o(TASKS["crc-poly"], {"bench": str(exe), "gguf": "m"}, namespace(tmp), 1)
            finally:
                del os.environ["FAKE_OPENCODE_SCRIPT"]
        self.assertTrue(rec["success"], rec)
        self.assertEqual((rec["tool_calls"], rec["rea_calls"], rec["invalid_calls"], rec["tool_errors"]), (1, 1, 0, 0))
        self.assertEqual(rec["rea_tools_used"], ["decompile"])
        self.assertFalse(rec["timed_out"])

    def test_invalid_calls_unknown_tools_and_rea_errors_are_counted_apart(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "bench.exe"
            exe.write_bytes(b"MZ")
            os.environ["FAKE_OPENCODE_SCRIPT"] = json.dumps({
                "calls": [["decompile", {"binary": str(exe)}], ["decompile", {"binary": str(exe), "procedure": "nope"}], ["imports", {"binary": str(exe)}]],
                "unknown": ["bash"], "final": "ANSWER: yes"})
            try:
                rec = rea_bench.run_arm_o(TASKS["imports-network"], {"bench": str(exe), "gguf": "m"}, namespace(tmp), 1)
            finally:
                del os.environ["FAKE_OPENCODE_SCRIPT"]
        self.assertEqual((rec["invalid_args"], rec["unknown_tool_calls"], rec["invalid_calls"], rec["tool_errors"]), (1, 1, 2, 1))
        self.assertEqual(rec["builtin_calls"], 1)
        self.assertTrue(rec["success"])

    def test_events_parser(self):
        log = "\n".join([json.dumps({"type": "text", "part": {"type": "text", "text": "first"}}),
                         json.dumps({"type": "tool_use", "part": {"type": "tool", "tool": "rea_imports", "state": {"status": "completed"}}}),
                         "not json", json.dumps({"type": "text", "part": {"type": "text", "text": "last ANSWER: 1"}})])
        ev = rea_bench.parse_events(log)
        self.assertEqual(ev["final_text"], "last ANSWER: 1")
        self.assertEqual(ev["tools"], [{"tool": "rea_imports", "status": "completed"}])


class Report(unittest.TestCase):
    def test_table_and_paired_test(self):
        def run(arm, task, ok, wall=60.0, inv=0, cat="constant"):
            return {"type": "run", "arm": arm, "task": task, "run": 1, "success": ok, "wall_s": wall, "invalid_calls": inv, "tool_errors": 0, "tool_calls": 2,
                    "tool_s": 30.0, "completion_tokens": 100, "category": cat}
        arms = {"R": [run("R", "a", True), run("R", "b", True), run("R", "c", False)],
                "O": [run("O", "a", True, 300, 2), run("O", "b", False, 600, 1), run("O", "c", False, 600)]}
        text = rea_bench.build_report(arms)
        self.assertIn("2/3 = 67%", text)
        self.assertIn("1/3 = 33%", text)
        self.assertIn("only R 1 ['b'], only O 0 []", text)
        self.assertIn("| constant | 1/3 | 2/3 |", text)


if __name__ == "__main__":
    unittest.main()

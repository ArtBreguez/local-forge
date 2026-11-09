"""The phase 3 quality harness (03-agent/quality): every problem against its reference and its broken solution, answer extraction,
test execution (including timeouts), statistics, the runner against a fake OpenAI-compatible server, the report and the grid driver."""
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .helpers import ROOT

QUALITY = ROOT / "03-agent" / "quality"
sys.path.insert(0, str(QUALITY))
import qbench  # noqa: E402
import qgrid  # noqa: E402
import qreport  # noqa: E402
import run_quality  # noqa: E402

HAVE_NODE = qbench.find_node() is not None
PROBLEMS = qbench.load_problems()
BY_ID = {p.id: p for p in PROBLEMS}


class Manifest(unittest.TestCase):
    def test_forty_problems_half_python_half_javascript(self):
        self.assertEqual(len(PROBLEMS), 40)
        self.assertEqual(sum(p.lang == "py" for p in PROBLEMS), 20)
        self.assertEqual(sum(p.lang == "js" for p in PROBLEMS), 20)
        self.assertEqual(len({p.id for p in PROBLEMS}), 40)

    def test_every_problem_has_its_files_and_names_its_entry_points_in_the_prompt(self):
        for p in PROBLEMS:
            with self.subTest(problem=p.id):
                self.assertTrue(p.id.startswith(p.lang + "_"))
                self.assertTrue(p.tests_path.is_file())
                self.assertTrue(p.ref_path("good").is_file())
                self.assertTrue(p.ref_path("bad").is_file())
                self.assertGreater(len(p.prompt), 200)
                for name in p.entries:
                    self.assertIn(name, p.prompt, "the model can only know the names the tests import from the prompt")
                self.assertIn(p.difficulty, ("easy", "medium", "hard"))
                self.assertTrue(256 <= p.max_tokens <= 2048)

    def test_directories_match_the_manifest(self):
        on_disk = sorted(d.name for d in (QUALITY / "problems").iterdir() if d.is_dir())
        self.assertEqual(on_disk, sorted(p.id for p in PROBLEMS))
        self.assertEqual(sorted(d.name for d in (QUALITY / "_ref").iterdir() if d.is_dir()), on_disk)

    def test_the_prompts_do_not_leak_the_tests(self):
        for p in PROBLEMS:
            self.assertNotIn("unittest", p.prompt)
            self.assertNotIn("node:test", p.prompt)

    def test_prompt_variants_differ_only_by_the_python_indentation_sentence(self):
        py = next(p for p in PROBLEMS if p.lang == "py")
        js = next(p for p in PROBLEMS if p.lang == "js")
        v1, v2 = qbench.build_messages(py, "v1"), qbench.build_messages(py, "v2")
        self.assertEqual(v1[0], v2[0])
        self.assertEqual(v2[1]["content"].replace(" Indent with 4 spaces per level.", ""), v1[1]["content"])
        self.assertEqual(qbench.build_messages(js, "v1"), qbench.build_messages(js, "v2"))
        v3 = qbench.build_messages(py, "v3")
        self.assertEqual(v3[1], v1[1])
        self.assertTrue(v3[0]["content"].startswith(v1[0]["content"]))
        self.assertIn("    if low > high:", v3[0]["content"])           # the example uses 4-space indentation

    def test_messages_are_identical_for_every_arm(self):
        p = PROBLEMS[0]
        self.assertEqual(qbench.build_messages(p), qbench.build_messages(p))
        msgs = qbench.build_messages(p)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn(p.prompt, msgs[1]["content"])


@unittest.skipUnless(HAVE_NODE, "node is needed for the JavaScript problems")
class ReferenceSolutions(unittest.TestCase):
    """A problem is only usable if a correct solution passes ALL hidden tests and a wrong one fails at least one."""

    def test_every_good_reference_passes_and_every_broken_one_fails(self):
        for p in PROBLEMS:
            with self.subTest(problem=p.id):
                good = qbench.run_tests(p, p.ref_path("good").read_text(encoding="utf-8"), timeout=120)
                self.assertTrue(good.passed, f"{p.id} good reference failed: {good.kind} {good.detail}")
                self.assertGreaterEqual(good.tests_total or 0, 4)
                bad = qbench.run_tests(p, p.ref_path("bad").read_text(encoding="utf-8"), timeout=120)
                self.assertFalse(bad.passed, f"{p.id} broken reference passed all tests")
                self.assertEqual(bad.kind, "test_fail", f"{p.id}: {bad.kind} {bad.detail}")

    def test_an_empty_solution_fails_every_problem(self):
        for p in PROBLEMS:
            with self.subTest(problem=p.id):
                out = qbench.run_tests(p, "", timeout=60)
                self.assertFalse(out.passed)


class Extraction(unittest.TestCase):
    def test_fenced_python(self):
        code, how = qbench.extract_code("Here you go:\n```python\ndef f():\n    return 1\n```\nDone.", "py")
        self.assertEqual(code, "def f():\n    return 1\n")
        self.assertEqual(how, "fenced block")

    def test_first_block_with_a_matching_tag_wins(self):
        text = "```bash\npip install x\n```\n```python\ndef f(): pass\n```\n```python\nprint(f())\n```"
        self.assertEqual(qbench.extract_code(text, "py")[0], "def f(): pass\n")

    def test_javascript_tags_and_untagged_blocks(self):
        self.assertEqual(qbench.extract_code("```js\nconst a = 1;\n```", "js")[0], "const a = 1;\n")
        self.assertEqual(qbench.extract_code("```javascript\nconst a = 1;\n```", "js")[0], "const a = 1;\n")
        self.assertEqual(qbench.extract_code("```\nconst a = 1;\n```", "js")[0], "const a = 1;\n")

    def test_a_block_with_another_language_is_used_when_nothing_matches(self):
        code, how = qbench.extract_code("```ruby\ndef f; end\n```", "py")
        self.assertEqual(code, "def f; end\n")
        self.assertIn("did not match", how)

    def test_unterminated_fence_is_the_cut_off_answer(self):
        code, how = qbench.extract_code("```python\ndef f():\n    return [1,", "py")
        self.assertEqual(how, "unterminated fence")
        self.assertTrue(code.startswith("def f():"))

    def test_whole_answer_when_it_looks_like_code(self):
        code, how = qbench.extract_code("def f():\n    return 1", "py")
        self.assertEqual((code, how), ("def f():\n    return 1\n", "whole answer"))
        self.assertIsNotNone(qbench.extract_code("const f = (x) => x * 2;", "js")[0])

    def test_prose_is_no_code(self):
        self.assertIsNone(qbench.extract_code("I would use a binary search here.", "py")[0])
        self.assertIsNone(qbench.extract_code("   \n", "js")[0])
        self.assertIsNone(qbench.extract_code("", "js")[0])

    def test_thinking_is_removed_and_never_mined_for_code(self):
        text = "Let me think. ```python\nwrong = 1\n```\n</think>\n\n```python\ndef f(): return 2\n```"
        self.assertEqual(qbench.extract_code(text, "py", thinking=True)[0], "def f(): return 2\n")
        self.assertEqual(qbench.extract_code("<think>x ```python\nwrong\n```</think>```python\ndef g(): pass\n```", "py")[0], "def g(): pass\n")

    def test_unfinished_thinking_has_no_answer(self):
        code, how = qbench.extract_code("Let me think about it ```python\ndef f(): pass\n``` and then", "py", thinking=True)
        self.assertEqual((code, how), (None, "thinking did not finish"))
        self.assertIsNone(qbench.extract_code("<think>still going", "py")[0])

    def test_server_side_split_reasoning_means_content_is_the_answer(self):
        self.assertEqual(qbench.extract_code("```python\ndef f(): pass\n```", "py", thinking=True, reasoning_split=True)[0], "def f(): pass\n")
        self.assertIsNone(qbench.extract_code("", "py", thinking=True, reasoning_split=True)[0])

    def test_javascript_module_syntax_is_forgiven_but_nothing_else(self):
        code, changed = qbench.normalize_js("export function f() { return 1; }\n", ("f",))
        self.assertTrue(changed)
        self.assertIn("function f()", code)
        self.assertNotIn("export function", code)
        self.assertIn("module.exports = { f };", code)
        code, changed = qbench.normalize_js("const f = () => 1;\nexport { f };\n", ("f",))
        self.assertNotIn("export {", code)
        self.assertIn("module.exports = { f };", code)
        code, changed = qbench.normalize_js("export default function f() {}\n", ("f",))
        self.assertNotIn("export", code.replace("module.exports", ""))
        same, changed = qbench.normalize_js("function f() {}\nmodule.exports = { f };\n", ("f",))
        self.assertEqual((same, changed), ("function f() {}\nmodule.exports = { f };\n", False))
        same, changed = qbench.normalize_js("exports.f = () => 1;\n", ("f",))
        self.assertFalse(changed)


class Execution(unittest.TestCase):
    def test_python_counts_failed_test_methods_not_subtests(self):
        # py_01 has 5 test methods; wrong answers fail many subtests inside them but each method counts once
        out = qbench.run_tests(BY_ID["py_01_parse_duration"], "def parse_duration(t):\n    return 0\n")
        self.assertEqual(out.kind, "test_fail")
        self.assertEqual(out.tests_total, 5)
        self.assertLessEqual(out.tests_failed, out.tests_total)
        self.assertGreaterEqual(out.tests_failed, 1)
        self.assertEqual(out.fraction, (out.tests_total - out.tests_failed) / out.tests_total)
        ok = qbench.run_tests(BY_ID["py_01_parse_duration"], BY_ID["py_01_parse_duration"].ref_path("good").read_text())
        self.assertEqual((ok.tests_failed, ok.fraction), (0, 1.0))

    def test_fraction_is_zero_when_nothing_ran(self):
        self.assertEqual(qbench.TestOutcome(False, "syntax", "x").fraction, 0.0)
        self.assertEqual(qbench.TestOutcome(False, "timeout", "x", 5, 0).fraction, 0.0)
        self.assertEqual(qbench.TestOutcome(False, "test_fail", "x", 4, 1).fraction, 0.75)
        self.assertEqual(qbench.TestOutcome(True, None, "", 4, 0).fraction, 1.0)

    def test_python_syntax_error_is_syntax(self):
        out = qbench.run_tests(BY_ID["py_01_parse_duration"], "def parse_duration(t:\n    return 1\n")
        self.assertEqual((out.passed, out.kind), (False, "syntax"))

    def test_python_infinite_loop_is_timeout_and_returns_promptly(self):
        t0 = time.monotonic()
        out = qbench.run_tests(BY_ID["py_01_parse_duration"], "def parse_duration(t):\n    while True:\n        pass\n", timeout=3)
        self.assertEqual((out.passed, out.kind), (False, "timeout"))
        self.assertLess(time.monotonic() - t0, 180)        # not a speed test: on the shared VM a spinning child can delay the parent by tens of seconds
        if not qbench.IS_WINDOWS:
            time.sleep(1)
            import subprocess
            self.assertEqual(subprocess.run(["pgrep", "-f", "qbench-.*test_hidde[n]"], capture_output=True, text=True).stdout.split(), [])

    def test_python_missing_function_is_a_test_failure_with_a_detail(self):
        out = qbench.run_tests(BY_ID["py_01_parse_duration"], "def something_else():\n    pass\n")
        self.assertEqual((out.passed, out.kind), (False, "test_fail"))
        self.assertTrue(out.detail)

    def test_python_reading_stdin_does_not_hang(self):
        out = qbench.run_tests(BY_ID["py_01_parse_duration"], "def parse_duration(t):\n    return int(input())\n", timeout=20)
        self.assertEqual(out.kind, "test_fail")

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_javascript_syntax_error_is_syntax(self):
        out = qbench.run_tests(BY_ID["js_03_chunk"], "function chunk(a, s) {\n  return [;\n}\nmodule.exports = { chunk };\n")
        self.assertEqual((out.passed, out.kind), (False, "syntax"))

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_javascript_infinite_loop_is_timeout_and_the_child_is_killed(self):
        t0 = time.monotonic()
        out = qbench.run_tests(BY_ID["js_03_chunk"], "function chunk(a, s) { while (true) {} }\nmodule.exports = { chunk };\n", timeout=4)
        self.assertEqual((out.passed, out.kind), (False, "timeout"))
        self.assertLess(time.monotonic() - t0, 120)        # generous: a loaded machine took 44 s here
        if not qbench.IS_WINDOWS:                         # no spinning node left behind (`node --test` runs the file in a child process)
            import subprocess
            time.sleep(1)
            left = subprocess.run(["pgrep", "-f", "qbench-.*test_hidde[n]"], capture_output=True, text=True).stdout.split()
            self.assertEqual(left, [])

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_javascript_esm_exports_are_normalised_and_flagged(self):
        p = BY_ID["js_03_chunk"]
        code = p.ref_path("good").read_text(encoding="utf-8").replace("module.exports = { chunk };", "export { chunk };")
        out = qbench.run_tests(p, code)
        self.assertTrue(out.passed, out.detail)
        self.assertTrue(out.normalized)

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_a_parent_package_json_with_type_module_does_not_turn_the_solution_into_an_es_module(self):
        """The Windows host has C:\\Users\\arthu\\package.json with "type": "module"; the temp dir is below it."""
        with tempfile.TemporaryDirectory() as parent:
            (Path(parent) / "package.json").write_text('{"type": "module"}\n', encoding="utf-8")
            with mock.patch.object(tempfile, "tempdir", parent):
                p = BY_ID["js_03_chunk"]
                out = qbench.run_tests(p, p.ref_path("good").read_text(encoding="utf-8"))
            self.assertTrue(out.passed, out.detail)

    def test_missing_node_is_a_clear_error(self):
        with mock.patch.object(qbench, "find_node", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "node not found"):
                qbench.run_tests(BY_ID["js_03_chunk"], "function chunk() {}\nmodule.exports = { chunk };\n")


class Statistics(unittest.TestCase):
    def test_wilson_matches_published_values(self):
        lo, hi = qbench.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        lo, hi = qbench.wilson(0, 40)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0881, places=3)
        lo, hi = qbench.wilson(40, 40)
        self.assertAlmostEqual(lo, 0.9119, places=3)
        self.assertEqual(hi, 1.0)
        lo, hi = qbench.wilson(30, 40)
        self.assertAlmostEqual(lo, 0.5979, places=3)
        self.assertAlmostEqual(hi, 0.8581, places=3)
        self.assertEqual(qbench.wilson(0, 0), (0.0, 1.0))

    def test_mcnemar_exact(self):
        self.assertEqual(qbench.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(qbench.mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(qbench.mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(qbench.mcnemar_exact(2, 8), 0.109375)
        self.assertAlmostEqual(qbench.mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(qbench.mcnemar_exact(1, 1), 1.0)
        self.assertAlmostEqual(qbench.mcnemar_exact(0, 10), 2 / 1024)
        self.assertLessEqual(qbench.mcnemar_exact(100, 100), 1.0)

    def test_pass_at_k(self):
        self.assertEqual(qbench.pass_at_k(3, 0, 3), 0.0)
        self.assertEqual(qbench.pass_at_k(3, 1, 3), 1.0)
        self.assertAlmostEqual(qbench.pass_at_k(3, 1, 1), 1 / 3)
        self.assertAlmostEqual(qbench.pass_at_k(3, 2, 1), 2 / 3)
        self.assertAlmostEqual(qbench.pass_at_k(10, 2, 2), 1 - (8 * 7) / (10 * 9))

    def test_median(self):
        self.assertEqual(qbench.median([3, 1, 2]), 2)
        self.assertEqual(qbench.median([1, 2, 3, 4]), 2.5)
        self.assertIsNone(qbench.median([]))


class FakeOpenAI:
    """A tiny /v1/chat/completions server. `mode`: good | bad | prose | long | error. Records every request body."""

    def __init__(self, mode="good"):
        self.mode = mode
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"total_slots": 1, "default_generation_settings": {"n_ctx": 16384}, "chat_template": "x" * 10, "build_info": "fake"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if outer.mode == "error":
                    return self._send(500, {"error": "boom"})
                user = body["messages"][1]["content"]
                p = next(p for p in PROBLEMS if p.prompt in user)
                if outer.mode == "prose":
                    text, finish = "I would solve this with a loop.", "stop"
                elif outer.mode == "long":
                    text, finish = "```python\n" + "x = 1\n" * 50, "length"
                else:
                    text, finish = f"```{'python' if p.lang == 'py' else 'javascript'}\n{p.ref_path(outer.mode).read_text(encoding='utf-8')}```", "stop"
                msg = {"role": "assistant", "content": text}
                if body.get("chat_template_kwargs", {}).get("enable_thinking"):
                    msg["reasoning_content"] = "Let me think about " + p.id
                self._send(200, {"choices": [{"message": msg, "finish_reason": finish}],
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                                 "timings": {"predicted_per_second": 9.5, "predicted_n": 50, "prompt_per_second": 150.0}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "A.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, server, *extra, arm="A"):
        argv = ["--endpoint", server.endpoint, "--arm", arm, "--out", str(self.out), "--request-timeout", "20", *extra]
        return run_quality.main(argv)

    def rows(self):
        meta, rows = qbench.read_results(self.out)
        return meta, rows

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_good_answers_pass_and_the_record_has_everything_the_journal_needs(self):
        with FakeOpenAI("good") as srv:
            self.assertEqual(self.run_runner(srv, "--problems", "py_01,js_01"), 0)
            meta, rows = self.rows()
            self.assertEqual(sorted(r["problem"] for r in rows), ["js_01_camel_to_snake", "py_01_parse_duration"])
            for r in rows:
                self.assertTrue(r["pass"], r)
                self.assertIsNone(r["kind"])
                for key in ("arm", "sample", "tok_s", "wall_s", "completion_tokens", "prompt_tokens", "finish_reason", "cap_hit", "response", "code", "seed"):
                    self.assertIn(key, r)
                self.assertEqual((r["tok_s"], r["completion_tokens"], r["cap_hit"], r["arm"]), (9.5, 50, False, "A"))
            self.assertEqual(meta["server_props"]["n_ctx"], 16384)
            self.assertEqual(meta["arm"], "A")
            req = srv.requests[0]
            self.assertEqual((req["temperature"], req["seed"], req["stream"], req["cache_prompt"]), (0.0, 1234, False, False))
            self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": False})
            p = BY_ID[rows[0]["problem"]]                     # requests are sent in manifest order
            self.assertEqual(req["max_tokens"], p.max_tokens)
            self.assertEqual(req["messages"], qbench.build_messages(p))
            self.assertEqual(rows[0]["prompt_variant"], qbench.DEFAULT_VARIANT)
            self.assertEqual(qbench.DEFAULT_VARIANT, "v3")

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_the_same_requests_go_to_every_arm(self):
        with FakeOpenAI("good") as srv:
            self.run_runner(srv, "--problems", "py_01,js_01")
            first = [dict(r) for r in srv.requests]
            srv.requests.clear()
            self.out = Path(self.tmp.name) / "other.jsonl"
            self.run_runner(srv, "--problems", "py_01,js_01", arm="C")
            self.assertEqual(first, srv.requests)

    def test_wrong_answers_fail_as_test_fail(self):
        with FakeOpenAI("bad") as srv:
            self.run_runner(srv, "--problems", "py_01,py_02")
            _, rows = self.rows()
            self.assertEqual([(r["pass"], r["kind"]) for r in rows], [(False, "test_fail")] * 2)
            self.assertTrue(all(r["detail"] for r in rows))

    def test_prose_is_no_code(self):
        with FakeOpenAI("prose") as srv:
            self.run_runner(srv, "--problems", "py_01")
            _, rows = self.rows()
            self.assertEqual((rows[0]["pass"], rows[0]["kind"]), (False, "no_code"))

    def test_hitting_the_token_cap_is_recorded(self):
        with FakeOpenAI("long") as srv:
            self.run_runner(srv, "--problems", "py_01")
            _, rows = self.rows()
            self.assertTrue(rows[0]["cap_hit"])
            self.assertEqual(rows[0]["finish_reason"], "length")
            self.assertFalse(rows[0]["pass"])

    def test_a_server_error_is_an_api_error_not_a_crash(self):
        with mock.patch.object(run_quality.time, "sleep"), FakeOpenAI("error") as srv:
            self.run_runner(srv, "--problems", "py_01")
            _, rows = self.rows()
            self.assertEqual((rows[0]["pass"], rows[0]["kind"]), (False, "api_error"))
            self.assertEqual(len(srv.requests), 3)          # 1 try + 2 retries

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_thinking_arm_sends_the_switch_and_adds_the_cap(self):
        with FakeOpenAI("good") as srv:
            self.run_runner(srv, "--problems", "py_01,js_01", "--thinking", "on", "--think-budget", "1000", arm="B")
            _, rows = self.rows()
            self.assertTrue(all(r["pass"] and r["thinking"] for r in rows), rows)
            self.assertTrue(all(r["reasoning"].startswith("Let me think") for r in rows))
            for req, row in zip(srv.requests, rows):
                self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": True})
                self.assertEqual(req["max_tokens"], BY_ID[row["problem"]].max_tokens + 1000)

    def test_budget_exhaustion_is_detected_from_the_injected_message(self):
        with FakeOpenAI("bad") as srv:
            self.run_runner(srv, "--problems", "py_01", "--thinking", "on", "--budget-message", "Let me think", arm="B")
            self.run_runner(srv, "--problems", "py_02", "--thinking", "off", "--budget-message", "Let me think", arm="B")
            _, rows = self.rows()
            self.assertEqual([r["budget_exhausted"] for r in rows], [True, False])     # the fake only returns reasoning when thinking is on

    def test_a_rerun_resumes_instead_of_repeating(self):
        with FakeOpenAI("bad") as srv:
            self.run_runner(srv, "--problems", "py_01")
            self.assertEqual(len(srv.requests), 1)
            self.run_runner(srv, "--problems", "py_01,py_02")
            self.assertEqual(len(srv.requests), 2)
            _, rows = self.rows()
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(1 for line in self.out.read_text().splitlines() if '"type": "meta"' in line), 1)

    def test_samples_use_different_seeds_and_record_the_temperature(self):
        with FakeOpenAI("bad") as srv:
            self.run_runner(srv, "--problems", "py_01", "--samples", "3", "--temperature", "0.6", arm="A-t0.6")
            _, rows = self.rows()
            self.assertEqual([r["sample"] for r in rows], [0, 1, 2])
            self.assertEqual(sorted({r["seed"] for r in rows}), [1234, 1235, 1236])
            self.assertEqual({r["temperature"] for r in rows}, {0.6})

    def test_a_stop_file_stops_between_requests_and_a_rerun_resumes(self):
        stop = Path(self.tmp.name) / "STOP"
        with FakeOpenAI("bad") as srv:
            stop.write_text("x")
            self.assertEqual(self.run_runner(srv, "--problems", "py_01", "--samples", "3", "--stop-file", str(stop)), 0)
            self.assertEqual(self.rows()[1], [])                      # stopped before the first request
            stop.unlink()
            self.run_runner(srv, "--problems", "py_01", "--samples", "3", "--stop-file", str(stop))
            self.assertEqual([r["sample"] for r in self.rows()[1]], [0, 1, 2])

    def test_max_problems_and_lang_filters(self):
        with FakeOpenAI("prose") as srv:
            self.run_runner(srv, "--lang", "js", "--max-problems", "3")
            _, rows = self.rows()
            self.assertEqual([r["problem"] for r in rows], [p.id for p in PROBLEMS if p.lang == "js"][:3])


def ids_flip(pid):
    return pid < "js_1"        # a handful of problems differ in text between arm C and the rest


class Report(unittest.TestCase):
    def make(self, d, arm, passes, thinking=False, cap=(), tok=9.0, temperature=0.0, samples=1):
        lines = [json.dumps({"type": "meta", "arm": arm, "thinking": thinking, "temperature": temperature})]
        for p in PROBLEMS:
            for s in range(samples):
                ok = passes(p.id, s)
                lines.append(json.dumps({"type": "result", "problem": p.id, "test_fraction": 1.0 if ok else 0.0, "response": f"{arm if (arm == 'C' and ids_flip(p.id)) else 'same'}", "lang": p.lang, "difficulty": p.difficulty, "subtle": p.subtle, "arm": arm,
                                         "sample": s, "pass": ok, "kind": None if ok else "test_fail", "cap_hit": p.id in cap, "tok_s": tok,
                                         "completion_tokens": 100, "wall_s": 10.0}))
        (d / f"{arm}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_table_has_ci_mcnemar_and_cap_hits(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ids = [p.id for p in PROBLEMS]
            self.make(d, "A", lambda pid, s: pid in ids[:30])                              # 30/40
            self.make(d, "B", lambda pid, s: pid in ids[:30], thinking=True, cap=ids[:5])   # identical results -> p = 1
            self.make(d, "C", lambda pid, s: pid in ids[:20], tok=7.0)                      # loses 10 problems
            text = qreport.build_report(d, "A")
            self.assertIn("| A | off | 40 | 30 | 75% | 60%–86% |", text)
            self.assertIn("| B | on | 40 | 30 | 75% |", text)
            self.assertRegex(text, r"\| C \| off \| 40 \| 20 \| 50% \|.*\| 10 / 0 \| 0\.002 \| worse \(p<0\.05\) \| 31/40 \| 7\.00 \|")
            self.assertIn("within noise", text)
            self.assertIn("| Arm | mean share of tests passed |", text)
            self.assertRegex(text, r"\| C \| 50\.0% \| 10 / 0 \| 0\.002 \| worse \(p<0\.05\) \|")
            self.assertRegex(text, r"\| B \| on \|.*\| 40/40 \|")           # identical text to A on every problem
            self.assertRegex(text, r"\| C \| off \|.*\| 31/40 \|")           # arm C differs on the 9 js_0x problems
            self.assertIn("| 5 |", text)                                                   # B's cap hits

    def test_a_retried_api_error_is_replaced_by_the_later_record(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "B.jsonl"
            rows = [{"type": "meta", "arm": "B"},
                    {"type": "result", "problem": "p1", "sample": 0, "pass": False, "kind": "api_error"},
                    {"type": "result", "problem": "p2", "sample": 0, "pass": True, "kind": None},
                    {"type": "result", "problem": "p1", "sample": 0, "pass": True, "kind": None}]
            f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            meta, got = qbench.read_results(f)
            self.assertEqual([(r["problem"], r["pass"]) for r in got], [("p2", True), ("p1", True)])

    def test_server_errors_are_left_out_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ids = [p.id for p in PROBLEMS]
            self.make(d, "A", lambda pid, s: pid in ids[:20])                 # 20/40
            self.make(d, "B", lambda pid, s: pid in ids[:20])
            rows = [json.loads(line) for line in (d / "B.jsonl").read_text().splitlines()]
            for r in rows[1:4]:
                r.update({"pass": False, "kind": "api_error", "test_fraction": 0.0})
            (d / "B.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            text = qreport.build_report(d, "A")
            self.assertRegex(text, r"\| B \| off \| 37 \| 17 \| 46% \|.*\| 3 \| test_fail: 20 \|")     # n = 37, three server errors listed, not failures
            self.assertIn("x = server error", text)

    def test_pass_at_k_section(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.make(d, "A", lambda pid, s: True)
            self.make(d, "A-t0.6", lambda pid, s: s == 1 and pid.endswith("01_parse_duration") or s == 2, temperature=0.6, samples=3)
            text = qreport.build_report(d, "A")
            self.assertIn("A-t0.6: 40 problems x [3] samples at temperature 0.6: pass@1 = 34%, pass@3 = 100%", text)


class Grid(unittest.TestCase):
    def test_arms_cover_the_issue_and_are_fully_specified(self):
        arms = qgrid.load_arms()
        self.assertEqual([k for k in arms if "-" not in k], ["A", "B", "C", "D", "E", "A2", "S10"])
        self.assertEqual([k for k in arms if k.startswith("P-")], ["P-v1", "P-v2", "P-v3"])
        a, b, c, d, e = (arms[x] for x in "ABCDE")
        for key in ("serve", "thinking", "temperature", "samples", "think_budget"):
            self.assertEqual(arms["A2"][key], a[key], "A2 must be an exact repeat of A")
        self.assertEqual((a["serve"], a["thinking"]), ([], "off"))
        self.assertEqual((b["serve"], b["thinking"]), (a["serve"], "on"))
        self.assertEqual(b["reasoning_budget"], 1024)
        self.assertEqual(qgrid.arm_server_env(a), {})
        self.assertEqual(qgrid.arm_server_env(b)["LLAMA_ARG_THINK_BUDGET"], "1024")
        self.assertIn("thinking budget", qgrid.arm_server_env(b)["LLAMA_ARG_THINK_BUDGET_MESSAGE"])
        rcb = qgrid.runner_command(b, 8081, "py", Path("out"), None)
        self.assertEqual(rcb[rcb.index("--think-budget") + 1], str(1024 + qgrid.REASONING_MARGIN))
        self.assertEqual(rcb[rcb.index("--budget-message") + 1], b["reasoning_budget_message"])
        self.assertEqual(c["serve"], ["--kv", "q8_0", "--fa", "on", "--ubatch", "256"])
        self.assertEqual(d["serve"], ["--ctx", "32768", "--ubatch", "256"])
        self.assertEqual(e["serve"], ["--spec", "ngram-mod"])
        self.assertEqual((arms["A-t0.6"]["temperature"], arms["A-t0.6"]["samples"]), (0.6, 3))
        s10 = arms["S10"]        # phase 6 best-of-N candidates: config A, 10 samples at 0.7, seeds 6000-6009
        self.assertEqual((s10["serve"], s10["thinking"], s10["temperature"], s10["samples"], s10["seed"]), (a["serve"], "off", 0.7, 10, 6000))
        rc10 = qgrid.runner_command(s10, 8081, "py", Path("out"), None)
        self.assertEqual((rc10[rc10.index("--seed") + 1], rc10[rc10.index("--samples") + 1]), ("6000", "10"))
        self.assertIn("--stop-file", rc10)
        for arm in arms.values():
            self.assertIn(arm["thinking"], ("on", "off"))
            self.assertEqual(arm["temperature"] if arm["id"] not in ("A-t0.6", "S10") else 0.0, 0.0)

    def test_every_forge_flag_an_arm_uses_exists(self):
        import subprocess
        helptxt = subprocess.run([sys.executable, "-m", "forge", "serve", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
        for arm in qgrid.load_arms().values():
            for flag in (x for x in arm["serve"] if x.startswith("--")):
                self.assertIn(flag, helptxt, arm["id"])

    def test_parse_serve_json_finds_the_object_after_the_progress_lines(self):
        text = 'select: plan gpu\nattempt 1: ngl=16\n{\n  "endpoint": "http://127.0.0.1:8081/v1",\n  "command": ["a", "b"],\n  "config": {"ngl": 16}\n}\n'
        self.assertEqual(qgrid.parse_serve_json(text)["command"], ["a", "b"])
        # the real output nests objects (config, attempts): the top-level object is the unindented brace, not the last brace
        real = json.dumps({"endpoint": "http://127.0.0.1:8081/v1", "plan": "gpu", "source": "tuned", "config": {"ngl": 16, "ctx": 16384},
                           "attempts": [{"ngl": 16, "ctx": 16384, "ok": True, "detail": "healthy"}], "command": ["llama-server", "-ngl", "16"],
                           "log": "x.log"}, indent=2)
        parsed = qgrid.parse_serve_json("select: x\nattempt 1: ngl=16\n" + real + "\n")
        self.assertEqual((parsed["config"]["ngl"], parsed["command"][0], len(parsed["attempts"])), (16, "llama-server", 1))
        self.assertIsNone(qgrid.parse_serve_json("attempt 1\nnothing yet\n"))
        self.assertIsNone(qgrid.parse_serve_json('{\n  "endpoint": "x",\n'))

    def test_commands(self):
        arm = {**qgrid.load_arms()["C"]}
        cmd = qgrid.forge_command(arm, 8081, "py")
        self.assertEqual(cmd[:6], ["py", "-m", "forge", "serve", "--port", "8081"])
        self.assertEqual(cmd[-6:], ["--kv", "q8_0", "--fa", "on", "--ubatch", "256"])
        rc = qgrid.runner_command(arm, 8081, "py", Path("out"), "node.exe", 3)
        self.assertIn("--thinking", rc)
        self.assertEqual(rc[rc.index("--arm") + 1], "C")
        self.assertEqual(rc[rc.index("--max-problems") + 1], "3")
        self.assertEqual(rc[rc.index("--node") + 1], "node.exe")


if __name__ == "__main__":
    unittest.main()

"""Phase 4 (03-agent): the task suite and its validators, the runner (time limit on the process tree, hashes, judgement), the metering proxy,
the orchestrator and its indentation repair (against an oracle chat server), the report and the grid driver. No model is needed."""
import json
import os
import shutil
import stat
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .helpers import ROOT

AGENT = ROOT / "03-agent"
for sub in ("harness", "orchestrator", "quality"):
    sys.path.insert(0, str(AGENT / sub))
import grid  # noqa: E402
import meter  # noqa: E402
import orchestrate  # noqa: E402
import proctree  # noqa: E402
import pyindent  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402
import taskbench  # noqa: E402

HAVE_NODE = taskbench.qbench.find_node() is not None
TASKS = taskbench.load_tasks()
BY_ID = {t.id: t for t in TASKS}
PY_TASKS = [t for t in TASKS if t.lang == "py"]
JS_TASKS = [t for t in TASKS if t.lang == "js"]


def tmpdir(test):
    d = tempfile.TemporaryDirectory(prefix="p4test-")
    test.addCleanup(d.cleanup)
    return Path(d.name)


# ------------------------------------------------------------------------------------------------ the suite

class Suite(unittest.TestCase):
    def test_twelve_tasks_of_every_kind_in_both_languages(self):
        self.assertEqual(len(TASKS), 12)
        self.assertEqual(len({t.id for t in TASKS}), 12)
        self.assertEqual({t.kind for t in TASKS}, set(taskbench.KINDS))
        self.assertEqual(len(PY_TASKS), 6)
        self.assertEqual(len(JS_TASKS), 6)

    def test_every_task_is_well_formed(self):
        for t in TASKS:
            with self.subTest(task=t.id):
                self.assertEqual(taskbench.validate_task(t), [])

    def test_reference_passes_broken_and_start_fail(self):
        for t in TASKS:
            if t.lang == "js" and not HAVE_NODE:
                continue
            with self.subTest(task=t.id):
                v = taskbench.validate_behaviour(t)
                self.assertEqual(v.problems, [])
                self.assertGreaterEqual(v.reference_total or 0, 8)

    def test_hidden_tests_never_in_the_agents_copy(self):
        for t in TASKS:
            with self.subTest(task=t.id):
                d = tmpdir(self) / "r"
                taskbench.materialize(t, d)
                for p in t.hidden_tests:
                    self.assertFalse((d / p).exists(), p)
                for p in t.visible_tests:
                    self.assertTrue((d / p).is_file(), p)

    def test_refactor_task_starts_green_the_others_red(self):
        self.assertEqual({t.id for t in TASKS if t.start_state == "green"}, {"py-refactor-split"})

    def test_prompts_do_not_leak_hidden_file_names(self):
        for t in TASKS:
            for p in t.hidden_tests:
                self.assertNotIn(Path(p).name, t.prompt)


class Validators(unittest.TestCase):
    def make_task(self, **over):
        src = BY_ID["py-intervals"]
        d = tmpdir(self) / "py-intervals"
        shutil.copytree(src.root, d)
        data = json.loads((d / "task.json").read_text())
        data.update(over)
        (d / "task.json").write_text(json.dumps(data))
        return taskbench.load_task(d), d

    def test_clean_copy_is_valid(self):
        t, _ = self.make_task()
        self.assertEqual(taskbench.validate_task(t), [])

    def test_hidden_test_inside_repo_is_flagged(self):
        t, d = self.make_task()
        shutil.copy(d / "hidden/tests/test_intervals_hidden.py", d / "repo/tests/")
        self.assertTrue(any("also in repo" in p for p in taskbench.validate_task(t)))

    def test_missing_hidden_file_and_unlisted_file(self):
        t, d = self.make_task(hidden_tests=["tests/nope.py"])
        probs = taskbench.validate_task(t)
        self.assertTrue(any("not in hidden/" in p for p in probs))
        self.assertTrue(any("not listed" in p for p in probs))

    def test_solution_may_not_overwrite_a_visible_test(self):
        t, d = self.make_task()
        (d / "solution/tests").mkdir()
        (d / "solution/tests/test_intervals.py").write_text("x = 1\n")
        self.assertTrue(any("overwrites protected" in p for p in taskbench.validate_task(t)))

    def test_bad_fields(self):
        t, _ = self.make_task(kind="magic", start_state="maybe")
        probs = taskbench.validate_task(t)
        self.assertTrue(any("kind" in p for p in probs) and any("start_state" in p for p in probs))

    def test_missing_key_is_a_task_error(self):
        _, d = self.make_task()
        data = json.loads((d / "task.json").read_text())
        del data["lang"]
        (d / "task.json").write_text(json.dumps(data))
        with self.assertRaises(taskbench.TaskError):
            taskbench.load_task(d)

    def test_unknown_task_id(self):
        with self.assertRaises(taskbench.TaskError):
            taskbench.load_tasks(ids=["nope"])

    def test_broken_that_passes_is_caught(self):
        t, d = self.make_task()
        shutil.copy(d / "solution/intervals.py", d / "broken/intervals.py")
        self.assertTrue(any("broken solution passes" in p for p in taskbench.validate_behaviour(t).problems))

    def test_reference_that_fails_is_caught(self):
        t, d = self.make_task()
        shutil.copy(d / "broken/intervals.py", d / "solution/intervals.py")
        self.assertTrue(any("reference solution fails" in p for p in taskbench.validate_behaviour(t).problems))

    def test_hashes_detect_edit_and_delete(self):
        t = BY_ID["py-wrap"]
        d = tmpdir(self) / "r"
        taskbench.materialize(t, d)
        before = taskbench.hash_files(d, t.visible_tests)
        self.assertEqual(taskbench.changed_files(before, d), [])
        (d / "tests/test_wrap.py").write_text("# edited\n")
        self.assertEqual(taskbench.changed_files(before, d), ["tests/test_wrap.py"])
        (d / "tests/__init__.py").unlink()
        self.assertEqual(set(taskbench.changed_files(before, d)), {"tests/test_wrap.py", "tests/__init__.py"})

    def test_materialize_refuses_a_dirty_target(self):
        d = tmpdir(self)
        (d / "x").write_text("1")
        with self.assertRaises(taskbench.TaskError):
            taskbench.materialize(BY_ID["py-wrap"], d)

    def test_check_runs_only_the_tasks_test_files(self):
        t = BY_ID["py-wrap"]
        d = tmpdir(self) / "r"
        taskbench.materialize(t, d)
        taskbench.copy_tree(t.solution_dir, d)
        (d / "tests/test_agent_added.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_fails(self):\n        self.fail('x')\n")
        run = taskbench.run_check(t, d)
        self.assertTrue(run.ok, run.output)

    def test_score_counts_unloadable_modules_as_failures(self):
        t = BY_ID["py-wrap"]
        d = tmpdir(self) / "r"
        taskbench.materialize(t, d)
        run = taskbench.run_check(t, d)
        passed, total = taskbench.score(t, run, 9)
        self.assertEqual((passed < total, total), (True, 9))
        self.assertEqual(taskbench.score(t, taskbench.TestRun(None, None, None, "", 1.0), 9), (0, 9))


# ------------------------------------------------------------------------------------------------ indentation repair

def flatten(code, unit=" "):
    """What phase 3 saw: every nested line at one level of one space."""
    return "\n".join((unit if ln.startswith((" ", "\t")) else "") + ln.strip() if ln.strip() else "" for ln in code.split("\n"))


class Repair(unittest.TestCase):
    def test_valid_code_is_untouched(self):
        code = "def f(x):\n    if x:\n        return 1\n    return 2\n"
        self.assertEqual(pyindent.repair(code), (code, False, False))

    def test_mixed_tabs_and_spaces(self):
        fixed, changed, repaired = pyindent.repair("def f():\n    x = 1\n\treturn x\n")
        self.assertTrue(repaired)
        ns = {}
        exec(fixed, ns)
        self.assertEqual(ns["f"](), 1)

    def test_flat_function_with_loop_and_final_return(self):
        flat = "def total(xs):\n s = 0\n for x in xs:\n if x > 0:\n s += x\n return s\n"
        fixed, _, repaired = pyindent.repair(flat)
        self.assertTrue(repaired)
        ns = {}
        exec(fixed, ns)
        self.assertEqual(ns["total"]([1, -2, 3]), 4)

    def test_flat_if_elif_else_and_try(self):
        flat = ("def f(x):\n try:\n n = int(x)\n except ValueError:\n return 'bad'\n if n < 0:\n return 'neg'\n elif n == 0:\n return 'zero'\n"
                " else:\n return 'pos'\n")
        fixed, _, repaired = pyindent.repair(flat)
        self.assertTrue(repaired)
        ns = {}
        exec(fixed, ns)
        self.assertEqual([ns["f"](v) for v in ("a", "-1", "0", "5")], ["bad", "neg", "zero", "pos"])

    def test_flat_class_with_methods(self):
        flat = "class C:\n def __init__(self, n):\n if n < 1:\n raise ValueError('n')\n self.n = n\n\n def double(self):\n return self.n * 2\n"
        fixed, _, repaired = pyindent.repair(flat)
        self.assertTrue(repaired)
        ns = {}
        exec(fixed, ns)
        self.assertEqual(ns["C"](4).double(), 8)
        with self.assertRaises(ValueError):
            ns["C"](0)

    def test_brackets_and_docstrings_are_carried_along(self):
        flat = 'def f():\n if True:\n """doc\n  more"""\n d = {\n "a": 1,\n "b": 2,\n }\n return d\n'
        fixed, _, repaired = pyindent.repair(flat)
        self.assertTrue(repaired)
        ns = {}
        exec(fixed, ns)
        self.assertEqual(ns["f"](), {"a": 1, "b": 2})

    def test_hopeless_input_is_returned_unchanged(self):
        junk = "def f(:\n    return )\n"
        self.assertEqual(pyindent.repair(junk), (junk, False, False))

    def test_reference_solutions_flattened_to_one_space(self):
        """The regression guard for the measured behaviour: flatten every Python reference file, repair it, and require that it compiles
        and that most of the files come back byte for byte."""
        files = [p for t in PY_TASKS for p in t.solution_dir.rglob("*.py")] + [p for t in PY_TASKS for p in t.repo_dir.rglob("*.py")
                                                                                if "tests" not in p.parts and p.stat().st_size > 0]
        exact = compiled = total = 0
        for p in files:
            code = p.read_text(encoding="utf-8")
            if "\n    " not in code:
                continue
            total += 1
            fixed, _, repaired = pyindent.repair(flatten(code))
            compiled += repaired
            exact += fixed == code
        self.assertGreater(total, 10)
        # measured: 18 of 20 compile again, 5 of 20 come back byte for byte (the rest differ where a block ends: flat text cannot say)
        self.assertGreaterEqual(compiled / total, 0.85, (compiled, total))
        self.assertGreaterEqual(exact / total, 0.2, (exact, compiled, total))


# ------------------------------------------------------------------------------------------------ the oracle chat server

class Oracle:
    """An OpenAI-compatible endpoint that answers like a perfect model: the next (at most two) reference files that the prompt shows
    differently, as FILE blocks; `DONE` when nothing differs. `mode`: solve | flat (every file flattened to one space) | garbage | slow."""

    def __init__(self, mode="solve"):
        self.mode, self.requests, self.bodies = mode, 0, []
        oracle = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                oracle.requests += 1
                oracle.bodies.append(body)
                user = body["messages"][-1]["content"]
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    ev = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "bash", "arguments": "{}"}}]}}]}
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    fin = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 50, "completion_tokens": 7}}
                    self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())
                    return
                text = oracle.reply(user)
                out = json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def reply(self, user):
        if self.mode == "garbage":
            return "I think you should consider the problem carefully."
        task = next((t for t in TASKS if t.prompt[:60] in user), None)
        if task is None:
            return "DONE"
        out = []
        for p in sorted(taskbench.list_files(task.solution_dir)):
            want = (task.solution_dir / p).read_text(encoding="utf-8")
            shown = f"=== {p} ===\n{want.rstrip()}\n"
            if shown in user:
                continue
            if self.mode == "flat":
                want = flatten(want)
            out.append(f"FILE: {p}\n```\n{want.rstrip()}\n```")
            if len(out) == 2:
                break
        return "\n\n".join(out) if out else "DONE"


class Orchestrator(unittest.TestCase):
    def run_arm(self, arm, task_id, mode="solve", limit=600.0, extra=()):
        oracle = Oracle(mode)
        self.addCleanup(oracle.close)
        out = tmpdir(self)
        rc = runner.main(["--arm", arm, "--tasks", task_id, "--endpoint", oracle.url, "--out-dir", str(out / "res"), "--work-dir", str(out / "w"),
                          "--time-limit", str(limit), "--shuffle-seed", "0", *extra])
        rows = [json.loads(ln) for ln in (out / "res" / f"{arm}.jsonl").read_text().splitlines()]
        return rc, rows[0], rows[1], oracle

    def test_every_python_task_is_solved_by_the_oracle(self):
        for t in PY_TASKS:
            with self.subTest(task=t.id):
                rc, meta, rec, oracle = self.run_arm("T", t.id)
                self.assertTrue(rec["success"], rec)
                self.assertEqual(rec["reason"], "ok")
                self.assertEqual(rec["tests_passed"], rec["tests_total"])
                self.assertEqual(rec["tests_total"], meta["reference_totals"][t.id])
                self.assertGreaterEqual(rec["steps"], 1)

    @unittest.skipUnless(HAVE_NODE, "node not found")
    def test_every_javascript_task_is_solved_by_the_oracle(self):
        for t in JS_TASKS:
            with self.subTest(task=t.id):
                rc, meta, rec, oracle = self.run_arm("T", t.id)
                self.assertTrue(rec["success"], rec)

    def test_refactor_needs_the_review_round_to_finish(self):
        rc, meta, rec, oracle = self.run_arm("T", "py-refactor-split")
        self.assertTrue(rec["success"], rec)
        self.assertGreaterEqual(rec["orchestrator"]["reviews"], 1)
        self.assertGreaterEqual(rec["steps"], 2)

    def test_flat_indentation_is_repaired_and_without_the_repair_it_is_not(self):
        rc, meta, rec, oracle = self.run_arm("T", "py-fix-paginate", mode="flat")
        self.assertTrue(rec["success"], rec)
        self.assertGreater(rec["orchestrator"]["repairs"], 0)
        rc, meta, rec2, oracle2 = self.run_arm("T-norepair", "py-fix-paginate", mode="flat")
        self.assertFalse(rec2["success"])
        self.assertGreater(rec2["orchestrator"]["syntax_rejects"], 0)
        self.assertEqual(rec2["orchestrator"]["repairs"], 0)

    def test_a_model_that_never_answers_in_format_stops_without_progress(self):
        rc, meta, rec, oracle = self.run_arm("T", "py-wrap", mode="garbage")
        self.assertFalse(rec["success"])
        self.assertEqual(rec["reason"], "no_change")
        self.assertEqual(rec["orchestrator"]["stop_reason"], "no progress")
        self.assertLessEqual(oracle.requests, 4)

    def test_tests_cannot_be_edited_by_the_model(self):
        so = orchestrate.is_test_path
        self.assertTrue(so("tests/test_x.py", set()) and so("test/a.test.js", set()) and so("src/x.test.js", set()))
        self.assertFalse(so("src/wrap.py", set()))
        self.assertIsNone(orchestrate.safe_path(Path("."), "../x.py"))
        self.assertIsNone(orchestrate.safe_path(Path("."), "/etc/passwd.py"))
        self.assertIsNone(orchestrate.safe_path(Path("."), "C:/x.py"))
        self.assertIsNone(orchestrate.safe_path(Path("."), "run.sh"))
        self.assertEqual(orchestrate.safe_path(Path("."), "`src/a.py`"), "src/a.py")

    def test_parse_edits_formats(self):
        a = "FILE: a.py\n```python\nx = 1\n```\n\nFILE: `b/c.js`\n```js\ny = 2\n```"
        self.assertEqual(orchestrate.parse_edits(a, None), [("a.py", "x = 1"), ("b/c.js", "y = 2")])
        self.assertEqual(orchestrate.parse_edits("**File: a.py**\n```\nz\n```", None), [("a.py", "z")])
        self.assertEqual(orchestrate.parse_edits("Here:\n```python\nq = 1\n```", "only.py"), [("only.py", "q = 1")])
        self.assertEqual(orchestrate.parse_edits("Here:\n```python\nq = 1\n```", None), [])
        self.assertEqual(orchestrate.parse_edits("no code", "only.py"), [])

    def test_failure_output_is_trimmed_to_the_first_failure(self):
        err = textwrap.dedent("""\
            ======================================================================
            FAIL: test_a (tests.test_x.T.test_a)
            ----------------------------------------------------------------------
            Traceback (most recent call last):
              File "/x/tests/test_x.py", line 5, in test_a
                self.assertEqual(f(1), 2)
            AssertionError: 1 != 2

            ======================================================================
            ERROR: test_b (tests.test_x.T.test_b)
            ----------------------------------------------------------------------
            Traceback (most recent call last):
              File "/x/tests/test_x.py", line 9, in test_b
            ValueError: boom

            ----------------------------------------------------------------------
            Ran 3 tests in 0.001s

            FAILED (failures=1, errors=1)
            """)
        res = orchestrate.parse_python(1, err)
        self.assertEqual((res.total, res.failed, res.passed), (3, 2, 1))
        self.assertEqual([n for n, _ in res.blocks], ["FAIL: test_a (tests.test_x.T.test_a)", "ERROR: test_b (tests.test_x.T.test_b)"])
        self.assertIn("1 != 2", res.blocks[0][1])
        self.assertNotIn("boom", res.blocks[0][1])

    def test_windows_crlf_output_is_parsed_like_unix_output(self):
        """Phase 4 first pass: on Windows the pipes give CRLF, no failure block was found and the model was told nothing failed."""
        err = ("======================================================================\r\nFAIL: test_a (tests.test_x.T.test_a)\r\n"
               "----------------------------------------------------------------------\r\nTraceback (most recent call last):\r\n"
               "  File \"/x/tests/test_x.py\", line 5, in test_a\r\nAssertionError: 1 != 2\r\n\r\n"
               "----------------------------------------------------------------------\r\nRan 2 tests in 0.001s\r\n\r\nFAILED (failures=1)\r\n")
        res = orchestrate.parse_python(1, err)
        self.assertEqual((res.total, res.failed, len(res.blocks)), (2, 1, 1))
        self.assertIn("1 != 2", res.blocks[0][1])
        tap = "TAP version 13\r\nnot ok 1 - adds\r\n  ---\r\n  error: 'boom'\r\n  ...\r\n1..1\r\n# tests 1\r\n# pass 0\r\n# fail 1\r\n"
        res = orchestrate.parse_node(1, tap, "")
        self.assertEqual((res.total, res.failed, len(res.blocks)), (1, 1, 1))

    def test_a_failure_without_a_parsable_block_is_never_reported_as_passing(self):
        d = tmpdir(self)
        res = orchestrate.run_tests(d, [sys.executable, "-c", "import sys; sys.stderr.write('boom happened'); sys.exit(1)"], "py", timeout=30)
        self.assertFalse(res.ok)
        self.assertIn("boom happened", res.blocks[0][1])
        prompt = orchestrate.build_prompt("task", d, set(), res, [], [], "fix")
        self.assertNotIn("visible tests pass", prompt)
        self.assertIn("boom happened", prompt)

    def test_the_example_in_the_system_prompt_matches_the_language(self):
        self.assertIn("```javascript", orchestrate.system_prompt("js"))
        self.assertNotIn("```python", orchestrate.system_prompt("js"))
        self.assertIn("```python", orchestrate.system_prompt("py"))
        self.assertIn("4 spaces", orchestrate.system_prompt("py"))

    def test_node_failure_blocks(self):
        tap = textwrap.dedent("""\
            TAP version 13
            # Subtest: adds
            not ok 1 - adds
              ---
              duration_ms: 0.5
              failureType: 'testCodeFailure'
              error: 'Expected values to be strictly equal:'
              code: 'ERR_ASSERTION'
              stack: |-
                TestContext.<anonymous> (/x/test/a.test.js:5:10)
                node:internal/test_runner/test:1:1
              ...
            ok 2 - subtracts
            1..2
            # tests 2
            # pass 1
            # fail 1
            """)
        res = orchestrate.parse_node(1, tap, "")
        self.assertEqual((res.total, res.failed, res.passed), (2, 1, 1))
        self.assertEqual(res.blocks[0][0], "adds")
        self.assertIn("strictly equal", res.blocks[0][1])
        self.assertNotIn("node:internal", res.blocks[0][1])
        self.assertNotIn("duration_ms", res.blocks[0][1])


# ------------------------------------------------------------------------------------------------ runner: time limit, hashes, proxy

FAKE_AGENT = r'''#!{python}
import json, os, subprocess, sys, time, urllib.request
mode = os.environ.get("FAKE_MODE", "solve")
sol = os.environ.get("FAKE_SOLUTION")
cfg = json.load(open(os.path.join(os.environ["XDG_CONFIG_HOME"], "opencode", "opencode.json")))
base = cfg["provider"]["forge"]["options"]["baseURL"]
def call():
    body = {{"model": "x", "stream": True, "temperature": 0.7, "top_p": 0.9, "tools": [{{"type": "function"}}],
            "messages": [{{"role": "user", "content": "hi"}}]}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={{"Content-Type": "application/json"}})
    return urllib.request.urlopen(req).read()
call()
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time,pathlib;time.sleep(6);pathlib.Path(os.environ['FAKE_MARK']).write_text('alive')".replace("import time", "import os,time")])
    time.sleep(600)
if mode == "edit_tests":
    open("tests/test_wrap.py", "a").write("# changed\n")
if mode in ("solve", "edit_tests") and sol:
    import shutil
    shutil.copytree(sol, ".", dirs_exist_ok=True)
print("done")
'''


class Runner(unittest.TestCase):
    def fake_agent(self, mode, **env):
        d = tmpdir(self)
        exe = d / "fake_opencode.py"
        exe.write_text(FAKE_AGENT.format(python=sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        os.environ["FAKE_MODE"] = mode
        for k, v in env.items():
            os.environ[k] = str(v)
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("FAKE_MODE", *env)])
        return exe

    def go(self, exe, task="py-wrap", limit=30.0):
        oracle = Oracle()
        self.addCleanup(oracle.close)
        out = tmpdir(self)
        runner.main(["--arm", "O", "--tasks", task, "--endpoint", oracle.url, "--out-dir", str(out / "res"), "--work-dir", str(out / "w"),
                     "--time-limit", str(limit), "--shuffle-seed", "0", "--opencode", str(exe)])
        rows = [json.loads(ln) for ln in (out / "res" / "O.jsonl").read_text().splitlines()]
        return rows[1], oracle

    @unittest.skipIf(sys.platform.startswith("win"), "the fake agent is a POSIX script")
    def test_agent_that_solves_it_succeeds_and_the_proxy_counts(self):
        exe = self.fake_agent("solve", FAKE_SOLUTION=BY_ID["py-wrap"].solution_dir)
        rec, oracle = self.go(exe)
        self.assertTrue(rec["success"], rec)
        self.assertEqual((rec["steps"], rec["tool_calls"], rec["prompt_tokens"], rec["completion_tokens"]), (1, 1, 50, 7))
        seen = oracle.bodies[0]
        self.assertEqual(seen["temperature"], 0)
        self.assertIs(seen["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotIn("top_p", seen)
        self.assertTrue(seen["stream_options"]["include_usage"])

    @unittest.skipIf(sys.platform.startswith("win"), "the fake agent is a POSIX script")
    def test_time_limit_kills_the_whole_tree(self):
        mark = tmpdir(self) / "mark.txt"
        exe = self.fake_agent("sleep", FAKE_MARK=mark)
        t0 = time.monotonic()
        rec, _ = self.go(exe, limit=3.0)
        self.assertLess(time.monotonic() - t0, 20)
        self.assertTrue(rec["timed_out"])
        self.assertFalse(rec["success"])
        self.assertTrue(rec["reason"].startswith("timeout"))
        self.assertLessEqual(rec["wall_s"], 3.0)
        time.sleep(7)                           # the grandchild would write its mark after 6 s if it had survived
        self.assertFalse(mark.exists(), "a descendant of the agent outlived the time limit")

    @unittest.skipIf(sys.platform.startswith("win"), "the fake agent is a POSIX script")
    def test_editing_a_test_file_fails_even_when_everything_passes(self):
        exe = self.fake_agent("edit_tests", FAKE_SOLUTION=BY_ID["py-wrap"].solution_dir)
        rec, _ = self.go(exe)
        self.assertEqual(rec["tests_passed"], rec["tests_total"])
        self.assertFalse(rec["success"])
        self.assertEqual(rec["reason"], "test_modified")
        self.assertEqual(rec["modified_tests"], ["tests/test_wrap.py"])

    def test_a_missing_agent_is_a_recorded_failure(self):
        rec, _ = self.go(Path("/nonexistent/opencode"))
        self.assertFalse(rec["success"])
        self.assertEqual(rec["reason"], "agent_start_failed")

    def test_finished_runs_are_skipped_on_resume(self):
        oracle = Oracle()
        self.addCleanup(oracle.close)
        out = tmpdir(self)
        args = ["--arm", "T", "--tasks", "py-intervals", "--endpoint", oracle.url, "--out-dir", str(out / "res"), "--work-dir", str(out / "w"),
                "--shuffle-seed", "0"]
        runner.main(args)
        n = oracle.requests
        runner.main(args)
        self.assertEqual(oracle.requests, n)
        self.assertEqual(len((out / "res" / "T.jsonl").read_text().splitlines()), 2)

    @unittest.skipUnless(proctree.IS_WINDOWS, "job objects exist on Windows only")
    def test_windows_job_object_is_used(self):
        tree = proctree.ProcTree([sys.executable, "-c", "import time; time.sleep(30)"], ".", dict(os.environ), open(os.devnull, "wb"))
        try:
            self.assertTrue(tree.job_ok)
        finally:
            tree.kill()
            tree.close()


# ------------------------------------------------------------------------------------------------ report and grid

def fake_run(task, arm, run, ok, wall=100.0, reason=None, passed=None, total=10, kind="function"):
    return {"type": "run", "task": task, "arm": arm, "run": run, "success": ok, "reason": reason or ("ok" if ok else "tests_failed"),
            "tests_passed": total if ok else (passed or 0), "tests_total": total, "wall_s": wall, "timed_out": False, "kind": kind,
            "steps": 5, "tool_calls": 7, "completion_tokens": 900}


class Report(unittest.TestCase):
    def write(self, d, arm, rows):
        with (d / f"{arm}.jsonl").open("w") as f:
            f.write(json.dumps({"type": "meta", "arm": arm, "time_limit_s": 900}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_summary_and_verdict(self):
        rows = [fake_run(f"t{i}", "T", 1, i < 7, wall=60 * (3 + i)) for i in range(10)]
        s = report.summarize(rows, 900)
        self.assertEqual((s["successes"], s["tasks"]), (7, 10))
        self.assertAlmostEqual(s["wilson"][0], 0.3968, places=3)
        self.assertEqual(s["median_min"], 7.5)
        self.assertTrue(s["usable"])

    def test_slow_or_rare_is_not_usable(self):
        slow = [fake_run(f"t{i}", "O", 1, True, wall=60 * 20) for i in range(10)]
        self.assertFalse(report.summarize(slow, 1500)["usable"])
        rare = [fake_run(f"t{i}", "O", 1, i < 3, wall=60) for i in range(10)]
        s = report.summarize(rare, 900)
        self.assertFalse(s["usable"])
        self.assertTrue(s["usable_time"] and not s["usable_rate"])

    def test_timeouts_count_at_the_limit_in_the_median(self):
        rows = [fake_run(f"t{i}", "O", 1, False, wall=5000) for i in range(3)]
        self.assertEqual(report.summarize(rows, 900)["median_min"], 15.0)

    def test_paired_mcnemar(self):
        a = [fake_run(f"t{i}", "O", 1, i < 2) for i in range(12)]
        b = [fake_run(f"t{i}", "T", 1, i < 2 or i >= 8) for i in range(12)]
        p = report.paired(a, b)
        self.assertEqual((len(p["only_a"]), len(p["only_b"]), len(p["both"]), p["neither"]), (0, 4, 2, 6))
        self.assertAlmostEqual(p["p"], 0.125)

    def test_second_run_only_feeds_stability(self):
        rows = [fake_run("t0", "T", 1, True), fake_run("t0", "T", 2, False), fake_run("t1", "T", 1, False), fake_run("t1", "T", 2, False)]
        self.assertEqual(report.first_runs(rows)["t0"]["run"], 1)
        self.assertEqual(report.stability(rows), (2, 1))

    def test_full_report_renders(self):
        d = tmpdir(self)
        self.write(d, "O", [fake_run(f"t{i}", "O", 1, i < 1, wall=900, reason="ok" if i < 1 else "timeout") for i in range(5)])
        self.write(d, "T", [fake_run(f"t{i}", "T", 1, i < 3) for i in range(5)])
        text = report.build_report(d)
        self.assertIn("| O | 5 | 1/5 |", text)
        self.assertIn("| T | 5 | 3/5 |", text)
        self.assertIn("McNemar", text)
        self.assertIn("timeout 4", text)


class Grid(unittest.TestCase):
    def test_stages(self):
        self.assertEqual(grid.parse_stages("O+T@1; T-norepair@1;O+T@2"), [(["O", "T"], 1), (["T-norepair"], 1), (["O", "T"], 2)])
        with self.assertRaises(SystemExit):
            grid.parse_stages("X@1")
        with self.assertRaises(SystemExit):
            grid.parse_stages("O@x")

    def test_runner_command_carries_the_limit_and_endpoint(self):
        ns = type("N", (), dict(python="py", port=8081, out_dir=Path("o"), work_dir=Path("w"), time_limit=900.0, note="n", node=None,
                                opencode="oc", oc_cache_seed=None))()
        cmd = grid.runner_cmd(ns, "O", "py-wrap", 2)
        self.assertEqual(cmd[cmd.index("--arm") + 1], "O")
        self.assertEqual(cmd[cmd.index("--runs") + 1], "2")
        self.assertEqual(cmd[cmd.index("--endpoint") + 1], "http://127.0.0.1:8081/v1")
        self.assertEqual(cmd[cmd.index("--time-limit") + 1], "900.0")

    def test_only_t_ngram_changes_the_server(self):
        self.assertEqual({a: v.get("serve", []) for a, v in runner.ARMS.items()},
                         {"O": [], "T": [], "T-norepair": [], "T-ngram": ["--spec", "ngram-mod"]})

    def test_opencode_config_points_at_the_given_url_with_context_limit(self):
        c = runner.opencode_config("http://127.0.0.1:9/v1")
        self.assertEqual(c["provider"]["forge"]["options"]["baseURL"], "http://127.0.0.1:9/v1")
        self.assertEqual(c["provider"]["forge"]["models"]["dsv4"]["limit"]["context"], 16384)
        self.assertEqual(c["model"], runner.OPENCODE_MODEL)


if __name__ == "__main__":
    unittest.main()

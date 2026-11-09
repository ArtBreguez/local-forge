"""Phase 6 (03-agent/quality/split.py, bestofn.py): the visible/hidden split of the tests, scoring both halves, best-of-N selection,
its time accounting and the tables."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from .helpers import ROOT

QUALITY = ROOT / "03-agent" / "quality"
sys.path.insert(0, str(QUALITY))
import bestofn  # noqa: E402
import qbench  # noqa: E402
import split  # noqa: E402

HAVE_NODE = qbench.find_node() is not None
BY_ID = {p.id: p for p in qbench.load_problems()}
SAMPLE_IDS = ["py_02_roman", "py_15_rotated_search", "py_04_add_months", "js_01_camel_to_snake", "js_14_emitter", "js_05_parse_query"]


class Split(unittest.TestCase):
    def test_the_committed_split_is_what_the_rule_gives_for_the_current_tests(self):
        self.assertEqual(split.load_split(), split.build_split(), "run `python 03-agent/quality/split.py --write` and commit before any run")

    def test_the_rule_alternates_starting_with_visible_and_no_half_is_empty(self):
        data = split.load_split()
        self.assertEqual(set(data["problems"]), set(BY_ID))
        for pid, e in data["problems"].items():
            self.assertEqual(e["visible"], list(range(0, e["n_tests"], 2)), pid)
            self.assertEqual(sorted(e["visible"] + e["hidden"]), list(range(e["n_tests"])), pid)
            self.assertGreaterEqual(len(e["visible"]), 2, pid)
            self.assertGreaterEqual(len(e["hidden"]), 2, pid)
            self.assertEqual(len(e["tests"]), e["n_tests"])

    def test_python_and_javascript_test_lists(self):
        py = "import unittest\n\nclass A(unittest.TestCase):\n    @unittest.skip('x')\n    def test_one(self):\n        pass\n\n    def helper(self):\n        pass\n\n    def test_two(self):\n        pass\n\nclass B(unittest.TestCase):\n    def test_three(self):\n        pass\n"
        self.assertEqual([t[0] for t in split.py_tests(py)], ["A.test_one", "A.test_two", "B.test_three"])
        self.assertEqual(split.py_tests(py)[0][1:], (4, 6), "the decorator belongs to the test")
        kept = split.build_part("py", py, {0})
        self.assertIn("test_one", kept)
        self.assertNotIn("test_two", kept)
        self.assertNotIn("class B", kept, "a TestCase that lost all its tests is dropped")
        self.assertIn("def helper", kept)
        js = "const test = require('node:test');\nconst h = () => 1;\n\ntest('a', () => {\n  h();\n});\n\ntest('b', () => {\n  if (h()) {\n    h();\n  }\n});\n\ntest('c', () => { h(); });\n"
        self.assertEqual([t[0] for t in split.js_tests(js)], ["a", "b", "c"])
        part = split.build_part("js", js, {1})
        self.assertIn("test('b'", part)
        self.assertNotIn("test('a'", part)
        self.assertNotIn("test('c'", part)
        self.assertIn("const h", part)

    def test_a_helper_class_without_tests_stays(self):
        src = "import unittest\n\nclass Probe(list):\n    pass\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        Probe()\n\n    def test_b(self):\n        Probe()\n"
        self.assertIn("class Probe", split.build_part("py", src, {1}))

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_the_reference_solution_passes_both_halves_and_the_counts_add_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts = split.write_parts(QUALITY, Path(tmp))
            for pid in SAMPLE_IDS:
                p = BY_ID[pid]
                good = p.ref_path("good").read_text(encoding="utf-8")
                full = qbench.run_tests(p, good)
                v = qbench.run_tests(p, good, tests_file=parts[pid]["visible"])
                h = qbench.run_tests(p, good, tests_file=parts[pid]["hidden"])
                self.assertTrue(full.passed and v.passed and h.passed, pid)
                self.assertEqual(v.tests_total + h.tests_total, full.tests_total, pid)

    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_score_row_reports_each_half_and_never_runs_code_that_is_not_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts = split.write_parts(QUALITY, Path(tmp))
            p = BY_ID["py_02_roman"]
            good = p.ref_path("good").read_text(encoding="utf-8")
            ok = bestofn.score_row(p, {"problem": p.id, "sample": 3, "code": good}, parts[p.id])
            self.assertEqual((ok["visible_pass"], ok["hidden_pass"], ok["kind"], ok["sample"]), (True, True, None, 3))
            broken = bestofn.score_row(p, {"problem": p.id, "sample": 0, "code": "def to_roman(n):\n    return ''\n"}, parts[p.id])
            self.assertFalse(broken["visible_pass"] and broken["hidden_pass"])
            self.assertIsNotNone(broken["kind"])
            none = bestofn.score_row(p, {"problem": p.id, "sample": 1, "kind": "no_code"}, parts[p.id])
            self.assertEqual((none["visible_pass"], none["hidden_pass"], none["kind"]), (False, False, "no_code"))


def row(pid, sample, passed, wall, **kw):
    return {"type": "result", "problem": pid, "sample": sample, "pass": passed, "wall_s": wall, "completion_tokens": 100, "tok_s": 10.0, **kw}


def score(pid, sample, visible, hidden):
    return {"type": "scored", "problem": pid, "sample": sample, "visible_pass": visible, "hidden_pass": hidden}


class Selection(unittest.TestCase):
    def test_first_visible_pass_wins_and_time_stops_there(self):
        g = row("p", 0, False, 40.0)
        cands = [row("p", 2, True, 30.0), row("p", 0, False, 10.0), row("p", 1, False, 20.0)]      # file order is not sample order
        vis = {("p", 0): score("p", 0, False, False), ("p", 1): score("p", 1, True, False), ("p", 2): score("p", 2, True, True)}
        sel = bestofn.select(g, cands, vis, 5)
        self.assertEqual((sel["source"], sel["row"]["sample"], sel["tried"], sel["time_s"]), ("sample", 1, 2, 30.0))
        self.assertFalse(sel["row"]["pass"], "sample 1 passes the visible half only, so the selection takes a wrong answer: that is the gap")

    def test_n_limits_the_candidates_and_the_fallback_pays_for_the_greedy_answer(self):
        g = row("p", 0, True, 40.0)
        cands = [row("p", i, False, 10.0) for i in range(4)] + [row("p", 4, True, 10.0)]
        vis = {("p", i): score("p", i, i == 4, i == 4) for i in range(5)}
        three = bestofn.select(g, cands, vis, 3)
        self.assertEqual((three["source"], three["tried"], three["time_s"]), ("greedy", 3, 70.0))
        five = bestofn.select(g, cands, vis, 5)
        self.assertEqual((five["source"], five["row"]["sample"], five["time_s"]), ("sample", 4, 50.0))

    def test_evaluate_counts_flips_in_both_directions(self):
        ids = ["a", "b", "c", "d"]
        greedy = {"a": row("a", 0, True, 40), "b": row("b", 0, False, 40), "c": row("c", 0, False, 40), "d": row("d", 0, True, 40)}
        gvis = {(k, 0): score(k, 0, v, v) for k, v in {"a": True, "b": False, "c": False, "d": True}.items()}
        samples = {"a": {0: row("a", 0, False, 10)},                      # a: no sample passes visible -> greedy (passes)
                   "b": {0: row("b", 0, True, 10)},                      # b: sample passes everything -> gain
                   "c": {0: row("c", 0, False, 10)},                     # c: nothing helps
                   "d": {0: row("d", 0, False, 10)}}                     # d: sample passes visible only -> chosen and wrong -> loss
        vis = {("a", 0): score("a", 0, False, False), ("b", 0): score("b", 0, True, True), ("c", 0): score("c", 0, False, False), ("d", 0): score("d", 0, True, False)}
        per = bestofn.evaluate(ids, greedy, samples, vis, gvis, 1)
        self.assertEqual(bestofn.paired(per, "greedy_full", "bon_full")[:2], (1, 1))
        self.assertEqual([per[k]["source"] for k in ids], ["greedy", "sample", "greedy", "sample"])
        self.assertAlmostEqual(per["a"]["time_s"], 50.0)
        table = bestofn.host_table(ids, greedy, samples, vis, gvis, [1])
        self.assertIn("best-of-1", table)
        self.assertIn("2/4 = 50%", table)

    def test_complete_rounds_ignores_server_errors_and_stops_at_the_first_gap(self):
        ids = ["a", "b"]
        rows = [row("a", 0, True, 1), row("b", 0, True, 1), row("a", 1, True, 1), row("b", 1, False, 1, kind="api_error"), row("a", 2, True, 1)]
        samples = bestofn.by_problem(rows)
        self.assertEqual(bestofn.complete_rounds(samples, ids), 1)

    def test_wilson_cell_format(self):
        self.assertEqual(bestofn.ci(9, 40), "9/40 = 22% (12-38%)")


if __name__ == "__main__":
    unittest.main()

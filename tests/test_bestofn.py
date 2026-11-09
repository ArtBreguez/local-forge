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


class Resume(unittest.TestCase):
    """A run killed in the middle of a write leaves a torn last line; resuming must neither trust it nor glue the next record to it."""

    def test_repair_tail_cuts_only_an_incomplete_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.jsonl"
            p.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": ')
            self.assertEqual(qbench.repair_tail(p), 6)
            self.assertEqual(p.read_bytes(), b'{"a": 1}\n{"b": 2}\n')
            self.assertEqual(qbench.repair_tail(p), 0, "a file that ends in a newline is left alone")
            p.write_bytes(b'{"a": 1')
            self.assertEqual(qbench.repair_tail(p), 7, "a file whose only line is torn becomes empty")
            self.assertEqual(p.read_bytes(), b"")
            self.assertEqual(qbench.repair_tail(Path(tmp) / "missing.jsonl"), 0)

    def test_read_jsonl_drops_a_torn_last_line_but_not_damage_in_the_middle(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.jsonl"
            p.write_text('{"a": 1}\n{"b": 2}\n{"c"', encoding="utf-8")
            self.assertEqual(qbench.read_jsonl(p), [{"a": 1}, {"b": 2}])
            p.write_text('{"a": 1}\n{"b"\n{"c": 3}\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                qbench.read_jsonl(p)

    def test_read_results_and_load_scored_survive_a_torn_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp) / "S.jsonl"
            r.write_text(json.dumps({"type": "meta", "arm": "S"}) + "\n" + json.dumps({"type": "result", "problem": "p", "sample": 0, "pass": True}) + '\n{"type": "res', encoding="utf-8")
            meta, rows = qbench.read_results(r)
            self.assertEqual((meta["arm"], len(rows)), ("S", 1))
            s = Path(tmp) / "S.scored.jsonl"
            s.write_text(json.dumps({"type": "scored", "problem": "p", "sample": 0, "visible_pass": True, "hidden_pass": True}) + '\n{"type": "sco', encoding="utf-8")
            self.assertEqual(list(bestofn.load_scored(s)), [("p", 0)])

    def test_score_file_repairs_its_output_resumes_and_honours_the_pause(self):
        from unittest import mock
        pid = "py_02_roman"
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "S.jsonl", Path(tmp) / "S.scored.jsonl"
            rows = [{"type": "meta", "arm": "S"}] + [{"type": "result", "problem": pid, "sample": i, "pass": False, "kind": "test_fail", "code": "x = 1"} for i in range(3)]
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out.write_text(json.dumps({"type": "scored", "problem": pid, "sample": 0, "visible_pass": False, "hidden_pass": False}) + '\n{"type": "scored", "pro', encoding="utf-8")
            fake = lambda p, row, parts, python=None, node=None: {"type": "scored", "problem": row["problem"], "sample": row["sample"], "visible_pass": True, "hidden_pass": False}  # noqa: E731
            with mock.patch.object(bestofn, "score_row", fake), mock.patch.object(bestofn.split, "write_parts", lambda *_: {pid: {}}), mock.patch.object(bestofn.time, "sleep") as sleep:
                done = bestofn.score_file(src, out, workers=1, pause=0.25)
            self.assertEqual(sorted(k[1] for k in done), [0, 1, 2])
            self.assertEqual(sleep.call_count, 2, "only the two answers that were not scored yet are scored (and paused after)")
            sleep.assert_called_with(0.25)
            self.assertEqual([json.loads(l)["sample"] for l in out.read_text(encoding="utf-8").splitlines()], [0, 1, 2], "the torn line is gone, nothing is doubled")


    @unittest.skipUnless(HAVE_NODE, "node is needed")
    def test_qbench_tz_runs_the_tests_in_the_given_zone(self):
        """phase 3 found js_08_age_on failing on the UTC-3 host for an answer that passes at UTC: the answer parses 'YYYY-MM-DD' as UTC and reads it back in local time."""
        import os
        from unittest import mock
        p = BY_ID["js_08_age_on"]
        code = ("function ageOn(birth, on) {\n  const b = new Date(birth), o = new Date(on);\n  let age = o.getFullYear() - b.getFullYear();\n"
                "  if (o.getMonth() < b.getMonth() || (o.getMonth() === b.getMonth() && o.getDate() < b.getDate())) age--;\n  return age;\n}\nmodule.exports = { ageOn };\n")
        with tempfile.TemporaryDirectory() as tmp:
            parts = split.write_parts(QUALITY, Path(tmp))
            half = parts["js_08_age_on"]["visible"]
            with mock.patch.dict(os.environ, {"QBENCH_TZ": "UTC"}):
                utc = qbench.run_tests(p, code, tests_file=half)
            with mock.patch.dict(os.environ, {"QBENCH_TZ": "America/Sao_Paulo"}):
                brt = qbench.run_tests(p, code, tests_file=half)
        self.assertTrue(utc.tests_total and brt.tests_total)
        self.assertNotEqual((utc.passed, utc.detail), (brt.passed, brt.detail), "the timezone must reach the node process")


class Passes(unittest.TestCase):
    def test_pass_at_k_is_the_unbiased_estimate(self):
        self.assertEqual(bestofn.pass_at_k(10, 0, 5), 0.0)
        self.assertEqual(bestofn.pass_at_k(10, 10, 5), 1.0)
        self.assertEqual(bestofn.pass_at_k(10, 6, 5), 1.0, "fewer than k failures: every draw of k contains a pass")
        self.assertAlmostEqual(bestofn.pass_at_k(10, 1, 5), 0.5)          # 1 - C(9,5)/C(10,5)
        self.assertAlmostEqual(bestofn.pass_at_k(4, 1, 1), 0.25)

    def test_passes_table_from_stored_flags(self):
        ids = [p.id for p in qbench.load_problems()]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)

            def write(name: str, greedy_pass: set[str], sample_pass: dict[str, int]) -> None:
                g = [{"type": "meta", "arm": name}] + [{"type": "result", "problem": pid, "sample": 0, "pass": pid in greedy_pass, "tok_s": 10.0, "wall_s": 20.0} for pid in ids]
                s = [{"type": "meta", "arm": name}] + [{"type": "result", "problem": pid, "sample": i, "pass": i < sample_pass.get(pid, 0), "wall_s": 30.0}
                                                      for pid in ids for i in range(10)]
                (d / f"{name}.greedy.jsonl").write_text("\n".join(json.dumps(r) for r in g) + "\n", encoding="utf-8")
                (d / f"{name}.sample.jsonl").write_text("\n".join(json.dumps(r) for r in s) + "\n", encoding="utf-8")
            write("base", set(ids[:10]), {pid: 1 for pid in ids[:20]})
            write("better", set(ids[:30]), {pid: 5 for pid in ids[:30]})
            write("censored", set(ids[:40]), {pid: 10 for pid in ids})
            table = bestofn.passes_table(d, "base", {"base": {"uncensored": True, "size_gb": 5.2, "licence": "Apache-2.0 (card)"},
                                                      "better": {"uncensored": True, "size_gb": 11.0, "licence": "Apache-2.0 (card)"}}, [5, 10])
        rows = [r for r in table.splitlines()[2:]]
        self.assertEqual([r.split(" | ")[0].strip("| ") for r in rows], ["better", "base", "censored (censored: reference row only)"],
                         "uncensored models first (best first), a censored model only as a reference row after them")
        better = rows[0]
        self.assertIn("30/40 = 75%", better)
        self.assertIn("0 / 20, p = 0.000", better)                       # only base: 0, only model: 20 (ids 10..29)
        self.assertIn("(reference)", rows[1])
        self.assertIn("10/40 = 25%", rows[1])
        self.assertIn("| no |", rows[2])
        self.assertIn("5.2 GB, Apache-2.0", rows[1])
        self.assertIn("pass@10", table)


if __name__ == "__main__":
    unittest.main()

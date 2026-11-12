"""Round 15, m8: the owner's rule about teachers, as code. Any open-weight teacher may FILTER in the default mode, whatever its origin; in the reasoning and thinking modes the teacher's reply is stored, so only a
teacher not trained by or distilled from an OpenAI or Anthropic model may be used (gpt-oss-* and the GPT-4o distill are filter-only).

  Catalogue    `Teacher.reasoning_ok`: the fifth field, False by default, True for the shipped teachers, never True for an entry whose name says OpenAI or Anthropic
  Refusal      `teachers.require_reasoning`: by name, by repo and file, strict for the Kaggle kit
  ModalStage   the REA entrypoint and the generic sample stage refuse `--reasoning` / `--thinking`, and a default run is not touched
  KaggleKit    `build_tasks` refuses a teacher without it, and a GGUF that is in no entry
  Docs         the rule in docs/TEACHER_CANDIDATES.md, distill/teachers.py, distill/README.md, and the open question about CodeForge
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .test_rea_modal import ModalCase, StubbedSampleStage
from .thinking_pilot import build_tasks

from distill import teachers  # noqa: E402

GPT_OSS = teachers.Teacher("gpt-oss-120b-derestricted", "ArliAI/gpt-oss-120b-Derestricted-GGUF", "gpt-oss-120b-Derestricted-Q4_K_M.gguf", "L40S", False)
GPT4O = teachers.Teacher("llama33-70b-gpt4o-distill", "MuXodious/gpt-4o-distil-Llama-3.3-70B-Instruct-PaperWitch-heresy", "gpt-4o-distil.Q4_K_M.gguf", "L40S")


class Catalogue(unittest.TestCase):
    def test_the_flag_is_the_fifth_field_and_false_unless_said(self):
        self.assertEqual(teachers.Teacher._fields, ("name", "repo", "file", "gpu", "reasoning_ok"))
        self.assertFalse(teachers.Teacher("x", "r", "f", "L4").reasoning_ok, "a teacher nobody has looked at is filter-only")
        self.assertEqual(len(teachers.TEACHERS[0]), 5)
        self.assertEqual(teachers.TEACHERS[0][3], "L4", "the entries are still (name, repo, file, gpu, ...) tuples")
        name, repo, fname, gpu, *_ = teachers.TEACHERS[0]
        self.assertEqual((name, gpu), ("heretic-9b", "L4"))

    def test_every_shipped_teacher_says_what_it_may_do_and_none_with_an_openai_or_anthropic_name_may_store_text(self):
        self.assertEqual([t.name for t in teachers.TEACHERS if not t.reasoning_ok], [], "the shipped teachers are Qwen-based and may store text (CodeForge's lineage is the owner's open question)")
        for t in teachers.TEACHERS:
            self.assertFalse(teachers.marked_filter_only(t), f"{t.name} names an OpenAI or Anthropic model: its flag cannot be True")
        for t in (GPT_OSS, GPT4O, teachers.Teacher("claude-distill-9b", "x/y", "z.gguf", "L4", True), teachers.Teacher("a", "unsloth/gpt-oss-20b-GGUF", "f.gguf", "L4", True), teachers.Teacher("a", "r", "Opus-distill.gguf", "L4", True)):
            self.assertTrue(teachers.marked_filter_only(t), t)
        self.assertFalse(teachers.marked_filter_only(teachers.find("qwen35-27b")))

    def test_the_names_of_the_four_unpacking_sites_still_work(self):
        """modal_qbench, modal_rea and modal_rea_data unpack the entries: five fields must not break them."""
        for entry in (*teachers.TEACHERS, ("heretic-9b-base-q4km", "vol:", "/lora/full/gguf/base-Q4_K_M.gguf", "L4")):
            name, repo, fname, gpu, *_ = entry
            self.assertEqual(entry[:4], (name, repo, fname, gpu))
        text = "".join((ROOT / "tools" / "modal" / n).read_text() for n in ("modal_qbench.py", "modal_rea.py", "modal_rea_data.py"))
        self.assertNotIn("for name, repo, fname, gpu in todo", text)
        self.assertNotIn("name, repo, fname, _ = m[0]", text)


class Refusal(unittest.TestCase):
    def patched(self, *extra):
        return mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, *extra])

    def test_a_filter_only_teacher_is_refused_by_name_and_by_its_gguf(self):
        with self.patched(GPT_OSS, GPT4O):
            for kw in ({"name": GPT_OSS.name}, {"repo": GPT_OSS.repo, "file": GPT_OSS.file}, {"name": "another-name", "repo": GPT_OSS.repo, "file": GPT_OSS.file}, {"name": GPT4O.name}):
                with self.assertRaisesRegex(SystemExit, "may only FILTER"):
                    teachers.require_reasoning(**kw)
            with self.assertRaisesRegex(SystemExit, "--thinking stores the teacher's reply verbatim"):
                teachers.require_reasoning(GPT_OSS.name, why="--thinking")
            try:
                teachers.require_reasoning(GPT_OSS.name)
            except SystemExit as e:
                self.assertIn("qwen35-27b", str(e), "it names the teachers that may")
                self.assertNotIn("gpt-oss-120b-derestricted'", str(e).split("Teachers that may")[1])

    def test_a_flag_that_says_true_for_an_openai_name_is_still_refused(self):
        liar = teachers.Teacher("gpt-oss-20b", "unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf", "L4", True)
        with self.patched(liar), self.assertRaisesRegex(SystemExit, "may only FILTER"):
            teachers.require_reasoning("gpt-oss-20b")

    def test_a_teacher_that_may_store_passes_and_an_unknown_one_only_fails_strictly(self):
        teachers.require_reasoning("codeforge-27b-heretic-q4")
        teachers.require_reasoning(repo=teachers.find("qwen35-27b").repo, file=teachers.find("qwen35-27b").file)
        teachers.require_reasoning("m", "repo", "f.gguf")                       # the generic stage's own names in the tests: nothing in the catalogue says no
        with self.assertRaisesRegex(SystemExit, "is in no entry of distill/teachers.py"):
            teachers.require_reasoning("m", "repo", "f.gguf", strict=True)
        with self.assertRaisesRegex(SystemExit, "x/y/z.gguf"):
            teachers.require_reasoning("", "x/y", "z.gguf", strict=True)

    def test_matching_takes_the_same_gguf_under_another_name(self):
        q4 = teachers.find("codeforge-27b-heretic-q4")
        self.assertEqual([t.name for t in teachers.matching("", q4.repo, q4.file)], ["codeforge-27b-heretic-q4"])
        self.assertEqual([t.name for t in teachers.matching("codeforge-27b-heretic-iq3", q4.repo, q4.file)], ["codeforge-27b-heretic-iq3", "codeforge-27b-heretic-q4"])
        self.assertEqual(teachers.matching("", q4.repo, ""), [], "a repo alone is not a GGUF")


class ModalStage(StubbedSampleStage):
    """The generic body of the stage (`sample_answers`) with a teacher that is in the catalogue and may only filter; the other tests of the stage call it with the name "m", which is in no entry."""

    def stage(self, **kw):
        with mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, GPT_OSS]):
            self.requests.clear()
            return self.data.sample_answers(GPT_OSS.name, GPT_OSS.repo, GPT_OSS.file, 2, self.tag, **kw)

    def test_the_stage_refuses_a_reasoning_or_thinking_run_of_a_filter_only_teacher_before_it_asks_anything(self):
        for kw in (dict(reasoning=True, max_len=4096), dict(reasoning=True, thinking=True, max_len=8192)):
            with self.assertRaisesRegex(SystemExit, "may only FILTER"):
                self.stage(**kw)
            self.assertEqual(self.requests, [])

    def test_a_default_run_of_the_same_teacher_is_not_touched(self):
        with mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, GPT_OSS]):
            self.run_stage(False)             # the default run, a teacher called "m"
        self.assertTrue(self.requests)
        calls = []
        real = teachers.require_reasoning
        with mock.patch.object(teachers, "require_reasoning", lambda *a, **k: calls.append(a) or real(*a, **k)):
            self.run_stage(False)
        self.assertEqual(calls, [], "no reasoning: the teacher is not asked about its origin, any open-weight model may filter")


class Entrypoint(ModalCase):
    def call(self, model, **kw):
        with tempfile.TemporaryDirectory() as d:
            return self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", model=model, k=4, **kw)

    def test_the_rea_entrypoint_refuses_before_a_gpu_is_rented_and_the_default_run_goes_through(self):
        with mock.patch.object(self.data, "MODELS", [*self.data.MODELS, GPT_OSS]), mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, GPT_OSS]):
            for kw in (dict(reasoning=True, max_len=4096), dict(reasoning=True, thinking=True, max_len=8192)):
                with self.assertRaisesRegex(SystemExit, "may only FILTER"):
                    self.call(GPT_OSS.name, **kw)
            remote, _ = self.call(GPT_OSS.name)
            self.assertEqual(remote.call_args.args[0], GPT_OSS.name, "default mode: it filters, whatever its origin")
            remote, _ = self.call("codeforge-27b-heretic-q4", reasoning=True, thinking=True, max_len=8192)
            self.assertTrue(remote.call_args.kwargs["thinking"])

    def test_a_run_config_with_a_filter_only_teacher_is_refused_the_same_way(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            cfg = pathlib.Path(d) / "c.json"
            cfg.write_text(json.dumps({**json.loads((ROOT / "distill" / "configs" / "rea-thinking.json").read_text()), "teacher": GPT_OSS.name}))
            with mock.patch.object(self.data, "MODELS", [*self.data.MODELS, GPT_OSS]), mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, GPT_OSS]):
                with self.assertRaisesRegex(SystemExit, "may only FILTER"):
                    self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", run_config=str(cfg))


class KaggleKit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        from .test_rea_kaggle import write_tasks
        from .test_rea_kaggle_kit import make_tasks
        self.tasks = write_tasks(self.d, make_tasks((("crc", 2),)))

    def build(self, out="kb", **kw):
        return build_tasks.build([self.tasks], self.d / out, "me", quota="crc=2", k=2, **kw)

    def test_the_kit_refuses_a_filter_only_teacher_by_name_or_by_gguf_and_writes_nothing(self):
        with mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, GPT_OSS]):
            for kw in ({"teacher": GPT_OSS.name}, {"model_repo": GPT_OSS.repo, "model_file": GPT_OSS.file}):
                with self.assertRaisesRegex(SystemExit, "may only FILTER"):
                    self.build(**kw)
        self.assertFalse((self.d / "kb").exists(), "refused before anything was written")

    def test_the_kit_refuses_a_gguf_that_is_in_no_entry(self):
        with self.assertRaisesRegex(SystemExit, "is in no entry of distill/teachers.py"):
            self.build(model_repo="someone/some-model-GGUF", model_file="some-model.gguf")
        with self.assertRaisesRegex(SystemExit, "is in no entry"):
            self.build(model_repo="someone/some-model-GGUF")        # a repo that is not the default's: the default's file is not its file

    def test_the_default_and_a_catalogue_teacher_build(self):
        cfg = self.build()
        self.assertIn("CodeForge", cfg["model_repo"])
        cfg = self.build("kb2", teacher="qwen35-9b")
        self.assertEqual(cfg["model_repo"], teachers.find("qwen35-9b").repo)

    def test_the_kernels_default_is_a_teacher_that_may_store_text(self):
        sys.path.insert(0, str(ROOT / "distill" / "backends" / "kaggle"))
        try:
            import run_teacher
        finally:
            sys.path.remove(str(ROOT / "distill" / "backends" / "kaggle"))
        found = teachers.matching("", run_teacher.DEFAULTS["model_repo"], run_teacher.DEFAULTS["model_file"])
        self.assertEqual([t.name for t in found], ["codeforge-27b-heretic-q4"])
        self.assertTrue(found[0].reasoning_ok)


class Docs(unittest.TestCase):
    def test_the_rule_is_in_the_catalogue_the_code_and_the_readmes(self):
        cand = (ROOT / "docs" / "TEACHER_CANDIDATES.md").read_text()
        for part in ("What a teacher may do depends on what the run stores", "any open-weight model may FILTER, whatever its origin", "not trained by or distilled from an OpenAI or Anthropic model",
                     "gpt-oss-\\* (OpenAI's own models", "the GPT-4o distill are filter-only", "`Teacher.reasoning_ok`", "**Filter only** (an OpenAI model: its text is not stored)",
                     "**Filter only** (a distill of GPT-4o", "its lineage is not settled", "Opus 5.5 and GPT-6 Astra", "is the owner's call"):
            self.assertIn(part, cand)
        self.assertNotIn("any open-weight model can be a teacher, whatever its origin", cand)
        code = (ROOT / "distill" / "teachers.py").read_text()
        for part in ("any open-weight teacher may FILTER, whatever its origin", "NOT trained by or distilled from an OpenAI or Anthropic model", "gpt-oss-*", "the GPT-4o distill", "PROVENANCE OPEN"):
            self.assertIn(part, code)
        for name in ("distill/README.md", "tools/modal/README.md"):
            text = (ROOT / name).read_text()
            self.assertIn("reasoning_ok", text, name)
            self.assertIn("gpt-oss-*", text, name)
            self.assertNotIn("No output of an OpenAI or Anthropic model enters a dataset", text)
            self.assertNotIn("does not forbid it", text)
        self.assertNotIn("no output of an OpenAI or Anthropic model enters a dataset", (ROOT / "distill" / "backends" / "modal_sample.py").read_text())

    def test_the_candidate_rows_that_are_openai_or_a_gpt4o_distill_say_filter_only(self):
        rows = [l for l in (ROOT / "docs" / "TEACHER_CANDIDATES.md").read_text().splitlines() if l.startswith("|") and ("gpt-oss" in l or "gpt-4o" in l)]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all("Filter only" in r for r in rows), rows)


if __name__ == "__main__":
    unittest.main()

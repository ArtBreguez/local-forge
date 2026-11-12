"""Round 15, m8 (and round 12, M2 and M3): the owner's rule about teachers, as code. Any open-weight teacher may FILTER in the default mode, whatever its origin; in the reasoning and thinking modes the teacher's
reply is stored, and the dataset rule is that no DIRECT OpenAI or Anthropic output is stored (gpt-oss-* and the GPT-4o distill are filter-only); a lineage that includes fine-tunes on such data is accepted case by
case by the owner (CodeForge, 2026-11-12).

  Catalogue    `Teacher.reasoning_ok`: the fifth field, False by default, True for the shipped teachers, never True for an entry whose name says OpenAI or Anthropic
  Refusal      `teachers.require_reasoning`: by name, by repo and file, strict for the Kaggle kit
  ModalStage   the REA entrypoint and the generic sample stage refuse `--reasoning` / `--thinking`, and a default run is not touched
  KaggleKit    `build_tasks` refuses a teacher without it, and a GGUF that is in no entry
  Bypasses     round 12, M2: a catalogue name that carries another GGUF (`--teacher` / `--run-config` with `--model-repo` / `--model-file`), a GGUF that says gpt-oss and is in no entry, the
               generic sample stage, and `kaggle-score` reading the GGUF run.json names
  Docs         the rule in docs/TEACHER_CANDIDATES.md, distill/teachers.py, distill/README.md, tools/modal/README.md, and the owner's decision about CodeForge with its lineage (round 12, M3)
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from .helpers import ROOT
from .test_rea_kaggle import stamp, write_replies, write_tasks
from .test_rea_kaggle_kit import make_tasks
from .test_rea_modal import ModalCase, StubbedSampleStage
from .test_rea_reasoning import reply
from .thinking_pilot import build_tasks, run_teacher

from distill import teachers  # noqa: E402
from rea_bench import kaggle_score  # noqa: E402

GPT_OSS = teachers.Teacher("gpt-oss-120b-derestricted", "ArliAI/gpt-oss-120b-Derestricted-GGUF", "gpt-oss-120b-Derestricted-Q4_K_M.gguf", "L40S", False)
GPT_OSS_GGUF = ("unsloth/gpt-oss-120b-GGUF", "gpt-oss-120b-Q4_K_M.gguf")        # the review's: a GGUF that no entry of the catalogue has, and that says it is OpenAI's
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
        self.assertEqual([t.name for t in teachers.TEACHERS if not t.reasoning_ok], [], "the shipped teachers may store text (CodeForge's lineage is accepted by the owner, 2026-11-12)")
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


class Bypasses(unittest.TestCase):
    """Round 12, M2: the gate judged only the catalogue entries it found, by name OR by repo and file, so a name that may store text carried any GGUF with it, and a GGUF that says gpt-oss but is in no entry
    passed the Modal stage's generic body. One test per way the review found; each is refused with the catalogue as it ships (no entry added for the occasion)."""

    def test_a_catalogue_name_does_not_carry_another_gguf(self):
        for name in ("qwen35-9b", "codeforge-27b-heretic-q4"):
            for strict in (True, False):
                with self.assertRaisesRegex(SystemExit, "name and the GGUF disagree"):
                    teachers.require_reasoning(name, *GPT_OSS_GGUF, strict=strict)
        other = teachers.find("qwen35-27b")
        with self.assertRaisesRegex(SystemExit, "name and the GGUF disagree.*qwen35-9b is "):
            teachers.require_reasoning("qwen35-9b", other.repo, other.file, strict=True)       # another entry that may store text is still not this name's GGUF
        own = teachers.find("qwen35-9b")
        teachers.require_reasoning("qwen35-9b", own.repo, own.file, strict=True)                # its own GGUF, and the name alone, pass
        teachers.require_reasoning("qwen35-9b", strict=True)
        teachers.require_reasoning("a-name-no-entry-has", own.repo, own.file, strict=True)      # the same GGUF under another name is the same teacher (`matching`)

    def test_the_text_given_is_read_for_the_markers_even_when_no_entry_has_it(self):
        for kw in (dict(repo=GPT_OSS_GGUF[0], file=GPT_OSS_GGUF[1]), dict(name="m", repo=GPT_OSS_GGUF[0], file=GPT_OSS_GGUF[1]), dict(name="claude-distill-9b", repo="x/y", file="z.gguf"),
                   dict(name="m", repo="x/y", file="Opus-9b.gguf"), dict(repo=GPT_OSS_GGUF[0])):
            for strict in (True, False):
                with self.assertRaisesRegex(SystemExit, "may only FILTER"):
                    teachers.require_reasoning(strict=strict, **kw)
        teachers.require_reasoning("m", "repo", "f.gguf")                       # the generic stage's own names in the tests: nothing says no

    def test_the_kit_refuses_the_flags_the_review_used_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            tasks = write_tasks(d, make_tasks((("crc", 2),)), "decompiled-seed9-all.jsonl")
            thinking = str(ROOT / "distill" / "configs" / "rea-thinking.json")
            base = ["--tasks", str(tasks), "--owner", "me"]
            own = teachers.find("codeforge-27b-heretic-q4")
            run = lambda out, *args: build_tasks.main([*base, "--out", str(d / out), *args])
            for out, args, why in (
                    ("a", ("--run-config", thinking, "--model-repo", GPT_OSS_GGUF[0], "--model-file", GPT_OSS_GGUF[1]), "do not also give another GGUF \\(--model-repo, --model-file\\)"),
                    ("b", ("--run-config", thinking, "--model-file", GPT_OSS_GGUF[1]), "do not also give another GGUF \\(--model-file\\)"),
                    ("c", ("--run-config", thinking, "--model-repo", teachers.find("qwen35-9b").repo), "do not also give another GGUF \\(--model-repo\\)"),
                    ("d", ("--quota", "crc=1", "--teacher", "qwen35-9b", "--model-repo", GPT_OSS_GGUF[0], "--model-file", GPT_OSS_GGUF[1]), "name and the GGUF disagree"),
                    ("e", ("--quota", "crc=1", "--teacher", "qwen35-9b", "--model-file", own.file), "name and the GGUF disagree"),
                    ("f", ("--quota", "crc=1", "--model-repo", GPT_OSS_GGUF[0], "--model-file", GPT_OSS_GGUF[1]), "may only FILTER")):
                with self.subTest(out), self.assertRaisesRegex(SystemExit, why):
                    run(out, *args)
                self.assertFalse((d / out).exists(), f"{out}: refused before anything was written")
            with contextlib.redirect_stdout(io.StringIO()):
                run("ok1", "--run-config", thinking)
                run("ok2", "--run-config", thinking, "--model-repo", own.repo, "--model-file", own.file)          # the entry's own GGUF, given again: the same run
                run("ok3", "--quota", "crc=1", "--teacher", "qwen35-9b", "--model-repo", teachers.find("qwen35-9b").repo)
            for out, repo in (("ok1", own.repo), ("ok2", own.repo), ("ok3", teachers.find("qwen35-9b").repo)):
                self.assertEqual(json.loads((d / out / "data" / "config.json").read_text())["model_repo"], repo)

    def run_info(self, repo, file):
        return {"thinking": True, "model_repo": repo, "model_file": file, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}, "max_tokens": 4096}

    def score(self, model, info):
        with tempfile.TemporaryDirectory() as d:
            tasks = make_tasks((("structs", 2),))
            steps = ("The function reads a field at a fixed offset from the base.", "Following the loop gives the key material directly.")       # two distinct replies: the agreement rule keeps the task
            lines = stamp([{"id": t["id"], "kind": t["kind"], "outs": [{"text": run_teacher.stored_text(reply(t, step), f"Reasoning {i} about {t['id']}: it reads the field.", True),
                                                                     "finish": "stop", "tokens": 400, "prompt_tokens": 200, "secs": 1.0} for i, step in enumerate(steps)]} for t in tasks], tasks)
            return kaggle_score.score(write_replies(d, lines), write_tasks(d, tasks, "decompiled-seed9-all.jsonl"), model, thinking=True, run_info=info)

    def test_kaggle_score_reads_the_gguf_run_json_names_before_it_stores_the_replies(self):
        default = (run_teacher.DEFAULTS["model_repo"], run_teacher.DEFAULTS["model_file"])
        for model, info, why in (("codeforge-27b-heretic-q4", self.run_info(*GPT_OSS_GGUF), "name and the GGUF disagree"),          # the label says CodeForge, the run asked gpt-oss
                                 ("m", self.run_info(*GPT_OSS_GGUF), "may only FILTER"),
                                 ("qwen35-9b", self.run_info(*default), "name and the GGUF disagree"),                             # the label is another catalogue entry than the GGUF that ran
                                 ("gpt-oss-120b", self.run_info(*default), "may only FILTER"),                                    # the label itself says OpenAI
                                 ("m", self.run_info("someone/some-model-GGUF", "some-model.gguf"), "is in no entry of distill/teachers.py")):
            with self.subTest(model=model, why=why), self.assertRaisesRegex(SystemExit, why):
                self.score(model, info)
        kept, stats = self.score("codeforge-27b-heretic-q4", self.run_info(*default))
        self.assertTrue(kept and stats["run"]["model_repo"] == default[0], "the teacher of the catalogue, under its own name")
        kept, stats = self.score("m", {k: v for k, v in self.run_info(*default).items() if k not in ("model_repo", "model_file")})
        self.assertTrue(kept, "a run.json that does not say which GGUF ran (an older kernel) is not judged: there is nothing to judge")


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

    def test_the_stage_refuses_a_catalogue_name_with_another_gguf_and_a_gguf_that_says_gpt_oss_whatever_its_name(self):
        """Round 12, M2: another subject calls the generic body with a catalogue name and any repo and file (and the kit's strictness does not apply here)."""
        for name, why in (("qwen35-9b", "name and the GGUF disagree"), ("codeforge-27b-heretic-q4", "name and the GGUF disagree"), ("m", "may only FILTER"), ("", "may only FILTER")):
            for kw in (dict(reasoning=True, max_len=4096), dict(reasoning=True, thinking=True, max_len=8192)):
                with self.subTest(name=name), self.assertRaisesRegex(SystemExit, why):
                    self.requests.clear()
                    self.data.sample_answers(name, *GPT_OSS_GGUF, 2, self.tag, **kw)
                self.assertEqual(self.requests, [])
        self.run_stage(False)               # and a default run of any name and GGUF still filters
        self.assertTrue(self.requests)

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
        for part in ("What a teacher may do depends on what the run stores", "any open-weight model may FILTER, whatever its origin", "no DIRECT OpenAI or Anthropic output is stored", "case by case by the owner, with the decision recorded",
                     "A dataset card that publishes teacher-reasoning records must state the teacher's lineage", "gpt-oss-\\* (OpenAI's own models", "the GPT-4o distill are filter-only", "`Teacher.reasoning_ok`", "**Filter only** (an OpenAI model: its text is not stored)",
                     "**Filter only** (a distill of GPT-4o", "The teacher in use: accepted by the owner (2026-11-12), lineage recorded", "Decision (the owner, 2026-11-12): CodeForge is accepted",
                     "Coder390 EfficientThink", "Opus 5.5 and GPT-6 Astra", "Fable Distill", "claude-fable-5-claude-code", "Fable-5-Chat", "Qwopus Coder Flash", "Qwen, DeepSeek and NVIDIA teachers",
                     "no dataset-card template"):
            self.assertIn(part, cand)
        for gone in ("its lineage is not settled", "is the owner's call", "a teacher that was not trained by or distilled from"):
            self.assertNotIn(gone, cand)
        self.assertNotIn("any open-weight model can be a teacher, whatever its origin", cand)
        code = (ROOT / "distill" / "teachers.py").read_text()
        for part in ("any open-weight teacher may FILTER, whatever its origin", "no DIRECT OpenAI or Anthropic output is stored", "accepted case by case by the", "gpt-oss-*", "the GPT-4o distill",
                     "ACCEPTED BY THE OWNER on 2026-11-12", "Coder390", "Opus 5.5 and GPT-6 Astra", "claude-fable-5-claude-code and Fable-5-Chat", "Qwen, DeepSeek and",
                     "A dataset card that publishes teacher-reasoning records must state the teacher's lineage"):
            self.assertIn(part, code)
        for gone in ("PROVENANCE OPEN", "NOT trained by or distilled from", "If the owner reads that"):
            self.assertNotIn(gone, code)
        for name in ("distill/README.md", "tools/modal/README.md"):
            text = (ROOT / name).read_text()
            self.assertIn("reasoning_ok", text, name)
            self.assertIn("gpt-oss-*", text, name)
            self.assertRegex(text, r"[Nn]o DIRECT (OpenAI or Anthropic output|output of an OpenAI or Anthropic model)", name)
            self.assertIn("CodeForge was accepted on 2026-11-12", text, name)
            self.assertIn("a dataset card that publishes teacher-reasoning records must state the teacher's lineage", text, name)
            self.assertNotIn("open question for the owner", text, name)
            self.assertNotIn("lineage is not settled", text, name)
            self.assertNotIn("No output of an OpenAI or Anthropic model enters a dataset", text)
            self.assertNotIn("does not forbid it", text)
        stage = (ROOT / "distill" / "backends" / "modal_sample.py").read_text()
        self.assertNotIn("no output of an OpenAI or Anthropic model enters a dataset", stage)
        self.assertIn("a direct OpenAI or Anthropic model: gpt-oss-*, the GPT-4o distill; a lineage that includes fine-tunes on such data is the owner's call", stage)
        self.assertNotIn("(trained by or distilled from an OpenAI or Anthropic model: gpt-oss-*", stage)
        self.assertIn("and it must state the teacher's lineage (CodeForge's: `docs/TEACHER_CANDIDATES.md`, last section)", (ROOT / "tools" / "modal" / "README.md").read_text(), "next to what the dataset card must say about the steps")

    def test_the_decision_about_codeforge_is_in_the_catalogue_and_both_entries_say_true(self):
        """The owner accepted it (2026-11-12): the flag stays True for both quantizations, the thinking config and the kernel's default are not refused, and the filter-only ones still are."""
        for name in ("codeforge-27b-heretic-iq3", "codeforge-27b-heretic-q4"):
            self.assertTrue(teachers.find(name).reasoning_ok, name)
            teachers.require_reasoning(name, strict=True)
        import json
        cfg = json.loads((ROOT / "distill" / "configs" / "rea-thinking.json").read_text())
        teachers.require_reasoning(cfg["teacher"], strict=True)
        for t in (GPT_OSS, GPT4O):
            with mock.patch.object(teachers, "TEACHERS", [*teachers.TEACHERS, t]), self.assertRaisesRegex(SystemExit, "may only FILTER"):
                teachers.require_reasoning(t.name)

    def test_the_gate_is_described_where_the_rule_is(self):
        text = (ROOT / "distill" / "README.md").read_text()
        for part in ("judges what was GIVEN", "name and GGUF disagree", "`--run-config` refuses any `--model-repo` / `--model-file`", "`kaggle-score` reads the GGUF that run.json names"):
            self.assertIn(part, text)
        code = (ROOT / "distill" / "teachers.py").read_text()
        for part in ('name and the GGUF disagree', "FILTER_ONLY_MARKERS", "the text given (name, repo, file) is read for those markers too"):
            self.assertIn(part, code)

    def test_the_candidate_rows_that_are_openai_or_a_gpt4o_distill_say_filter_only(self):
        rows = [l for l in (ROOT / "docs" / "TEACHER_CANDIDATES.md").read_text().splitlines() if l.startswith("|") and ("gpt-oss" in l or "gpt-4o" in l)]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all("Filter only" in r for r in rows), rows)


if __name__ == "__main__":
    unittest.main()

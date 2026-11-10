"""Tests for tools/rea-bench: the frozen legacy family, the scorer, and every task family.

Per family (generated classes `Family_<name>`):
  (a) the C compiles cleanly with gcc -Wall -Werror at -O0..-O3 and -Os (and with clang when it is installed);
  (b) a small harness compiled from the same source (at -O0, -O2 and -O1 under UBSan) prints what a Python reference
      computed from the task's parameters, which proves the truth the question is built on;
  (c) the scorer accepts the canonical answer and rejects a plausible wrong one.
Compiling is sequential and each file is tiny; the whole module takes about a minute.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from .helpers import ROOT

RB = ROOT / "tools" / "rea-bench"
sys.path.insert(0, str(RB))

from rea_bench import core, families, scoring, verify  # noqa: E402
from rea_bench.verify import CLANG, compile_check, run_harness  # noqa: E402


def digest(directory: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.glob("*")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def build_binary(src: str, opt: str, out: pathlib.Path, extra=()) -> None:
    r = subprocess.run(["gcc", opt, "-fno-lto", *extra, "-o", str(out), "-x", "c", "-"], input=src, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def global_text_symbols(src: str, opt: str) -> set:
    with tempfile.TemporaryDirectory() as d:
        exe = pathlib.Path(d, "t.sym")
        build_binary(src, opt, exe)
        nm = subprocess.run(["nm", str(exe)], capture_output=True, text=True).stdout
    return {line.split()[2] for line in nm.splitlines() if len(line.split()) == 3 and line.split()[1] == "T"}


UBSAN = verify.ubsan_available()


class LegacyPin(unittest.TestCase):
    """The ten original templates must keep producing byte-identical files for a seed (the held-out split is seed 1001)."""
    PINS = {1001: "97beddfa668013b6a82176c4bf4108b3726b2d9a1c1ba3714d4aaef24a9b2418",
            7: "9505981ce3199e87db6490807a798603b08c84a391ee579b5c9f25c0b91a592a"}

    def test_in_process(self):
        for seed, want in self.PINS.items():
            with tempfile.TemporaryDirectory() as d:
                core.write_tasks(families.generate("legacy", 60, seed), pathlib.Path(d))
                self.assertEqual(len(list(pathlib.Path(d).glob("*"))), 120)
                self.assertEqual(digest(pathlib.Path(d)), want, f"legacy output for seed {seed} changed")

    def test_cli_one_command_and_old_invocation(self):
        for args in (["gen", "--family", "legacy"], []):   # `[]` = the old gen_rea_tasks.py command line
            with tempfile.TemporaryDirectory() as d:
                r = subprocess.run([sys.executable, str(RB), *args, "--n", "60", "--seed", "1001", "--out", d], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(digest(pathlib.Path(d)), self.PINS[1001])

    def test_legacy_json_has_no_new_fields(self):
        t = families.generate("legacy", 10, 1001)[0]
        self.assertEqual(list(t.meta()), ["id", "kind", "question", "opt", "function", "score", "accept"])


class Scorer(unittest.TestCase):
    def test_original_modes_unchanged(self):
        s = scoring.score
        self.assertTrue(s("the key is Abc123", {"score": "exact", "accept": ["abc123"]}))
        self.assertTrue(s("magic 0xDEADBEEF version 3", {"score": "all", "accept": ["deadbeef", "3"]}))
        self.assertTrue(s("it is 0x9E3779B9", {"score": "all", "accept": ["2654435769"]}))   # decimal truth, hex answer
        self.assertTrue(s("djb2 and 0x1234abcd", {"score": "all_groups", "accept": [["djb2", "djb"], ["1234abcd"]]}))
        self.assertFalse(s("fnv and 0x1234abcd", {"score": "all_groups", "accept": [["djb2", "djb"], ["1234abcd"]]}))
        self.assertTrue(s("skips 0x20", {"score": "any", "accept": ["0x20", "20", "32"]}))
        self.assertTrue({"9e3779b9", "0x9e3779b9"} <= set(scoring.variants("2654435769")))

    def test_tokens_are_whole_tokens(self):
        t = {"score": "tokens", "accept": [["8", "0x8"]]}
        self.assertTrue(scoring.score("The field is at offset 8.", t))
        self.assertTrue(scoring.score("offset **0x8**", t))
        self.assertFalse(scoring.score("offset 0x18", t))      # substring scoring would accept this ("0x18" contains "8")
        self.assertFalse(scoring.score("offset 18", t))
        self.assertFalse(scoring.score("offset 80", t))

    def test_tokens_reject_and_hyphen_words(self):
        tea = {"score": "tokens", "accept": [["tea"]], "reject": ["xtea", "x-tea", "xxtea", "rc4"]}
        self.assertTrue(scoring.score("It is TEA (Tiny Encryption Algorithm).", tea))
        self.assertFalse(scoring.score("This is XTEA", tea))
        self.assertFalse(scoring.score("This is X-TEA", tea))
        self.assertFalse(scoring.score("TEA or maybe RC4", tea))
        b64 = {"score": "tokens", "accept": [["base64", "base-64"]]}
        self.assertTrue(scoring.score("Base-64 encoder", b64))
        self.assertTrue(scoring.score("base64", b64))

    def test_tokens_signs_and_ranges(self):
        t = {"score": "tokens", "accept": [["-37"], ["40"]]}
        self.assertTrue(scoring.score("lower bound is -37, upper 40", t))
        self.assertTrue(scoring.score("(-37, 40)", t))
        self.assertFalse(scoring.score("lower bound 37, upper 40", t))
        r = {"score": "tokens", "accept": [["5"], ["9"]]}
        self.assertTrue(scoring.score("range 5-9", r))        # a hyphen between digits separates the numbers

    def test_exact_case_is_case_sensitive(self):
        t = {"score": "exact_case", "accept": ["aBcD"]}
        self.assertTrue(scoring.score("alphabet: `aBcD`", t))
        self.assertTrue(scoring.score("a B c D", t))          # whitespace is ignored
        self.assertFalse(scoring.score("AbCd", t))
        self.assertFalse(scoring.score("abcd", t))

    def test_only_the_headline_counts(self):
        """B1: numbers a justification restates must not turn a wrong headline into a match (reviewer's three repros)."""
        s = scoring.score
        tok = lambda *g, **k: {"score": "tokens", "accept": list(g), **k}
        self.assertFalse(s("1 byte; the loop stops at n - 2 and compares p[n - 2]", tok(core.dec(2))))
        self.assertFalse(s("8 states (numbered 0 to 7), from the 8-row transition table", tok(core.dec(7))))
        self.assertFalse(s("Lowest 1571, highest 1750 (the code tests v > 1570 && v < 1751)", tok(core.dec(1570), core.dec(1751))))
        self.assertTrue(s("**2**\nThe loop stops at n - 2, so 7 bytes are skipped in p[n - 1].", tok(core.dec(2))))
        self.assertFalse(s("The loop stops at n - 3.\n2", tok(core.dec(2))), "the headline is the first line, not any line")
        self.assertTrue(s("Answer:\n\n`0x10`\nit is read at p + 0x20", tok(core.hx(0x10))), "a bare lead-in line is skipped")
        self.assertEqual(scoring.headline("```\n- **0x18** bytes\n```"), "0x18 bytes")
        self.assertEqual(scoring.headline("  \n\n"), "")

    def test_a_reply_that_lists_candidates_is_not_an_answer(self):
        s = scoring.score
        one = {"score": "tokens", "accept": [core.dec(2)]}
        for reply in ("0 1 2", "1 2 3", "2 or 3", "2, maybe 0x10", "0 1 2 3 4 8 16 32 64 255 256 0x10 0x20"):
            self.assertFalse(s(reply, one), reply)
        self.assertTrue(s("2", one) and s("0x2", one) and s("it is 2.", one))
        off = {"score": "tokens", "accept": [core.hx(0x10)]}
        self.assertFalse(s("0x10 0x20", off) or s("0x10 or edb88320", off))
        fsm = {"score": "tokens", "accept": [["bcd", "bdc"]], "cand": "[bcd]{2,}"}
        self.assertTrue(s("bcd", fsm) and s("b c d", fsm) and s("b, d, c", fsm))
        self.assertFalse(s("bcd or ccd", fsm))
        self.assertTrue(s("bcd or bdc", fsm), "both are shortest inputs")

    def test_bounds_must_come_in_order(self):
        both = {"score": "tokens", "accept": [core.dec(1570), core.dec(1751)]}
        self.assertTrue(scoring.score("Lowest 1570, highest 1751", both))
        self.assertTrue(scoring.score("1570 to 1751", both))
        self.assertFalse(scoring.score("highest 1751, lowest 1570", both))
        self.assertFalse(scoring.score("1751 1570", both))
        self.assertFalse(scoring.score("1570", both))

    def test_number_spellings(self):
        """M3: padded hex, C suffixes and digit separators are the same number; units are not converted."""
        s = scoring.score
        small = {"score": "tokens", "accept": [core.hx(8)]}
        for r in ("0x8", "0x08", "0x0008", "0X8"):
            self.assertTrue(s(r, small), r)
        self.assertTrue(s("8", small), "the same number in decimal")
        self.assertFalse(s("10", {"score": "tokens", "accept": [core.hx(0x10)]}), "bare 10 is decimal ten, not 0x10")
        self.assertTrue(s("16", {"score": "tokens", "accept": [core.hx(0x10)]}))
        poly = {"score": "tokens", "accept": [core.hx(0xedb88320, 8), ["reflected"]], "reject": ["normal"]}
        for r in ("0xedb88320, reflected", "0xEDB88320u, reflected", "0xedb88320UL reflected", "edb88320 reflected", "0x0edb88320 reflected"):
            self.assertTrue(s(r, poly), r)
        self.assertFalse(s("0xedb88320, normal", poly))
        thr = {"score": "tokens", "accept": [core.dec(250000000)]}
        for r in ("250000000", "250,000,000 ns", "250_000_000", "250'000'000", "250000000ull"):
            self.assertTrue(s(r, thr), r)
        self.assertFalse(s("250 ms", thr) or s("250000", thr) or s("250,000", thr))
        self.assertTrue(s("-37", {"score": "tokens", "accept": [core.dec(-37)]}))
        self.assertFalse(s("37", {"score": "tokens", "accept": [core.dec(-37)]}))

    def test_reject_looks_at_the_headline_only(self):
        tracer = {"score": "tokens", "accept": [["tracerpid"], core.dec(3)], "reject": ["ptrace", "env", "timing"]}
        self.assertTrue(scoring.score("tracerpid, returns 3\nIt notices that a debugger is ptrace-attached via /proc.", tracer))
        self.assertFalse(scoring.score("tracerpid or ptrace, returns 3", tracer))

    def test_bare_hex_words_are_not_accepted(self):
        """m1: 'a', 'be', 'ad', 'dead' are words and '10' is decimal ten; only 0x... (or an unambiguous long bare form) counts."""
        self.assertEqual(core.hx(0xa), ["0xa"])
        self.assertEqual(core.hx(0xbe, 2), ["0xbe"])
        self.assertEqual(core.hx(0x10), ["0x10"])
        self.assertEqual(core.hx(0xdead, 4), ["0xdead"])
        self.assertEqual(core.hx(0xa001, 4), ["0xa001", "a001"])
        self.assertEqual(core.hx(0), ["0x0", "0"])
        wordy = {"score": "tokens", "accept": [core.hx(0xa)]}
        self.assertFalse(scoring.score("It is a field, and a flag", wordy))
        self.assertTrue(scoring.score("0xa", wordy))

    def test_exact_head_and_unknown_modes(self):
        t = {"score": "exact_head", "accept": ["admin_login42"]}
        self.assertTrue(scoring.score("`admin_login42`\nXOR 0x37", t))
        self.assertTrue(scoring.score("Here is the string:\nAdmin_Login42", t))
        self.assertFalse(scoring.score("I do not know.\nadmin_login42", t))
        with self.assertRaises(ValueError):
            scoring.score("x", {"score": "fuzzy", "accept": ["x"], "id": "t1"})
        self.assertEqual(scoring.canonical(t), "admin_login42")

    def test_legacy_modes_still_score_the_whole_reply(self):
        """The 34/60 -> 50/60 numbers were measured with whole-reply substring scoring; that must not change."""
        self.assertTrue(scoring.score("Probably a guess.\nThe key is 0xAB", {"score": "exact", "accept": ["ab"]}))
        self.assertTrue(scoring.score("first line says nothing\nseed 7 and 0x1f", {"score": "all", "accept": ["7", "1f"]}))

    def test_canonical_roundtrip(self):
        for t in ({"score": "tokens", "accept": [["a1", "0xa1"], ["tea"]]}, {"score": "exact", "accept": ["x"]},
                  {"score": "all", "accept": ["12", "ff"]}, {"score": "all_groups", "accept": [["djb"], ["12ab"]]}):
            self.assertTrue(scoring.score(scoring.canonical(t), t), t)

    def test_modal_module_uses_the_package_scorer(self):
        data = (ROOT / "tools" / "modal" / "modal_rea_data.py").read_text()
        self.assertIn("from rea_bench.scoring import check_mode, score", data)
        self.assertNotIn("def _score", data)
        self.assertIn("task.get(\"cc\", \"gcc\")", data)
        rea = (ROOT / "tools" / "modal" / "modal_rea.py").read_text()
        self.assertIn("from rea_bench.scoring import check_mode, score", rea)
        self.assertIn('add_local_dir(str(RB / "rea_bench"), "/root/rea_bench")', rea, "the container needs the package mounted")
        self.assertRegex(rea, r"(?m)^def sh\(")
        self.assertNotRegex(rea, r"(?m)^def (score|_variants)\(", "modal_rea.py must not keep a private copy of the scorer")
        self.assertIn('"clang"', rea)
        body = data[data.index("def sample_answers"):data.index("srv = subprocess.Popen")]
        self.assertLess(body.index("check_mode(t)"), body.index("hf_hub_download(repo"), "validate the tasks before the model download")


class ContentFreeReplies(unittest.TestCase):
    """B1: replies with no answer in them ("0 1 2", "0x10 0x20", every common number at once) must fail every task of every new
    family. The reviewer measured 56% of pktsum tasks passing "0 1 2" with whole-reply token matching."""
    SEEDS = (3, 4, 5)
    N = 600

    def test_no_new_family_accepts_a_filler_reply(self):
        for fam in families.modules():
            if fam == "legacy":
                continue
            total = passed = 0
            for seed in self.SEEDS:
                for t in families.generate(fam, self.N, seed):
                    total += 1
                    hit = verify.filler_passes(t)
                    passed += bool(hit)
                    self.assertFalse(hit, f"{t.id} ({t.variant}) accepts {hit}: {t.truth}")
            self.assertEqual(passed, 0, f"{fam}: {passed}/{total} tasks accept a filler reply")

    def test_the_fillers_are_what_the_review_used(self):
        self.assertEqual(verify.FILLER_REPLIES[:3], ["0 1 2", "1 2 3", "0x10 0x20"])
        self.assertIn("0 1 2 3 4 8 16 32 64 255 256 0x10 0x20", verify.FILLER_REPLIES)


class Regressions(unittest.TestCase):
    """Things the independent review (.review-phase-a.md) found, kept as tests."""

    def test_m2_m3_every_task_is_consistent_for_seeds_that_used_to_break(self):
        # m2: bounds-77-0008 (index_limit form 5 with limit == a) accepted its own wrong answer; m3: fsm seed 3 died with StopIteration
        for fam, n, seeds in (("bounds", 240, (2, 7, 77)), ("fsm", 1200, (3,)), ("antidebug", 120, (7,)), ("pktsum", 240, (7,))):
            for seed in seeds:
                for t in families.generate(fam, n, seed):
                    self.assertTrue(scoring.score(scoring.canonical(t.truth), t.truth), f"{t.id} canonical rejected")
                    self.assertFalse(scoring.score(t.wrong, t.truth), f"{t.id} ({t.variant}) accepts its own wrong answer {t.wrong!r}: {t.truth}")

    def test_m2_two_bounds_always_differ(self):
        for t in families.generate("bounds", 600, 77):
            acc = t.truth["accept"]
            if len(acc) == 2:
                self.assertNotEqual(acc[0], acc[1], f"{t.id} ({t.variant})")

    def test_m5_env_names_are_not_english_words(self):
        from rea_bench import f_antidebug
        for name in f_antidebug.ENV_VARS:
            self.assertTrue("_" in name or len(name) >= 10, f"{name} would be accepted from any prose that happens to contain it")
        self.assertNotIn("LINES", f_antidebug.ENV_VARS)
        self.assertNotIn("COLUMNS", f_antidebug.ENV_VARS)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(RB), *args], capture_output=True, text=True)

    def test_m6_help_is_help(self):
        r = self.run_cli("--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("{gen,list,check", r.stdout)
        self.assertNotIn("--family legacy", r.stdout)

    def test_m6_gen_refuses_a_non_empty_out_and_force_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_cli("gen", "--family", "crc", "--n", "6", "--seed", "5", "--out", d).returncode, 0)
            r = self.run_cli("gen", "--family", "crc", "--n", "3", "--seed", "5", "--out", d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--force", r.stderr)
            self.assertEqual(len(list(pathlib.Path(d).glob("*"))), 12, "nothing was touched")
            pathlib.Path(d, "notes.txt").write_text("mine")
            self.assertEqual(self.run_cli("gen", "--family", "crc", "--n", "3", "--seed", "5", "--out", d, "--force").returncode, 0)
            self.assertEqual(sorted(p.name for p in pathlib.Path(d).glob("*")),
                             sorted(["notes.txt"] + [f"crc-5-000{i}.{e}" for i in range(3) for e in ("c", "json")]), "stale tasks removed, other files kept")

    def test_m6_n_smaller_than_the_family_count(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("gen", "--family", "all", "--n", "5", "--seed", "5", "--out", d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("smaller than the 11 families", r.stderr)
            self.assertEqual(list(pathlib.Path(d).glob("*")), [])
            self.assertNotEqual(self.run_cli("check", "--family", "all", "--n", "5").returncode, 0)


class ModalScripts(unittest.TestCase):
    """M1: modal_rea.py kept its own scorer and every `tokens` task crashed it. Imported here against a stub `modal` module."""

    @classmethod
    def setUpClass(cls):
        from unittest import mock
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

    def test_one_scorer_everywhere(self):
        self.assertIs(self.rea.score, scoring.score)
        self.assertIs(self.data.score, scoring.score)
        self.assertIs(self.rea.check_mode, scoring.check_mode)

    def test_tokens_and_exact_case_tasks_no_longer_crash_or_misscore(self):
        for t in families.generate("crc", 12, 99) + families.generate("encoding", 12, 99):
            meta = t.meta()
            self.assertTrue(self.rea.score(scoring.canonical(t.truth), meta), t.id)      # TypeError before: accept is a list of lists
            self.assertFalse(self.rea.score(t.wrong, meta), t.id)
        case = {"score": "exact_case", "accept": ["aBcD"]}
        self.assertFalse(self.rea.score("abcd", case), "the old copy case-folded alphabets")
        self.assertTrue(self.rea.score("aBcD", case))

    def test_fixed_questions_keep_their_legacy_scoring(self):
        fixed = json.loads((ROOT / "tools" / "modal" / "rea-bench" / "rea-tasks.json").read_text())["tasks"]
        self.assertTrue(fixed)
        for t in fixed:
            scoring.check_mode(t)
            self.assertTrue(self.rea.score(scoring.canonical(t) if t["score"] != "any" else t["accept"][0], t), t["id"])

    def test_main_stops_on_an_unknown_score_mode_before_renting_a_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "tools" / "modal" / "rea-bench").mkdir(parents=True)
            shutil.copy(ROOT / "tools" / "modal" / "rea-bench" / "rea-tasks.json", root / "tools" / "modal" / "rea-bench")
            held = root / "results" / "phase6" / "rea-data"
            held.mkdir(parents=True)
            (held / "decompiled-seed1001.jsonl").write_text(json.dumps({"id": "x-1", "kind": "x", "score": "fuzzy", "accept": ["x"],
                                                                         "pseudocode": "void f(void) {}"}) + "\n")
            old, self.rea.ROOT = self.rea.ROOT, root
            try:
                with self.assertRaisesRegex(ValueError, "unknown score mode 'fuzzy' in task x-1"):
                    self.rea.main("")
            finally:
                self.rea.ROOT = old


def sample_tasks(family: str, seeds=(101, 202)) -> list:
    n = len(families.modules()[family].VARIANTS)
    return [t for s in seeds for t in families.generate(family, n, s)]


class FamilyChecks:
    """Mixin turned into one TestCase per new family below."""
    family = ""

    @classmethod
    def setUpClass(cls):
        cls.tasks = sample_tasks(cls.family)

    def test_schema_and_determinism(self):
        m = families.modules()[self.family]
        self.assertEqual({t.variant for t in self.tasks}, set(m.VARIANTS), "every variant is generated")
        again = sample_tasks(self.family)
        self.assertEqual([t.source() for t in again], [t.source() for t in self.tasks])
        self.assertEqual([t.meta() for t in again], [t.meta() for t in self.tasks])
        for t in self.tasks:
            meta = json.loads(json.dumps(t.meta()))
            self.assertEqual(meta["function"], "proc")
            self.assertEqual(meta["kind"], self.family)
            self.assertIn(meta["cc"], core.CCS)
            self.assertIn(meta["opt"], core.OPTS)
            self.assertIn(meta["score"], ("tokens", "exact_case", "exact_head"), "new families use headline-only modes")
            self.assertTrue(meta["accept"])
            self.assertLessEqual(len(t.question), 400, "short, self-contained question")
            self.assertRegex(t.source(), r"\bproc\s*\(")
            for h in t.helpers:
                self.assertRegex(t.source(), r"__attribute__\(\(noinline\)\)[^;{]*\b" + h + r"\s*\(")
        self.assertEqual(len({t.id for t in self.tasks}), len(self.tasks), "task ids are unique")

    def test_flags_vary(self):
        tasks = families.generate(self.family, 60, 5)
        self.assertEqual({t.cc for t in tasks}, {"gcc", "clang"})
        self.assertGreaterEqual(len({t.opt for t in tasks}), 4)
        self.assertEqual({t.cc for t in families.generate(self.family, 6, 5, "gcc")}, {"gcc"})

    def test_compiles_clean_with_gcc(self):
        for t in self.tasks:
            for opt in core.OPTS:
                r = compile_check(t.source(), opt)
                self.assertEqual(r.returncode, 0, f"{t.id} ({t.variant}) gcc {opt}:\n{r.stderr}\n{t.source()}")

    @unittest.skipUnless(CLANG, "clang is not installed here")
    def test_compiles_clean_with_clang(self):
        for t in self.tasks:
            for opt in ("-O0", "-O2"):
                r = compile_check(t.source(), opt, "clang")
                self.assertEqual(r.returncode, 0, f"{t.id} ({t.variant}) clang {opt}:\n{r.stderr}")

    def test_harness_proves_the_truth(self):
        proven = 0
        for t in self.tasks:
            if not t.check:
                continue
            main_c, want, *pre = t.check
            src = t.source(main_c, pre[0] if pre else "")
            for opt, extra in verify.harness_runs(UBSAN):
                r = run_harness(src, opt, extra)
                self.assertEqual(r.returncode, 0, f"{t.id} ({t.variant}) {opt} {extra}:\n{r.stderr}")
                self.assertEqual(r.stdout, want, f"{t.id} ({t.variant}) {opt} {extra}")
            proven += 1
        self.assertGreater(proven, 0, "at least some variants have a harness")

    def test_proc_is_a_global_text_symbol(self):
        """modal_rea_data.py finds the function with `nm t.sym | grep ' T proc$'`; check that for the stub-main build."""
        for t in self.tasks:
            for opt in ("-O0", "-O2"):
                self.assertIn("proc", global_text_symbols(t.source(), opt), f"{t.id} {opt}")

    def test_scorer_accepts_canonical_and_rejects_wrong(self):
        for t in self.tasks:
            good = scoring.canonical(t.truth)
            self.assertTrue(scoring.score(good, t.truth), f"{t.id}: canonical {good!r} rejected")
            self.assertTrue(t.wrong, f"{t.id}: task needs a plausible wrong answer")
            self.assertFalse(scoring.score(t.wrong, t.truth), f"{t.id}: wrong answer {t.wrong!r} accepted")
            self.assertFalse(scoring.score("I cannot tell.", t.truth))
            self.assertFalse(scoring.score("", t.truth))


# ---- family-specific reference checks (things the generic harness cannot see) -------------------------------------------

class StructsModel(unittest.TestCase):
    def test_layout_matches_hand_computation(self):
        from rea_bench import f_structs as fs
        u8, u16, u32, u64 = (fs.Scalar(c, n) for c, n in (("uint8_t", 1), ("uint16_t", 2), ("uint32_t", 4), ("uint64_t", 8)))
        st = fs.Struct("t", [("a", u8), ("b", u32), ("c", u16)])
        self.assertEqual((st.offsets, st.size, st.align), ({"a": 0, "b": 4, "c": 8}, 12, 4))
        st = fs.Struct("t", [("a", u8), ("b", u64), ("c", fs.Arr(u8, 3)), ("d", u16)])
        self.assertEqual((st.offsets["b"], st.offsets["c"], st.offsets["d"], st.size), (8, 16, 20, 24))
        inner = fs.Struct("i", [("x", u8), ("y", u16)])           # size 4, align 2
        outer = fs.Struct("o", [("h", u8), ("sub", inner), ("t", u32)])
        self.assertEqual((outer.offsets["sub"], outer.offsets["t"], outer.size), (2, 8, 12))


class CrcReference(unittest.TestCase):
    """The Python reference the CRC harness is compared with, anchored to independent implementations and published check values."""
    def test_known_check_values(self):
        import binascii
        import zlib

        from rea_bench import f_crc
        v = b"123456789"
        by_name = {n: (w, p, i, r, x) for n, w, p, i, r, x in f_crc.STANDARD}
        self.assertEqual(f_crc.crc_ref(v, *by_name["crc32"]), zlib.crc32(v))
        self.assertEqual(f_crc.crc_ref(b"\x00\xff hello", *by_name["crc32"]), zlib.crc32(b"\x00\xff hello"))
        self.assertEqual(f_crc.crc_ref(v, *by_name["crc16-xmodem"]), binascii.crc_hqx(v, 0))
        self.assertEqual(f_crc.crc_ref(v, *by_name["crc16-ccitt-false"]), binascii.crc_hqx(v, 0xFFFF))
        for name, check in (("crc32c", 0xE3069283), ("crc32-mpeg2", 0x0376E6E7), ("crc16-arc", 0xBB3D), ("crc16-modbus", 0x4B37),
                            ("crc16-ccitt-false", 0x29B1), ("crc16-xmodem", 0x31C3)):
            self.assertEqual(f_crc.crc_ref(v, *by_name[name]), check, name)

    def test_reflect(self):
        from rea_bench import f_crc
        self.assertEqual(f_crc.reflect(0x04C11DB7, 32), 0xEDB88320)
        self.assertEqual(f_crc.reflect(0x8005, 16), 0xA001)
        self.assertEqual(f_crc.reflect(0x1021, 16), 0x8408)

    def test_literal_in_source_is_the_reflected_one(self):
        from rea_bench import f_crc
        for t in families.generate("crc", 24, 9):
            p = t.params
            want = f"0x{p['lit']:0{p['width'] // 4}x}u"
            self.assertIn(want, t.body)
            self.assertEqual(p["lit"], f_crc.reflect(p["poly"], p["width"]) if p["refl"] else p["poly"])


class CipherReference(unittest.TestCase):
    """Python references behind the cipher harness, checked against published vectors and round trips."""
    def test_rc4_vector(self):
        from rea_bench import f_cipher
        self.assertEqual(f_cipher.rc4(b"Key", b"Plaintext").hex(), "bbf316e8d940af0ad3")

    def test_chacha20_rfc7539_block(self):
        from rea_bench import f_cipher
        key = [int.from_bytes(bytes(range(i, i + 4)), "little") for i in range(0, 32, 4)]
        nonce = [int.from_bytes(bytes.fromhex(h), "little") for h in ("00000009", "0000004a", "00000000")]   # bytes 00 00 00 09 / 00 00 00 4a / 0
        out = f_cipher.chacha_block(key, 1, nonce, 20)
        self.assertEqual([f"{w:08x}" for w in out[:4]], ["e4e7f110", "15593bd1", "1fdd0f50", "c47120a3"])
        self.assertEqual(f"{out[15]:08x}", "4e3c50a2")

    def test_salsa_core_zero_and_difference_from_chacha(self):
        from rea_bench import f_cipher
        self.assertEqual(f_cipher.salsa_core([0] * 16, 20), [0] * 16)
        key = list(range(1, 9))
        self.assertNotEqual(f_cipher.salsa_block(key, 1, [2, 3], 20), f_cipher.chacha_block(key, 1, [2, 3, 4], 20))

    def test_tea_xtea_zero_key_vectors_and_round_trip(self):
        from rea_bench import f_cipher
        z = [0, 0, 0, 0]
        self.assertEqual(f_cipher.tea((0, 0), z, 32), (0x41EA3A0A, 0x94BAA940))
        self.assertEqual(f_cipher.xtea((0, 0), z, 32), (0xDEE9D4D8, 0xF7131ED9))
        key = [0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210]
        for fn in (f_cipher.tea, f_cipher.xtea):
            for rounds in (16, 32, 48):
                enc = fn((0xDEADBEEF, 0x0BADF00D), key, rounds)
                self.assertEqual(fn(enc, key, rounds, decrypt=True), (0xDEADBEEF, 0x0BADF00D))

    def test_name_rejection_rules(self):
        from rea_bench import f_cipher
        tea = f_cipher._truth("tea", [["32", "0x20"]], "enc")
        xtea = f_cipher._truth("xtea", [["32", "0x20"]], "dec")
        s = scoring.score
        self.assertTrue(s("TEA (Tiny Encryption Algorithm), encrypts, 32 iterations", tea))
        self.assertFalse(s("XTEA, encrypts, 32", tea))
        self.assertFalse(s("TEA, decrypts, 32", tea))
        self.assertTrue(s("XTEA (eXtended TEA): it decrypts, 32 rounds", xtea))
        self.assertFalse(s("TEA, decrypts, 32", xtea))
        self.assertFalse(s("XTEA, encrypts, 32", xtea))


class EncodingReference(unittest.TestCase):
    def test_base58_known_values(self):
        from rea_bench import f_encoding as fe
        self.assertEqual(fe.b58encode(b"Hello World!", fe.B58), "2NEpo7TZRRrLZSi2U")
        self.assertEqual(fe.b58encode(b"\x00\x00\x01", fe.B58), "112")
        self.assertEqual(fe.b58encode(b"", fe.B58), "")

    def test_standard_alphabets_match_stdlib(self):
        import base64
        from rea_bench import f_encoding as fe
        self.assertEqual(fe.encode_ref("base64", b"\xfb\xff\xfe", fe.B64, True), base64.b64encode(b"\xfb\xff\xfe").decode())
        self.assertEqual(fe.encode_ref("base32", b"foobar", fe.B32, True), "MZXW6YTBOI======")
        self.assertEqual(fe.encode_ref("base32", b"foobar", fe.B32, False), "MZXW6YTBOI")
        custom = fe.B64[::-1]
        self.assertEqual(fe.encode_ref("base64", b"abc", custom, True), base64.b64encode(b"abc").decode().translate(str.maketrans(fe.B64, custom)))

    def test_both_alphabet_kinds_and_both_questions_occur(self):
        tasks = families.generate("encoding", 120, 4)
        self.assertEqual({t.params["custom"] for t in tasks}, {True, False})
        self.assertEqual({t.params["asked"] for t in tasks}, {"alphabet", "ident"})
        self.assertEqual({t.truth["score"] for t in tasks}, {"tokens", "exact_case"})
        for t in tasks:
            if t.params["asked"] == "alphabet":
                self.assertTrue(t.params["custom"])
                self.assertEqual(len(set(t.params["alphabet"])), len(t.params["alphabet"]))

    def test_direction_and_base_rules(self):
        enc = next(t for t in families.generate("encoding", 60, 4) if t.params["asked"] == "ident" and t.params["direction"] == "encode" and t.params["base"] == "base64")
        dec = next(t for t in families.generate("encoding", 60, 4) if t.params["asked"] == "ident" and t.params["direction"] == "decode" and t.params["base"] == "base58")
        s = scoring.score
        self.assertTrue(s("It is a Base64 encoder.", enc.truth))
        self.assertTrue(s("encodes using base-64", enc.truth))
        self.assertFalse(s("It is a Base64 decoder.", enc.truth))
        self.assertFalse(s("It is a Base32 encoder.", enc.truth))
        self.assertTrue(s("It decodes a Base58 encoding.", dec.truth))   # "encoding" as a noun is fine for a decoder
        self.assertFalse(s("It encodes Base58.", dec.truth))


class StrobfModel(unittest.TestCase):
    def test_plaintexts_are_plain_and_varied(self):
        tasks = families.generate("strobf", 70, 8)
        plains = {t.params["plain"] for t in tasks}
        self.assertGreater(len(plains), 60)
        for t in tasks:
            p = t.params["plain"]
            self.assertLessEqual(len(p), 40)
            self.assertTrue(all(c.isalnum() and c.islower() or c.isdigit() or c in "_ ./-:" for c in p), p)
            self.assertNotIn(p, t.body, "the plaintext must not appear in the C source")

    def test_scorer_ignores_spacing_but_not_a_changed_character(self):
        t = families.generate("strobf", 1, 8)[0]
        p = t.params["plain"]
        self.assertTrue(scoring.score(f"The string is `{p}`.", t.truth))
        self.assertFalse(scoring.score(t.wrong, t.truth))


class PktsumModel(unittest.TestCase):
    def test_every_question_kind_and_both_byte_orders_occur(self):
        tasks = families.generate("pktsum", 140, 6)
        self.assertEqual({t.params["ask"] for t in tasks}, {"start", "tail", "endian", "len_offset", "len_start"})
        self.assertEqual({t.params["endian"] for t in tasks if t.params["W"] == 2}, {"little", "big"})

    def test_harness_packets_exercise_the_range_and_the_byte_order(self):
        for t in families.generate("pktsum", 70, 6):
            verdicts = t.check[1].split()
            self.assertIn("1", verdicts, "a valid packet")
            self.assertIn("0", verdicts, "a corrupted covered byte")
            self.assertIn("-1", verdicts, "malformed packets")
            if t.params["W"] == 2:
                self.assertGreaterEqual(verdicts.count("0"), 3, "covered flips and the swapped checksum bytes all fail")


class BoundsModel(unittest.TestCase):
    def test_negative_bounds_need_their_sign(self):
        seen = 0
        for t in families.generate("bounds", 120, 11):
            lo = t.params.get("lo")
            if t.variant in ("range_chain", "window", "clamp") and lo is not None and lo < 0:
                seen += 1
                self.assertTrue(scoring.score(f"lowest {lo}, highest {t.params['hi']}", t.truth))
                self.assertFalse(scoring.score(f"lowest {-lo}, highest {t.params['hi']}", t.truth))
        self.assertGreater(seen, 3, "the sample should contain negative bounds")

    def test_every_loop_and_guard_form_occurs(self):
        tasks = families.generate("bounds", 240, 12)
        self.assertEqual({t.params["form"] for t in tasks if t.variant == "loop_iterations"}, {"up_lt", "up_le", "down_gt", "down_ge"})
        self.assertEqual({t.params["form"] for t in tasks if t.variant == "index_limit"}, set(range(6)))
        self.assertEqual({t.params["form"] for t in tasks if t.variant == "range_chain"}, set(range(5)))


class MultiModel(unittest.TestCase):
    """The helper must stay a separate, uncloned function that proc really calls, at every optimisation level (gcc)."""
    OBJDUMP = shutil.which("objdump")

    def calls_in(self, disasm: str, fn: str) -> set:
        section, on = [], False
        for line in disasm.splitlines():
            if line.endswith(f"<{fn}>:"):
                on = True
            elif on and line.endswith(">:"):
                break
            elif on:
                section.append(line)
        return {m for l in section for m in __import__("re").findall(r"(?:call|jmp)\s+[0-9a-f]+ <([^>+]+)>", l)}

    @unittest.skipUnless(OBJDUMP, "objdump is not installed")
    def test_helpers_are_called_not_inlined_or_cloned(self):
        tasks = [t for s in (21, 22) for t in families.generate("multi", 5, s)]
        for t in tasks:
            for opt in core.OPTS:
                with tempfile.TemporaryDirectory() as d:
                    exe = pathlib.Path(d, "t.sym")
                    build_binary(t.source(), opt, exe)
                    syms = global_text_symbols(t.source(), opt)
                    dis = subprocess.run([self.OBJDUMP, "-d", "--no-show-raw-insn", str(exe)], capture_output=True, text=True).stdout
                for h in t.helpers:
                    self.assertIn(h, syms, f"{t.id} {opt}: {h} must be a global symbol")
                callees = self.calls_in(dis, "proc")
                self.assertIn(t.helpers[0], callees, f"{t.id} {opt}: proc must call {t.helpers[0]}, calls {callees}")
                if len(t.helpers) > 1:
                    self.assertIn(t.helpers[1], self.calls_in(dis, t.helpers[0]), f"{t.id} {opt}: helper must call helper2")
                self.assertFalse([c for c in callees if "constprop" in c or "isra" in c or "part" in c], callees)


class AntidebugModel(unittest.TestCase):
    def test_details_are_in_the_code_and_techniques_rejected_across(self):
        tasks = families.generate("antidebug", 60, 13)
        for t in tasks:
            p = t.params
            if t.variant == "env":
                self.assertIn(f'getenv("{p["name"]}")', t.body)
            if t.variant.startswith("timing"):
                self.assertIn(str(p["thr"]), t.body)
        one = next(t for t in tasks if t.variant == "ptrace")
        s = scoring.score
        self.assertTrue(s(f"ptrace, returns {one.params['rv']}", one.truth))
        self.assertTrue(s(f"PTRACE_TRACEME self-attach; return value {one.params['rv']}", one.truth))
        self.assertFalse(s(f"env, {one.params['rv']}", one.truth))
        self.assertFalse(s("ptrace", one.truth), "the detail is missing")
        env = next(t for t in tasks if t.variant == "env")
        self.assertTrue(s(f"env: {env.params['name']}", env.truth))
        self.assertTrue(s(f"It probes the environment variable {env.params['name'].lower()}.", env.truth))

    def test_harness_uses_fakes_not_the_real_process(self):
        for t in families.generate("antidebug", 6, 13):
            self.assertEqual(len(t.check), 3 if t.variant != "env" else 2)
            if t.variant != "env":
                self.assertRegex(t.check[2], r"#define (ptrace|syscall|clock_gettime|gettimeofday|fopen) fake_")


for _fam in families.modules():
    if _fam != "legacy":
        globals()[f"Family_{_fam}"] = type(f"Family_{_fam}", (FamilyChecks, unittest.TestCase), {"family": _fam})


if __name__ == "__main__":
    unittest.main()

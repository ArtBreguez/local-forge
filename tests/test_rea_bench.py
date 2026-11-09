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

from rea_bench import core, families, scoring  # noqa: E402

CLANG = shutil.which("clang")


def digest(directory: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.glob("*")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def cc_syntax(src: str, opt: str, cc: str = "gcc") -> subprocess.CompletedProcess:
    """-Wall -Werror compile of the generated file alone (no link): any warning at this level fails."""
    return subprocess.run([cc, opt, "-Wall", "-Werror", "-c", "-o", "/dev/null", "-x", "c", "-"], input=src,
                          capture_output=True, text=True, timeout=60)


def run_harness(src: str, opt: str, extra=(), cc: str = "gcc") -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as d:
        exe = pathlib.Path(d, "h")
        b = subprocess.run([cc, opt, "-Wall", *extra, "-o", str(exe), "-x", "c", "-"], input=src, capture_output=True, text=True, timeout=60)
        if b.returncode:
            return b
        return subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)


def _ubsan_works() -> bool:
    r = subprocess.run(["gcc", "-fsanitize=undefined", "-fno-sanitize-recover=all", "-o", "/dev/null", "-x", "c", "-"],
                       input="int main(void){return 0;}", capture_output=True, text=True)
    return r.returncode == 0


UBSAN = _ubsan_works()


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

    def test_canonical_roundtrip(self):
        for t in ({"score": "tokens", "accept": [["a1", "0xa1"], ["tea"]]}, {"score": "exact", "accept": ["x"]},
                  {"score": "all", "accept": ["12", "ff"]}, {"score": "all_groups", "accept": [["djb"], ["12ab"]]}):
            self.assertTrue(scoring.score(scoring.canonical(t), t), t)

    def test_modal_module_uses_the_package_scorer(self):
        src = (ROOT / "tools" / "modal" / "modal_rea_data.py").read_text()
        self.assertIn("from rea_bench.scoring import _score, _variants", src)
        self.assertNotIn("def _score", src)
        self.assertIn("task.get(\"cc\", \"gcc\")", src)
        self.assertIn('"clang"', (ROOT / "tools" / "modal" / "modal_rea.py").read_text())


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
            self.assertIn(meta["score"], ("exact", "any", "all", "all_groups", "tokens"))
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
                r = cc_syntax(t.source(), opt)
                self.assertEqual(r.returncode, 0, f"{t.id} ({t.variant}) gcc {opt}:\n{r.stderr}\n{t.source()}")

    @unittest.skipUnless(CLANG, "clang is not installed here")
    def test_compiles_clean_with_clang(self):
        for t in self.tasks:
            for opt in ("-O0", "-O2"):
                r = cc_syntax(t.source(), opt, "clang")
                self.assertEqual(r.returncode, 0, f"{t.id} ({t.variant}) clang {opt}:\n{r.stderr}")

    def test_harness_proves_the_truth(self):
        proven = 0
        for t in self.tasks:
            if not t.check:
                continue
            main_c, want = t.check
            src = t.source(main_c)
            runs = [("-O0", ()), ("-O2", ())] + ([("-O1", ("-fsanitize=undefined", "-fno-sanitize-recover=all"))] if UBSAN else [])
            for opt, extra in runs:
                r = run_harness(src, opt, extra)
                self.assertEqual(r.returncode, 0, f"{t.id} ({t.variant}) {opt} {extra}:\n{r.stderr}")
                self.assertEqual(r.stdout, want, f"{t.id} ({t.variant}) {opt} {extra}")
            proven += 1
        self.assertGreater(proven, 0, "at least some variants have a harness")

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


for _fam in families.modules():
    if _fam != "legacy":
        globals()[f"Family_{_fam}"] = type(f"Family_{_fam}", (FamilyChecks, unittest.TestCase), {"family": _fam})


if __name__ == "__main__":
    unittest.main()

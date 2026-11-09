import ast
import pathlib
import unittest

import formatting
import parsing
import util

ROOT = pathlib.Path(util.__file__).resolve().parent


class SplitHiddenTests(unittest.TestCase):
    def test_new_modules_behave_like_the_old_functions(self):
        self.assertEqual(parsing.parse_kv("k = v"), ("k", "v"))
        self.assertEqual(parsing.parse_kv_block("a=1\n#x\nb=2"), {"a": "1", "b": "2"})
        self.assertEqual(parsing.parse_size("1G"), 1024 ** 3)
        self.assertEqual(formatting.format_size(2 * 1024 ** 2), "2M")
        self.assertEqual(formatting.format_kv({"b": 1, "a": 2}), "a=2\nb=1")
        self.assertEqual(formatting.format_table([("x", 1)], ["k", "v"]), "k  v\n-  -\nx  1")

    def test_util_reexports_the_same_objects(self):
        for name in ("parse_kv", "parse_kv_block", "parse_size"):
            self.assertIs(getattr(util, name), getattr(parsing, name), name)
        for name in ("format_size", "format_table", "format_kv"):
            self.assertIs(getattr(util, name), getattr(formatting, name), name)
        self.assertIs(util.UNITS, parsing.UNITS)
        self.assertEqual(util.DEFAULT_SEP, "=")

    def test_util_has_no_definitions_of_its_own(self):
        tree = ast.parse((ROOT / "util.py").read_text(encoding="utf-8"))
        defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        self.assertEqual(defs, [])

    def test_modules_hold_the_right_names(self):
        for name in ("DEFAULT_SEP", "UNITS", "parse_kv", "parse_kv_block", "parse_size"):
            self.assertTrue(hasattr(parsing, name), name)
        for name in ("format_size", "format_table", "format_kv"):
            self.assertTrue(hasattr(formatting, name), name)
        self.assertFalse(hasattr(parsing, "format_size"))
        self.assertFalse(hasattr(formatting, "parse_size") and formatting.parse_size is not parsing.parse_size)

    def test_formatting_uses_the_constants_of_parsing(self):
        self.assertIs(formatting.UNITS, parsing.UNITS)
        self.assertIs(formatting.DEFAULT_SEP, parsing.DEFAULT_SEP)

    def test_default_separator_still_applies(self):
        self.assertEqual(formatting.format_kv({"a": 1}), "a=1")
        with self.assertRaises(ValueError):
            parsing.parse_kv("=value")

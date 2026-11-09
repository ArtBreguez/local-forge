import unittest

import util


class ParsingTests(unittest.TestCase):
    def test_parse_kv(self):
        self.assertEqual(util.parse_kv(" name = Ada "), ("name", "Ada"))
        self.assertEqual(util.parse_kv("a:b", sep=":"), ("a", "b"))
        with self.assertRaises(ValueError):
            util.parse_kv("novalue")

    def test_parse_kv_block(self):
        text = "# comment\na=1\n\nb = 2\na=3\n"
        self.assertEqual(util.parse_kv_block(text), {"a": "3", "b": "2"})

    def test_parse_size(self):
        self.assertEqual(util.parse_size("10k"), 10240)
        self.assertEqual(util.parse_size("2M"), 2 * 1024 * 1024)
        self.assertEqual(util.parse_size("512"), 512)
        with self.assertRaises(ValueError):
            util.parse_size("12x")


class FormattingTests(unittest.TestCase):
    def test_format_size(self):
        self.assertEqual(util.format_size(10240), "10k")
        self.assertEqual(util.format_size(1536), "1536")
        self.assertEqual(util.format_size(3 * 1024 ** 3), "3G")

    def test_format_kv(self):
        self.assertEqual(util.format_kv({"b": 2, "a": 1}), "a=1\nb=2")
        self.assertEqual(util.format_kv({"x": 1}, sep=": "), "x: 1")

    def test_format_table(self):
        out = util.format_table([("ada", 36), ("bob", 5)], ["name", "age"])
        self.assertEqual(out, "name  age\n----  ---\nada   36\nbob   5")

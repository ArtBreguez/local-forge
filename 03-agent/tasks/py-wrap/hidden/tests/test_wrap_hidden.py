import unittest

from wrap import wrap


class WrapHiddenTests(unittest.TestCase):
    def test_exact_fit_stays_on_one_line(self):
        self.assertEqual(wrap("abc def", 7), ["abc def"])
        self.assertEqual(wrap("abc def", 6), ["abc", "def"])

    def test_whitespace_is_collapsed(self):
        self.assertEqual(wrap("a  b\tc\nd", 20), ["a b c d"])
        self.assertEqual(wrap("  padded   text  ", 20), ["padded text"])

    def test_long_word_is_split_on_its_own_lines(self):
        self.assertEqual(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"])
        self.assertEqual(wrap("ab abcdefghij cd", 4), ["ab", "abcd", "efgh", "ij", "cd"])
        self.assertEqual(wrap("abcdefgh", 4), ["abcd", "efgh"])

    def test_width_one(self):
        self.assertEqual(wrap("ab c", 1), ["a", "b", "c"])

    def test_types(self):
        for bad in (5, ["a"], b"abc"):
            with self.assertRaises(TypeError):
                wrap(bad, 5)
        for bad in (0, -3, 2.5, "5", None):
            with self.assertRaises(ValueError):
                wrap("abc", bad)

    def test_many_lines(self):
        text = " ".join(["word"] * 10)
        self.assertEqual(wrap(text, 14), ["word word word", "word word word", "word word word", "word"])

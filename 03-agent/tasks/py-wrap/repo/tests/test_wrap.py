import unittest

from wrap import wrap


class WrapTests(unittest.TestCase):
    def test_greedy_lines(self):
        self.assertEqual(wrap("the quick brown fox", 9), ["the quick", "brown fox"])
        self.assertEqual(wrap("a b c d", 3), ["a b", "c d"])

    def test_no_words(self):
        self.assertEqual(wrap("", 10), [])
        self.assertEqual(wrap("   \n ", 10), [])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            wrap("abc", 0)
        with self.assertRaises(TypeError):
            wrap(None, 5)

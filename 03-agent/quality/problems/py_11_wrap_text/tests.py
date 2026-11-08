import unittest
from solution import wrap_text


class WrapTextTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(wrap_text("the quick brown fox", 9), ["the quick", "brown fox"])
        self.assertEqual(wrap_text("aa bb cc", 5), ["aa bb", "cc"])
        self.assertEqual(wrap_text("aa bb cc", 4), ["aa", "bb", "cc"])
        self.assertEqual(wrap_text("aa bb cc", 8), ["aa bb cc"])

    def test_exact_fit_is_allowed(self):
        self.assertEqual(wrap_text("abc def", 7), ["abc def"])
        self.assertEqual(wrap_text("abc def", 6), ["abc", "def"])

    def test_long_word_on_its_own_line(self):
        self.assertEqual(wrap_text("aa bbbbbbbb cc", 4), ["aa", "bbbbbbbb", "cc"])
        self.assertEqual(wrap_text("abcdefghij", 3), ["abcdefghij"])
        self.assertEqual(wrap_text("x abcdefghij y", 5), ["x", "abcdefghij", "y"])

    def test_whitespace_runs_collapse(self):
        self.assertEqual(wrap_text("  one   two\tthree\nfour  ", 11), ["one two", "three four"])
        self.assertEqual(wrap_text("a\n\n\nb", 10), ["a b"])

    def test_empty(self):
        self.assertEqual(wrap_text("", 10), [])
        self.assertEqual(wrap_text("   \n\t ", 10), [])

    def test_width_one(self):
        self.assertEqual(wrap_text("a b c", 1), ["a", "b", "c"])

    def test_no_line_is_over_width_unless_single_word(self):
        text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod"
        for width in range(1, 30):
            lines = wrap_text(text, width)
            self.assertEqual(" ".join(lines).split(), text.split())
            for line in lines:
                self.assertTrue(len(line) <= width or " " not in line, (width, line))

    def test_greedy_not_balanced(self):
        self.assertEqual(wrap_text("a bb ccc dddd", 6), ["a bb", "ccc", "dddd"])
        self.assertEqual(wrap_text("aaa bb c", 6), ["aaa bb", "c"])

    def test_bad_width(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                wrap_text("x", bad)
        for bad in (2.5, "3", None):
            with self.assertRaises(TypeError):
                wrap_text("x", bad)


if __name__ == "__main__":
    unittest.main()

import unittest
from solution import top_words


class TopWordsTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(top_words("the cat and the hat and the bat", 2), [("the", 3), ("and", 2)])

    def test_tie_break_alphabetical(self):
        self.assertEqual(top_words("b a c b a c d", 3), [("a", 2), ("b", 2), ("c", 2)])
        self.assertEqual(top_words("zeta alpha mid", 3), [("alpha", 1), ("mid", 1), ("zeta", 1)])

    def test_case_insensitive(self):
        self.assertEqual(top_words("Apple apple APPLE banana", 2), [("apple", 3), ("banana", 1)])

    def test_apostrophes(self):
        self.assertEqual(top_words("Don't stop, don't stop. 'tis the don't", 3), [("don't", 3), ("stop", 2), ("the", 1)])
        self.assertEqual(top_words("rock 'n' roll", 3), [("n", 1), ("rock", 1), ("roll", 1)])
        self.assertEqual(top_words("it's 'quoted' text''", 5), [("it's", 1), ("quoted", 1), ("text", 1)])

    def test_digits_and_punctuation_separate(self):
        self.assertEqual(top_words("a1b 2c!!d-e_f", 10), [("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1), ("f", 1)])

    def test_k_larger_than_distinct(self):
        self.assertEqual(top_words("one two two", 10), [("two", 2), ("one", 1)])

    def test_k_zero_and_empty(self):
        self.assertEqual(top_words("a b c", 0), [])
        self.assertEqual(top_words("", 3), [])
        self.assertEqual(top_words("123 !!! ---", 3), [])

    def test_returns_tuples(self):
        result = top_words("x x y", 2)
        self.assertEqual(result, [("x", 2), ("y", 1)])
        self.assertIsInstance(result[0], tuple)

    def test_non_ascii_letters_separate(self):
        self.assertEqual(top_words("café caf", 3), [("caf", 2)])

    def test_errors(self):
        with self.assertRaises(ValueError):
            top_words("a", -1)
        for bad in (1.5, "2", None):
            with self.assertRaises(TypeError):
                top_words("a", bad)
        with self.assertRaises(TypeError):
            top_words(None, 1)


if __name__ == "__main__":
    unittest.main()

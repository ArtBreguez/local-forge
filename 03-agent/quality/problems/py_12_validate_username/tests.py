import unittest
from solution import validate_username


class UsernameTests(unittest.TestCase):
    def test_valid_and_normalised(self):
        cases = {
            "alice": "alice",
            "  Alice_01 ": "alice_01",
            "BOB": "bob",
            "a.b_c": "a.b_c",
            "user.name99": "user.name99",
            "x" * 20: "x" * 20,
            "abc": "abc",
            "\tTom\n": "tom",
            "a1b": "a1b",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(validate_username(raw), expected)

    def test_length(self):
        for raw in ["", "ab", "  ab  ", "x" * 21, "x" * 40]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    validate_username(raw)

    def test_length_counts_after_stripping(self):
        self.assertEqual(validate_username("  " + "y" * 20 + "  "), "y" * 20)

    def test_start_and_end(self):
        for raw in ["1abc", "_abc", ".abc", "abc_", "abc.", "9", "a_b_"]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    validate_username(raw)

    def test_consecutive_specials(self):
        for raw in ["a__b", "a..b", "a._b", "a_.b", "ab__cd"]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    validate_username(raw)

    def test_characters(self):
        for raw in ["a b", "a-b", "a@b", "a/b", "Ünal", "naïve", "ａｂｃ", "abcé", "a١b", "a b c"]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    validate_username(raw)

    def test_type(self):
        for value in (None, 12345, b"alice", ["alice"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    validate_username(value)


if __name__ == "__main__":
    unittest.main()

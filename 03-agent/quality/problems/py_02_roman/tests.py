import unittest
from solution import roman_to_int


class RomanTests(unittest.TestCase):
    def test_valid(self):
        cases = {
            "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "IX": 9,
            "X": 10, "XL": 40, "L": 50, "XC": 90, "C": 100, "CD": 400,
            "D": 500, "CM": 900, "M": 1000, "XIV": 14, "XIX": 19,
            "MCMXCIV": 1994, "MMXXIV": 2024, "MMMCMXCIX": 3999,
            "CDXLIV": 444, "MMMDCCCLXXXVIII": 3888,
        }
        for text, value in cases.items():
            with self.subTest(text=text):
                self.assertEqual(roman_to_int(text), value)

    def test_non_canonical_rejected(self):
        bad = ["", "IIII", "VV", "IC", "IL", "XM", "MMMM", "IIX", "XIIII",
               "VX", "LC", "DM", "CCCC", "XXXX", "IVI", "IXI", "CMM", "IM"]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    roman_to_int(text)

    def test_bad_characters(self):
        for text in ["iv", "Iv", "A", "X1", " X", "X ", "MCMXCIVa"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    roman_to_int(text)

    def test_type(self):
        for value in [None, 4, 4.0, b"IV"]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    roman_to_int(value)


if __name__ == "__main__":
    unittest.main()

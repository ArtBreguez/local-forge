import unittest
from solution import parse_duration


class ParseDurationTests(unittest.TestCase):
    def test_single_units(self):
        self.assertEqual(parse_duration("90s"), 90)
        self.assertEqual(parse_duration("2d"), 172800)
        self.assertEqual(parse_duration("0s"), 0)
        self.assertEqual(parse_duration("5m"), 300)
        self.assertEqual(parse_duration("3h"), 10800)

    def test_combined(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("1d2h3m4s"), 86400 + 7200 + 180 + 4)
        self.assertEqual(parse_duration("1h15s"), 3615)
        self.assertEqual(parse_duration("0d0h0m0s"), 0)

    def test_large_values_are_not_normalised(self):
        self.assertEqual(parse_duration("120m"), 7200)
        self.assertEqual(parse_duration("100000s"), 100000)

    def test_value_errors(self):
        bad = ["", "h", "10", "1x", "1ms", "30m1h", "1h1h", "-5s", "+5s",
               "1.5h", " 1h", "1h ", "1 h", "1h30", "s", "1H", "1h-30m"]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_duration(text)

    def test_type_errors(self):
        for value in [None, 5, 1.5, b"1h", ["1h"]]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    parse_duration(value)


if __name__ == "__main__":
    unittest.main()

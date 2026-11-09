import unittest
from solution import to_24h


class To24hTests(unittest.TestCase):
    def test_midnight_and_noon(self):
        self.assertEqual(to_24h("12:00 AM"), "00:00")
        self.assertEqual(to_24h("12:30 AM"), "00:30")
        self.assertEqual(to_24h("12:00 PM"), "12:00")
        self.assertEqual(to_24h("12:59 PM"), "12:59")
        self.assertEqual(to_24h("12:00 am"), "00:00")

    def test_morning(self):
        self.assertEqual(to_24h("1:05 AM"), "01:05")
        self.assertEqual(to_24h("9:00AM"), "09:00")
        self.assertEqual(to_24h("11:59 AM"), "11:59")
        self.assertEqual(to_24h("07:30 am"), "07:30")
        self.assertEqual(to_24h("10:10 Am"), "10:10")

    def test_afternoon(self):
        self.assertEqual(to_24h("1:05pm"), "13:05")
        self.assertEqual(to_24h("11:59 PM"), "23:59")
        self.assertEqual(to_24h("6:45 pM"), "18:45")
        self.assertEqual(to_24h("09:15 PM"), "21:15")

    def test_whitespace(self):
        self.assertEqual(to_24h("  7:45 Pm "), "19:45")
        self.assertEqual(to_24h("\t3:00 AM\n"), "03:00")

    def test_all_valid_hours(self):
        for h in range(1, 13):
            am = f"{h % 12:02d}:15"
            pm = f"{h % 12 + 12:02d}:15"
            with self.subTest(h=h):
                self.assertEqual(to_24h(f"{h}:15 AM"), am)
                self.assertEqual(to_24h(f"{h}:15 PM"), pm)

    def test_invalid(self):
        bad = ["", "0:30 AM", "00:30 AM", "13:00 PM", "7:60 AM", "7:5 AM", "7 AM", "7:30", "07:30 XM",
               "7:30 A", "7:30 AMM", "730 AM", "7.30 AM", "-1:30 AM", "7:30  AM", "7:30 A M", "AM", "12:00",
               "1:2:3 AM", "7:30 AM PM", "24:00 AM", "7:300 AM"]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    to_24h(text)

    def test_type(self):
        for value in (None, 730, b"7:30 AM"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    to_24h(value)


if __name__ == "__main__":
    unittest.main()

import unittest
from solution import iso_week


class IsoWeekTests(unittest.TestCase):
    def test_year_boundaries(self):
        cases = {
            "2021-01-01": "2020-W53",
            "2021-01-03": "2020-W53",
            "2021-01-04": "2021-W01",
            "2018-12-31": "2019-W01",
            "2024-12-30": "2025-W01",
            "2016-01-01": "2015-W53",
            "2015-12-31": "2015-W53",
            "2020-12-31": "2020-W53",
            "2026-01-01": "2026-W01",
            "2012-01-01": "2011-W52",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(iso_week(text), expected)

    def test_ordinary_days(self):
        self.assertEqual(iso_week("2024-02-29"), "2024-W09")
        self.assertEqual(iso_week("2024-07-04"), "2024-W27")
        self.assertEqual(iso_week("2023-06-15"), "2023-W24")
        self.assertEqual(iso_week("2024-01-01"), "2024-W01")

    def test_bad_shape(self):
        for text in ["2021-1-1", "20210101", "2021-01-01T10:00", "21-01-01",
                     "2021/01/01", " 2021-01-01", "2021-01-01 ", "", "abcd-ef-gh"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    iso_week(text)

    def test_not_a_real_date(self):
        for text in ["2023-02-29", "2021-13-01", "2021-00-10", "2021-04-31", "2021-01-00"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    iso_week(text)

    def test_type(self):
        for value in [None, 20210101, b"2021-01-01"]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    iso_week(value)


if __name__ == "__main__":
    unittest.main()

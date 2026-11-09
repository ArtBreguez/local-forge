import unittest
from solution import add_months


class AddMonthsTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(add_months("2024-05-10", 1), "2024-06-10")
        self.assertEqual(add_months("2024-05-10", 0), "2024-05-10")
        self.assertEqual(add_months("2024-05-10", 12), "2025-05-10")

    def test_year_rollover(self):
        self.assertEqual(add_months("2024-12-15", 1), "2025-01-15")
        self.assertEqual(add_months("2024-01-15", -1), "2023-12-15")
        self.assertEqual(add_months("2024-01-15", -13), "2022-12-15")
        self.assertEqual(add_months("2024-11-30", 14), "2026-01-30")

    def test_clamping(self):
        self.assertEqual(add_months("2024-01-31", 1), "2024-02-29")
        self.assertEqual(add_months("2023-01-31", 1), "2023-02-28")
        self.assertEqual(add_months("2024-03-31", -1), "2024-02-29")
        self.assertEqual(add_months("2024-05-31", 1), "2024-06-30")
        self.assertEqual(add_months("2024-02-29", 12), "2025-02-28")
        self.assertEqual(add_months("2024-02-29", 48), "2028-02-29")
        self.assertEqual(add_months("2024-10-31", -2), "2024-08-31")

    def test_range(self):
        self.assertEqual(add_months("9999-11-30", 1), "9999-12-30")
        with self.assertRaises(ValueError):
            add_months("9999-12-31", 1)
        with self.assertRaises(ValueError):
            add_months("0001-01-15", -1)

    def test_invalid_date(self):
        for text in ["2024-02-30", "not a date", "2024-13-01", ""]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    add_months(text, 1)

    def test_type(self):
        for n in [1.0, "1", None, True]:
            with self.subTest(n=n):
                with self.assertRaises(TypeError):
                    add_months("2024-01-01", n)


if __name__ == "__main__":
    unittest.main()

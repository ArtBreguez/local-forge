import unittest
from solution import business_days


class BusinessDaysTests(unittest.TestCase):
    def test_plain_weeks(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-08"), 5)
        self.assertEqual(business_days("2024-01-01", "2024-01-15"), 10)
        self.assertEqual(business_days("2024-01-01", "2024-01-06"), 5)
        self.assertEqual(business_days("2024-01-01", "2024-01-05"), 4)

    def test_weekend_only(self):
        self.assertEqual(business_days("2024-01-06", "2024-01-08"), 0)
        self.assertEqual(business_days("2024-01-06", "2024-01-07"), 0)

    def test_empty_range(self):
        self.assertEqual(business_days("2024-01-03", "2024-01-03"), 0)

    def test_end_is_exclusive_start_is_inclusive(self):
        self.assertEqual(business_days("2024-01-03", "2024-01-04"), 1)
        self.assertEqual(business_days("2024-01-04", "2024-01-05"), 1)
        self.assertEqual(business_days("2024-01-05", "2024-01-08"), 1)

    def test_full_year(self):
        self.assertEqual(business_days("2024-01-01", "2025-01-01"), 262)

    def test_holidays(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-08", ["2024-01-03"]), 4)
        self.assertEqual(business_days("2024-01-01", "2024-01-08", ("2024-01-03", "2024-01-04")), 3)

    def test_weekend_holiday_not_double_subtracted(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-08", ["2024-01-06"]), 5)
        self.assertEqual(business_days("2024-01-01", "2024-01-08", ["2024-01-06", "2024-01-07"]), 5)

    def test_duplicate_and_outside_holidays(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-08", ["2024-01-03", "2024-01-03"]), 4)
        self.assertEqual(business_days("2024-01-01", "2024-01-08", ["2023-12-25", "2024-02-01"]), 5)

    def test_holiday_on_end_date_ignored(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-03", ["2024-01-03"]), 2)

    def test_holiday_on_start_date_counts(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-03", ["2024-01-01"]), 1)

    def test_generator_holidays(self):
        self.assertEqual(business_days("2024-01-01", "2024-01-08", (h for h in ["2024-01-02"])), 4)

    def test_errors(self):
        with self.assertRaises(ValueError):
            business_days("2024-01-08", "2024-01-01")
        with self.assertRaises(ValueError):
            business_days("2024-02-30", "2024-03-01")
        with self.assertRaises(ValueError):
            business_days("2024-01-01", "2024-01-08", ["2024-13-01"])


if __name__ == "__main__":
    unittest.main()

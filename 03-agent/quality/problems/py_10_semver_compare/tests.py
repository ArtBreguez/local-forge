import unittest
from solution import compare_versions


class SemverTests(unittest.TestCase):
    def test_core_numbers(self):
        self.assertEqual(compare_versions("1.0.0", "2.0.0"), -1)
        self.assertEqual(compare_versions("2.0.0", "1.0.0"), 1)
        self.assertEqual(compare_versions("1.2.3", "1.2.3"), 0)
        self.assertEqual(compare_versions("1.10.0", "1.9.0"), 1)
        self.assertEqual(compare_versions("1.0.10", "1.0.9"), 1)
        self.assertEqual(compare_versions("0.0.1", "0.0.2"), -1)
        self.assertEqual(compare_versions("10.0.0", "9.99.99"), 1)

    def test_prerelease_is_lower_than_release(self):
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0"), -1)
        self.assertEqual(compare_versions("1.0.0", "1.0.0-rc.1"), 1)
        self.assertEqual(compare_versions("1.0.1-alpha", "1.0.0"), 1)

    def test_spec_example_chain(self):
        chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
                 "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        for i, lower in enumerate(chain):
            for higher in chain[i + 1:]:
                with self.subTest(lower=lower, higher=higher):
                    self.assertEqual(compare_versions(lower, higher), -1)
                    self.assertEqual(compare_versions(higher, lower), 1)
            self.assertEqual(compare_versions(lower, lower), 0)

    def test_numeric_identifiers_compare_numerically(self):
        self.assertEqual(compare_versions("1.0.0-2", "1.0.0-10"), -1)
        self.assertEqual(compare_versions("1.0.0-alpha.2", "1.0.0-alpha.10"), -1)

    def test_numeric_below_alphanumeric(self):
        self.assertEqual(compare_versions("1.0.0-1", "1.0.0-a"), -1)
        self.assertEqual(compare_versions("1.0.0-a", "1.0.0-1"), 1)
        self.assertEqual(compare_versions("1.0.0-alpha.1", "1.0.0-alpha.beta"), -1)

    def test_ascii_order_not_case_insensitive(self):
        self.assertEqual(compare_versions("1.0.0-Alpha", "1.0.0-alpha"), -1)
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0-Alpha"), 1)
        self.assertEqual(compare_versions("1.0.0-a-b", "1.0.0-a"), 1)

    def test_build_metadata_ignored(self):
        self.assertEqual(compare_versions("1.0.0+a", "1.0.0+b"), 0)
        self.assertEqual(compare_versions("1.0.0+20130313144700", "1.0.0"), 0)
        self.assertEqual(compare_versions("1.0.0-alpha+001", "1.0.0-alpha+002"), 0)
        self.assertEqual(compare_versions("1.0.0-alpha+zzz", "1.0.0"), -1)

    def test_invalid_versions(self):
        bad = ["1.0", "1", "1.0.0.0", "01.0.0", "1.00.0", "1.0.00", "v1.0.0", "1.0.0-", "1.0.0-01",
               "1.0.0-alpha..1", "1.0.0-alpha.", "1.0.0+", "1.0.0+a..b", "", "a.b.c", "1.0.0-al pha",
               "1.0.0 ", "-1.0.0", "1.0.0-alpha_1"]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    compare_versions(text, "1.0.0")
                with self.assertRaises(ValueError):
                    compare_versions("1.0.0", text)


if __name__ == "__main__":
    unittest.main()

import unittest

from pager import paginate


class PagerTests(unittest.TestCase):
    def test_first_page(self):
        r = paginate(list(range(9)), 1, 3)
        self.assertEqual(r["items"], [0, 1, 2])
        self.assertEqual((r["pages"], r["has_prev"], r["has_next"]), (3, False, True))

    def test_empty_list_has_one_page(self):
        r = paginate([], 1, 5)
        self.assertEqual((r["items"], r["pages"], r["has_next"]), ([], 1, False))

    def test_last_partial_page(self):
        r = paginate(list(range(10)), 4, 3)
        self.assertEqual(r["items"], [9])
        self.assertEqual((r["pages"], r["has_prev"], r["has_next"]), (4, True, False))

    def test_bad_per_page(self):
        with self.assertRaises(ValueError):
            paginate([1], 1, 0)

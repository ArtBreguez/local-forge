import unittest

from pager import paginate


class PagerHiddenTests(unittest.TestCase):
    def test_exact_multiple_has_no_extra_page(self):
        r = paginate(list(range(6)), 2, 3)
        self.assertEqual((r["items"], r["pages"], r["has_next"]), ([3, 4, 5], 2, False))
        with self.assertRaises(ValueError):
            paginate(list(range(6)), 3, 3)

    def test_fewer_items_than_one_page(self):
        r = paginate([1, 2], 1, 10)
        self.assertEqual((r["items"], r["pages"], r["has_prev"], r["has_next"]), ([1, 2], 1, False, False))

    def test_one_more_than_a_full_page(self):
        r = paginate(list(range(7)), 3, 3)
        self.assertEqual((r["items"], r["pages"]), ([6], 3))
        self.assertEqual(paginate(list(range(7)), 2, 3)["has_next"], True)

    def test_page_out_of_range(self):
        for page in (0, -1, 5):
            with self.assertRaises(ValueError):
                paginate(list(range(10)), page, 3)
        with self.assertRaises(ValueError):
            paginate([], 2, 3)

    def test_per_page_one_and_tuple_input(self):
        r = paginate((7, 8, 9), 3, 1)
        self.assertEqual((r["items"], r["pages"], r["page"]), ([9], 3, 3))

    def test_negative_per_page(self):
        with self.assertRaises(ValueError):
            paginate([1, 2, 3], 1, -2)

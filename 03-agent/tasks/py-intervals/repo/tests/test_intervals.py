import unittest

from intervals import merge_intervals


class MergeTests(unittest.TestCase):
    def test_overlapping(self):
        self.assertEqual(merge_intervals([(1, 3), (2, 4)]), [(1, 4)])

    def test_unsorted_input(self):
        self.assertEqual(merge_intervals([(5, 8), (1, 3), (2, 4)]), [(1, 4), (5, 8)])

    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])

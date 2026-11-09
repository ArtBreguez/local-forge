import unittest

from intervals import merge_intervals


class MergeHiddenTests(unittest.TestCase):
    def test_touching_intervals_merge(self):
        self.assertEqual(merge_intervals([(1, 3), (3, 5)]), [(1, 5)])
        self.assertEqual(merge_intervals([(3, 5), (1, 3), (5, 6)]), [(1, 6)])

    def test_contained_interval(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3), (4, 5)]), [(1, 10)])

    def test_gap_keeps_intervals_apart(self):
        self.assertEqual(merge_intervals([(1, 2), (3, 4)]), [(1, 2), (3, 4)])

    def test_lists_in_tuples_out_and_input_untouched(self):
        data = [[4, 6], [1, 2], [2, 3]]
        self.assertEqual(merge_intervals(data), [(1, 3), (4, 6)])
        self.assertEqual(data, [[4, 6], [1, 2], [2, 3]])

    def test_point_intervals_and_floats(self):
        self.assertEqual(merge_intervals([(2, 2), (2, 2)]), [(2, 2)])
        self.assertEqual(merge_intervals([(0.5, 1.5), (1.5, 2.5)]), [(0.5, 2.5)])

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            merge_intervals([(1, 2), (5, 3)])

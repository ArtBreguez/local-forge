import copy
import unittest
from solution import merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]])
        self.assertEqual(merge_intervals([]), [])
        self.assertEqual(merge_intervals([[4, 7]]), [[4, 7]])

    def test_touching_intervals_merge(self):
        self.assertEqual(merge_intervals([[1, 4], [4, 5]]), [[1, 5]])
        self.assertEqual(merge_intervals([[1, 2], [2, 3], [3, 4]]), [[1, 4]])

    def test_disjoint_stay_separate(self):
        self.assertEqual(merge_intervals([[1, 2], [3, 4]]), [[1, 2], [3, 4]])

    def test_unsorted_input(self):
        self.assertEqual(merge_intervals([[8, 10], [1, 3], [2, 6]]), [[1, 6], [8, 10]])

    def test_nested_interval_does_not_shrink_the_end(self):
        self.assertEqual(merge_intervals([[1, 10], [2, 3], [4, 5]]), [[1, 10]])
        self.assertEqual(merge_intervals([[1, 10], [2, 3], [11, 12]]), [[1, 10], [11, 12]])

    def test_zero_length(self):
        self.assertEqual(merge_intervals([[2, 2], [2, 2]]), [[2, 2]])
        self.assertEqual(merge_intervals([[1, 3], [3, 3]]), [[1, 3]])
        self.assertEqual(merge_intervals([[5, 5], [1, 2]]), [[1, 2], [5, 5]])

    def test_floats_and_negatives(self):
        self.assertEqual(merge_intervals([[-5, -1], [-2, 0.5], [1.5, 2.5]]), [[-5, 0.5], [1.5, 2.5]])

    def test_tuples_in_lists_out(self):
        result = merge_intervals([(3, 5), (1, 3)])
        self.assertEqual(result, [[1, 5]])
        self.assertIsInstance(result[0], list)

    def test_input_not_modified(self):
        data = [[8, 10], [1, 3], [2, 6]]
        snapshot = copy.deepcopy(data)
        result = merge_intervals(data)
        self.assertEqual(data, snapshot)
        result[0][0] = -100
        self.assertEqual(data, snapshot)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            merge_intervals([[5, 1]])
        with self.assertRaises(ValueError):
            merge_intervals([[1, 2], [9, 3]])


if __name__ == "__main__":
    unittest.main()

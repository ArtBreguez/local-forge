import unittest
from solution import spiral_order


class SpiralTests(unittest.TestCase):
    def test_square(self):
        self.assertEqual(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [1, 2, 3, 6, 9, 8, 7, 4, 5])
        self.assertEqual(spiral_order([[1, 2], [3, 4]]), [1, 2, 4, 3])

    def test_wide(self):
        self.assertEqual(spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7])

    def test_tall(self):
        self.assertEqual(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), [1, 2, 3, 6, 9, 12, 11, 10, 7, 4, 5, 8])

    def test_single_row_and_column(self):
        self.assertEqual(spiral_order([[1, 2, 3, 4]]), [1, 2, 3, 4])
        self.assertEqual(spiral_order([[1], [2], [3]]), [1, 2, 3])
        self.assertEqual(spiral_order([[7]]), [7])

    def test_odd_inner_cases(self):
        self.assertEqual(spiral_order([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
                         [1, 2, 3, 4, 5, 10, 15, 14, 13, 12, 11, 6, 7, 8, 9])
        self.assertEqual(spiral_order([[1, 2], [3, 4], [5, 6]]), [1, 2, 4, 6, 5, 3])
        self.assertEqual(spiral_order([[1, 2, 3], [4, 5, 6]]), [1, 2, 3, 6, 5, 4])

    def test_empty(self):
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[]]), [])
        self.assertEqual(spiral_order([[], []]), [])

    def test_every_element_once(self):
        for r in range(1, 7):
            for c in range(1, 7):
                m = [[i * c + j for j in range(c)] for i in range(r)]
                with self.subTest(r=r, c=c):
                    self.assertEqual(sorted(spiral_order(m)), list(range(r * c)))

    def test_ragged_rejected(self):
        for bad in ([[1, 2], [3]], [[1], [2, 3]], [[1, 2, 3], [4, 5, 6], [7, 8]], [[], [1]]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    spiral_order(bad)

    def test_input_not_modified(self):
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        spiral_order(m)
        self.assertEqual(m, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


if __name__ == "__main__":
    unittest.main()

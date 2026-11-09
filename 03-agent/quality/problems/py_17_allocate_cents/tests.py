import unittest
from solution import allocate


class AllocateTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(allocate(100, [1, 1, 1]), [34, 33, 33])
        self.assertEqual(allocate(10, [3, 2, 2]), [4, 3, 3])
        self.assertEqual(allocate(5, [0, 3, 0]), [0, 5, 0])

    def test_exact_division(self):
        self.assertEqual(allocate(10, [1, 1]), [5, 5])
        self.assertEqual(allocate(100, [33, 33, 34]), [33, 33, 34])
        self.assertEqual(allocate(7, [3, 3, 1]), [3, 3, 1])

    def test_largest_remainder_not_first_index(self):
        self.assertEqual(allocate(10, [3, 2, 2]), [4, 3, 3])
        self.assertEqual(allocate(10, [1, 3, 3]), [2, 4, 4])
        self.assertEqual(allocate(2, [5, 1, 1]), [2, 0, 0])
        self.assertEqual(allocate(100, [1, 2, 3]), [17, 33, 50])

    def test_ties_go_to_lower_index(self):
        self.assertEqual(allocate(1, [1, 1]), [1, 0])
        self.assertEqual(allocate(3, [1, 1, 4]), [1, 0, 2])
        self.assertEqual(allocate(10, [1] * 7), [2, 2, 2, 1, 1, 1, 1])
        self.assertEqual(allocate(2, [1, 1, 1]), [1, 1, 0])

    def test_zero_total(self):
        self.assertEqual(allocate(0, [1, 2]), [0, 0])

    def test_single_recipient(self):
        self.assertEqual(allocate(99, [7]), [99])

    def test_big_numbers_exact(self):
        total = 10 ** 18 + 1
        out = allocate(total, [1, 1, 1])
        self.assertEqual(sum(out), total)
        self.assertEqual(out, [333333333333333334, 333333333333333334, 333333333333333333])

    def test_always_sums_to_total(self):
        for total in range(0, 40):
            for weights in ([1, 2, 3], [5, 5, 5, 1], [0, 7, 0, 3], [9, 1], [2, 2, 2, 2, 2]):
                with self.subTest(total=total, weights=weights):
                    out = allocate(total, weights)
                    self.assertEqual(sum(out), total)
                    self.assertTrue(all(isinstance(x, int) and x >= 0 for x in out))
                    for x, w in zip(out, weights):
                        exact = total * w / sum(weights)
                        self.assertTrue(abs(x - exact) < 1)

    def test_errors(self):
        for total, weights in [(-1, [1]), (5, []), (5, [1, -1]), (5, [0, 0]), (5, [0])]:
            with self.subTest(total=total, weights=weights):
                with self.assertRaises(ValueError):
                    allocate(total, weights)

    def test_input_not_modified(self):
        w = [3, 2, 2]
        allocate(10, w)
        self.assertEqual(w, [3, 2, 2])


if __name__ == "__main__":
    unittest.main()

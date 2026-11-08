import unittest
from solution import search_rotated


class ProbeList(list):
    """A list that counts how many elements the code under test looks at."""
    reads = 0

    def __getitem__(self, i):
        self.reads += len(range(*i.indices(len(self)))) if isinstance(i, slice) else 1
        return super().__getitem__(i)

    def __iter__(self):
        self.reads += len(self)
        return super().__iter__()

    def __contains__(self, x):
        self.reads += len(self)
        return super().__contains__(x)

    def index(self, *args):
        self.reads += len(self)
        return super().index(*args)


class RotatedSearchTests(unittest.TestCase):
    def test_reads_only_a_logarithmic_number_of_elements(self):
        base = list(range(0, 200000, 2))
        for shift in (0, 1, 70000, 99999):
            for target in (0, 123456, 199998, 77777):
                nums = ProbeList(base[shift:] + base[:shift])
                expected = -1 if target % 2 else list.index(nums, target)
                nums.reads = 0
                self.assertEqual(search_rotated(nums, target), expected)
                self.assertLessEqual(nums.reads, 150, f"shift {shift} target {target}")

    def test_examples(self):
        self.assertEqual(search_rotated([4, 5, 6, 7, 0, 1, 2], 0), 4)
        self.assertEqual(search_rotated([4, 5, 6, 7, 0, 1, 2], 3), -1)
        self.assertEqual(search_rotated([], 1), -1)

    def test_tiny(self):
        self.assertEqual(search_rotated([1], 1), 0)
        self.assertEqual(search_rotated([1], 0), -1)
        self.assertEqual(search_rotated([1, 3], 3), 1)
        self.assertEqual(search_rotated([3, 1], 1), 1)
        self.assertEqual(search_rotated([3, 1], 3), 0)
        self.assertEqual(search_rotated([3, 1], 2), -1)

    def test_every_rotation_every_value(self):
        for n in range(1, 14):
            base = [i * 3 - 7 for i in range(n)]
            for k in range(n):
                nums = base[k:] + base[:k]
                for idx, v in enumerate(nums):
                    with self.subTest(nums=nums, target=v):
                        self.assertEqual(search_rotated(nums, v), idx)
                for missing in (-100, -6, -5, 0, 1, 2, 100):
                    if missing not in nums:
                        with self.subTest(nums=nums, target=missing):
                            self.assertEqual(search_rotated(nums, missing), -1)

    def test_not_rotated(self):
        nums = list(range(0, 100, 5))
        for i, v in enumerate(nums):
            self.assertEqual(search_rotated(nums, v), i)

    def test_large(self):
        base = list(range(0, 200000, 2))
        nums = base[70000:] + base[:70000]
        self.assertEqual(search_rotated(nums, 0), len(base) - 70000)
        self.assertEqual(search_rotated(nums, 199998), 29999)
        self.assertEqual(search_rotated(nums, 140001), -1)

    def test_input_not_modified(self):
        nums = [6, 7, 1, 2, 3, 4, 5]
        copy = list(nums)
        search_rotated(nums, 3)
        self.assertEqual(nums, copy)


if __name__ == "__main__":
    unittest.main()

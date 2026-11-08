import unittest
from solution import LRUCache


class LRUCacheTests(unittest.TestCase):
    def test_basic_get_put(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("b"), 2)
        self.assertIsNone(c.get("zzz"))
        self.assertEqual(c.get("zzz", -1), -1)
        self.assertEqual(len(c), 2)

    def test_eviction_order(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        self.assertNotIn("a", c)
        self.assertEqual(c.keys(), ["b", "c"])

    def test_get_refreshes_recency(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.put("c", 3)
        self.assertIn("a", c)
        self.assertNotIn("b", c)
        self.assertEqual(c.keys(), ["a", "c"])

    def test_put_existing_updates_value_and_recency_without_evicting(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)
        self.assertEqual(len(c), 2)
        self.assertEqual(c.keys(), ["b", "a"])
        self.assertEqual(c.get("a"), 10)
        self.assertEqual(c.get("b"), 2)

    def test_contains_does_not_refresh(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertIn("a", c)
        self.assertEqual(c.keys(), ["a", "b"])
        c.put("c", 3)
        self.assertNotIn("a", c)

    def test_missing_get_does_not_change_anything(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.get("nope")
        self.assertEqual(c.keys(), ["a"])
        self.assertEqual(len(c), 1)

    def test_falsy_values_are_stored(self):
        c = LRUCache(2)
        c.put("zero", 0)
        c.put("none", None)
        self.assertEqual(c.get("zero", "missing"), 0)
        self.assertIsNone(c.get("none", "missing"))
        self.assertIn("none", c)

    def test_capacity_one(self):
        c = LRUCache(1)
        c.put(1, "x")
        c.put(2, "y")
        self.assertEqual(c.keys(), [2])
        self.assertEqual(c.get(1), None)

    def test_long_sequence(self):
        c = LRUCache(3)
        for i in range(10):
            c.put(i, i * i)
        self.assertEqual(c.keys(), [7, 8, 9])
        c.get(7)
        c.put(10, 100)
        self.assertEqual(c.keys(), [9, 7, 10])

    def test_bad_capacity(self):
        for bad in (0, -3):
            with self.subTest(capacity=bad):
                with self.assertRaises(ValueError):
                    LRUCache(bad)
        for bad in (1.5, "2", None):
            with self.subTest(capacity=bad):
                with self.assertRaises(TypeError):
                    LRUCache(bad)


if __name__ == "__main__":
    unittest.main()

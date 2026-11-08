import unittest
from solution import flatten


class FlattenTests(unittest.TestCase):
    def test_example(self):
        self.assertEqual(flatten({"a": {"b": 1, "c": [10, {"d": 2}]}, "e": []}),
                         {"a.b": 1, "a.c.0": 10, "a.c.1.d": 2, "e": []})

    def test_flat_input(self):
        self.assertEqual(flatten({"x": 1, "y": "two"}), {"x": 1, "y": "two"})
        self.assertEqual(flatten({}), {})

    def test_custom_separator(self):
        self.assertEqual(flatten({"a": {"b": {"c": 1}}}, sep="/"), {"a/b/c": 1})
        self.assertEqual(flatten({"a": [1, 2]}, sep="::"), {"a::0": 1, "a::1": 2})
        self.assertEqual(flatten({"a": {"b": 1}}, ""), {"ab": 1})

    def test_empty_containers_are_kept(self):
        self.assertEqual(flatten({"a": {}}), {"a": {}})
        self.assertEqual(flatten({"a": {"b": []}}), {"a.b": []})
        self.assertEqual(flatten({"a": [[], {}]}), {"a.0": [], "a.1": {}})

    def test_leaves_are_not_expanded(self):
        self.assertEqual(flatten({"t": (1, 2), "s": {1, 2}, "n": None, "f": False, "z": 0}),
                         {"t": (1, 2), "s": {1, 2}, "n": None, "f": False, "z": 0})

    def test_non_string_keys(self):
        self.assertEqual(flatten({1: {2: 3}, None: {True: "x"}}), {"1.2": 3, "None.True": "x"})

    def test_list_of_lists(self):
        self.assertEqual(flatten({"m": [[1, 2], [3]]}), {"m.0.0": 1, "m.0.1": 2, "m.1.0": 3})

    def test_deep(self):
        d = leaf = {}
        for i in range(30):
            leaf["k"] = {}
            leaf = leaf["k"]
        leaf["end"] = 1
        self.assertEqual(flatten(d), {".".join(["k"] * 30 + ["end"]): 1})

    def test_input_not_modified(self):
        data = {"a": {"b": [1, {"c": 2}]}}
        import copy
        snap = copy.deepcopy(data)
        flatten(data)
        self.assertEqual(data, snap)

    def test_top_level_must_be_dict(self):
        for bad in ([1, 2], "abc", None, 5, (1, 2)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    flatten(bad)


if __name__ == "__main__":
    unittest.main()

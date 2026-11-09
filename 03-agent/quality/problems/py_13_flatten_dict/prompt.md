Write a function `flatten(obj, sep=".")` that flattens nested dictionaries and lists into a single-level dictionary.

`obj` must be a `dict` (raise `TypeError` otherwise). Walk it recursively: the keys on the path to a value are joined with `sep` to form the new key; for a list, the position of the element (0, 1, 2, ...) is used as its key part. Keys are converted with `str()`. Only `dict` and `list` are containers: every other value, including tuples, sets and `None`, is a leaf and is stored as is. An empty dict or empty list that is nested inside `obj` is itself kept as a leaf value (it is not dropped). An empty top-level dict gives `{}`.

Example: `flatten({"a": {"b": 1, "c": [10, {"d": 2}]}, "e": []})` -> `{"a.b": 1, "a.c.0": 10, "a.c.1.d": 2, "e": []}`.

Implement `merge_intervals(intervals)` in `intervals.py`; the file holds a stub.

`intervals` is a list of `(start, end)` pairs (tuples or two-item lists) of numbers. Return a new list of `(start, end)` tuples, sorted by start, in which overlapping intervals and intervals that merely touch (one ends where the next starts) are merged into one. The input must not be modified. An empty input gives an empty list. Raise `ValueError` if any pair has `start > end` (an interval with `start == end` is valid).

Example: `merge_intervals([(5, 8), (1, 3), (2, 4)])` -> `[(1, 4), (5, 8)]`.

The visible tests are in `tests/test_intervals.py` (run them with `python -m unittest`); more tests that you cannot see check the same behaviour. Do not modify anything under `tests/`.

Write a function `merge_intervals(intervals)` that merges overlapping intervals.

`intervals` is a list of `[start, end]` pairs (lists or tuples of ints or floats, in any order). Return a new list of `[start, end]` lists, sorted by start, in which every group of intervals that overlap or merely touch (the end of one equals the start of the next) is replaced by a single interval covering all of them. Example: `[[1,3],[2,6],[8,10],[15,18]]` -> `[[1,6],[8,10],[15,18]]`; `[[1,4],[4,5]]` -> `[[1,5]]`. An empty list gives `[]`.

The input must not be modified. Raise `ValueError` if any pair has `start > end` (a zero-length interval with `start == end` is allowed).

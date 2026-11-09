Write a function `search_rotated(nums, target)` that finds a value in a rotated sorted list.

`nums` is a list of distinct integers that was sorted in ascending order and then rotated by an unknown amount (rotating `[0,1,2,4,5,6,7]` by 3 gives `[4,5,6,7,0,1,2]`; a rotation of 0 leaves it sorted). Return the index of `target` in `nums`, or `-1` if it is not present. The list may be empty. Use a binary search, O(log n): the tests count how many elements of the list your code reads (indexing, slicing, iterating, `in` and `.index` all count) and a lookup in a list of 100,000 numbers that reads more than 150 of them fails. The input must not be modified.

Examples: `search_rotated([4,5,6,7,0,1,2], 0)` -> `4`; `search_rotated([4,5,6,7,0,1,2], 3)` -> `-1`; `search_rotated([], 1)` -> `-1`.

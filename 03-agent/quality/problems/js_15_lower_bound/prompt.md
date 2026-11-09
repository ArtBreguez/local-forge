Write a function `lowerBound(arr, x)` that does a binary search in a sorted array and returns the first position where `x` could be inserted while keeping the array sorted.

`arr` is an array of numbers sorted in ascending order (duplicates allowed). Return the smallest index `i` such that `arr[i] >= x`, or `arr.length` if every element is smaller than `x`. For `lowerBound([1,2,2,2,3], 2)` the answer is `1`; for `lowerBound([1,2,3], 4)` it is `3`; for `lowerBound([], 7)` it is `0`. Use a binary search, O(log n): the tests count how many elements your code reads from a million-element array, and a lookup that reads more than 100 of them fails. Do not modify the array.

Throw a `TypeError` if `arr` is not an array or `x` is not a number.

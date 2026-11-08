Write a function `chunk(arr, size)` that splits an array into consecutive chunks.

Return a new array of arrays, each with `size` elements except possibly the last, which holds the remainder. `chunk([1,2,3,4,5], 2)` -> `[[1,2],[3,4],[5]]`; `chunk([], 3)` -> `[]`; `chunk([1,2], 5)` -> `[[1,2]]`. The input array must not be modified, and the chunks must be new arrays.

Throw a `TypeError` if `arr` is not an array or `size` is not a number. Throw a `RangeError` if `size` is a number but not a positive integer (0, negative, fractional, `NaN` or `Infinity`).

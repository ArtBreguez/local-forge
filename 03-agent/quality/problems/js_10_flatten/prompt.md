Write a function `flatten(arr, depth = 1)` that flattens nested arrays up to a given depth, without using `Array.prototype.flat` or `flatMap`.

Return a new array. `depth` is how many levels of nesting to remove: `flatten([1,[2,[3,[4]]]])` -> `[1,2,[3,[4]]]` (default depth 1), `flatten([1,[2,[3,[4]]]], 2)` -> `[1,2,3,[4]]`, `flatten([1,[2,[3,[4]]]], Infinity)` -> `[1,2,3,4]`. `depth` 0 returns a shallow copy. Only arrays are flattened (strings and array-like objects are left as elements), and empty nested arrays simply disappear. The input is not modified.

Throw a `TypeError` if `arr` is not an array. Throw a `RangeError` if `depth` is negative or `NaN`.

Write a function `deepEqual(a, b)` that returns `true` if two values are deeply equal and `false` otherwise.

Rules:
- Primitives are equal when `===` says so, except that `NaN` equals `NaN`. (`0` and `-0` are equal.)
- Arrays are equal when they have the same length and their elements are deeply equal in order. An array is never equal to a non-array.
- `Date` objects are equal when they hold the same timestamp; a `Date` is never equal to a non-`Date`.
- Plain objects are equal when they have exactly the same set of own enumerable string keys and the values for each key are deeply equal. Key order does not matter. A key whose value is `undefined` is not the same as a missing key.
- `null` is only equal to `null`. Functions are equal only when they are the same function.

You do not need to handle cyclic structures, `Map`, `Set` or typed arrays.

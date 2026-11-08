Write a function `groupBy(items, keyFn)` that groups array items by a computed key.

`keyFn` is called as `keyFn(item, index)` for every item, in order. Return a `Map` whose keys are the computed keys, in the order the keys were first seen, and whose values are arrays of the items with that key, in their original order. Keys are compared the way a `Map` compares them (so `1` and `'1'` are different keys and `NaN` is a single key). An empty array gives an empty `Map`. The input array is not modified.

Throw a `TypeError` if `items` is not an array or `keyFn` is not a function.

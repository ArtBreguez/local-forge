Write a function `memoize(fn, keyFn)` that returns a memoized version of `fn`.

- The cache key for a call is `keyFn(...args)` when `keyFn` is given, otherwise the first argument. Keys are compared the way a `Map` compares them.
- When the key has been seen before, return the stored result without calling `fn` again. Falsy results (`0`, `''`, `false`, `null`, `undefined`) are cached like any other result.
- If `fn` throws, nothing is cached and the error propagates, so the next call with that key calls `fn` again.
- `fn` is called with the same `this` and arguments as the memoized function.
- The returned function has a `clear()` method that empties the cache.

Throw a `TypeError` if `fn` is not a function, or if `keyFn` is given and is not a function.

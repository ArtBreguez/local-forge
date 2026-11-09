Write an async function `retry(fn, attempts)` that calls `fn` until it succeeds, up to a limited number of times.

`fn` is called as `fn(attemptNumber)` where `attemptNumber` is 1 for the first call, 2 for the second, and so on. `fn` may return a value or a promise, or throw synchronously. If the call succeeds (returns a value or its promise resolves), `retry` resolves with that value and `fn` is not called again. If it throws or rejects, call it again, up to `attempts` calls in total, one at a time (the next call starts only after the previous one has settled). There is no delay between attempts. If every attempt fails, reject with the error from the last attempt.

`attempts` must be an integer of at least 1; otherwise the returned promise rejects with a `RangeError` (and `fn` is never called). If `fn` is not a function the returned promise rejects with a `TypeError`.

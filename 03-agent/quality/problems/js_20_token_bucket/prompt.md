Write a function `createLimiter({ capacity, refillPerSec, now })` that builds a token-bucket rate limiter.

The bucket holds up to `capacity` tokens and starts full. `now()` returns the current time in milliseconds (it is injected so that tests can control the clock). Tokens are added continuously at `refillPerSec` tokens per second, including fractions of a token, but the bucket never holds more than `capacity`. The returned object has one method, `tryTake(n = 1)`: first bring the bucket up to date using `now()`; then, if it holds at least `n` tokens, remove `n` tokens and return `true`; otherwise return `false` and leave the bucket as it is (a refused request consumes nothing). Asking for more than `capacity` tokens is simply refused.

Throw a `RangeError` if `capacity` is not a positive integer or `refillPerSec` is not a positive finite number; throw a `TypeError` if `now` is not a function. `tryTake` throws a `RangeError` if `n` is not a positive integer.

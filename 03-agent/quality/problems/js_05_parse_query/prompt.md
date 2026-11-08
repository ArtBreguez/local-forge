Write a function `parseQuery(qs)` that parses a URL query string into a plain object.

- An optional leading `?` is ignored. The empty string (or just `?`) gives `{}`.
- Pairs are separated by `&`; empty pairs (as in `a=1&&b=2` or a trailing `&`) are skipped.
- A pair is split at its first `=`: `a=b=c` gives key `a`, value `b=c`. A pair without `=` has the value `''` (so `flag` -> `{ flag: '' }`).
- Keys and values are decoded: first each `+` becomes a space, then percent-escapes are decoded as UTF-8 (so `%2B` is a literal `+`, and `%E2%9C%93` is `✓`). A pair whose decoded key is empty is skipped.
- If a key occurs once, its value is a string. If it occurs more than once, the value is an array of all its values in order.
- Malformed percent-encoding (such as `%zz` or a truncated `%E2%9C`) makes the function throw a `URIError`.

Throw a `TypeError` if `qs` is not a string. The result is an object whose own properties are the parsed keys (a normal object or one created with `Object.create(null)` are both fine); keys such as `constructor` or `toString` are ordinary keys like any other.

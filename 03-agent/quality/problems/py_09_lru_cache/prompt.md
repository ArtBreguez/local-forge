Write a class `LRUCache` implementing a least-recently-used cache.

- `LRUCache(capacity)`: `capacity` must be an `int` of at least 1; raise `TypeError` if it is not an int and `ValueError` if it is smaller than 1.
- `get(key, default=None)`: return the value stored for `key` and mark the key as most recently used; return `default` if the key is absent.
- `put(key, value)`: store the value and mark the key as most recently used. Updating an existing key never evicts anything. If adding a new key would exceed the capacity, remove the least recently used key first.
- `len(cache)` is the number of stored keys.
- `key in cache` reports presence only; it does not change the recency order.
- `cache.keys()` returns a list of the keys ordered from least recently used to most recently used.

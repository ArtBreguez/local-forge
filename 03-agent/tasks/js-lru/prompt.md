Implement the class `LRUCache` in `src/lru.js` (CommonJS, `module.exports = { LRUCache }`); the file holds a stub.

- `new LRUCache(capacity)`: `capacity` must be a positive integer, otherwise throw `RangeError`.
- `get(key)`: the value, or `undefined` if the key is absent. A hit makes the key the most recently used.
- `set(key, value)`: stores the value and makes the key the most recently used (also when the key already existed). If the cache then holds more than `capacity` keys, the least recently used key is removed. Returns the cache itself.
- `has(key)`: true or false. It does not change the order of use.
- `delete(key)`: removes the key, returns true if it was there.
- `size` (a getter): the number of keys held.
- `keys()`: an array of the keys from least to most recently used.

The visible tests are in `test/lru.test.js` (run them with `node --test`); more tests that you cannot see check the same behaviour. Do not modify anything under `test/`.

function memoize(fn, keyFn) {
  if (typeof fn !== 'function') throw new TypeError('fn must be a function');
  if (keyFn !== undefined && typeof keyFn !== 'function') throw new TypeError('keyFn must be a function');
  const cache = new Map();
  function memoized(...args) {
    const key = keyFn ? keyFn(...args) : args[0];
    if (cache.has(key)) return cache.get(key);
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  }
  memoized.clear = () => cache.clear();
  return memoized;
}

module.exports = { memoize };

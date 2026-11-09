function memoize(fn, keyFn) {
  if (typeof fn !== 'function') throw new TypeError('fn must be a function');
  if (keyFn !== undefined && typeof keyFn !== 'function') throw new TypeError('keyFn must be a function');
  const cache = {};
  function memoized(...args) {
    const key = keyFn ? keyFn(...args) : args[0];
    if (cache[key]) return cache[key];
    const result = fn.apply(this, args);
    cache[key] = result;
    return result;
  }
  memoized.clear = () => { for (const k of Object.keys(cache)) delete cache[k]; };
  return memoized;
}

module.exports = { memoize };

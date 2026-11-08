function createLimiter({ capacity, refillPerSec, now }) {
  if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError('capacity must be a positive integer');
  if (typeof refillPerSec !== 'number' || !Number.isFinite(refillPerSec) || refillPerSec <= 0) {
    throw new RangeError('refillPerSec must be a positive finite number');
  }
  if (typeof now !== 'function') throw new TypeError('now must be a function');
  let tokens = capacity;
  let last = now();
  return {
    tryTake(n = 1) {
      if (!Number.isInteger(n) || n < 1) throw new RangeError('n must be a positive integer');
      const t = now();
      if (t > last) {
        tokens = Math.min(capacity, tokens + ((t - last) / 1000) * refillPerSec);
        last = t;
      }
      if (tokens >= n) {
        tokens -= n;
        return true;
      }
      return false;
    },
  };
}

module.exports = { createLimiter };

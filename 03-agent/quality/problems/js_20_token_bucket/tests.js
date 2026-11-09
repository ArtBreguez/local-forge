const test = require('node:test');
const assert = require('node:assert/strict');
const { createLimiter } = require('./solution.js');

function clock(start = 1000) {
  let t = start;
  const now = () => t;
  now.advance = (ms) => { t += ms; };
  return now;
}

test('starts full and drains', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 3, refillPerSec: 1, now });
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
});

test('refills over time', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 3, refillPerSec: 2, now });
  for (let i = 0; i < 3; i += 1) assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
  now.advance(500);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
  now.advance(1000);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
});

test('fractions of a token accumulate across calls', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 2, refillPerSec: 2, now });
  assert.equal(lim.tryTake(2), true);
  now.advance(250);
  assert.equal(lim.tryTake(), false);
  now.advance(250);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
});

test('the bucket never holds more than capacity', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 3, refillPerSec: 10, now });
  now.advance(3600 * 1000);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
});

test('a refused request consumes nothing', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 3, refillPerSec: 1, now });
  assert.equal(lim.tryTake(2), true);
  assert.equal(lim.tryTake(2), false);
  assert.equal(lim.tryTake(1), true);
  assert.equal(lim.tryTake(1), false);
});

test('requests larger than capacity are refused', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 2, refillPerSec: 100, now });
  assert.equal(lim.tryTake(3), false);
  now.advance(10000);
  assert.equal(lim.tryTake(3), false);
  assert.equal(lim.tryTake(2), true);
});

test('taking several tokens at once', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 5, refillPerSec: 1, now });
  assert.equal(lim.tryTake(5), true);
  now.advance(4000);
  assert.equal(lim.tryTake(5), false);
  now.advance(1000);
  assert.equal(lim.tryTake(5), true);
});

test('time is only read through now()', () => {
  const now = clock(5);
  const lim = createLimiter({ capacity: 1, refillPerSec: 8, now });
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
  now.advance(62.5);
  assert.equal(lim.tryTake(), false);
  now.advance(62.5);
  assert.equal(lim.tryTake(), true);
  assert.equal(lim.tryTake(), false);
});

test('limiters are independent', () => {
  const now = clock();
  const a = createLimiter({ capacity: 1, refillPerSec: 1, now });
  const b = createLimiter({ capacity: 1, refillPerSec: 1, now });
  assert.equal(a.tryTake(), true);
  assert.equal(b.tryTake(), true);
  assert.equal(a.tryTake(), false);
});

test('invalid configuration', () => {
  const now = clock();
  for (const capacity of [0, -1, 1.5, NaN, '3', undefined]) {
    assert.throws(() => createLimiter({ capacity, refillPerSec: 1, now }), RangeError, `capacity ${capacity}`);
  }
  for (const refillPerSec of [0, -1, NaN, Infinity, '1', undefined]) {
    assert.throws(() => createLimiter({ capacity: 1, refillPerSec, now }), RangeError, `refill ${refillPerSec}`);
  }
  assert.throws(() => createLimiter({ capacity: 1, refillPerSec: 1, now: 5 }), TypeError);
  assert.throws(() => createLimiter({ capacity: 1, refillPerSec: 1 }), TypeError);
});

test('invalid token counts', () => {
  const now = clock();
  const lim = createLimiter({ capacity: 3, refillPerSec: 1, now });
  for (const n of [0, -1, 1.5, NaN, '1', null]) {
    assert.throws(() => lim.tryTake(n), RangeError, `n ${n}`);
  }
});

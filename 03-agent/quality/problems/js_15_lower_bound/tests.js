const test = require('node:test');
const assert = require('node:assert/strict');
const { lowerBound } = require('./solution.js');

test('examples', () => {
  assert.equal(lowerBound([1, 2, 2, 2, 3], 2), 1);
  assert.equal(lowerBound([1, 2, 3], 4), 3);
  assert.equal(lowerBound([], 7), 0);
});

test('below, between, above', () => {
  const a = [10, 20, 30, 40];
  assert.equal(lowerBound(a, 5), 0);
  assert.equal(lowerBound(a, 10), 0);
  assert.equal(lowerBound(a, 15), 1);
  assert.equal(lowerBound(a, 30), 2);
  assert.equal(lowerBound(a, 31), 3);
  assert.equal(lowerBound(a, 40), 3);
  assert.equal(lowerBound(a, 41), 4);
});

test('runs of duplicates', () => {
  assert.equal(lowerBound([5, 5, 5, 5], 5), 0);
  assert.equal(lowerBound([5, 5, 5, 5], 6), 4);
  assert.equal(lowerBound([5, 5, 5, 5], 4), 0);
  assert.equal(lowerBound([1, 1, 2, 2, 2, 2, 3, 3], 2), 2);
  assert.equal(lowerBound([1, 1, 2, 2, 2, 2, 3, 3], 3), 6);
});

test('negative and fractional values', () => {
  const a = [-5, -2.5, 0, 0.5, 3];
  assert.equal(lowerBound(a, -3), 1);
  assert.equal(lowerBound(a, -2.5), 1);
  assert.equal(lowerBound(a, 0.25), 3);
  assert.equal(lowerBound(a, -Infinity), 0);
  assert.equal(lowerBound(a, Infinity), 5);
});

test('agrees with a linear scan on every position', () => {
  for (let n = 0; n < 12; n += 1) {
    const arr = Array.from({ length: n }, (_, i) => Math.floor(i / 2) * 2);
    for (let x = -1; x <= n + 1; x += 1) {
      let expected = arr.length;
      for (let i = 0; i < arr.length; i += 1) { if (arr[i] >= x) { expected = i; break; } }
      assert.equal(lowerBound(arr, x), expected, `${JSON.stringify(arr)} ${x}`);
    }
  }
});

test('reads only a logarithmic number of elements', () => {
  const big = Array.from({ length: 1000000 }, (_, i) => i * 2);
  for (const x of [-5, 0, 1, 777777, 1999998, 1999999, 5000000]) {
    let reads = 0;
    const probe = new Proxy(big, {
      get(target, key, receiver) {
        if (typeof key === 'string' && /^\d+$/.test(key)) reads += 1;
        return Reflect.get(target, key, receiver);
      },
    });
    const expected = x <= 0 ? 0 : Math.min(Math.ceil(x / 2), big.length);
    assert.equal(lowerBound(probe, x), expected, `x=${x}`);
    assert.ok(reads <= 100, `x=${x} read ${reads} elements`);
  }
});

test('large array', () => {
  const big = Array.from({ length: 1000000 }, (_, i) => i * 2);
  assert.equal(lowerBound(big, 0), 0);
  assert.equal(lowerBound(big, 999999), 500000);
  assert.equal(lowerBound(big, 1999998), 999999);
  assert.equal(lowerBound(big, 1999999), 1000000);
});

test('does not modify the array', () => {
  const a = [1, 2, 3];
  lowerBound(a, 2);
  assert.deepEqual(a, [1, 2, 3]);
});

test('bad arguments', () => {
  assert.throws(() => lowerBound('abc', 1), TypeError);
  assert.throws(() => lowerBound(null, 1), TypeError);
  assert.throws(() => lowerBound([1, 2], '1'), TypeError);
  assert.throws(() => lowerBound([1, 2]), TypeError);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const { chunk } = require('./solution.js');

test('splits with a remainder', () => {
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.deepEqual(chunk([1, 2, 3, 4, 5, 6], 3), [[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(chunk([1, 2, 3, 4, 5, 6, 7], 3), [[1, 2, 3], [4, 5, 6], [7]]);
});

test('edge sizes', () => {
  assert.deepEqual(chunk([], 3), []);
  assert.deepEqual(chunk([1, 2], 5), [[1, 2]]);
  assert.deepEqual(chunk([1, 2, 3], 1), [[1], [2], [3]]);
  assert.deepEqual(chunk([1, 2, 3], 3), [[1, 2, 3]]);
});

test('keeps element types', () => {
  assert.deepEqual(chunk(['a', null, undefined, { x: 1 }], 2), [['a', null], [undefined, { x: 1 }]]);
});

test('does not modify the input and returns fresh arrays', () => {
  const input = [1, 2, 3, 4];
  const out = chunk(input, 2);
  assert.deepEqual(input, [1, 2, 3, 4]);
  out[0].push(99);
  assert.deepEqual(input, [1, 2, 3, 4]);
});

test('bad size', () => {
  for (const size of [0, -1, 1.5, NaN, Infinity, -Infinity]) {
    assert.throws(() => chunk([1, 2, 3], size), RangeError, `size ${size}`);
  }
  for (const size of ['2', null, undefined, {}, [2]]) {
    assert.throws(() => chunk([1, 2, 3], size), TypeError, `size ${String(size)}`);
  }
});

test('bad array', () => {
  for (const arr of ['abc', null, undefined, { length: 3 }, 5]) {
    assert.throws(() => chunk(arr, 2), TypeError);
  }
});

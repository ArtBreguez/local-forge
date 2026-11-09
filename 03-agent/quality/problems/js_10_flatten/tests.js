const test = require('node:test');
const assert = require('node:assert/strict');
const { flatten } = require('./solution.js');

test('default depth is one', () => {
  assert.deepEqual(flatten([1, [2, [3, [4]]]]), [1, 2, [3, [4]]]);
  assert.deepEqual(flatten([[1, 2], [3], []]), [1, 2, 3]);
  assert.deepEqual(flatten([]), []);
  assert.deepEqual(flatten([1, 2, 3]), [1, 2, 3]);
});

test('explicit depths', () => {
  const input = [1, [2, [3, [4]]]];
  assert.deepEqual(flatten(input, 0), [1, [2, [3, [4]]]]);
  assert.deepEqual(flatten(input, 1), [1, 2, [3, [4]]]);
  assert.deepEqual(flatten(input, 2), [1, 2, 3, [4]]);
  assert.deepEqual(flatten(input, 3), [1, 2, 3, 4]);
  assert.deepEqual(flatten(input, 10), [1, 2, 3, 4]);
  assert.deepEqual(flatten(input, Infinity), [1, 2, 3, 4]);
});

test('depth 0 is a shallow copy', () => {
  const input = [1, [2]];
  const out = flatten(input, 0);
  assert.notEqual(out, input);
  assert.equal(out[1], input[1]);
});

test('empty nested arrays vanish', () => {
  assert.deepEqual(flatten([1, [], [[]], [[], [2]]], Infinity), [1, 2]);
  assert.deepEqual(flatten([[], []]), []);
  assert.deepEqual(flatten([[[]]], 1), [[]]);
});

test('only arrays are flattened', () => {
  assert.deepEqual(flatten(['ab', ['cd']], Infinity), ['ab', 'cd']);
  const arrayLike = { length: 2, 0: 'x', 1: 'y' };
  assert.deepEqual(flatten([arrayLike, [arrayLike]], Infinity), [arrayLike, arrayLike]);
  assert.deepEqual(flatten([null, [undefined, [0]]], Infinity), [null, undefined, 0]);
});

test('does not modify the input', () => {
  const input = [1, [2, [3]]];
  flatten(input, Infinity);
  assert.deepEqual(input, [1, [2, [3]]]);
});

test('does not use Array.prototype.flat', () => {
  const realFlat = Array.prototype.flat;
  const realFlatMap = Array.prototype.flatMap;
  Array.prototype.flat = function () { throw new Error('flat is not allowed'); };
  Array.prototype.flatMap = function () { throw new Error('flatMap is not allowed'); };
  try {
    assert.deepEqual(flatten([1, [2, [3]]], Infinity), [1, 2, 3]);
  } finally {
    Array.prototype.flat = realFlat;
    Array.prototype.flatMap = realFlatMap;
  }
});

test('deep nesting', () => {
  let nested = [1];
  for (let i = 0; i < 500; i += 1) nested = [nested];
  assert.deepEqual(flatten(nested, Infinity), [1]);
});

test('bad arguments', () => {
  for (const v of ['abc', null, undefined, { length: 1 }, 5]) {
    assert.throws(() => flatten(v), TypeError);
  }
  for (const d of [-1, NaN, -Infinity]) {
    assert.throws(() => flatten([1, [2]], d), RangeError, String(d));
  }
});

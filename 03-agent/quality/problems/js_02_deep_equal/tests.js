const test = require('node:test');
const assert = require('node:assert/strict');
const { deepEqual } = require('./solution.js');

test('primitives', () => {
  assert.equal(deepEqual(1, 1), true);
  assert.equal(deepEqual(1, 2), false);
  assert.equal(deepEqual('a', 'a'), true);
  assert.equal(deepEqual('a', 'b'), false);
  assert.equal(deepEqual(null, null), true);
  assert.equal(deepEqual(undefined, undefined), true);
  assert.equal(deepEqual(null, undefined), false);
  assert.equal(deepEqual(NaN, NaN), true);
  assert.equal(deepEqual(0, -0), true);
  assert.equal(deepEqual(1, '1'), false);
  assert.equal(deepEqual(true, 1), false);
  assert.equal(deepEqual(false, 0), false);
});

test('arrays', () => {
  assert.equal(deepEqual([1, 2, 3], [1, 2, 3]), true);
  assert.equal(deepEqual([1, 2], [1, 2, 3]), false);
  assert.equal(deepEqual([1, 2, 3], [3, 2, 1]), false);
  assert.equal(deepEqual([1, [2, [3]]], [1, [2, [3]]]), true);
  assert.equal(deepEqual([1, [2, [3]]], [1, [2, [4]]]), false);
  assert.equal(deepEqual([], []), true);
  assert.equal(deepEqual([NaN], [NaN]), true);
});

test('arrays and objects are different', () => {
  assert.equal(deepEqual([], {}), false);
  assert.equal(deepEqual({}, []), false);
  assert.equal(deepEqual([1, 2], { 0: 1, 1: 2 }), false);
  assert.equal(deepEqual({ 0: 1, 1: 2 }, [1, 2]), false);
});

test('objects', () => {
  assert.equal(deepEqual({ a: 1, b: 2 }, { b: 2, a: 1 }), true);
  assert.equal(deepEqual({ a: 1 }, { a: 2 }), false);
  assert.equal(deepEqual({ a: 1 }, { a: 1, b: 2 }), false);
  assert.equal(deepEqual({ a: 1, b: 2 }, { a: 1 }), false);
  assert.equal(deepEqual({ a: { b: [1, { c: 2 }] } }, { a: { b: [1, { c: 2 }] } }), true);
  assert.equal(deepEqual({ a: { b: [1, { c: 2 }] } }, { a: { b: [1, { c: 3 }] } }), false);
  assert.equal(deepEqual({}, {}), true);
});

test('undefined-valued keys are not missing keys', () => {
  assert.equal(deepEqual({ a: 1, b: undefined }, { a: 1 }), false);
  assert.equal(deepEqual({ a: 1 }, { a: 1, b: undefined }), false);
  assert.equal(deepEqual({ a: undefined }, { b: undefined }), false);
  assert.equal(deepEqual({ a: undefined }, { a: undefined }), true);
});

test('null and objects', () => {
  assert.equal(deepEqual(null, {}), false);
  assert.equal(deepEqual({}, null), false);
  assert.equal(deepEqual({ a: null }, { a: {} }), false);
  assert.equal(deepEqual({ a: null }, { a: null }), true);
});

test('dates', () => {
  assert.equal(deepEqual(new Date(1000), new Date(1000)), true);
  assert.equal(deepEqual(new Date(1000), new Date(2000)), false);
  assert.equal(deepEqual(new Date(1000), {}), false);
  assert.equal(deepEqual({}, new Date(1000)), false);
  assert.equal(deepEqual(new Date(1000), 1000), false);
  assert.equal(deepEqual({ when: new Date(5) }, { when: new Date(5) }), true);
});

test('functions', () => {
  const f = () => 1;
  const g = () => 1;
  assert.equal(deepEqual(f, f), true);
  assert.equal(deepEqual(f, g), false);
  assert.equal(deepEqual({ f }, { f }), true);
});

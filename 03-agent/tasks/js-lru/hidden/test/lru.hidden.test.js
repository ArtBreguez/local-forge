const test = require('node:test');
const assert = require('node:assert/strict');
const { LRUCache } = require('../src/lru.js');

test('hidden: set on an existing key updates the value and the order', () => {
  const c = new LRUCache(3);
  c.set('a', 1).set('b', 2).set('c', 3);
  c.set('a', 10);
  assert.deepEqual(c.keys(), ['b', 'c', 'a']);
  assert.equal(c.get('a'), 10);
  assert.equal(c.size, 3);
  c.set('d', 4);
  assert.deepEqual(c.keys(), ['c', 'a', 'd']);
});

test('hidden: has does not refresh a key', () => {
  const c = new LRUCache(2);
  c.set('a', 1);
  c.set('b', 2);
  assert.equal(c.has('a'), true);
  c.set('c', 3);
  assert.equal(c.has('a'), false);
  assert.deepEqual(c.keys(), ['b', 'c']);
});

test('hidden: a miss does not change anything', () => {
  const c = new LRUCache(2);
  c.set('a', 1);
  c.set('b', 2);
  assert.equal(c.get('x'), undefined);
  assert.deepEqual(c.keys(), ['a', 'b']);
});

test('hidden: delete', () => {
  const c = new LRUCache(2);
  c.set('a', 1);
  assert.equal(c.delete('a'), true);
  assert.equal(c.delete('a'), false);
  assert.equal(c.size, 0);
  assert.deepEqual(c.keys(), []);
});

test('hidden: capacity one and falsy values', () => {
  const c = new LRUCache(1);
  c.set('a', 0);
  assert.equal(c.get('a'), 0);
  assert.equal(c.has('a'), true);
  c.set('b', null);
  assert.equal(c.has('a'), false);
  assert.equal(c.get('b'), null);
  assert.equal(c.set('c', 1), c);
});

test('hidden: invalid capacities', () => {
  for (const bad of [0, -2, 1.5, '3', undefined, NaN]) {
    assert.throws(() => new LRUCache(bad), RangeError);
  }
});

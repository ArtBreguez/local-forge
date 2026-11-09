const test = require('node:test');
const assert = require('node:assert/strict');
const { LRUCache } = require('../src/lru.js');

test('stores and reads values', () => {
  const c = new LRUCache(2);
  c.set('a', 1);
  c.set('b', 2);
  assert.equal(c.get('a'), 1);
  assert.equal(c.get('zzz'), undefined);
  assert.equal(c.size, 2);
});

test('evicts the least recently used key', () => {
  const c = new LRUCache(2);
  c.set('a', 1);
  c.set('b', 2);
  c.get('a');
  c.set('c', 3);
  assert.equal(c.has('b'), false);
  assert.deepEqual(c.keys(), ['a', 'c']);
});

test('capacity must be a positive integer', () => {
  assert.throws(() => new LRUCache(0), RangeError);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const { TaskQueue } = require('../src/queue.js');

test('first in, first out without priorities', () => {
  const q = new TaskQueue();
  q.push('a').push('b').push('c');
  assert.equal(q.size, 3);
  assert.equal(q.pop(), 'a');
  assert.equal(q.pop(), 'b');
  assert.equal(q.pop(), 'c');
  assert.equal(q.pop(), undefined);
});

test('higher priority first, ties in push order', () => {
  const q = new TaskQueue();
  q.push('low', -1).push('first', 5).push('mid').push('second', 5);
  assert.deepEqual(q.items().map((t) => t.name), ['first', 'second', 'mid', 'low']);
  assert.equal(q.pop(), 'first');
});

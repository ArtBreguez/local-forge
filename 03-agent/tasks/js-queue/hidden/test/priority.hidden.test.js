const test = require('node:test');
const assert = require('node:assert/strict');
const { TaskQueue } = require('../src/queue.js');
const { describe } = require('../src/report.js');

test('hidden: interleaved pushes keep push order among equals', () => {
  const q = new TaskQueue();
  q.push('a', 1).push('b').push('c', 1).push('d').push('e', 2);
  const out = [];
  for (let n = q.size; n > 0; n--) out.push(q.pop());
  assert.deepEqual(out, ['e', 'a', 'c', 'b', 'd']);
});

test('hidden: negative and zero priorities', () => {
  const q = new TaskQueue();
  q.push('neg', -3).push('zero', 0).push('neg2', -3);
  assert.deepEqual(q.items(), [
    { name: 'zero', priority: 0 },
    { name: 'neg', priority: -3 },
    { name: 'neg2', priority: -3 },
  ]);
});

test('hidden: items does not change the queue', () => {
  const q = new TaskQueue();
  q.push('x', 2).push('y');
  const first = q.items();
  first.pop();
  assert.equal(q.size, 2);
  assert.deepEqual(q.items().map((t) => t.name), ['x', 'y']);
});

test('hidden: priority must be an integer', () => {
  const q = new TaskQueue();
  for (const bad of [1.5, '2', NaN, null, Infinity]) {
    assert.throws(() => q.push('t', bad), RangeError);
  }
  assert.equal(q.size, 0);
});

test('hidden: report formats negative priorities and an empty queue', () => {
  const q = new TaskQueue();
  assert.equal(describe(q), '');
  q.push('write docs').push('cleanup', -1).push('deploy', 5);
  assert.equal(describe(q), '1. [p5] deploy\n2. write docs\n3. [p-1] cleanup');
});

test('hidden: pop on an empty queue after draining', () => {
  const q = new TaskQueue();
  q.push('only', 3);
  assert.equal(q.pop(), 'only');
  assert.equal(q.pop(), undefined);
  assert.equal(q.size, 0);
});

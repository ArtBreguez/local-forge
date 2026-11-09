const test = require('node:test');
const assert = require('node:assert/strict');
const { TaskQueue } = require('../src/queue.js');
const { describe } = require('../src/report.js');

test('plain report', () => {
  const q = new TaskQueue();
  q.push('write docs').push('ship');
  assert.equal(describe(q), '1. write docs\n2. ship');
});

test('report shows priorities', () => {
  const q = new TaskQueue();
  q.push('write docs').push('deploy', 5);
  assert.equal(describe(q), '1. [p5] deploy\n2. write docs');
});

const test = require('node:test');
const assert = require('node:assert/strict');
const { groupBy } = require('./solution.js');

test('groups in first-seen key order', () => {
  const m = groupBy([1, 2, 3, 4, 5], (x) => x % 2);
  assert.ok(m instanceof Map);
  assert.deepEqual([...m], [[1, [1, 3, 5]], [0, [2, 4]]]);
});

test('passes the index as the second argument', () => {
  const m = groupBy(['a', 'b', 'c', 'd', 'e'], (x, i) => i % 2);
  assert.deepEqual([...m], [[0, ['a', 'c', 'e']], [1, ['b', 'd']]]);
});

test('keys are compared like Map keys', () => {
  const m = groupBy([1, '1', 1, NaN, NaN, '1'], (x) => x);
  assert.deepEqual([...m.keys()].map(String), ['1', '1', 'NaN']);
  assert.equal(m.get(1).length, 2);
  assert.equal(m.get('1').length, 2);
  assert.equal(m.get(NaN).length, 2);
});

test('groups objects by property', () => {
  const people = [
    { name: 'ann', team: 'x' },
    { name: 'bob', team: 'y' },
    { name: 'cy', team: 'x' },
  ];
  const m = groupBy(people, (p) => p.team);
  assert.deepEqual(m.get('x').map((p) => p.name), ['ann', 'cy']);
  assert.deepEqual(m.get('y').map((p) => p.name), ['bob']);
  assert.equal(m.get('z'), undefined);
});

test('empty input and falsy keys', () => {
  assert.equal(groupBy([], (x) => x).size, 0);
  const m = groupBy([0, 1, 2], (x) => (x === 0 ? 0 : undefined));
  assert.deepEqual([...m], [[0, [0]], [undefined, [1, 2]]]);
});

test('key function is called once per item', () => {
  let calls = 0;
  groupBy([1, 2, 3], (x) => { calls += 1; return x; });
  assert.equal(calls, 3);
});

test('does not modify the input', () => {
  const input = [3, 1, 2];
  groupBy(input, (x) => x % 2);
  assert.deepEqual(input, [3, 1, 2]);
});

test('bad arguments', () => {
  assert.throws(() => groupBy('abc', (x) => x), TypeError);
  assert.throws(() => groupBy(null, (x) => x), TypeError);
  assert.throws(() => groupBy([1], 'length'), TypeError);
  assert.throws(() => groupBy([1]), TypeError);
});

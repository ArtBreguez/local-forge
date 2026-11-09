const test = require('node:test');
const assert = require('node:assert/strict');
const { memoize } = require('./solution.js');

test('caches by first argument', () => {
  let calls = 0;
  const sq = memoize((x) => { calls += 1; return x * x; });
  assert.equal(sq(4), 16);
  assert.equal(sq(4), 16);
  assert.equal(sq(5), 25);
  assert.equal(calls, 2);
});

test('falsy and undefined results are cached', () => {
  for (const value of [0, '', false, null, undefined, NaN]) {
    let calls = 0;
    const f = memoize(() => { calls += 1; return value; });
    assert.ok(Object.is(f('k'), value));
    assert.ok(Object.is(f('k'), value));
    assert.equal(calls, 1, `value ${String(value)}`);
  }
});

test('falsy keys work', () => {
  let calls = 0;
  const f = memoize((x) => { calls += 1; return String(x); });
  f(0); f(0); f(''); f(''); f(undefined); f(undefined); f(null); f(null);
  assert.equal(calls, 4);
});

test('keys compare like Map keys', () => {
  let calls = 0;
  const f = memoize((x) => { calls += 1; return typeof x; });
  f(1); f('1'); f(1); f('1');
  assert.equal(calls, 2);
  f(NaN); f(NaN);
  assert.equal(calls, 3);
  const a = {}; const b = {};
  f(a); f(a); f(b);
  assert.equal(calls, 5);
});

test('default key is only the first argument', () => {
  let calls = 0;
  const add = memoize((a, b) => { calls += 1; return a + b; });
  assert.equal(add(1, 2), 3);
  assert.equal(add(1, 100), 3);
  assert.equal(calls, 1);
});

test('custom key function sees all arguments', () => {
  let calls = 0;
  const add = memoize((a, b) => { calls += 1; return a + b; }, (a, b) => `${a},${b}`);
  assert.equal(add(1, 2), 3);
  assert.equal(add(1, 100), 101);
  assert.equal(add(1, 2), 3);
  assert.equal(calls, 2);
});

test('errors are not cached', () => {
  let calls = 0;
  const f = memoize((x) => {
    calls += 1;
    if (calls < 3) throw new Error(`fail ${calls}`);
    return x;
  });
  assert.throws(() => f('a'), /fail 1/);
  assert.throws(() => f('a'), /fail 2/);
  assert.equal(f('a'), 'a');
  assert.equal(f('a'), 'a');
  assert.equal(calls, 3);
});

test('this and all arguments are forwarded', () => {
  const obj = {
    base: 10,
    add: memoize(function (x, y, z) { return this.base + x + y + z; }),
  };
  assert.equal(obj.add(1, 2, 3), 16);
});

test('clear empties the cache', () => {
  let calls = 0;
  const f = memoize((x) => { calls += 1; return x; });
  f(1); f(1);
  assert.equal(calls, 1);
  f.clear();
  f(1);
  assert.equal(calls, 2);
});

test('bad arguments', () => {
  assert.throws(() => memoize('nope'), TypeError);
  assert.throws(() => memoize(), TypeError);
  assert.throws(() => memoize((x) => x, 'key'), TypeError);
  assert.doesNotThrow(() => memoize((x) => x, undefined));
});

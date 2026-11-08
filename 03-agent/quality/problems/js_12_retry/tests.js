const test = require('node:test');
const assert = require('node:assert/strict');
const { retry } = require('./solution.js');

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('resolves on first success without further calls', async () => {
  const seen = [];
  const out = await retry((n) => { seen.push(n); return 'ok'; }, 5);
  assert.equal(out, 'ok');
  assert.deepEqual(seen, [1]);
});

test('retries until success and passes attempt numbers', async () => {
  const seen = [];
  const out = await retry(async (n) => {
    seen.push(n);
    if (n < 3) throw new Error(`boom ${n}`);
    return n * 10;
  }, 5);
  assert.equal(out, 30);
  assert.deepEqual(seen, [1, 2, 3]);
});

test('rejects with the last error after all attempts', async () => {
  const seen = [];
  const errors = [];
  await assert.rejects(
    retry(async (n) => {
      seen.push(n);
      const e = new Error(`fail ${n}`);
      errors.push(e);
      throw e;
    }, 3),
    (err) => err === errors[2] && err.message === 'fail 3',
  );
  assert.deepEqual(seen, [1, 2, 3]);
});

test('a single attempt', async () => {
  let calls = 0;
  await assert.rejects(retry(() => { calls += 1; throw new Error('once'); }, 1), /once/);
  assert.equal(calls, 1);
});

test('synchronous throws count as failures', async () => {
  let calls = 0;
  const out = await retry((n) => {
    calls += 1;
    if (n === 1) throw new TypeError('sync');
    return 'second';
  }, 2);
  assert.equal(out, 'second');
  assert.equal(calls, 2);
});

test('falsy results are successes', async () => {
  for (const value of [0, '', false, null, undefined]) {
    let calls = 0;
    const out = await retry(() => { calls += 1; return value; }, 3);
    assert.ok(Object.is(out, value));
    assert.equal(calls, 1);
  }
});

test('attempts are sequential, never overlapping', async () => {
  const log = [];
  await retry(async (n) => {
    log.push(`start ${n}`);
    await tick();
    await tick();
    log.push(`end ${n}`);
    if (n < 3) throw new Error('again');
  }, 3);
  assert.deepEqual(log, ['start 1', 'end 1', 'start 2', 'end 2', 'start 3', 'end 3']);
});

test('invalid attempts reject with RangeError and never call fn', async () => {
  let calls = 0;
  const fn = () => { calls += 1; return 1; };
  for (const bad of [0, -1, 1.5, NaN, Infinity, '3', null, undefined]) {
    await assert.rejects(retry(fn, bad), RangeError, String(bad));
  }
  assert.equal(calls, 0);
});

test('fn must be a function', async () => {
  await assert.rejects(retry('nope', 3), TypeError);
  await assert.rejects(retry(undefined, 3), TypeError);
});

test('returns a promise even for invalid input', () => {
  const p = retry(() => 1, 0);
  assert.ok(p instanceof Promise);
  p.catch(() => {});
});

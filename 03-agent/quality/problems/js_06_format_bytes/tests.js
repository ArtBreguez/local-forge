const test = require('node:test');
const assert = require('node:assert/strict');
const { formatBytes } = require('./solution.js');

test('bytes', () => {
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(1), '1 B');
  assert.equal(formatBytes(999), '999 B');
  assert.equal(formatBytes(1023), '1023 B');
});

test('kilobytes', () => {
  assert.equal(formatBytes(1024), '1 KB');
  assert.equal(formatBytes(1536), '1.5 KB');
  assert.equal(formatBytes(10240), '10 KB');
  assert.equal(formatBytes(1127), '1.1 KB');
  assert.equal(formatBytes(1331), '1.3 KB');
});

test('trailing .0 is dropped', () => {
  assert.equal(formatBytes(1075), '1 KB');
  assert.equal(formatBytes(2048 + 50), '2 KB');
  assert.equal(formatBytes(1048576 + 10000), '1 MB');
});

test('larger units', () => {
  assert.equal(formatBytes(1048576), '1 MB');
  assert.equal(formatBytes(1572864), '1.5 MB');
  assert.equal(formatBytes(1073741824), '1 GB');
  assert.equal(formatBytes(1610612736), '1.5 GB');
  assert.equal(formatBytes(1099511627776), '1 TB');
  assert.equal(formatBytes(5497558138880), '5 TB');
});

test('rounding up to the next unit', () => {
  assert.equal(formatBytes(1048575), '1 MB');
  assert.equal(formatBytes(1073741823), '1 GB');
  assert.equal(formatBytes(1099511627775), '1 TB');
  assert.equal(formatBytes(1048000), '1023.4 KB');
});

test('terabytes are the last unit', () => {
  assert.equal(formatBytes(1125899906842624), '1024 TB');
  assert.equal(formatBytes(2251799813685248), '2048 TB');
});

test('non-numbers and non-finite numbers throw TypeError', () => {
  for (const v of ['100', null, undefined, NaN, Infinity, -Infinity, {}, [5]]) {
    assert.throws(() => formatBytes(v), TypeError, String(v));
  }
});

test('negative and fractional numbers throw RangeError', () => {
  for (const v of [-1, -1024, 1.5, 0.1]) {
    assert.throws(() => formatBytes(v), RangeError, String(v));
  }
});

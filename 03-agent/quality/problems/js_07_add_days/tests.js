const test = require('node:test');
const assert = require('node:assert/strict');
const { addDays } = require('./solution.js');

test('simple additions', () => {
  assert.equal(addDays('2024-01-01', 0), '2024-01-01');
  assert.equal(addDays('2024-01-01', 1), '2024-01-02');
  assert.equal(addDays('2024-01-31', 1), '2024-02-01');
  assert.equal(addDays('2024-02-28', 2), '2024-03-01');
  assert.equal(addDays('2024-02-28', 1), '2024-02-29');
  assert.equal(addDays('2023-02-28', 1), '2023-03-01');
});

test('year boundaries', () => {
  assert.equal(addDays('2023-12-31', 1), '2024-01-01');
  assert.equal(addDays('2024-01-01', -1), '2023-12-31');
  assert.equal(addDays('2024-03-01', -1), '2024-02-29');
});

test('long spans and leap years', () => {
  assert.equal(addDays('2024-01-31', 30), '2024-03-01');
  assert.equal(addDays('2024-02-29', 365), '2025-02-28');
  assert.equal(addDays('2024-02-29', 366), '2025-03-01');
  assert.equal(addDays('2000-01-01', 36524), '2099-12-31');
  assert.equal(addDays('2024-05-05', -400), '2023-04-01');
  assert.equal(addDays('1900-02-28', 1), '1900-03-01');
  assert.equal(addDays('2000-02-28', 1), '2000-02-29');
});

test('daylight-saving transitions do not matter', () => {
  assert.equal(addDays('2024-03-09', 1), '2024-03-10');
  assert.equal(addDays('2024-03-10', 1), '2024-03-11');
  assert.equal(addDays('2024-03-31', 1), '2024-04-01');
  assert.equal(addDays('2024-10-26', 1), '2024-10-27');
  assert.equal(addDays('2024-10-27', 1), '2024-10-28');
  assert.equal(addDays('2024-11-02', 1), '2024-11-03');
  assert.equal(addDays('2024-11-03', 1), '2024-11-04');
});

test('years below 1000 keep four digits', () => {
  assert.equal(addDays('0099-12-31', 1), '0100-01-01');
  assert.equal(addDays('0001-01-01', 0), '0001-01-01');
  assert.equal(addDays('0001-01-01', 1), '0001-01-02');
  assert.equal(addDays('0999-12-31', 1), '1000-01-01');
});

test('results outside 0001..9999 throw RangeError', () => {
  assert.throws(() => addDays('9999-12-31', 1), RangeError);
  assert.throws(() => addDays('0001-01-01', -1), RangeError);
  assert.equal(addDays('9999-12-30', 1), '9999-12-31');
});

test('invalid dates throw RangeError', () => {
  for (const s of ['2023-02-30', '2023-13-01', '2023-00-10', '2023-04-31', '2023-02-29', '0000-01-01', '2023-01-00']) {
    assert.throws(() => addDays(s, 1), RangeError, s);
  }
});

test('wrong shape throws RangeError', () => {
  for (const s of ['2024-2-3', '20240203', '2024-02-03T00:00', '24-02-03', '2024/02/03', ' 2024-02-03', '2024-02-03 ', '', 'tomorrow']) {
    assert.throws(() => addDays(s, 1), RangeError, JSON.stringify(s));
  }
});

test('bad argument types throw TypeError', () => {
  for (const d of [1.5, '1', null, undefined, NaN, Infinity]) {
    assert.throws(() => addDays('2024-01-01', d), TypeError, String(d));
  }
  for (const s of [null, undefined, 20240101, new Date(0)]) {
    assert.throws(() => addDays(s, 1), TypeError);
  }
});

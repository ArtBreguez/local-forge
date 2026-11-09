const test = require('node:test');
const assert = require('node:assert/strict');
const { ageOn } = require('./solution.js');

test('ordinary birthdays', () => {
  assert.equal(ageOn('2000-01-15', '2024-01-14'), 23);
  assert.equal(ageOn('2000-01-15', '2024-01-15'), 24);
  assert.equal(ageOn('2000-01-15', '2024-01-16'), 24);
  assert.equal(ageOn('2000-12-31', '2024-12-30'), 23);
  assert.equal(ageOn('2000-12-31', '2024-12-31'), 24);
  assert.equal(ageOn('1990-06-15', '2024-06-14'), 33);
  assert.equal(ageOn('1990-06-15', '2024-07-01'), 34);
  assert.equal(ageOn('1990-06-15', '2024-05-30'), 33);
});

test('same day and first year', () => {
  assert.equal(ageOn('2024-05-05', '2024-05-05'), 0);
  assert.equal(ageOn('2024-05-05', '2025-05-04'), 0);
  assert.equal(ageOn('2024-05-05', '2025-05-05'), 1);
});

test('leap-day births in common years', () => {
  assert.equal(ageOn('2000-02-29', '2023-02-28'), 22);
  assert.equal(ageOn('2000-02-29', '2023-03-01'), 23);
  assert.equal(ageOn('2000-02-29', '2023-12-31'), 23);
  assert.equal(ageOn('2004-02-29', '2005-02-28'), 0);
  assert.equal(ageOn('2004-02-29', '2005-03-01'), 1);
});

test('leap-day births in leap years', () => {
  assert.equal(ageOn('2000-02-29', '2024-02-28'), 23);
  assert.equal(ageOn('2000-02-29', '2024-02-29'), 24);
  assert.equal(ageOn('2000-02-29', '2024-03-01'), 24);
  assert.equal(ageOn('2000-02-29', '2000-02-29'), 0);
});

test('leap-day as the date being asked about', () => {
  assert.equal(ageOn('2001-02-28', '2024-02-29'), 23);
  assert.equal(ageOn('2001-03-01', '2024-02-29'), 22);
  assert.equal(ageOn('2001-03-01', '2024-03-01'), 23);
});

test('on before birth throws RangeError', () => {
  assert.throws(() => ageOn('2024-05-05', '2024-05-04'), RangeError);
  assert.throws(() => ageOn('2024-05-05', '2023-12-31'), RangeError);
});

test('invalid dates throw RangeError', () => {
  for (const [b, o] of [['2023-02-30', '2024-01-01'], ['2000-01-01', '2023-02-29'], ['2000-13-01', '2024-01-01'],
    ['2000-1-1', '2024-01-01'], ['2000-01-01', '20240101'], ['', '2024-01-01'], ['2000-02-29', 'today']]) {
    assert.throws(() => ageOn(b, o), RangeError, `${b} ${o}`);
  }
});

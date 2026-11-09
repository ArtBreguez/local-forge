import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseDuration } from '../src/duration.js';

test('single units', () => {
  assert.equal(parseDuration('45m'), 2700);
  assert.equal(parseDuration('90s'), 90);
  assert.equal(parseDuration('2h'), 7200);
});

test('combined units in any order, with or without spaces', () => {
  assert.equal(parseDuration('1h30m'), 5400);
  assert.equal(parseDuration('2h 5s'), 7205);
  assert.equal(parseDuration('10s 1m'), 70);
  assert.equal(parseDuration(' 1h 1m 1s '), 3661);
});

test('a bare number is seconds', () => {
  assert.equal(parseDuration('120'), 120);
  assert.equal(parseDuration('0'), 0);
});

test('not a string throws TypeError', () => {
  assert.throws(() => parseDuration(30), TypeError);
  assert.throws(() => parseDuration(null), TypeError);
});

test('invalid text throws RangeError', () => {
  for (const bad of ['', '   ', '5x', '1h1h', '-5m', '1.5h', 'h', '1h30', 'abc']) {
    assert.throws(() => parseDuration(bad), RangeError, `expected RangeError for ${JSON.stringify(bad)}`);
  }
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseDuration } from '../src/duration.js';

test('hidden: every unit once, any order', () => {
  assert.equal(parseDuration('1h30m15s'), 5415);
  assert.equal(parseDuration('15s30m1h'), 5415);
  assert.equal(parseDuration('1m1s1h'), 3661);
  assert.equal(parseDuration('0h'), 0);
  assert.equal(parseDuration('007s'), 7);
});

test('hidden: whitespace around and between parts', () => {
  assert.equal(parseDuration('\t5m\n'), 300);
  assert.equal(parseDuration('1h   30m'), 5400);
  assert.equal(parseDuration('  42  '), 42);
});

test('hidden: large values', () => {
  assert.equal(parseDuration('100h'), 360000);
  assert.equal(parseDuration('99999s'), 99999);
});

test('hidden: units are lower case only', () => {
  for (const bad of ['1H', '5M', '10S', '1h30M']) {
    assert.throws(() => parseDuration(bad), RangeError, `expected RangeError for ${JSON.stringify(bad)}`);
  }
});

test('hidden: signs, exponents, repeated and unknown units', () => {
  for (const bad of ['+5m', '1e3', '1h-5m', '1m1m', '5ms', '1d', '1h h', '1h30m5', '5toString', '1 .5h']) {
    assert.throws(() => parseDuration(bad), RangeError, `expected RangeError for ${JSON.stringify(bad)}`);
  }
});

test('hidden: other non-strings throw TypeError', () => {
  for (const bad of [undefined, NaN, {}, [], true, 5n]) {
    assert.throws(() => parseDuration(bad), TypeError);
  }
});

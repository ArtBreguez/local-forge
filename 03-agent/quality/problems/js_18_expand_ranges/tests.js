const test = require('node:test');
const assert = require('node:assert/strict');
const { expandRanges } = require('./solution.js');

test('examples', () => {
  assert.deepEqual(expandRanges('1-3,5,7-9'), [1, 2, 3, 5, 7, 8, 9]);
  assert.deepEqual(expandRanges('5,1-3,2'), [1, 2, 3, 5]);
  assert.deepEqual(expandRanges('1-5,3-8'), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('single numbers and one-element ranges', () => {
  assert.deepEqual(expandRanges('7'), [7]);
  assert.deepEqual(expandRanges('2-2'), [2]);
  assert.deepEqual(expandRanges('0'), [0]);
  assert.deepEqual(expandRanges('3,3,3'), [3]);
  assert.deepEqual(expandRanges('10-12'), [10, 11, 12]);
  assert.deepEqual(expandRanges('100'), [100]);
});

test('numbers sort numerically, not as text', () => {
  assert.deepEqual(expandRanges('10,9,100,8-9'), [8, 9, 10, 100]);
  assert.deepEqual(expandRanges('9-11,2'), [2, 9, 10, 11]);
});

test('whitespace around parts', () => {
  assert.deepEqual(expandRanges(' 1-3 , 5 '), [1, 2, 3, 5]);
  assert.deepEqual(expandRanges('\t4\n,\t6-7 '), [4, 6, 7]);
});

test('leading zeros', () => {
  assert.deepEqual(expandRanges('007'), [7]);
  assert.deepEqual(expandRanges('01-03'), [1, 2, 3]);
});

test('empty specs', () => {
  assert.deepEqual(expandRanges(''), []);
  assert.deepEqual(expandRanges('   '), []);
  assert.deepEqual(expandRanges('\n'), []);
});

test('results are numbers', () => {
  const out = expandRanges('1-2,4');
  assert.deepEqual(out.map((x) => typeof x), ['number', 'number', 'number']);
});

test('descending ranges throw RangeError', () => {
  assert.throws(() => expandRanges('5-3'), RangeError);
  assert.throws(() => expandRanges('1,9-2'), RangeError);
});

test('malformed parts throw SyntaxError', () => {
  for (const s of ['1,,2', '1,', ',1', ',', 'a', '1-', '-3', '1-2-3', '1.5', '1a', '--', '1,x-3', '1 2', '+4', '-']) {
    assert.throws(() => expandRanges(s), SyntaxError, JSON.stringify(s));
  }
});

test('non-strings throw TypeError', () => {
  for (const v of [null, undefined, 5, [1], {}]) {
    assert.throws(() => expandRanges(v), TypeError);
  }
});

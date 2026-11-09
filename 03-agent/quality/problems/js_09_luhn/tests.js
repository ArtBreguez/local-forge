const test = require('node:test');
const assert = require('node:assert/strict');
const { isValidCardNumber } = require('./solution.js');

test('well-known valid test numbers', () => {
  assert.equal(isValidCardNumber('4111111111111111'), true);
  assert.equal(isValidCardNumber('5500000000000004'), true);
  assert.equal(isValidCardNumber('378282246310005'), true);
  assert.equal(isValidCardNumber('6011111111111117'), true);
  assert.equal(isValidCardNumber('4012888888881881'), true);
});

test('wrong check digit', () => {
  assert.equal(isValidCardNumber('4111111111111112'), false);
  assert.equal(isValidCardNumber('5500000000000005'), false);
  assert.equal(isValidCardNumber('378282246310006'), false);
});

test('single transposition is detected', () => {
  assert.equal(isValidCardNumber('4111111111111111'.replace('41', '14')), false);
});

test('separators', () => {
  assert.equal(isValidCardNumber('4111 1111 1111 1111'), true);
  assert.equal(isValidCardNumber('4111-1111-1111-1111'), true);
  assert.equal(isValidCardNumber('4111 1111-1111 1111'), true);
  assert.equal(isValidCardNumber(' 4111111111111111 '), true);
  assert.equal(isValidCardNumber('3782 822463 10005'), true);
});

test('length limits', () => {
  assert.equal(isValidCardNumber('79927398713'), false);
  assert.equal(isValidCardNumber('100000000008'), true);
  assert.equal(isValidCardNumber('10000000000'), false);
  assert.equal(isValidCardNumber('4111111111111111110'), true);
  assert.equal(isValidCardNumber('4111111111111111111'), false);
  assert.equal(isValidCardNumber('1000000000000000009'), true);
  assert.equal(isValidCardNumber('41111111111111111115'), false);
  assert.equal(isValidCardNumber('0000000000000000'), true);
});

test('non-digit characters', () => {
  for (const s of ['4111a11111111111', '4111.1111.1111.1111', '4111_1111_1111_1111', '4111111111111111x', '+4111111111111111', '4111\t1111 1111 1111']) {
    assert.equal(isValidCardNumber(s), false, s);
  }
});

test('empty and separator-only input', () => {
  assert.equal(isValidCardNumber(''), false);
  assert.equal(isValidCardNumber('   '), false);
  assert.equal(isValidCardNumber('----'), false);
});

test('non-strings are false, never an exception', () => {
  for (const v of [4111111111111111, null, undefined, {}, ['4111111111111111'], true]) {
    assert.equal(isValidCardNumber(v), false);
  }
});

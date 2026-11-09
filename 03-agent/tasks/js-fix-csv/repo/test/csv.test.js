const test = require('node:test');
const assert = require('node:assert/strict');
const { parseCsvLine } = require('../src/csv.js');

test('plain fields', () => {
  assert.deepEqual(parseCsvLine('a,b,c'), ['a', 'b', 'c']);
});

test('quoted field with a comma', () => {
  assert.deepEqual(parseCsvLine('a,"b,c",d'), ['a', 'b,c', 'd']);
});

test('trailing comma and empty line', () => {
  assert.deepEqual(parseCsvLine('a,b,'), ['a', 'b', '']);
  assert.deepEqual(parseCsvLine(''), ['']);
});

test('escaped quotes', () => {
  assert.deepEqual(parseCsvLine('a,"say ""hi""",b'), ['a', 'say "hi"', 'b']);
});

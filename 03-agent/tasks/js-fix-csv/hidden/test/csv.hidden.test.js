const test = require('node:test');
const assert = require('node:assert/strict');
const { parseCsvLine } = require('../src/csv.js');

test('hidden: a field that is only an escaped quote', () => {
  assert.deepEqual(parseCsvLine('""""'), ['"']);
  assert.deepEqual(parseCsvLine('x,"""",y'), ['x', '"', 'y']);
});

test('hidden: escaped quotes at the start and the end of a field', () => {
  assert.deepEqual(parseCsvLine('"""start",mid,"end"""'), ['"start', 'mid', 'end"']);
});

test('hidden: several escaped quotes and commas together', () => {
  assert.deepEqual(parseCsvLine('"a ""b"", c ""d""",e'), ['a "b", c "d"', 'e']);
});

test('hidden: empty quoted field and empty fields', () => {
  assert.deepEqual(parseCsvLine('"",a,,"",'), ['', 'a', '', '', '']);
  assert.deepEqual(parseCsvLine(',,'), ['', '', '']);
});

test('hidden: behaviour that already worked', () => {
  assert.deepEqual(parseCsvLine('one'), ['one']);
  assert.deepEqual(parseCsvLine('"only,commas,here"'), ['only,commas,here']);
  assert.deepEqual(parseCsvLine('  spaced , out '), ['  spaced ', ' out ']);
});

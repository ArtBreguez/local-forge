const test = require('node:test');
const assert = require('node:assert/strict');
const { formatTable } = require('./solution.js');

test('the documented example', () => {
  assert.equal(
    formatTable([['name', 'qty'], ['apple', 3], ['kiwi', 12]]),
    ['name  | qty', 'apple | 3', 'kiwi  | 12'].join('\n'),
  );
});

test('three columns', () => {
  assert.equal(
    formatTable([['id', 'name', 'ok'], [1, 'Al', true], [200, 'Barbara', false]]),
    ['id  | name    | ok', '1   | Al      | true', '200 | Barbara | false'].join('\n'),
  );
});

test('no trailing spaces on any line', () => {
  const out = formatTable([['a', 'bbbbbb'], ['cccccc', 'd']]);
  for (const line of out.split('\n')) assert.equal(line, line.trimEnd());
  assert.equal(out, ['a      | bbbbbb', 'cccccc | d'].join('\n'));
});

test('ragged rows', () => {
  assert.equal(formatTable([['a', 'b', 'c'], ['d']]), ['a | b | c', 'd |   |'].join('\n'));
  assert.equal(formatTable([['x'], ['y', 'zz']]), ['x |', 'y | zz'].join('\n'));
});

test('null and undefined are empty', () => {
  assert.equal(formatTable([['a', null, 'c'], [undefined, 'b', 'dd']]), ['a |   | c', '  | b | dd'].join('\n'));
});

test('single column and single row', () => {
  assert.equal(formatTable([['one'], ['three'], ['x']]), ['one', 'three', 'x'].join('\n'));
  assert.equal(formatTable([['a', 'b']]), 'a | b');
});

test('empty input', () => {
  assert.equal(formatTable([]), '');
});

test('non-string cells', () => {
  assert.equal(formatTable([[1.5, false], [0, null]]), ['1.5 | false', '0   |'].join('\n'));
});

test('width is measured with length', () => {
  assert.equal(formatTable([['éé', 'x'], ['a', 'y']]), ['éé | x', 'a  | y'].join('\n'));
});

test('bad arguments', () => {
  assert.throws(() => formatTable('abc'), TypeError);
  assert.throws(() => formatTable(null), TypeError);
  assert.throws(() => formatTable([['a'], 'b']), TypeError);
  assert.throws(() => formatTable([1, 2]), TypeError);
});

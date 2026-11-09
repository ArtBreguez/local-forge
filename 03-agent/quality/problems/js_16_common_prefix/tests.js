const test = require('node:test');
const assert = require('node:assert/strict');
const { longestCommonPrefix } = require('./solution.js');

test('ordinary words', () => {
  assert.equal(longestCommonPrefix(['flower', 'flow', 'flight']), 'fl');
  assert.equal(longestCommonPrefix(['dog', 'racecar', 'car']), '');
  assert.equal(longestCommonPrefix(['interview', 'internet', 'interval', 'intern']), 'inter');
  assert.equal(longestCommonPrefix(['abc', 'abc']), 'abc');
  assert.equal(longestCommonPrefix(['abc', 'abcd', 'ab']), 'ab');
});

test('degenerate inputs', () => {
  assert.equal(longestCommonPrefix([]), '');
  assert.equal(longestCommonPrefix(['only']), 'only');
  assert.equal(longestCommonPrefix(['']), '');
  assert.equal(longestCommonPrefix(['', 'a']), '');
  assert.equal(longestCommonPrefix(['a', '']), '');
});

test('case-sensitive', () => {
  assert.equal(longestCommonPrefix(['Abc', 'abc']), '');
  assert.equal(longestCommonPrefix(['aBc', 'aBd']), 'aB');
});

test('surrogate pairs are never split', () => {
  assert.equal(longestCommonPrefix(['\u{1F600}a', '\u{1F601}a']), '');
  assert.equal(longestCommonPrefix(['\u{1F600}a', '\u{1F600}b']), '\u{1F600}');
  assert.equal(longestCommonPrefix(['a\u{1F600}', 'a\u{1F601}']), 'a');
  assert.equal(longestCommonPrefix(['\u{1F600}', '\u{1F600}']), '\u{1F600}');
  assert.equal(longestCommonPrefix(['x\u{1F600}\u{1F4A9}', 'x\u{1F600}\u{1F4AA}', 'x\u{1F600}']), 'x\u{1F600}');
});

test('results are always well-formed strings', () => {
  const out = longestCommonPrefix(['\u{1F600}a', '\u{1F601}a']);
  assert.equal(out.length, 0);
  const out2 = longestCommonPrefix(['\u{10000}z', '\u{10001}z', '\u{10000}y']);
  assert.equal(out2, '');
});

test('bad arguments', () => {
  assert.throws(() => longestCommonPrefix('abc'), TypeError);
  assert.throws(() => longestCommonPrefix(null), TypeError);
  assert.throws(() => longestCommonPrefix(['a', 1]), TypeError);
  assert.throws(() => longestCommonPrefix(['a', null]), TypeError);
  assert.throws(() => longestCommonPrefix([['a']]), TypeError);
});

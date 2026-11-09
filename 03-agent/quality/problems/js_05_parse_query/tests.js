const test = require('node:test');
const assert = require('node:assert/strict');
const solution = require('./solution.js');

// compare own properties only, so that a null-prototype result is as acceptable as an object literal
const parseQuery = (qs) => ({ ...solution.parseQuery(qs) });

test('basic pairs', () => {
  assert.deepEqual(parseQuery('a=1&b=2'), { a: '1', b: '2' });
  assert.deepEqual(parseQuery('?a=1&b=2'), { a: '1', b: '2' });
  assert.deepEqual(parseQuery(''), {});
  assert.deepEqual(parseQuery('?'), {});
});

test('repeated keys become arrays', () => {
  assert.deepEqual(parseQuery('a=1&a=2&a=3'), { a: ['1', '2', '3'] });
  assert.deepEqual(parseQuery('a=1&b=2&a=3'), { a: ['1', '3'], b: '2' });
  assert.deepEqual(parseQuery('a=&a='), { a: ['', ''] });
});

test('missing values and empty pairs', () => {
  assert.deepEqual(parseQuery('flag'), { flag: '' });
  assert.deepEqual(parseQuery('a=&b'), { a: '', b: '' });
  assert.deepEqual(parseQuery('a=1&&b=2&'), { a: '1', b: '2' });
  assert.deepEqual(parseQuery('&&'), {});
});

test('split at the first equals sign', () => {
  assert.deepEqual(parseQuery('a=b=c'), { a: 'b=c' });
  assert.deepEqual(parseQuery('token=abc==&x=1'), { token: 'abc==', x: '1' });
});

test('empty keys are skipped', () => {
  assert.deepEqual(parseQuery('=x&a=1'), { a: '1' });
  assert.deepEqual(parseQuery('=x'), {});
});

test('decoding', () => {
  assert.deepEqual(parseQuery('q=hello+world'), { q: 'hello world' });
  assert.deepEqual(parseQuery('q=a%2Bb'), { q: 'a+b' });
  assert.deepEqual(parseQuery('q=a%20b'), { q: 'a b' });
  assert.deepEqual(parseQuery('q=%E2%9C%93'), { q: '✓' });
  assert.deepEqual(parseQuery('my+key=1'), { 'my key': '1' });
  assert.deepEqual(parseQuery('a%5B0%5D=x'), { 'a[0]': 'x' });
  assert.deepEqual(parseQuery('q=100%25'), { q: '100%' });
  assert.deepEqual(parseQuery('q=%2B%2B+%2B'), { q: '++ +' });
});

test('malformed percent-encoding throws URIError', () => {
  assert.throws(() => solution.parseQuery('a=%zz'), URIError);
  assert.throws(() => solution.parseQuery('a=%E2%9C'), URIError);
  assert.throws(() => solution.parseQuery('a=%'), URIError);
  assert.throws(() => solution.parseQuery('%=1'), URIError);
});

test('keys that exist on Object.prototype are ordinary keys', () => {
  assert.deepEqual(parseQuery('constructor=1&toString=2'), { constructor: '1', toString: '2' });
  assert.deepEqual(parseQuery('hasOwnProperty=1&hasOwnProperty=2'), { hasOwnProperty: ['1', '2'] });
});

test('non-strings throw TypeError', () => {
  for (const v of [null, undefined, 5, {}, ['a=1']]) {
    assert.throws(() => solution.parseQuery(v), TypeError);
  }
});

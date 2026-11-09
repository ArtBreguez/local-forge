const test = require('node:test');
const assert = require('node:assert/strict');
const { slugify } = require('../src/slugify.js');

test('hidden: digits, underscores and symbols', () => {
  assert.equal(slugify('Top 10 tips_and-tricks'), 'top-10-tips-and-tricks');
  assert.equal(slugify('C++ & C#'), 'c-c');
  assert.equal(slugify('v2.0.1'), 'v2-0-1');
});

test('hidden: custom separator', () => {
  assert.equal(slugify('Hello World', { separator: '_' }), 'hello_world');
  assert.equal(slugify('  --a--b-- ', { separator: '.' }), 'a.b');
});

test('hidden: more accents', () => {
  assert.equal(slugify('Ação São João'), 'acao-sao-joao');
  assert.equal(slugify('Ünïcödé'), 'unicode');
});

test('hidden: maxLength cuts and removes a trailing separator', () => {
  assert.equal(slugify('hello world', { maxLength: 6 }), 'hello');
  assert.equal(slugify('hello world', { maxLength: 8 }), 'hello-wo');
  assert.equal(slugify('hello world', { maxLength: 100 }), 'hello-world');
  assert.equal(slugify('hello world', { maxLength: 5 }), 'hello');
});

test('hidden: errors', () => {
  for (const bad of [null, undefined, 42, ['a'], {}]) {
    assert.throws(() => slugify(bad), TypeError);
  }
  for (const bad of [0, -1, 2.5, '5', null]) {
    assert.throws(() => slugify('abc', { maxLength: bad }), RangeError);
  }
});

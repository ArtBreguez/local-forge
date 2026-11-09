const test = require('node:test');
const assert = require('node:assert/strict');
const { slugify } = require('../src/slugify.js');

test('basic words and punctuation', () => {
  assert.equal(slugify('Hello, World!'), 'hello-world');
  assert.equal(slugify('  many   spaces  '), 'many-spaces');
});

test('accents are removed', () => {
  assert.equal(slugify('Crème Brûlée'), 'creme-brulee');
});

test('nothing left gives an empty string', () => {
  assert.equal(slugify(''), '');
  assert.equal(slugify('!!!'), '');
});

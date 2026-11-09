const test = require('node:test');
const assert = require('node:assert/strict');
const { compile } = require('../src/pattern.js');

test('static and param segments', () => {
  assert.deepEqual(compile('/users/:id')('/users/7'), { id: '7' });
  assert.equal(compile('/users/:id')('/posts/7'), null);
  assert.equal(compile('/users/:id')('/users'), null);
  assert.deepEqual(compile('/a/b')('/a/b/'), {});
});

test('params are URI-decoded', () => {
  assert.deepEqual(compile('/t/:name')('/t/a%20b'), { name: 'a b' });
});

const test = require('node:test');
const assert = require('node:assert/strict');
const { Router } = require('../src/router.js');

test('dispatches, 404 and 405', () => {
  const r = new Router();
  r.add('get', '/users/:id', (p) => `user ${p.id}`).add('POST', '/users/:id', () => 'created');
  assert.deepEqual(r.handle('GET', '/users/7'), { status: 200, result: 'user 7' });
  assert.deepEqual(r.handle('DELETE', '/users/7'), { status: 405, allow: ['GET', 'POST'] });
  assert.deepEqual(r.handle('GET', '/nope'), { status: 404 });
});

test('wildcard routes', () => {
  const r = new Router();
  r.add('GET', '/files/*', (p) => p.splat);
  assert.deepEqual(r.handle('GET', '/files/docs/2024/a.txt'), { status: 200, result: 'docs/2024/a.txt' });
  assert.deepEqual(r.handle('GET', '/files'), { status: 404 });
});

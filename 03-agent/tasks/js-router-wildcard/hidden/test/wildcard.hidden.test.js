const test = require('node:test');
const assert = require('node:assert/strict');
const { compile } = require('../src/pattern.js');
const { Router } = require('../src/router.js');

test('hidden: splat is joined with / and each segment is decoded', () => {
  const m = compile('/files/*');
  assert.deepEqual(m('/files/a.txt'), { splat: 'a.txt' });
  assert.deepEqual(m('/files/my%20docs/a%2Bb.txt'), { splat: 'my docs/a+b.txt' });
  assert.deepEqual(m('/files//x///y/'), { splat: 'x/y' });
});

test('hidden: the wildcard needs at least one segment', () => {
  const m = compile('/files/*');
  assert.equal(m('/files'), null);
  assert.equal(m('/files/'), null);
  assert.equal(m('/other/a'), null);
  assert.equal(m('/'), null);
});

test('hidden: params next to a wildcard', () => {
  const m = compile('/users/:id/files/*');
  assert.deepEqual(m('/users/7/files/a/b'), { id: '7', splat: 'a/b' });
  assert.equal(m('/users/7/docs/a/b'), null);
  assert.equal(m('/users/7/files'), null);
});

test('hidden: a bare wildcard matches any non-empty path', () => {
  const m = compile('/*');
  assert.deepEqual(m('/a/b/c'), { splat: 'a/b/c' });
  assert.equal(m('/'), null);
});

test('hidden: * must be the last segment', () => {
  assert.throws(() => compile('/a/*/b'), Error);
  assert.throws(() => compile('/*/b'), Error);
  assert.doesNotThrow(() => compile('/a/b'));
});

test('hidden: router order, 405 and static behaviour are unchanged', () => {
  const r = new Router();
  r.add('GET', '/files/index', () => 'index');
  r.add('GET', '/files/*', (p) => `file ${p.splat}`);
  r.add('PUT', '/files/*', () => 'put');
  assert.deepEqual(r.handle('GET', '/files/index'), { status: 200, result: 'index' });
  assert.deepEqual(r.handle('GET', '/files/a/b'), { status: 200, result: 'file a/b' });
  assert.deepEqual(r.handle('POST', '/files/a'), { status: 405, allow: ['GET', 'PUT'] });
  assert.deepEqual(r.handle('GET', '/files'), { status: 404 });
  const first = new Router();
  first.add('GET', '/x/*', () => 'wild').add('GET', '/x/y', () => 'static');
  assert.equal(first.handle('GET', '/x/y').result, 'wild');
});

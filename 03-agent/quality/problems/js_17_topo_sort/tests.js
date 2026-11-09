const test = require('node:test');
const assert = require('node:assert/strict');
const { topoSort } = require('./solution.js');

test('chains', () => {
  assert.deepEqual(topoSort({ a: [], b: ['a'], c: ['b'] }), ['a', 'b', 'c']);
  assert.deepEqual(topoSort({ c: ['b'], b: ['a'], a: [] }), ['a', 'b', 'c']);
  assert.deepEqual(topoSort({ c: ['a', 'b'], b: ['a'], a: [] }), ['a', 'b', 'c']);
});

test('smallest ready name goes first at every step', () => {
  assert.deepEqual(topoSort({ x: [], a: [] }), ['a', 'x']);
  assert.deepEqual(topoSort({ b: [], a: ['b'], c: [] }), ['b', 'a', 'c']);
  assert.deepEqual(topoSort({ m: [], z: [], a: ['z'] }), ['m', 'z', 'a']);
  assert.deepEqual(topoSort({ d: ['b', 'c'], b: ['a'], c: ['a'], a: [] }), ['a', 'b', 'c', 'd']);
  assert.deepEqual(topoSort({ q: [], p: ['q'], r: ['q'], a: ['r'] }), ['q', 'p', 'r', 'a']);
});

test('names that only appear as dependencies are tasks too', () => {
  assert.deepEqual(topoSort({ a: ['z'] }), ['z', 'a']);
  assert.deepEqual(topoSort({ app: ['lib', 'util'], lib: ['util'] }), ['util', 'lib', 'app']);
});

test('duplicate dependencies count once', () => {
  assert.deepEqual(topoSort({ b: ['a', 'a'], a: [] }), ['a', 'b']);
});

test('independent tasks come out sorted', () => {
  assert.deepEqual(topoSort({ delta: [], alpha: [], charlie: [], bravo: [] }), ['alpha', 'bravo', 'charlie', 'delta']);
  assert.deepEqual(topoSort({ B: [], a: [], A: [], b: [] }), ['A', 'B', 'a', 'b']);
});

test('empty graph', () => {
  assert.deepEqual(topoSort({}), []);
});

test('cycles throw an error mentioning the cycle', () => {
  assert.throws(() => topoSort({ a: ['b'], b: ['a'] }), /cycle/i);
  assert.throws(() => topoSort({ a: ['a'] }), /cycle/i);
  assert.throws(() => topoSort({ a: [], b: ['a', 'c'], c: ['b'] }), /cycle/i);
  assert.throws(() => topoSort({ a: ['b'], b: ['c'], c: ['a'], d: [] }), /cycle/i);
});

test('input is not modified', () => {
  const g = { c: ['b', 'a'], b: ['a'], a: [] };
  const copy = JSON.parse(JSON.stringify(g));
  topoSort(g);
  assert.deepEqual(g, copy);
});

test('a longer realistic graph', () => {
  const g = {
    deploy: ['test', 'package'],
    package: ['build'],
    test: ['build', 'lint'],
    build: ['compile'],
    lint: [],
    compile: ['fetch'],
    fetch: [],
    docs: ['build'],
  };
  assert.deepEqual(topoSort(g), ['fetch', 'compile', 'build', 'docs', 'lint', 'package', 'test', 'deploy']);
});

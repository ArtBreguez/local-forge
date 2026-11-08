const test = require('node:test');
const assert = require('node:assert/strict');
const { render } = require('./solution.js');

test('simple substitution', () => {
  assert.equal(render('Hello {{name}}!', { name: 'Ann' }), 'Hello Ann!');
  assert.equal(render('{{a}}{{b}}', { a: 1, b: 2 }), '12');
  assert.equal(render('{{a}} and {{a}}', { a: 'x' }), 'x and x');
  assert.equal(render('no placeholders', { a: 1 }), 'no placeholders');
  assert.equal(render('', {}), '');
});

test('whitespace inside the braces', () => {
  assert.equal(render('{{ name }}', { name: 'Ann' }), 'Ann');
  assert.equal(render('{{\tname\n}}', { name: 'Ann' }), 'Ann');
  assert.equal(render('{{   user.name   }}', { user: { name: 'Bo' } }), 'Bo');
});

test('nested paths and arrays', () => {
  const data = { user: { name: 'Cy', address: { city: 'Oslo' } }, items: ['a', 'b', 'c'] };
  assert.equal(render('{{user.address.city}}', data), 'Oslo');
  assert.equal(render('{{items.1}}', data), 'b');
});

test('falsy values are rendered, null and undefined are empty', () => {
  assert.equal(render('[{{n}}]', { n: 0 }), '[0]');
  assert.equal(render('[{{f}}]', { f: false }), '[false]');
  assert.equal(render('[{{s}}]', { s: '' }), '[]');
  assert.equal(render('[{{x}}]', { x: null }), '[]');
  assert.equal(render('[{{x}}]', { x: undefined }), '[]');
  assert.equal(render('[{{n}}]', { n: NaN }), '[NaN]');
});

test('missing paths are empty', () => {
  assert.equal(render('[{{nope}}]', {}), '[]');
  assert.equal(render('[{{a.b.c}}]', { a: {} }), '[]');
  assert.equal(render('[{{a.b.c}}]', { a: null }), '[]');
  assert.equal(render('[{{a.b}}]', { a: 5 }), '[]');
  assert.equal(render('[{{a}}]', null), '[]');
  assert.equal(render('[{{a}}]', undefined), '[]');
});

test('only own properties are read', () => {
  assert.equal(render('[{{toString}}]', {}), '[]');
  assert.equal(render('[{{constructor}}]', {}), '[]');
  assert.equal(render('[{{__proto__}}]', {}), '[]');
  assert.equal(render('[{{a.hasOwnProperty}}]', { a: {} }), '[]');
  assert.equal(render('[{{toString}}]', { toString: 'mine' }), '[mine]');
});

test('invalid placeholders are left alone', () => {
  assert.equal(render('{{}}', {}), '{{}}');
  assert.equal(render('{{ a b }}', { a: 1, b: 2 }), '{{ a b }}');
  assert.equal(render('{{a}', { a: 1 }), '{{a}');
  assert.equal(render('{ {a}}', { a: 1 }), '{ {a}}');
  assert.equal(render('{{ }}', { a: 1 }), '{{ }}');
});

test('values are inserted literally', () => {
  assert.equal(render('{{x}}', { x: '<b>&amp;</b>' }), '<b>&amp;</b>');
  assert.equal(render('{{x}}', { x: '$& and $1 and $$' }), '$& and $1 and $$');
  assert.equal(render('{{x}}', { x: '{{y}}', y: 'no' }), '{{y}}');
});

test('numbers and objects are converted with String()', () => {
  assert.equal(render('{{n}}', { n: 1.5 }), '1.5');
  assert.equal(render('{{a}}', { a: [1, 2] }), '1,2');
  assert.equal(render('{{b}}', { b: true }), 'true');
});

test('template must be a string', () => {
  for (const t of [null, undefined, 5, {}, ['x']]) {
    assert.throws(() => render(t, {}), TypeError);
  }
});

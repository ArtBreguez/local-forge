const test = require('node:test');
const assert = require('node:assert/strict');
const { camelToSnake } = require('./solution.js');

test('simple words', () => {
  assert.equal(camelToSnake('camelCase'), 'camel_case');
  assert.equal(camelToSnake('PascalCase'), 'pascal_case');
  assert.equal(camelToSnake('simple'), 'simple');
  assert.equal(camelToSnake('aB'), 'a_b');
});

test('acronyms', () => {
  assert.equal(camelToSnake('parseHTTPResponse'), 'parse_http_response');
  assert.equal(camelToSnake('userID'), 'user_id');
  assert.equal(camelToSnake('XMLHttpRequest'), 'xml_http_request');
  assert.equal(camelToSnake('HTML'), 'html');
  assert.equal(camelToSnake('ABc'), 'a_bc');
  assert.equal(camelToSnake('iOS'), 'i_os');
});

test('digits', () => {
  assert.equal(camelToSnake('getV2Value'), 'get_v2_value');
  assert.equal(camelToSnake('HTTP2Server'), 'http2_server');
  assert.equal(camelToSnake('A1B2'), 'a1_b2');
  assert.equal(camelToSnake('item2'), 'item2');
});

test('already snake or odd characters', () => {
  assert.equal(camelToSnake('already_snake'), 'already_snake');
  assert.equal(camelToSnake('already_snakeCase'), 'already_snake_case');
  assert.equal(camelToSnake('with-dash'), 'with-dash');
  assert.equal(camelToSnake(''), '');
  assert.equal(camelToSnake('a'), 'a');
  assert.equal(camelToSnake('A'), 'a');
});

test('non-strings throw TypeError', () => {
  for (const v of [null, undefined, 5, {}, ['a']]) {
    assert.throws(() => camelToSnake(v), TypeError);
  }
});

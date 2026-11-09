const test = require('node:test');
const assert = require('node:assert/strict');
const { Emitter } = require('./solution.js');

test('listeners run in registration order with arguments', () => {
  const e = new Emitter();
  const log = [];
  e.on('x', (...a) => log.push(['first', ...a]));
  e.on('x', (...a) => log.push(['second', ...a]));
  assert.equal(e.emit('x', 1, 2), true);
  assert.deepEqual(log, [['first', 1, 2], ['second', 1, 2]]);
});

test('emit returns false without listeners', () => {
  const e = new Emitter();
  assert.equal(e.emit('nothing'), false);
  e.on('a', () => {});
  assert.equal(e.emit('b'), false);
});

test('this is the emitter', () => {
  const e = new Emitter();
  let seen;
  e.on('x', function () { seen = this; });
  e.emit('x');
  assert.equal(seen, e);
});

test('on and once are chainable', () => {
  const e = new Emitter();
  assert.equal(e.on('a', () => {}), e);
  assert.equal(e.once('a', () => {}), e);
  assert.equal(e.off('a', () => {}), e);
});

test('once runs only once', () => {
  const e = new Emitter();
  let n = 0;
  e.once('x', () => { n += 1; });
  assert.equal(e.emit('x'), true);
  assert.equal(e.emit('x'), false);
  assert.equal(n, 1);
  assert.equal(e.listenerCount('x'), 0);
});

test('same function registered twice runs twice', () => {
  const e = new Emitter();
  let n = 0;
  const f = () => { n += 1; };
  e.on('x', f).on('x', f);
  assert.equal(e.listenerCount('x'), 2);
  e.emit('x');
  assert.equal(n, 2);
});

test('off removes all registrations including once', () => {
  const e = new Emitter();
  let n = 0;
  const f = () => { n += 1; };
  e.on('x', f).once('x', f).on('x', f);
  e.off('x', f);
  assert.equal(e.listenerCount('x'), 0);
  assert.equal(e.emit('x'), false);
  assert.equal(n, 0);
});

test('off of a once listener by its original function', () => {
  const e = new Emitter();
  let n = 0;
  const f = () => { n += 1; };
  e.once('x', f);
  e.off('x', f);
  e.emit('x');
  assert.equal(n, 0);
});

test('off leaves other listeners and other events alone', () => {
  const e = new Emitter();
  const log = [];
  const a = () => log.push('a');
  const b = () => log.push('b');
  e.on('x', a).on('x', b).on('y', a);
  e.off('x', a);
  e.off('x', () => {});
  e.off('never', a);
  e.emit('x');
  e.emit('y');
  assert.deepEqual(log, ['b', 'a']);
});

test('removing a listener during emit does not skip the next one', () => {
  const e = new Emitter();
  const log = [];
  const a = () => { log.push('a'); e.off('x', a); };
  const b = () => log.push('b');
  const c = () => log.push('c');
  e.on('x', a).on('x', b).on('x', c);
  e.emit('x');
  assert.deepEqual(log, ['a', 'b', 'c']);
  log.length = 0;
  e.emit('x');
  assert.deepEqual(log, ['b', 'c']);
});

test('a listener removed by an earlier listener still runs in the current emit', () => {
  const e = new Emitter();
  const log = [];
  const b = () => log.push('b');
  e.on('x', () => { log.push('a'); e.off('x', b); });
  e.on('x', b);
  e.emit('x');
  assert.deepEqual(log, ['a', 'b']);
  log.length = 0;
  e.emit('x');
  assert.deepEqual(log, ['a']);
});

test('a listener added during emit runs from the next emit', () => {
  const e = new Emitter();
  const log = [];
  e.on('x', () => {
    log.push('a');
    e.on('x', () => log.push('late'));
  });
  e.emit('x');
  assert.deepEqual(log, ['a']);
  log.length = 0;
  e.emit('x');
  assert.deepEqual(log, ['a', 'late']);
});

test('re-emitting from a once listener does not call it again', () => {
  const e = new Emitter();
  let n = 0;
  e.once('x', () => { n += 1; if (n < 5) e.emit('x'); });
  e.emit('x');
  assert.equal(n, 1);
});

test('any value can be an event name', () => {
  const e = new Emitter();
  const sym = Symbol('s');
  const obj = {};
  const log = [];
  e.on(sym, () => log.push('sym')).on(obj, () => log.push('obj')).on(0, () => log.push('zero')).on('0', () => log.push('str'));
  e.emit(sym); e.emit(obj); e.emit(0); e.emit('0');
  assert.deepEqual(log, ['sym', 'obj', 'zero', 'str']);
});

test('listener must be a function', () => {
  const e = new Emitter();
  assert.throws(() => e.on('x', 'nope'), TypeError);
  assert.throws(() => e.once('x'), TypeError);
});

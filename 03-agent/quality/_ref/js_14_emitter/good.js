class Emitter {
  constructor() {
    this._events = new Map();
  }

  _add(event, fn, once) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    if (!this._events.has(event)) this._events.set(event, []);
    this._events.get(event).push({ fn, once });
    return this;
  }

  on(event, fn) {
    return this._add(event, fn, false);
  }

  once(event, fn) {
    return this._add(event, fn, true);
  }

  off(event, fn) {
    const list = this._events.get(event);
    if (list) {
      const kept = list.filter((l) => l.fn !== fn);
      if (kept.length) this._events.set(event, kept);
      else this._events.delete(event);
    }
    return this;
  }

  emit(event, ...args) {
    const list = this._events.get(event);
    if (!list || list.length === 0) return false;
    const snapshot = list.slice();
    const kept = list.filter((l) => !l.once);
    if (kept.length) this._events.set(event, kept);
    else this._events.delete(event);
    for (const l of snapshot) l.fn.apply(this, args);
    return true;
  }

  listenerCount(event) {
    const list = this._events.get(event);
    return list ? list.length : 0;
  }
}

module.exports = { Emitter };

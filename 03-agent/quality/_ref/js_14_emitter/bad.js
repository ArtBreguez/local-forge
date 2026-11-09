class Emitter {
  constructor() {
    this._events = new Map();
  }

  on(event, fn) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    if (!this._events.has(event)) this._events.set(event, []);
    this._events.get(event).push(fn);
    return this;
  }

  once(event, fn) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    const wrapper = (...args) => {
      this.off(event, wrapper);
      return fn.apply(this, args);
    };
    return this.on(event, wrapper);
  }

  off(event, fn) {
    const list = this._events.get(event);
    if (list) {
      const i = list.indexOf(fn);
      if (i !== -1) list.splice(i, 1);
    }
    return this;
  }

  emit(event, ...args) {
    const list = this._events.get(event);
    if (!list || list.length === 0) return false;
    for (const fn of list) fn.apply(this, args);
    return true;
  }

  listenerCount(event) {
    const list = this._events.get(event);
    return list ? list.length : 0;
  }
}

module.exports = { Emitter };

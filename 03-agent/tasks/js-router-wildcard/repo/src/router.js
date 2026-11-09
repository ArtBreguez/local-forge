const { compile } = require('./pattern.js');

class Router {
  constructor() {
    this.routes = [];
  }

  add(method, pattern, handler) {
    this.routes.push({ method: method.toUpperCase(), pattern, match: compile(pattern), handler });
    return this;
  }

  handle(method, path) {
    const verb = method.toUpperCase();
    const allowed = new Set();
    for (const route of this.routes) {
      const params = route.match(path);
      if (params === null) continue;
      if (route.method === verb) return { status: 200, result: route.handler(params) };
      allowed.add(route.method);
    }
    if (allowed.size > 0) return { status: 405, allow: [...allowed].sort() };
    return { status: 404 };
  }
}

module.exports = { Router };

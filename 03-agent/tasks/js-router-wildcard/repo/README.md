# router

A tiny URL router.

    const { Router } = require('./src/router.js');
    const r = new Router();
    r.add('GET', '/users/:id', (params) => `user ${params.id}`);
    r.handle('GET', '/users/7');   // { status: 200, result: 'user 7' }

`src/pattern.js` turns a pattern such as `/users/:id` into a matching function; `src/router.js` keeps the routes and
answers `handle(method, path)` with `{ status: 200, result }`, `{ status: 404 }` or `{ status: 405, allow: [...] }`.

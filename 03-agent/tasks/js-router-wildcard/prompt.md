`src/` holds a small URL router (read `README.md` and the code first). Add wildcard routes:

- A pattern may end with a `*` segment: `/files/*` matches `/files/a.txt` and `/files/docs/2024/a.txt`, and the matched rest of the path is returned in the params as `splat`, joined with `/` and with each segment URI-decoded (`/files/my%20docs/a.txt` gives `splat` `"my docs/a.txt"`). The wildcard needs at least one segment: `/files` and `/files/` do not match.
- Other `:name` params of the pattern are still returned next to `splat` (`/users/:id/files/*`).
- A `*` anywhere but in the last segment makes `compile` throw an `Error`.
- Everything else behaves as before: routes are tried in the order they were added, and a path that matches a pattern under another method gives `405`.

The visible tests are in `test/` (run them with `node --test`); more tests that you cannot see check the same behaviour. Do not modify anything under `test/`.

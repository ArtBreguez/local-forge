Add priorities to the task queue: `src/queue.js` and `src/report.js` both need a change.

- `push(name, priority = 0)`: `priority` must be an integer (negative values are allowed), otherwise throw `RangeError`. Still returns the queue.
- `pop()` removes and returns the name of the task with the highest priority; among tasks with the same priority, the one pushed first comes first. It returns `undefined` when the queue is empty.
- `items()` returns `{ name, priority }` objects in the order in which the tasks would be popped, and does not change the queue.
- `describe(queue)` in `src/report.js` prints one line per task as before (`"1. write docs"`); a task whose priority is not 0 gets `[p<priority>]` between the number and the name: `"1. [p5] deploy"`, `"3. [p-1] cleanup"`.

The visible tests are in `test/` (run them with `node --test`); more tests that you cannot see check the same behaviour. Do not modify anything under `test/`.

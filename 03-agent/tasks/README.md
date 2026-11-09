# Phase 4 task suite

Twelve small tasks, six Python and six JavaScript, standard library only (`python -m unittest`, `node --test`). Each task is a directory:

| file | what it is |
|---|---|
| `task.json` | id, language, kind, difficulty, the command the agent is told to run, the visible and the hidden test files, whether the visible tests fail at the start (`failing`) or already pass (`green`, the refactor) |
| `prompt.md` | what the agent is told; identical for every arm |
| `repo/` | the starting repository, including the visible tests (these are protected: byte-identical or the run fails) |
| `hidden/` | the hidden tests, laid over a copy of the agent's directory by the runner after the run; the agent never has them |
| `solution/` | files that, laid over `repo/`, make every test pass |
| `broken/` | files that, laid over `repo/`, still fail a hidden test |

`python 03-agent/harness/taskbench.py validate` runs the three promises for all tasks (`tests/test_agent.py` does the same): the reference passes the visible and the hidden tests, the broken solution fails at least one hidden test, the starting repository is in its stated state and fails the hidden tests. The hidden tests are a superset of what the visible ones check: they add the edge cases the prompt already specifies in words (rounding, ties, empty input, error types), so a solution that satisfies the prompt passes them.

| id | kind | difficulty | what the agent has to do |
|---|---|---|---|
| `js-duration` | function | medium | `parseDuration` (the task of OpenCode run 1, 5 visible tests, 6 hidden groups) |
| `py-wrap` | function | medium | greedy word wrap with long-word splitting |
| `py-intervals` | function | easy | merge overlapping or touching intervals |
| `js-slugify` | function | easy | slug with accents, separator, max length |
| `js-lru` | function | medium | an LRU cache class |
| `py-fix-paginate` | bugfix | easy | a failing visible test; the bug is a floor division |
| `js-fix-csv` | bugfix | medium | a failing visible test; doubled quotes inside a quoted field |
| `py-discount` | two-file | medium | per-line discounts in `shop/cart.py` and `shop/receipt.py` |
| `js-queue` | two-file | medium | priorities in `src/queue.js` and `src/report.js` |
| `py-refactor-split` | refactor | medium | split `util.py` into `parsing.py` and `formatting.py`; the visible tests are green and must stay green |
| `py-todo-tags` | codebase | hard | read a 5-file todo list, add tags to model, commands and CLI |
| `js-router-wildcard` | codebase | hard | read a small router, add `*` wildcard routes |

Success of a run (`03-agent/harness/runner.py`): every visible and hidden test passes, the visible test files are byte-identical, and the agent finished inside the time limit.

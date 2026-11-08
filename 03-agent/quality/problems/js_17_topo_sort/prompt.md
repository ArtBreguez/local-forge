Write a function `topoSort(graph)` that orders tasks so that every task comes after the tasks it depends on.

`graph` is a plain object. Each key is a task name and its value is an array of the names of tasks that must come **before** it. A name that appears only inside a dependency list (never as a key) is also a task, with no dependencies of its own. Duplicate names in a dependency list count once.

Return an array containing every task exactly once. Build the order step by step: at each step, among all tasks whose dependencies have already been placed, pick the one whose name is smallest in plain string order (`<`), and place it. An empty graph gives `[]`.

If the dependencies contain a cycle (including a task that depends on itself), throw an `Error` whose message contains the word `cycle`. The input must not be modified.

Examples: `{a: [], b: ['a'], c: ['b']}` -> `['a','b','c']`; `{b: [], a: ['b'], c: []}` -> `['b','a','c']`; `{a: ['z']}` -> `['z','a']`.

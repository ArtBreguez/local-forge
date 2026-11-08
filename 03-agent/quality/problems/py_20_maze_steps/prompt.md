Write a function `maze_steps(grid)` that returns the length of the shortest path through a maze.

`grid` is a list of equal-length strings. `S` marks the start, `E` the exit, `#` a wall and `.` an open cell. You may move one cell at a time up, down, left or right (no diagonals) and only through open cells, `S` and `E`. Return the minimum number of moves needed to get from `S` to `E`, or `-1` if `E` cannot be reached.

Example: `["S.#", "..#", "#.E"]` -> `4`.

Raise `ValueError` if the rows are not all the same length, if any character other than `S E # .` appears, or if there is not exactly one `S` and exactly one `E`.

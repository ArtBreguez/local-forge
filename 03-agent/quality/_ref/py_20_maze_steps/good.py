from collections import deque


def maze_steps(grid):
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("rows must be non-empty and the same length")
    start = end = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch not in "SE#.":
                raise ValueError(f"bad cell {ch!r}")
            if ch == "S":
                if start is not None:
                    raise ValueError("more than one S")
                start = (r, c)
            elif ch == "E":
                if end is not None:
                    raise ValueError("more than one E")
                end = (r, c)
    if start is None or end is None:
        raise ValueError("need exactly one S and one E")
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == end:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#" and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return -1

def maze_steps(grid):
    start = end = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "E":
                end = (r, c)
    if start is None or end is None:
        raise ValueError("need S and E")
    seen = set()

    def dfs(r, c, d):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] == "#" or (r, c) in seen:
            return -1
        if (r, c) == end:
            return d
        seen.add((r, c))
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            res = dfs(r + dr, c + dc, d + 1)
            if res != -1:
                return res
        return -1

    return dfs(start[0], start[1], 0)

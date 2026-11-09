def merge_intervals(intervals):
    cleaned = []
    for item in intervals:
        start, end = item
        if start > end:
            raise ValueError(f"invalid interval: {item!r}")
        cleaned.append((start, end))
    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged

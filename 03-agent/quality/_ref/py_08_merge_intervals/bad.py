def merge_intervals(intervals):
    pairs = []
    for iv in intervals:
        start, end = iv
        if start > end:
            raise ValueError(f"invalid interval {iv!r}")
        pairs.append((start, end))
    pairs.sort()
    merged = []
    for start, end in pairs:
        if merged and start < merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged

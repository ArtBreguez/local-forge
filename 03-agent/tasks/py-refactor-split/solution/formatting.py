"""Formatting helpers."""
from parsing import DEFAULT_SEP, UNITS


def format_size(n):
    """10240 -> '10k'; only exact multiples of a unit are shortened, anything else stays a plain number."""
    for unit in ("G", "M", "k"):
        if n >= UNITS[unit] and n % UNITS[unit] == 0:
            return f"{n // UNITS[unit]}{unit}"
    return str(n)


def format_kv(mapping, sep=DEFAULT_SEP):
    """One 'key=value' line per item, sorted by key."""
    return "\n".join(f"{key}{sep}{value}" for key, value in sorted(mapping.items()))


def format_table(rows, headers):
    """Left-aligned text table: header line, a dashes line, then the rows (columns separated by two spaces)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)

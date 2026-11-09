"""Small text utilities for the report tool."""

DEFAULT_SEP = "="
UNITS = {"k": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def parse_kv(line, sep=DEFAULT_SEP):
    """Split 'key=value' into ('key', 'value'), both stripped. ValueError without the separator or with an empty key."""
    if sep not in line:
        raise ValueError(f"missing {sep!r} in {line!r}")
    key, value = line.split(sep, 1)
    key = key.strip()
    if not key:
        raise ValueError("empty key")
    return key, value.strip()


def parse_kv_block(text, sep=DEFAULT_SEP):
    """Parse many lines into a dict; blank lines and lines starting with '#' are skipped, later keys win."""
    result = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = parse_kv(stripped, sep)
        result[key] = value
    return result


def parse_size(text):
    """'10k' -> 10240, '2M' -> 2097152, '512' -> 512 (k, M and G are powers of 1024)."""
    text = text.strip()
    if not text:
        raise ValueError("empty size")
    if text[-1] in UNITS:
        number, factor = text[:-1], UNITS[text[-1]]
    else:
        number, factor = text, 1
    if not number.isdigit():
        raise ValueError(f"bad size: {text!r}")
    return int(number) * factor


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

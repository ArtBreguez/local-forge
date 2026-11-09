"""Parsing helpers."""

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
